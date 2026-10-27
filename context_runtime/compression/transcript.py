"""Transcript compaction that archives instead of discarding (SPEC §5.5, principle 3).

An agent loop appends tool output and tool-call arguments (often whole documents) to the conversation. Near the
model's window something has to give. The rule here: **nothing is lost**. Before an old item is replaced by a
short stub, its full text is written to an ``Archive`` (one file per item, content addressed by an id) and
indexed by a retriever. The stub names the id, the size and the first characters, and the manager offers two
read-only tools so the model can get the content back:

    search_context(query, k)   retrieval over everything archived (BM25 offline, redevops-rag when available)
    read_archive(id, offset)   the exact original text, in slices

``TranscriptCompactor`` satisfies ``agent_harness.context.ContextManager`` structurally; this module does not
import ``agent_harness`` at module load except for the tool contract.
"""
from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..types import Hit
from ..tools.base import ToolPlugin, ToolResult, function_tool
from .contract import CLIPPED_NOTE, ResultView, is_placeholder as _is_placeholder, result_view

_CHARS_PER_TOKEN = 3          # same conservative estimate agent_harness uses
RECOVERY_TOOLS = ("read_archive", "search_context")
STUB = re.compile(r"^\[archived (A\d+)")
_CHUNK = 1200


@dataclass
class ArchiveEntry:
    id: str
    kind: str                 # tool_output | tool_args
    tool: str
    turn: int
    chars: int
    derived_from: tuple[str, ...] = ()
    path: str = ""


class Archive:
    """Append-only store of everything compaction removed. Files on disk are the source of truth."""

    def __init__(self, directory: str | Path | None = None, retriever: Any | None = None):
        self.directory = Path(directory) if directory else Path(tempfile.mkdtemp(prefix="cr-archive-"))
        self.directory.mkdir(parents=True, exist_ok=True)
        if retriever is None:
            from ..adapters.store_inmemory import InMemoryStore
            retriever = InMemoryStore(source="archive")
        self.retriever = retriever
        self.entries: dict[str, ArchiveEntry] = {}
        self._dirty = False

    def add(self, text: str, *, kind: str, tool: str, turn: int, derived_from: tuple[str, ...] = ()) -> str:
        aid = f"A{len(self.entries) + 1:04d}"
        path = self.directory / f"{aid}__{kind}__{re.sub(r'[^A-Za-z0-9_-]', '_', tool)[:40]}.txt"
        path.write_text(text, encoding="utf-8")
        self.entries[aid] = ArchiveEntry(aid, kind, tool, turn, len(text), derived_from, str(path))
        docs = getattr(self.retriever, "docs", None)
        if isinstance(docs, list):                       # in-memory BM25: add passages directly
            for i in range(0, max(len(text), 1), _CHUNK):
                docs.append({"chunk_id": f"{aid}#{i // _CHUNK}", "filename": f"{aid} ({tool}, turn {turn})",
                             "text": text[i:i + _CHUNK], "created_at": None})
        else:                                            # file-indexing retriever (redevops-rag): re-index lazily
            self._dirty = True
        return aid

    def get(self, aid: str) -> str:
        e = self.entries[aid]
        return Path(e.path).read_text(encoding="utf-8")

    def search(self, query: str, k: int = 5) -> list[Hit]:
        if self._dirty:
            self.retriever.index(str(self.directory))
            self._dirty = False
        return self.retriever.search(query, k, "hybrid")


def _head(text: str, n: int) -> str:
    return " ".join(text[:n].split())


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _lines(text: str) -> set[str]:
    return {_norm(l) for l in text.splitlines() if len(l.strip()) > 20}


class TranscriptCompactor:
    """``policy="legacy"`` (default) is the behaviour the stage-1 benchmarks ran. ``policy="v2"`` changes only the stub that replaces a
    long tool-call ARGUMENT: it is worded so it cannot be mistaken for content (a model was seen to copy the legacy stub into a new
    ``write_file``). Pins and input-evidence pinning are opt-in and apply under either policy."""

    def __init__(self, archive: Archive | None = None, *, keep_recent_outputs: int = 4, keep_recent_calls: int = 2,
                 min_archive_chars: int = 600, stub_head_chars: int = 240, arg_limit: int = 800, policy: str = "legacy",
                 pin_input_reads: bool = False, pin_input_max_chars: int = 8000, pin_budget_fraction: float = 0.25,
                 recovery_keep: int = 6):
        if policy not in ("legacy", "v2"):
            raise ValueError(f"unknown policy {policy!r}")
        self.archive = archive or Archive()
        self.keep_recent_outputs, self.keep_recent_calls = keep_recent_outputs, keep_recent_calls
        self.min_archive_chars, self.stub_head_chars, self.arg_limit = min_archive_chars, stub_head_chars, arg_limit
        self.policy = policy
        self.pin_input_reads, self.pin_input_max_chars, self.pin_budget_fraction = pin_input_reads, pin_input_max_chars, pin_budget_fraction
        self._pin_keys: list[str] = []
        self._input_lines: dict[str, set[str]] = {}
        self._turn = 0
        self.recovery_keep = recovery_keep

    # ---- pins ----------------------------------------------------------------------------------------------------
    def pin_text(self, text: str) -> None:
        """A tool result that contains the start of ``text`` (whitespace-insensitive) is never archived by ``prepare``."""
        key = _norm(text)[:160]
        if len(key) >= 20 and key not in self._pin_keys:
            self._pin_keys.append(key)

    def _is_pinned(self, content: str) -> bool:
        if not self._pin_keys or not isinstance(content, str):
            return False
        head = _norm(content[:5000])
        return any(k in head for k in self._pin_keys)

    def _input_line_sets(self) -> list[set[str]]:
        for aid, e in self.archive.entries.items():
            if e.kind == "input_file" and aid not in self._input_lines:
                self._input_lines[aid] = _lines(self.archive.get(aid))
        return list(self._input_lines.values())

    def _is_input_read(self, content: str) -> bool:
        """A short tool result that is (mostly) the text of an indexed input file, i.e. the agent reading its evidence."""
        if STUB.match(content) or len(content) > self.pin_input_max_chars:
            return False
        got = _lines(content)
        if len(got) < 2:
            return False
        return any(len(got & il) / len(got) >= 0.7 for il in self._input_line_sets())

    def _evidence_pins(self, messages: list[dict[str, Any]], target_tokens: int) -> set[int]:
        """Indices of small input-file reads to keep verbatim, newest first, within ``pin_budget_fraction`` of the target."""
        if not self.pin_input_reads:
            return set()
        budget = int(self.pin_budget_fraction * target_tokens * _CHARS_PER_TOKEN)
        pins, used = set(), 0
        for i in range(len(messages) - 1, -1, -1):
            m = messages[i]
            c = m.get("content")
            if m.get("role") == "tool" and isinstance(c, str) and self._is_input_read(c):
                if used + len(c) > budget:
                    continue                                # over budget: this (older) one stays compactable
                pins.add(i)
                used += len(c)
        return pins

    # ---- contract: what the model can see (owned here, asked by the harness) -----------------------------------------
    def is_placeholder(self, value: Any) -> bool:
        """Whether ``value`` is one of this manager's stand-ins for removed content (so a harness can refuse to run an action built from it)."""
        return _is_placeholder(value)

    def visibility(self, messages: list[dict[str, Any]], tool_call_id: str) -> ResultView:
        """visible | clipped | archived (with the archive id) | unavailable, for the tool result of ``tool_call_id`` in the current messages."""
        v = result_view(messages, tool_call_id)
        if v.state == "archived" and v.archive_id in self.archive.entries:       # a clipped result that was archived later: the full text is the id its note names
            m = CLIPPED_NOTE.search(self.archive.get(v.archive_id)[-800:])
            if m and m.group(1) in self.archive.entries and m.group(1) != v.archive_id:
                return ResultView("archived", m.group(1), self.archive.entries[m.group(1)].chars)
        return v

    # ---- ContextManager ----------------------------------------------------------------------------------------
    def prepare(self, messages: list[dict[str, Any]], *, target_tokens: int, estimated_tokens: int) -> int:
        """Archive the oldest compactable items until the estimate is at or under ``target_tokens``.

        If the normal protections (the newest outputs and calls stay verbatim, small items are left alone) leave the
        estimate above target, the protections are relaxed in two further steps. Still lossless: everything removed is
        archived. Only the system prompt, the task and the single newest tool output are never touched here; going
        beyond that is ``on_context_error``'s job.
        """
        self._turn += 1
        est, changed = estimated_tokens, 0
        self._excluded = self._evidence_pins(messages, target_tokens)
        steps = [(self.keep_recent_outputs, self.keep_recent_calls, self.min_archive_chars, self.arg_limit, self.stub_head_chars),
                 (2, 1, 300, 400, 120), (1, 1, 150, 200, 60)]       # relaxed steps also use shorter stub previews
        for keep_out, keep_calls, min_chars, arg_limit, head in steps:
            for kind, i, extra in list(self._candidates(messages, keep_out, keep_calls, min_chars, arg_limit)):
                if est <= target_tokens:
                    return changed
                saved = self._compact(messages, kind, i, extra, head)
                if saved:
                    est -= saved // _CHARS_PER_TOKEN
                    changed += 1
            if est <= target_tokens:
                break
        return changed

    def on_context_error(self, messages: list[dict[str, Any]], attempt: int) -> int:
        """Server said the prompt is too long: archive more aggressively (still lossless)."""
        keep_out, keep_calls, min_chars, arg_limit = {1: (2, 1, 300, 400), 2: (1, 1, 150, 200)}.get(attempt, (0, 0, 0, 120))
        changed = 0
        saved_pins = self._pin_keys
        if attempt >= 3:                                  # last resort: honour nothing, a run must not die on a pin
            self._pin_keys = []
        try:
            for kind, i, extra in self._candidates(messages, keep_out, keep_calls, min_chars, arg_limit, for_prepare=False):
                if self._compact(messages, kind, i, extra):
                    changed += 1
        finally:
            self._pin_keys = saved_pins
        if attempt >= 3:                                  # last resort: archive and clip any remaining large message
            for i, m in enumerate(messages):
                c = m.get("content")
                if i > 1 and isinstance(c, str) and len(c) > 1500 and not STUB.match(c):
                    aid = self.archive.add(c, kind="tool_output" if m.get("role") == "tool" else "message",
                                           tool=m.get("role", "?"), turn=self._turn)
                    messages[i] = {**m, "content": f"[archived {aid}: {len(c)} chars] " + _head(c, 700)
                                   + f" ... (rest kept: read_archive(\"{aid}\"))"}
                    changed += 1
        return changed

    def admit(self, tool: str, text: str, max_chars: int) -> str:
        """Every tool result passes here before entering the conversation. An oversized one is archived whole and
        the conversation gets its head and tail plus the id, so even the middle can be retrieved later."""
        if len(text) <= max_chars:
            return text
        aid = self.archive.add(text, kind="tool_output", tool=tool, turn=self._turn)
        note = f"\n[... {len(text) - max_chars} chars in the middle archived as {aid}: search_context(query) or read_archive(\"{aid}\") ...]\n"
        head = max_chars * 2 // 3
        return text[:head] + note + text[-(max_chars - head):]

    def tools(self) -> list[ToolPlugin]:
        s = {"type": "string"}
        return [
            function_tool(
                "search_context", self._search,
                "Search content that was removed from the conversation to save space (earlier tool outputs, "
                "large file contents). Use specific words from what you need.",
                {"type": "object", "properties": {"query": s, "k": {"type": "integer"}}, "required": ["query"]}),
            function_tool(
                "read_archive", self._read,
                "Read the exact original text of an archived item by id (e.g. A0003), in slices.",
                {"type": "object", "properties": {"id": s, "offset": {"type": "integer"}, "limit": {"type": "integer"}},
                 "required": ["id"]}),
        ]

    # ---- internals ---------------------------------------------------------------------------------------------
    def _search(self, args: dict) -> ToolResult:
        hits = self.archive.search(str(args["query"]), int(args.get("k", 5)))
        if not hits:
            return ToolResult(text="no archived content matches; try other words or read_archive(id)")
        return ToolResult(text="\n\n".join(f"[{h.chunk_id}] {h.filename}\n{h.text}" for h in hits), hits=list(hits))

    def _read(self, args: dict) -> ToolResult:
        aid = str(args["id"])
        if aid not in self.archive.entries:
            return ToolResult(ok=False, error=f"unknown archive id {aid!r}; known: {', '.join(self.archive.entries) or 'none'}")
        text, off, lim = self.archive.get(aid), max(0, int(args.get("offset", 0))), max(1, min(int(args.get("limit", 6000)), 12000))
        more = f"\n[... {len(text) - off - lim} more chars: read_archive(\"{aid}\", offset={off + lim})]" if off + lim < len(text) else ""
        head = (f"[{aid}: chars {off}-{min(off + lim, len(text))} of {len(text)}. This slice stays visible for your next {self.recovery_keep} tool "
                f"results, then it is archived again: use what you need from it now (note it or write it to a file).]\n")
        return ToolResult(text=head + text[off:off + lim] + more)

    @staticmethod
    def _tool_names(messages: list[dict[str, Any]]) -> dict[str, str]:
        names: dict[str, str] = {}
        for m in messages:
            for c in m.get("tool_calls") or []:
                names[c["id"]] = c["function"]["name"]
        return names

    def _candidates(self, messages, keep_out, keep_calls, min_chars, arg_limit, for_prepare: bool = True):
        names = self._tool_names(messages)
        outs = [i for i, m in enumerate(messages) if m.get("role") == "tool"
                and isinstance(m.get("content"), str) and not STUB.match(m["content"]) and len(m["content"]) >= max(min_chars, 1)]
        all_tool = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        protected = set(all_tool[-keep_out:]) if keep_out else set()
        if for_prepare and self.recovery_keep:           # evidence the model just asked for (read_archive / search_context) is not archived again straight away
            protected |= {i for i in all_tool[-self.recovery_keep:] if names.get(messages[i].get("tool_call_id", "")) in RECOVERY_TOOLS}
        calls = [i for i, m in enumerate(messages) if m.get("role") == "assistant" and m.get("tool_calls")]
        protected_calls = set(calls[-keep_calls:]) if keep_calls else set()
        keep = getattr(self, "_excluded", set()) if for_prepare else set()
        items = [(i, "out", names.get(messages[i].get("tool_call_id", ""), "tool")) for i in outs
                 if i not in protected and i not in keep and not self._is_pinned(messages[i]["content"])]
        for i in calls:
            if i in protected_calls:
                continue
            if any(isinstance(v, str) and len(v) > arg_limit for c in messages[i]["tool_calls"]
                   for v in _args(c).values()):
                items.append((i, "args", ""))
        for i, kind, extra in sorted(items):               # oldest first
            yield kind, i, extra

    def _compact(self, messages, kind: str, i: int, tool: str, head: int | None = None) -> int:
        head = self.stub_head_chars if head is None else head
        m = messages[i]
        if kind == "out":
            text = m["content"]
            aid = self.archive.add(text, kind="tool_output", tool=tool, turn=self._turn)
            stub = (f"[archived {aid}: {tool} output, {len(text)} chars. Starts: \"{_head(text, head)}\" "
                    f"Full text kept: search_context(query) or read_archive(\"{aid}\").]")
            messages[i] = {**m, "content": stub}
            return len(text) - len(stub)
        saved, calls = 0, []
        for c in m["tool_calls"]:
            args = _args(c)
            small = {}
            for k, v in args.items():
                if isinstance(v, str) and len(v) > self.arg_limit:
                    aid = self.archive.add(v, kind="tool_args", tool=f"{c['function']['name']}.{k}", turn=self._turn)
                    if self.policy == "v2":
                        small[k] = (f"<<ARCHIVED-PLACEHOLDER {aid}: {c['function']['name']}.{k}, {len(v)} chars. This is NOT the content: "
                                    f"never reuse this text as an argument. Original: read_archive(\"{aid}\"). Starts: {_head(v, 80)}>>")
                    else:
                        small[k] = f"[archived {aid}: {len(v)} chars] " + _head(v, 200) + f" ... (read_archive(\"{aid}\"))"
                    saved += len(v) - len(small[k])
                else:
                    small[k] = v
            calls.append({**c, "function": {**c["function"], "arguments": json.dumps(small)}})
        messages[i] = {**m, "tool_calls": calls}
        return saved


def _args(call: dict) -> dict:
    try:
        a = json.loads(call["function"].get("arguments") or "{}")
    except ValueError:
        return {}
    return a if isinstance(a, dict) else {}
