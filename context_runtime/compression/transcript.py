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

_CHARS_PER_TOKEN = 3          # same conservative estimate agent_harness uses
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


class TranscriptCompactor:
    def __init__(self, archive: Archive | None = None, *, keep_recent_outputs: int = 4, keep_recent_calls: int = 2,
                 min_archive_chars: int = 600, stub_head_chars: int = 240, arg_limit: int = 800):
        self.archive = archive or Archive()
        self.keep_recent_outputs, self.keep_recent_calls = keep_recent_outputs, keep_recent_calls
        self.min_archive_chars, self.stub_head_chars, self.arg_limit = min_archive_chars, stub_head_chars, arg_limit
        self._turn = 0

    # ---- ContextManager ----------------------------------------------------------------------------------------
    def prepare(self, messages: list[dict[str, Any]], *, target_tokens: int, estimated_tokens: int) -> int:
        """Archive the oldest compactable items until the estimate is at or under ``target_tokens``."""
        self._turn += 1
        est, changed = estimated_tokens, 0
        for kind, i, extra in self._candidates(messages, self.keep_recent_outputs, self.keep_recent_calls,
                                               self.min_archive_chars, self.arg_limit):
            if est <= target_tokens:
                break
            saved = self._compact(messages, kind, i, extra)
            if saved:
                est -= saved // _CHARS_PER_TOKEN
                changed += 1
        return changed

    def on_context_error(self, messages: list[dict[str, Any]], attempt: int) -> int:
        """Server said the prompt is too long: archive more aggressively (still lossless)."""
        keep_out, keep_calls, min_chars, arg_limit = {1: (2, 1, 300, 400), 2: (1, 1, 150, 200)}.get(attempt, (0, 0, 0, 120))
        changed = 0
        for kind, i, extra in self._candidates(messages, keep_out, keep_calls, min_chars, arg_limit):
            if self._compact(messages, kind, i, extra):
                changed += 1
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
        return ToolResult(text=text[off:off + lim] + more)

    @staticmethod
    def _tool_names(messages: list[dict[str, Any]]) -> dict[str, str]:
        names: dict[str, str] = {}
        for m in messages:
            for c in m.get("tool_calls") or []:
                names[c["id"]] = c["function"]["name"]
        return names

    def _candidates(self, messages, keep_out, keep_calls, min_chars, arg_limit):
        names = self._tool_names(messages)
        outs = [i for i, m in enumerate(messages) if m.get("role") == "tool"
                and isinstance(m.get("content"), str) and not STUB.match(m["content"]) and len(m["content"]) >= max(min_chars, 1)]
        all_tool = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        protected = set(all_tool[-keep_out:]) if keep_out else set()
        calls = [i for i, m in enumerate(messages) if m.get("role") == "assistant" and m.get("tool_calls")]
        protected_calls = set(calls[-keep_calls:]) if keep_calls else set()
        items = [(i, "out", names.get(messages[i].get("tool_call_id", ""), "tool")) for i in outs if i not in protected]
        for i in calls:
            if i in protected_calls:
                continue
            if any(isinstance(v, str) and len(v) > arg_limit for c in messages[i]["tool_calls"]
                   for v in _args(c).values()):
                items.append((i, "args", ""))
        for i, kind, extra in sorted(items):               # oldest first
            yield kind, i, extra

    def _compact(self, messages, kind: str, i: int, tool: str) -> int:
        m = messages[i]
        if kind == "out":
            text = m["content"]
            aid = self.archive.add(text, kind="tool_output", tool=tool, turn=self._turn)
            stub = (f"[archived {aid}: {tool} output, {len(text)} chars. Starts: \"{_head(text, self.stub_head_chars)}\" "
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
