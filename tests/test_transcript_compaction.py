"""Lossless transcript compaction + the agent-harness integration."""
from __future__ import annotations

import json

import pytest

from agent_harness.context import ContextManager
from agent_harness.llm import ChatError
from agent_harness.loop import run_loop
from agent_harness.tools import ApprovalPolicy, ToolRegistry, ToolResult, function_tool

from context_runtime.compression.transcript import STUB, Archive, TranscriptCompactor
from context_runtime.integrations.agent_harness import ContextRuntimeManager


def call(i, name, **args):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def convo(n_outputs=8, size=3000):
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "TASK"}]
    originals = {}
    for i in range(n_outputs):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [call(i, "bash", command=f"cmd{i}")]})
        text = f"output {i} " + ("lorem ipsum dolor " * (size // 18)) + f" UNIQUETOKEN{i}xyz"
        originals[i] = text
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": text})
    return msgs, originals


def test_manager_satisfies_the_agent_harness_interface():
    assert isinstance(TranscriptCompactor(), ContextManager)
    assert isinstance(ContextRuntimeManager(), ContextManager)


def test_compaction_is_lossless_and_keeps_the_recent_outputs():
    msgs, originals = convo()
    tc = TranscriptCompactor()
    changed = tc.prepare(msgs, target_tokens=2000, estimated_tokens=30000)
    assert changed > 0
    tool = [m for m in msgs if m["role"] == "tool"]
    stubs = [m for m in tool if STUB.match(m["content"])]
    assert len(stubs) == changed and all("search_context" in m["content"] and "bash output" in m["content"] for m in stubs)
    assert [m["content"] for m in tool[-4:]] == [originals[i] for i in range(4, 8)] or len(stubs) <= 4   # recent kept while enough
    # every archived original is recoverable byte for byte
    for m in stubs:
        aid = STUB.match(m["content"]).group(1)
        assert tc.archive.get(aid) in originals.values()
    assert msgs[0]["content"] == "SYS" and msgs[1]["content"] == "TASK"      # system and task untouched
    assert tc.prepare(msgs, target_tokens=10**9, estimated_tokens=0) == 0     # nothing to do under target


def test_stops_as_soon_as_the_target_is_met_oldest_first():
    msgs, _ = convo()
    tc = TranscriptCompactor()
    tc.prepare(msgs, target_tokens=100, estimated_tokens=700)   # one item saves ~890 tokens: one is enough
    stubbed = [i for i, m in enumerate(msgs) if m["role"] == "tool" and STUB.match(m["content"])]
    assert stubbed == [3]                                                       # only the oldest tool output


def test_large_tool_call_arguments_are_archived_and_json_stays_valid():
    big = "DOC " * 1000
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
    for i in range(4):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [call(i, "write_file", path=f"p{i}", content=big + str(i))]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "wrote"})
    tc = TranscriptCompactor()
    assert tc.prepare(msgs, target_tokens=10, estimated_tokens=10_000) >= 2
    first = json.loads(msgs[2]["tool_calls"][0]["function"]["arguments"])
    assert first["path"] == "p0" and first["content"].startswith("[archived A")
    assert tc.archive.get("A0001") == big + "0"
    assert json.loads(msgs[-2]["tool_calls"][0]["function"]["arguments"])["content"] == big + "3"   # recent call intact


def test_tools_search_and_read_back_archived_text():
    msgs, originals = convo()
    tc = TranscriptCompactor()
    tc.prepare(msgs, target_tokens=100, estimated_tokens=50_000)
    search, read = tc.tools()
    hit = search.run({"query": "UNIQUETOKEN2xyz"})
    assert hit.ok and "UNIQUETOKEN2xyz" in hit.text and hit.hits
    aid = STUB.match(msgs[3 + 2 * 2]["content"]).group(1)                        # the output of call 2
    full = read.run({"id": aid, "limit": 12000})
    assert full.text.startswith("output 2 ")
    sliced = read.run({"id": aid, "offset": 10, "limit": 20})
    assert len(sliced.text.split("\n[...")[0]) == 20 and "read_archive" in sliced.text
    bad = read.run({"id": "A9999"})
    assert not bad.ok and "unknown archive id" in bad.error


def test_context_errors_escalate_and_stay_lossless():
    msgs, originals = convo(n_outputs=6, size=900)
    tc = TranscriptCompactor()
    n = [tc.on_context_error(msgs, a) for a in (1, 2, 3)]
    assert n[0] > 0 and sum(n) >= n[0]
    assert all(STUB.match(m["content"]) for m in msgs if m["role"] == "tool")   # attempt 3 archives even the newest
    assert set(tc.archive.entries) and all(len(tc.archive.get(a)) > 0 for a in tc.archive.entries)


def test_index_files_makes_inputs_searchable(tmp_path):
    (tmp_path / "data.csv").write_text("id,name\n1,Zanzibar\n")
    (tmp_path / "img.png").write_bytes(b"\x89PNG")
    mgr = ContextRuntimeManager(tmp_path / "arch")
    assert mgr.index_files(tmp_path) == 1
    assert "Zanzibar" in mgr._search({"query": "Zanzibar"}).text


class Scripted:
    def __init__(self, replies, prompt_tokens):
        self.replies, self.i, self.prompt_tokens, self.seen = list(replies), 0, prompt_tokens, []

    def complete(self, messages, tools=None):
        self.seen.append((json.loads(json.dumps(messages)), [t["function"]["name"] for t in tools or []]))
        r = self.replies[self.i]
        self.i += 1
        if isinstance(r, Exception):
            raise r
        return {"message": r, "finish_reason": "stop", "usage": {"prompt_tokens": self.prompt_tokens, "completion_tokens": 5}}


def test_inside_the_real_agent_loop_the_model_can_recover_compacted_output():
    reg = ToolRegistry(ApprovalPolicy("bypass"))
    reg.register(function_tool("dump", lambda a: ToolResult(text=f"BIG{a['n']} " + "filler " * 700 + f" SECRET{a['n']}"),
                               parameters={"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}))
    mgr = ContextRuntimeManager()
    replies = [{"role": "assistant", "content": "", "tool_calls": [call(i, "dump", n=i)]} for i in range(6)]
    replies += [{"role": "assistant", "content": "", "tool_calls": [call(99, "search_context", query="SECRET1")]},
                {"role": "assistant", "content": "done"}]
    client = Scripted(replies, prompt_tokens=900)
    res = run_loop(client, reg, "sys", "task", max_context_tokens=1000, max_tool_chars=12000, context_manager=mgr)
    assert res.stop == "done"
    names = client.seen[0][1]
    assert "search_context" in names and "read_archive" in names and "dump" in names
    last_msgs = client.seen[-1][0]
    assert any(STUB.match(m["content"]) for m in last_msgs if m["role"] == "tool")              # old output was compacted
    assert any("SECRET1" in m["content"] and not STUB.match(m["content"]) for m in last_msgs if m["role"] == "tool")   # and recovered


def test_server_context_error_is_handled_by_the_manager_in_the_loop():
    reg = ToolRegistry(ApprovalPolicy("bypass"))
    reg.register(function_tool("dump", lambda a: ToolResult(text="x" * 5000), parameters={"type": "object", "properties": {}}))
    mgr = ContextRuntimeManager()
    err = ChatError("HTTP 400: This model's maximum context length is 40960 tokens")
    replies = [{"role": "assistant", "content": "", "tool_calls": [call(i, "dump")]} for i in range(6)] + [err, {"role": "assistant", "content": "ok"}]
    client = Scripted(replies, prompt_tokens=10)
    res = run_loop(client, reg, "s", "t", max_context_tokens=10**9, context_manager=mgr)
    assert res.stop == "done" and mgr.archive.entries


def test_oversized_tool_output_is_archived_whole_at_admission_so_the_middle_is_recoverable():
    tc = TranscriptCompactor()
    text = "HEAD " + "x" * 5000 + " NEEDLEINTHEMIDDLE " + "y" * 5000 + " TAIL"
    shown = tc.admit("bash", text, 1000)
    assert len(shown) < 1400 and shown.startswith("HEAD") and shown.endswith("TAIL") and "A0001" in shown
    assert "NEEDLEINTHEMIDDLE" not in shown and tc.archive.get("A0001") == text
    assert "NEEDLEINTHEMIDDLE" in tc._search({"query": "NEEDLEINTHEMIDDLE"}).text
    assert tc.admit("bash", "short", 1000) == "short" and len(tc.archive.entries) == 1


@pytest.mark.skipif(not __import__("os").environ.get("CR_TEST_RAG"), reason="downloads an embedding model; set CR_TEST_RAG=1")
def test_redevops_rag_backend_retrieves_paraphrases_from_the_archive(tmp_path):
    pytest.importorskip("redevops_rag")
    mgr = ContextRuntimeManager(tmp_path / "arch", use_redevops_rag=True)
    mgr.archive.add("Quarterly revenue by region: EMEA 4.2M, APAC 3.1M, Americas 7.9M.", kind="tool_output", tool="bash", turn=1)
    mgr.archive.add("Incident postmortem: database failover took 14 minutes after the storage upgrade.", kind="tool_output", tool="read", turn=1)
    search = mgr.tools()[0]
    assert "A0001" in search.run({"query": "sales per area of the world", "k": 1}).text      # no shared keywords: vector leg
    assert "A0002" in search.run({"query": "why was the outage so long", "k": 1}).text
