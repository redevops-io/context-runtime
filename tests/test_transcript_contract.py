"""The compactor's model-visible contract: what the harness may ask (is_placeholder, visibility) and recovery reads staying visible."""
import json

from context_runtime.compression.contract import STATE_ARCHIVED, STATE_CLIPPED, STATE_UNAVAILABLE, STATE_VISIBLE, is_placeholder, result_view
from context_runtime.compression.transcript import Archive, TranscriptCompactor
from context_runtime.integrations.agent_harness import ContextRuntimeManager

BIG = "\n".join(f"row {i} with bulky tool output text {i * 13}" for i in range(300))


def call(i, name="bash", args=None):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args or {})}}]}


def tool(i, content):
    return {"role": "tool", "tool_call_id": f"c{i}", "content": content}


def convo(items):
    m = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
    for i, (name, body) in enumerate(items):
        m += [call(i, name), tool(i, body)]
    return m


def test_visibility_reports_visible_archived_clipped_and_unavailable():
    c = ContextRuntimeManager(None, policy="v2")
    msgs = convo([("bash", BIG)] + [("bash", "short result")] * 6)
    assert c.visibility(msgs, "c0").state == STATE_VISIBLE
    c.prepare(msgs, target_tokens=1, estimated_tokens=10**6)
    v = c.visibility(msgs, "c0")
    assert v.state == STATE_ARCHIVED and v.archive_id == "A0001" and v.recoverable
    assert c.visibility(msgs, "c6").state == STATE_VISIBLE                      # the newest outputs stay verbatim
    assert c.visibility(msgs, "nope").state == STATE_UNAVAILABLE
    clipped = c.admit("web_fetch", BIG, 2000)
    assert result_view([tool(9, clipped)], "c9").state == STATE_CLIPPED and result_view([tool(9, clipped)], "c9").archive_id is not None


def test_visibility_looks_at_the_current_messages_not_at_history():
    c = ContextRuntimeManager(None)
    msgs = convo([("bash", BIG)])
    before = c.visibility(msgs, "c0")
    msgs[3] = {**msgs[3], "content": "[archived A0042: bash output, 9999 chars. Starts: \"x\" Full text kept: read_archive(\"A0042\").]"}
    assert before.state == STATE_VISIBLE and c.visibility(msgs, "c0").archive_id == "A0042"


def test_is_placeholder_recognises_both_stub_generations_and_nothing_else():
    c = ContextRuntimeManager(None)
    msgs = convo([("write_file", "ok")] + [("bash", "x")] * 5)
    msgs[2]["tool_calls"][0]["function"]["arguments"] = json.dumps({"path": "/w/a.py", "content": "z" * 3000})
    v2 = TranscriptCompactor(Archive(), policy="v2")
    v2.prepare(msgs, target_tokens=1, estimated_tokens=10**6)
    stub = json.loads(msgs[2]["tool_calls"][0]["function"]["arguments"])["content"]
    assert stub.startswith("<<ARCHIVED-PLACEHOLDER") and c.is_placeholder(stub) and is_placeholder(stub)
    assert c.is_placeholder("[archived A0003: 2000 chars] import x ... (read_archive(\"A0003\"))")
    assert not c.is_placeholder("print('hello')") and not c.is_placeholder("a ARCHIVED-PLACEHOLDER mention in the middle") and not c.is_placeholder(5)


def test_recovery_reads_are_not_archived_again_for_the_next_tool_results():
    c = ContextRuntimeManager(None, recovery_keep=6)
    msgs = convo([("bash", BIG), ("read_archive", BIG)] + [("bash", "r" * 700)] * 5)
    c.prepare(msgs, target_tokens=1, estimated_tokens=10**6)
    assert c.visibility(msgs, "c1").state == STATE_VISIBLE                      # the read_archive result is protected
    assert c.visibility(msgs, "c0").state == STATE_ARCHIVED                     # an ordinary old output is not
    more = msgs + [m for i in range(10, 17) for m in (call(i), tool(i, "s" * 700))]
    c.prepare(more, target_tokens=1, estimated_tokens=10**6)
    assert c.visibility(more, "c1").state == STATE_ARCHIVED                     # protection ends after recovery_keep further results
    off = ContextRuntimeManager(None, recovery_keep=0)
    m2 = convo([("bash", BIG), ("read_archive", BIG)] + [("bash", "r" * 700)] * 5)
    off.prepare(m2, target_tokens=1, estimated_tokens=10**6)
    assert off.visibility(m2, "c1").state == STATE_ARCHIVED


def test_read_archive_text_states_how_long_the_slice_stays():
    c = ContextRuntimeManager(None, recovery_keep=6)
    aid = c.archive.add("x" * 100, kind="tool_output", tool="bash", turn=1)
    out = c._read({"id": aid})
    assert "next 6 tool results" in out.text


def test_visibility_names_the_full_text_when_a_clipped_result_was_archived_later():
    c = ContextRuntimeManager(None)
    clipped = c.admit("web_fetch", BIG, 2000)                      # full text archived as A0001; the conversation holds head + tail + note
    msgs = convo([("web_fetch", clipped)] + [("bash", "r" * 700)] * 6)
    assert c.visibility(msgs, "c0").state == STATE_CLIPPED and c.visibility(msgs, "c0").archive_id == "A0001"
    c.prepare(msgs, target_tokens=1, estimated_tokens=10**6)         # the clipped message itself is now archived too (A0002) and replaced by a stub
    v = c.visibility(msgs, "c0")
    assert v.state == STATE_ARCHIVED and v.archive_id == "A0001"       # not the intermediate stub: the id that holds the complete original
    assert c.archive.get(v.archive_id) == BIG
