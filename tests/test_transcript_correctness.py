import json

import pytest

from context_runtime.compression.transcript import Archive, TranscriptCompactor
from context_runtime.integrations.agent_harness import ContextRuntimeManager

CSV = "\n".join(f"2025{i:03d},Applicant Number {i},WA,status pending background check notes row {i}" for i in range(12))
BIG = "\n".join(f"line {i} of some bulky tool output that is not an input file {i * 7}" for i in range(400))


def call(i, name="bash", args=None):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {})}}]}


def tool(i, content):
    return {"role": "tool", "tool_call_id": f"c{i}", "content": content}


def convo(bodies, args_for=None):
    m = [{"role": "system", "content": "s"}, {"role": "user", "content": "t"}]
    for i, b in enumerate(bodies):
        m += [call(i, args=(args_for or {}).get(i)), tool(i, b)]
    return m


def shown(messages):
    return [m["content"] for m in messages if m["role"] == "tool"]


# -- defaults are the legacy behaviour --------------------------------------------------------------------------------
def test_default_policy_is_legacy_and_its_argument_stub_is_unchanged():
    c = TranscriptCompactor(Archive())
    assert c.policy == "legacy"
    msgs = convo(["r"] * 6, {0: {"path": "/tmp/a.py", "content": "z" * 2000}})
    c.prepare(msgs, target_tokens=1, estimated_tokens=10**6)
    arg = json.loads(msgs[2]["tool_calls"][0]["function"]["arguments"])["content"]
    assert arg.startswith("[archived A0001: 2000 chars] ") and arg.endswith('... (read_archive("A0001"))')


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError):
        TranscriptCompactor(Archive(), policy="v3")


# -- v2 argument stub ---------------------------------------------------------------------------------------------------
def test_v2_argument_stub_says_it_is_not_content_and_the_original_stays_in_the_archive():
    c = TranscriptCompactor(Archive(), policy="v2")
    doc = "import openpyxl\n" + "x = 1\n" * 500
    msgs = convo(["r"] * 6, {0: {"path": "/tmp/a.py", "content": doc}})
    c.prepare(msgs, target_tokens=1, estimated_tokens=10**6)
    args = json.loads(msgs[2]["tool_calls"][0]["function"]["arguments"])                # still valid JSON, other args untouched
    assert args["path"] == "/tmp/a.py" and args["content"].startswith("<<ARCHIVED-PLACEHOLDER A0001: bash.content")
    assert "NOT the content" in args["content"] and 'read_archive("A0001")' in args["content"] and len(args["content"]) < 400
    assert c.archive.get("A0001") == doc                                                  # lossless


def test_both_stub_generations_are_recognised_by_the_managers_own_detector_which_the_harness_asks():
    """The stub FORMAT is owned here. The harness no longer carries a regex for it: it asks ``is_placeholder`` (agent_harness.context.ContextAware)."""
    c = TranscriptCompactor(Archive(), policy="v2")
    msgs = convo(["r"] * 6, {0: {"content": "q" * 3000}})
    c.prepare(msgs, target_tokens=1, estimated_tokens=10**6)
    assert c.is_placeholder(json.loads(msgs[2]["tool_calls"][0]["function"]["arguments"])["content"])
    legacy = TranscriptCompactor(Archive())
    m2 = convo(["r"] * 6, {0: {"content": "q" * 3000}})
    legacy.prepare(m2, target_tokens=1, estimated_tokens=10**6)
    assert legacy.is_placeholder(json.loads(m2[2]["tool_calls"][0]["function"]["arguments"])["content"])   # the legacy stub too
    ctx = pytest.importorskip("agent_harness.context")
    if hasattr(ctx, "ContextAware"):
        assert isinstance(c, ctx.ContextAware)                                         # structurally satisfies the harness's optional extension


# -- pins ---------------------------------------------------------------------------------------------------------------
def test_pinned_text_is_never_archived_by_prepare_but_other_old_results_are():
    instr = "TASK INSTRUCTIONS: build the three-sheet workbook and write a PDF report. Reference files: a.csv b.csv"
    c = TranscriptCompactor(Archive())
    c.pin_text(instr)
    msgs = convo(["$ cat TASK_INSTRUCTIONS.txt\n" + instr + "\n" + "pad " * 200, BIG, BIG, BIG, BIG, BIG, BIG, BIG])
    c.prepare(msgs, target_tokens=1, estimated_tokens=10**6)
    out = shown(msgs)
    assert "TASK INSTRUCTIONS" in out[0] and not out[0].startswith("[archived")
    assert out[1].startswith("[archived") and out[2].startswith("[archived")


def test_pins_yield_in_the_last_resort_so_a_run_cannot_die_on_them():
    instr = "TASK INSTRUCTIONS: build the three-sheet workbook and write a PDF report. Reference files: a.csv b.csv"
    c = TranscriptCompactor(Archive())
    c.pin_text(instr)
    msgs = convo([instr + "\n" + "pad " * 400, BIG, BIG])
    c.on_context_error(msgs, 1)
    assert "TASK INSTRUCTIONS" in shown(msgs)[0]                                          # attempts 1-2 honour the pin
    c.on_context_error(msgs, 3)
    assert shown(msgs)[0].startswith("[archived")                                          # attempt 3 releases it


# -- input-evidence pinning -------------------------------------------------------------------------------------------
def manager(tmp_path, **kw):
    inp = tmp_path / "task_folder"
    inp.mkdir(parents=True, exist_ok=True)
    (inp / "applications.csv").write_text(CSV)
    m = ContextRuntimeManager(tmp_path / "arch", **kw)
    assert m.index_files(inp) == 1
    return m


def test_small_input_reads_are_kept_when_pinning_is_on_and_archived_when_it_is_off(tmp_path):
    for flag, expect_kept in ((True, True), (False, False)):
        m = manager(tmp_path / str(flag), pin_input_reads=flag)
        msgs = convo(["$ cat applications.csv\n" + CSV, BIG, BIG, BIG, BIG, BIG, BIG, BIG])
        m.prepare(msgs, target_tokens=20000, estimated_tokens=10**6)
        first = shown(msgs)[0]
        assert (not first.startswith("[archived")) == expect_kept, (flag, first[:60])
        assert shown(msgs)[1].startswith("[archived")                                    # bulky non-input output goes either way


def test_pinning_respects_its_budget_and_size_limit(tmp_path):
    m = manager(tmp_path, pin_input_reads=True, pin_input_max_chars=100, pin_budget_fraction=0.25)
    msgs = convo(["$ cat applications.csv\n" + CSV, BIG, BIG, BIG, BIG, BIG, BIG, BIG])
    m.prepare(msgs, target_tokens=20000, estimated_tokens=10**6)
    assert shown(msgs)[0].startswith("[archived")                                          # larger than pin_input_max_chars: not pinned
    m2 = manager(tmp_path / "b", pin_input_reads=True, pin_budget_fraction=0.0001)
    msgs = convo(["$ cat applications.csv\n" + CSV, BIG, BIG, BIG, BIG, BIG, BIG, BIG])
    m2.prepare(msgs, target_tokens=20000, estimated_tokens=10**6)
    assert shown(msgs)[0].startswith("[archived")                                          # over the pinned budget: stays compactable


def test_an_unrelated_small_result_is_not_mistaken_for_an_input_read(tmp_path):
    m = manager(tmp_path, pin_input_reads=True)
    other = "\n".join(f"completely different tool output line number {i} with enough text to count" for i in range(20))
    msgs = convo([other, BIG, BIG, BIG, BIG, BIG, BIG, BIG])
    m.prepare(msgs, target_tokens=20000, estimated_tokens=10**6)
    assert shown(msgs)[0].startswith("[archived")


def test_everything_removed_is_still_archived_verbatim_with_pins_on(tmp_path):
    m = manager(tmp_path, pin_input_reads=True, policy="v2")
    bodies = ["$ cat applications.csv\n" + CSV, BIG + "a", BIG + "b", BIG + "c", BIG + "d", BIG + "e", BIG + "f", BIG + "g"]
    msgs = convo(bodies)
    m.prepare(msgs, target_tokens=20000, estimated_tokens=10**6)
    for original, now in zip(bodies, shown(msgs)):
        if now.startswith("[archived"):
            aid = now.split()[1].rstrip(":")
            assert m.archive.get(aid) == original
