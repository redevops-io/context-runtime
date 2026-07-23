"""Governance seam: Bedrock Guardrails + CloudWatch telemetry + the neutral GuardedModel wrapper.

All exercised with fakes — no boto3. Proves the neutral Protocols (Guardrail, TelemetryReader) work
and that GuardedModel composes a guardrail with any ModelPlugin.
"""
from context_runtime.providers.base import Guardrail, GuardrailVerdict, TelemetryReader
from context_runtime.providers.guarded_model import GuardedModel, GuardrailBlocked
from context_runtime.types import ModelRequest, ModelResult


# ── Bedrock Guardrail over a fake apply_guardrail ───────────────────────────────────────────────
class FakeBedrockGuard:
    def __init__(self, action="NONE", outputs=None, assessments=None):
        self.action, self.outputs, self.assessments = action, outputs or [], assessments or []
        self.calls = []

    def apply_guardrail(self, **kw):
        self.calls.append(kw)
        return {"action": self.action, "outputs": self.outputs, "assessments": self.assessments}


def test_bedrock_guardrail_allows_clean_text():
    from context_runtime.providers.aws.guardrail import BedrockGuardrail
    g = BedrockGuardrail(client=FakeBedrockGuard(action="NONE"), guardrail_id="g1")
    assert isinstance(g, Guardrail)
    assert g.check_input("hello").allowed is True


def test_bedrock_guardrail_blocks_and_reports_reasons():
    from context_runtime.providers.aws.guardrail import BedrockGuardrail
    fake = FakeBedrockGuard(action="GUARDRAIL_INTERVENED",
                            assessments=[{"contentPolicy": {"filters": [{"type": "VIOLENCE"}]}}])
    g = BedrockGuardrail(client=fake, guardrail_id="g1")
    v = g.check_output("something bad")
    assert v.allowed is False and v.action == "blocked" and "VIOLENCE" in v.reasons


def test_bedrock_guardrail_masks_pii():
    from context_runtime.providers.aws.guardrail import BedrockGuardrail
    fake = FakeBedrockGuard(action="GUARDRAIL_INTERVENED", outputs=[{"text": "call me at {PHONE}"}])
    g = BedrockGuardrail(client=fake, guardrail_id="g1")
    v = g.check_input("call me at 555-1234")
    assert v.allowed is True and v.action == "masked" and v.text == "call me at {PHONE}"


# ── CloudWatch telemetry over a fake logs client ────────────────────────────────────────────────
class FakeLogs:
    def __init__(self):
        self.started = []

    def start_query(self, **kw):
        self.started.append(kw)
        return {"queryId": "q1"}

    def get_query_results(self, queryId):
        return {"status": "Complete", "results": [
            [{"field": "@timestamp", "value": "t0"}, {"field": "level", "value": "ERROR"}],
        ]}


def test_cloudwatch_reader_runs_query_and_flattens_rows():
    from context_runtime.providers.aws.cloudwatch import CloudWatchReader
    fake = FakeLogs()
    r = CloudWatchReader(client=fake, log_group="/app", now=lambda: 1000, sleep=lambda s: None)
    assert isinstance(r, TelemetryReader)
    rows = r.query("fields @message | filter level='ERROR'", window_s=300)
    assert rows == [{"@timestamp": "t0", "level": "ERROR"}]
    started = fake.started[0]
    assert started["startTime"] == 700 and started["endTime"] == 1000
    assert started["logGroupName"] == "/app"


# ── the neutral wrapper: any Guardrail × any ModelPlugin ────────────────────────────────────────
class StubModelP:
    def __init__(self, text):
        self.text = text
    def complete(self, req):
        return ModelResult(text=self.text, model="m", tier="cheap")
    def capabilities(self, model):
        from context_runtime.types import ModelCapabilities
        return ModelCapabilities()
    def count_tokens(self, text, model):
        return len(text) // 4
    def info(self):
        from context_runtime.types import PluginInfo
        return PluginInfo(name="stub", kind="model")


class ScriptGuard:
    """Guardrail whose verdicts are supplied per side."""
    def __init__(self, on_input=None, on_output=None):
        self._in = on_input or GuardrailVerdict(allowed=True)
        self._out = on_output or GuardrailVerdict(allowed=True)
    def check_input(self, text):
        return self._in
    def check_output(self, text):
        return self._out


def _req():
    return ModelRequest(messages=({"role": "user", "content": "hi"},), system="sys")


def test_guarded_model_passes_clean_io_through():
    gm = GuardedModel(StubModelP("clean answer"), ScriptGuard())
    assert gm.complete(_req()).text == "clean answer"


def test_guarded_model_refuses_blocked_input_without_calling_model():
    guard = ScriptGuard(on_input=GuardrailVerdict(allowed=False, action="blocked", reasons=("PROMPT_ATTACK",)))
    gm = GuardedModel(StubModelP("should not appear"), guard)
    assert gm.complete(_req()).text.startswith("This request was blocked")


def test_guarded_model_masks_output():
    guard = ScriptGuard(on_output=GuardrailVerdict(allowed=True, action="masked", text="redacted {PII}"))
    gm = GuardedModel(StubModelP("my ssn is 111-22-3333"), guard)
    assert gm.complete(_req()).text == "redacted {PII}"


def test_guarded_model_raise_mode():
    guard = ScriptGuard(on_output=GuardrailVerdict(allowed=False, reasons=("TOXICITY",)))
    gm = GuardedModel(StubModelP("bad"), guard, on_block="raise")
    try:
        gm.complete(_req())
        assert False, "expected GuardrailBlocked"
    except GuardrailBlocked as e:
        assert "TOXICITY" in str(e)
