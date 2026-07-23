"""BedrockGuardrail — a ``Guardrail`` over Amazon Bedrock Guardrails (``apply_guardrail``).

Content safety (prompt-injection, toxicity, PII) on model I/O by leaning on the managed service rather
than rebuilding a classifier — the article's "you'd want all three; we don't reimplement AWS's"
posture. Satisfies the neutral ``Guardrail`` Protocol, so ``GuardedModel`` wraps any ModelPlugin with
it, and Azure Content Safety later drops in behind the same Protocol. Client injectable for tests.
"""
from __future__ import annotations

from ..base import GuardrailVerdict


class BedrockGuardrail:
    def __init__(self, session=None, *, guardrail_id: str, version: str = "DRAFT", client=None):
        self._session = session
        self.guardrail_id = guardrail_id
        self.version = version
        self._client = client

    def _bedrock(self):
        if self._client is None:
            self._client = self._session.client("bedrock-runtime")
        return self._client

    def _apply(self, text: str, source: str) -> GuardrailVerdict:
        resp = self._bedrock().apply_guardrail(
            guardrailIdentifier=self.guardrail_id,
            guardrailVersion=self.version,
            source=source,  # "INPUT" | "OUTPUT"
            content=[{"text": {"text": text}}],
        )
        intervened = resp.get("action") == "GUARDRAIL_INTERVENED"
        # the service returns (possibly masked/redacted) replacement text in outputs
        outs = resp.get("outputs", []) or []
        masked = outs[0].get("text") if outs and isinstance(outs[0], dict) else None
        reasons = tuple(
            a.get("type", "policy")
            for assess in (resp.get("assessments", []) or [])
            for a in _flatten_assessment(assess)
        )
        if not intervened:
            return GuardrailVerdict(allowed=True)
        if masked and masked != text:
            return GuardrailVerdict(allowed=True, action="masked", text=masked, reasons=reasons)
        return GuardrailVerdict(allowed=False, action="blocked", reasons=reasons or ("guardrail",))

    def check_input(self, text: str) -> GuardrailVerdict:
        return self._apply(text, "INPUT")

    def check_output(self, text: str) -> GuardrailVerdict:
        return self._apply(text, "OUTPUT")


def _flatten_assessment(assess: dict) -> list[dict]:
    """Pull the individual policy hits out of a Bedrock assessment block (best-effort, shape-tolerant)."""
    out: list[dict] = []
    for key in ("topicPolicy", "contentPolicy", "wordPolicy", "sensitiveInformationPolicy"):
        block = assess.get(key) or {}
        for listkey in ("topics", "filters", "customWords", "managedWordLists", "piiEntities", "regexes"):
            for item in block.get(listkey, []) or []:
                if isinstance(item, dict):
                    out.append({"type": item.get("type") or item.get("name") or key})
    return out
