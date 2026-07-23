"""ModelArmorGuardrail — a ``Guardrail`` over Google Cloud Model Armor.

Content safety (prompt-injection, jailbreak, sensitive data) on model I/O by leaning on the managed
service. Satisfies the neutral ``Guardrail`` Protocol, so ``GuardedModel`` wraps any ModelPlugin with
it. Model Armor screens against a *template*; the client + template path are injectable for tests.
"""
from __future__ import annotations

from ..base import GuardrailVerdict


def _blocked(result) -> bool:
    # filter_match_state is MATCH_FOUND (intervened) or NO_MATCH_FOUND (clean). Note NO_MATCH_FOUND
    # contains "MATCH_FOUND" as a substring, so match precisely.
    state = getattr(getattr(result, "sanitization_result", None), "filter_match_state", None)
    s = str(state)
    return "MATCH_FOUND" in s and "NO_MATCH_FOUND" not in s


class ModelArmorGuardrail:
    def __init__(self, session=None, *, template: str, client=None):
        self._session = session
        self.template = template          # projects/.../locations/.../templates/<id>
        self._client = client

    def _armor(self):
        if self._client is None:
            self._client = self._session.modelarmor_client()
        return self._client

    def check_input(self, text: str) -> GuardrailVerdict:
        r = self._armor().sanitize_user_prompt(
            request={"name": self.template, "user_prompt_data": {"text": text}})
        return GuardrailVerdict(allowed=not _blocked(r),
                                action="blocked" if _blocked(r) else "none",
                                reasons=("model_armor",) if _blocked(r) else ())

    def check_output(self, text: str) -> GuardrailVerdict:
        r = self._armor().sanitize_model_response(
            request={"name": self.template, "model_response_data": {"text": text}})
        return GuardrailVerdict(allowed=not _blocked(r),
                                action="blocked" if _blocked(r) else "none",
                                reasons=("model_armor",) if _blocked(r) else ())
