"""GuardedModel — wrap any ``ModelPlugin`` with any ``Guardrail`` (provider-neutral governance).

Content guardrails on model I/O without touching the model plane: ``GuardedModel`` checks the request
before the call and the response after, using whatever ``Guardrail`` it's handed (Bedrock Guardrails
today, Azure Content Safety later, a local classifier for self-hosted). Because both sides are
Protocols, this composes freely — a Bedrock guardrail can even guard a local model, or vice versa.

Blocked input/output returns a safe refusal (``on_block="refuse"``, the default) or raises
(``on_block="raise"``); a masked verdict substitutes the redacted text. The wrapper is itself a
``ModelPlugin``, so the runtime holds it transparently.
"""
from __future__ import annotations

from dataclasses import replace

from ..types import ModelRequest, ModelResult, PluginInfo


class GuardrailBlocked(RuntimeError):
    """Raised when content is blocked and the wrapper is configured to raise rather than refuse."""


_REFUSAL = "This request was blocked by the content guardrail."


class GuardedModel:
    def __init__(self, model, guardrail, *, on_block: str = "refuse", refusal: str = _REFUSAL):
        self.model = model
        self.guardrail = guardrail
        if on_block not in ("refuse", "raise"):
            raise ValueError("on_block must be 'refuse' or 'raise'")
        self.on_block = on_block
        self.refusal = refusal

    def _guard(self, text: str, check) -> tuple[str, str | None]:
        """Return (possibly-masked text, refusal-or-None). Raises when on_block='raise' and blocked."""
        v = check(text)
        if v.allowed:
            return (v.text if v.action == "masked" and v.text else text), None
        if self.on_block == "raise":
            raise GuardrailBlocked(", ".join(v.reasons) or "blocked")
        return text, self.refusal

    def complete(self, req: ModelRequest) -> ModelResult:
        # guard the input (the user turn(s) + system prompt, concatenated for the check)
        probe = "\n".join([req.system or ""] + [m.get("content", "") for m in req.messages]).strip()
        _, refusal = self._guard(probe, self.guardrail.check_input)
        if refusal is not None:
            return ModelResult(text=refusal, model="guardrail", tier="guardrail")

        res = self.model.complete(req)

        # guard the output
        masked, refusal = self._guard(res.text, self.guardrail.check_output)
        if refusal is not None:
            return replace(res, text=refusal)
        if masked != res.text:
            return replace(res, text=masked)
        return res

    def capabilities(self, model: str):
        return self.model.capabilities(model)

    def count_tokens(self, text: str, model: str) -> int:
        return self.model.count_tokens(text, model)

    def info(self) -> PluginInfo:
        inner = self.model.info()
        return PluginInfo(name=f"guarded:{inner.name}", kind="model", capabilities=inner.capabilities)
