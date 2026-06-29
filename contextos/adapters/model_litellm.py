"""LiteLLMModel — the real ModelPlugin binding (SPEC §4.3).

Wraps LiteLLM for transport + token counting + cost across 100+ providers, and uses
agentic-os ``router.py`` tier policy when available. Lazy-imports both so the core
package installs and the stub path runs without these deps.

Install:  pip install "contextos[litellm]"
"""
from __future__ import annotations

from dataclasses import dataclass

from ..types import ModelCapabilities, ModelRequest, ModelResult, PluginInfo


@dataclass
class Tier:
    name: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    cost_per_1k: float = 0.0


class LiteLLMModel:
    """ModelPlugin over LiteLLM. ``tiers`` mirrors agentic-os Tier policy."""

    def __init__(self, tiers: list[Tier] | None = None, default_tier: str = "cheap"):
        self.tiers = {t.name: t for t in (tiers or [])}
        self.default_tier = default_tier

    def _litellm(self):
        try:
            import litellm  # type: ignore
        except ImportError as e:  # pragma: no cover - exercised only with extra installed
            raise RuntimeError(
                "LiteLLMModel needs the 'litellm' extra: pip install 'contextos[litellm]'"
            ) from e
        return litellm

    def _tier_for(self, capability: str) -> Tier:
        # capability → tier mapping is the router's job; v0.1 keeps it simple
        return self.tiers.get(self.default_tier) or next(iter(self.tiers.values()))

    def complete(self, req: ModelRequest) -> ModelResult:
        litellm = self._litellm()
        tier = self._tier_for(req.capability)
        messages = list(req.messages)
        if req.system:
            messages = [{"role": "system", "content": req.system}, *messages]
        resp = litellm.completion(
            model=tier.model, messages=messages, max_tokens=req.max_tokens,
            base_url=tier.base_url, api_key=tier.api_key,
        )
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        ptoks = getattr(usage, "prompt_tokens", 0) if usage else 0
        ctoks = getattr(usage, "completion_tokens", 0) if usage else 0
        try:
            cost = litellm.completion_cost(completion_response=resp)
        except Exception:
            cost = (ptoks + ctoks) / 1000.0 * tier.cost_per_1k
        return ModelResult(
            text=text, model=tier.model, tier=tier.name,
            prompt_tokens=ptoks, completion_tokens=ctoks, est_cost_usd=round(cost or 0.0, 6),
        )

    def capabilities(self, model: str) -> ModelCapabilities:
        litellm = self._litellm()
        try:
            info = litellm.get_model_info(model)
            return ModelCapabilities(
                max_context_tokens=info.get("max_input_tokens", 8192),
                prompt_cache=bool(info.get("supports_prompt_caching", False)),
                tool_calling=bool(info.get("supports_function_calling", False)),
                vision=bool(info.get("supports_vision", False)),
            )
        except Exception:
            return ModelCapabilities()

    def count_tokens(self, text: str, model: str) -> int:
        try:
            return self._litellm().token_counter(model=model, text=text)
        except Exception:
            return max(1, len(text) // 4)

    def info(self) -> PluginInfo:
        return PluginInfo(name="litellm", kind="model", capabilities=frozenset(self.tiers))
