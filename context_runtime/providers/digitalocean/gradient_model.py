"""GradientModel — a ``ModelPlugin`` over DigitalOcean Gradient serverless inference.

DO's inference endpoint is OpenAI-compatible, so this speaks the ``/chat/completions`` wire format and
registers DO models as Context Runtime cost **Tiers**. One model access key reaches every hosted model
(Anthropic, OpenAI, Meta, Mistral, NVIDIA). The HTTP transport is injectable (via the DoSession) so
tests need no network. This keeps the model plane provider-neutral: the runtime just sees a ModelPlugin.
"""
from __future__ import annotations

from ...types import ModelCapabilities, ModelRequest, ModelResult, PluginInfo
from ...adapters.model_litellm import Tier

# DO model ids are configurable; these are reasonable defaults (override via config).
_DEFAULT_TIERS = [
    Tier(name="local", model="llama3.3-70b-instruct", cost_per_1k=0.0006),
    Tier(name="cheap", model="llama3.3-70b-instruct", cost_per_1k=0.0006),
    Tier(name="premium", model="anthropic-claude-3.5-sonnet", cost_per_1k=0.009),
]


class GradientModel:
    def __init__(self, tiers: list[Tier], *, session, default_tier: str = "cheap"):
        self.tiers = {t.name: t for t in tiers}
        self.default_tier = default_tier
        self.session = session

    @classmethod
    def from_config(cls, session, tiers=None, default_tier: str = "cheap") -> "GradientModel":
        built: list[Tier] = []
        for t in (tiers or _DEFAULT_TIERS):
            built.append(t if isinstance(t, Tier) else Tier(name=t[0], model=t[1],
                         cost_per_1k=(t[2] if len(t) > 2 else 0.0)))
        return cls(built, session=session, default_tier=default_tier)

    def per_tier_models(self) -> dict:
        tiers = list(self.tiers.values())
        return {name: GradientModel(tiers, session=self.session, default_tier=name) for name in self.tiers}

    def _tier_for(self, capability: str) -> Tier:
        return self.tiers.get(self.default_tier) or next(iter(self.tiers.values()))

    def complete(self, req: ModelRequest) -> ModelResult:
        tier = self._tier_for(req.capability)
        messages: list[dict] = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.extend(dict(m) for m in req.messages)
        payload = {"model": tier.model, "messages": messages, "max_tokens": req.max_tokens, "stream": False}
        headers = {"Authorization": f"Bearer {self.session.inference_key or ''}"}
        data = self.session.post(f"{self.session.inference_base}/chat/completions", payload, headers)

        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = data.get("usage") or {}
        ptoks = int(usage.get("prompt_tokens") or self.count_tokens(str(messages), tier.model))
        ctoks = int(usage.get("completion_tokens") or self.count_tokens(text, tier.model))
        cost = (ptoks + ctoks) / 1000.0 * tier.cost_per_1k
        return ModelResult(
            text=text.strip(), model=data.get("model", tier.model), tier=tier.name,
            prompt_tokens=ptoks, completion_tokens=ctoks,
            est_cost_usd=round(cost, 6), models_used=(tier.model,),
        )

    def capabilities(self, model: str) -> ModelCapabilities:
        return ModelCapabilities(max_context_tokens=128000, tool_calling=True, structured_outputs=True)

    def count_tokens(self, text: str, model: str) -> int:
        return max(1, len(text) // 4)

    def info(self) -> PluginInfo:
        return PluginInfo(name="gradient", kind="model", capabilities=frozenset(self.tiers))
