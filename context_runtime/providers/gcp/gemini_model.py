"""GeminiModel — a ``ModelPlugin`` over Gemini on Vertex AI (google-genai ``generate_content``).

Registers Gemini models as Context Runtime cost **Tiers**, so CR's core "which model tier" decision
includes Gemini alongside every other provider. The model plane stays provider-neutral: this is one
ModelPlugin among many; the runtime never learns it is Gemini.

The google-genai client is injectable: pass ``client=`` (a real ``genai.Client`` or a fake) for tests,
or a ``GcpSession`` to build one lazily. Inputs are plain dicts so tests need no google SDK.
"""
from __future__ import annotations

from ...types import ModelCapabilities, ModelRequest, ModelResult, PluginInfo
from ...adapters.model_litellm import Tier

_DEFAULT_TIERS = [
    Tier(name="local", model="gemini-2.0-flash-lite", cost_per_1k=0.00004),
    Tier(name="cheap", model="gemini-2.0-flash", cost_per_1k=0.0002),
    Tier(name="premium", model="gemini-2.5-pro", cost_per_1k=0.005),
]

# google-genai uses roles "user"/"model"; map assistant→model, everything else→user.
def _role(r: str) -> str:
    return "model" if r in ("assistant", "model") else "user"


class GeminiModel:
    def __init__(self, tiers: list[Tier], *, session=None, client=None, default_tier: str = "cheap"):
        self.tiers = {t.name: t for t in tiers}
        self.default_tier = default_tier
        self._session = session
        self._client = client

    @classmethod
    def from_config(cls, session, tiers=None, default_tier: str = "cheap") -> "GeminiModel":
        built: list[Tier] = []
        for t in (tiers or _DEFAULT_TIERS):
            built.append(t if isinstance(t, Tier) else Tier(name=t[0], model=t[1],
                         cost_per_1k=(t[2] if len(t) > 2 else 0.0)))
        return cls(built, session=session, default_tier=default_tier)

    def per_tier_models(self) -> dict:
        tiers = list(self.tiers.values())
        return {name: GeminiModel(tiers, session=self._session, client=self._client, default_tier=name)
                for name in self.tiers}

    def _genai(self):
        if self._client is None:
            self._client = self._session.genai_client()
        return self._client

    def _tier_for(self, capability: str) -> Tier:
        return self.tiers.get(self.default_tier) or next(iter(self.tiers.values()))

    def complete(self, req: ModelRequest) -> ModelResult:
        tier = self._tier_for(req.capability)
        contents = [{"role": _role(m["role"]), "parts": [{"text": m["content"]}]} for m in req.messages]
        config: dict = {"max_output_tokens": req.max_tokens}
        if req.system:
            config["system_instruction"] = req.system
        resp = self._genai().models.generate_content(model=tier.model, contents=contents, config=config)

        text = getattr(resp, "text", "") or ""
        usage = getattr(resp, "usage_metadata", None)
        ptoks = int(getattr(usage, "prompt_token_count", 0) or self.count_tokens(str(contents), tier.model))
        ctoks = int(getattr(usage, "candidates_token_count", 0) or self.count_tokens(text, tier.model))
        cost = (ptoks + ctoks) / 1000.0 * tier.cost_per_1k
        return ModelResult(
            text=text.strip(), model=tier.model, tier=tier.name,
            prompt_tokens=ptoks, completion_tokens=ctoks,
            est_cost_usd=round(cost, 6), models_used=(tier.model,),
        )

    def capabilities(self, model: str) -> ModelCapabilities:
        return ModelCapabilities(max_context_tokens=1000000, tool_calling=True,
                                 structured_outputs=True, vision=True)

    def count_tokens(self, text: str, model: str) -> int:
        return max(1, len(text) // 4)

    def info(self) -> PluginInfo:
        return PluginInfo(name="gemini", kind="model", capabilities=frozenset(self.tiers))
