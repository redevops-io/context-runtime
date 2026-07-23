"""BedrockModel — a ``ModelPlugin`` over Amazon Bedrock's ``converse`` API.

Registers Bedrock models as Context Runtime cost **Tiers**, so CR's core decision — *which model tier
to spend on* — includes Bedrock alongside every other provider, priced and learned identically. The
model plane stays provider-neutral: this is one ``ModelPlugin`` among many (OpenAI-compatible,
LiteLLM, stub); the runtime never learns it's Bedrock.

boto3 is optional and injectable: pass ``client=`` (a real bedrock-runtime client or a fake) for
tests, or an ``AwsSession`` to build one lazily.
"""
from __future__ import annotations

from ...types import ModelCapabilities, ModelRequest, ModelResult, PluginInfo
from ...adapters.model_litellm import Tier  # dataclass only — no litellm import


# Sensible default tier→Bedrock-model map (override via config). Costs are blended $/1k tokens,
# rounded; a deployment tunes them to its negotiated Bedrock pricing.
_DEFAULT_TIERS = [
    Tier(name="local", model="amazon.nova-micro-v1:0", cost_per_1k=0.00004),
    Tier(name="cheap", model="amazon.nova-lite-v1:0", cost_per_1k=0.00024),
    Tier(name="premium", model="anthropic.claude-3-5-sonnet-20241022-v2:0", cost_per_1k=0.009),
]


class BedrockModel:
    """ModelPlugin over Bedrock ``converse``. ``tiers`` maps CR tier names → Bedrock model ids."""

    def __init__(self, tiers: list[Tier], *, session=None, client=None, default_tier: str = "cheap"):
        self.tiers = {t.name: t for t in tiers}
        self.default_tier = default_tier
        self._session = session
        self._client = client

    @classmethod
    def from_config(cls, session, tiers=None, default_tier: str = "cheap") -> "BedrockModel":
        """Build from an AwsSession + optional tier config. ``tiers`` is a list of Tier or of
        (name, model_id, cost_per_1k) tuples; falls back to the default Nova/Claude map."""
        built: list[Tier] = []
        for t in (tiers or _DEFAULT_TIERS):
            built.append(t if isinstance(t, Tier) else Tier(name=t[0], model=t[1],
                         cost_per_1k=(t[2] if len(t) > 2 else 0.0)))
        return cls(built, session=session, default_tier=default_tier)

    def per_tier_models(self) -> dict:
        """Expand to a CR-tier → ModelPlugin dict so the runtime's tier choice (local/cheap/premium)
        actually maps to the matching Bedrock model, sharing one client. A single BedrockModel would
        collapse every tier onto ``default_tier``; this preserves the tier decision the planner made."""
        tiers = list(self.tiers.values())
        return {name: BedrockModel(tiers, session=self._session, client=self._client, default_tier=name)
                for name in self.tiers}

    def _bedrock(self):
        if self._client is None:
            self._client = self._session.client("bedrock-runtime")
        return self._client

    def _tier_for(self, capability: str) -> Tier:
        return self.tiers.get(self.default_tier) or next(iter(self.tiers.values()))

    def complete(self, req: ModelRequest) -> ModelResult:
        tier = self._tier_for(req.capability)
        messages = [{"role": m["role"], "content": [{"text": m["content"]}]} for m in req.messages]
        kwargs: dict = {
            "modelId": tier.model,
            "messages": messages,
            "inferenceConfig": {"maxTokens": req.max_tokens},
        }
        if req.system:
            kwargs["system"] = [{"text": req.system}]
        # tools are passed through only when already in Bedrock toolConfig shape (best-effort);
        # OpenAI-shaped tool schemas are ignored here rather than mis-translated.
        if req.tools and all(isinstance(t, dict) and "toolSpec" in t for t in req.tools):
            kwargs["toolConfig"] = {"tools": list(req.tools)}

        resp = self._bedrock().converse(**kwargs)
        blocks = (resp.get("output", {}).get("message", {}) or {}).get("content", []) or []
        text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))
        usage = resp.get("usage", {}) or {}
        ptoks = int(usage.get("inputTokens") or self.count_tokens(str(messages), tier.model))
        ctoks = int(usage.get("outputTokens") or self.count_tokens(text, tier.model))
        cost = (ptoks + ctoks) / 1000.0 * tier.cost_per_1k
        return ModelResult(
            text=text.strip(),
            model=tier.model,
            tier=tier.name,
            prompt_tokens=ptoks,
            completion_tokens=ctoks,
            est_cost_usd=round(cost, 6),
            models_used=(tier.model,),
        )

    def capabilities(self, model: str) -> ModelCapabilities:
        # Claude/Nova on Bedrock: large context, tool use, vision on the multimodal models.
        vision = "nova" in model or "claude-3" in model
        return ModelCapabilities(max_context_tokens=200000, tool_calling=True,
                                 structured_outputs=True, vision=vision)

    def count_tokens(self, text: str, model: str) -> int:
        return max(1, len(text) // 4)  # ~4 chars/token, matches the other adapters' estimate

    def info(self) -> PluginInfo:
        return PluginInfo(name="bedrock", kind="model", capabilities=frozenset(self.tiers))
