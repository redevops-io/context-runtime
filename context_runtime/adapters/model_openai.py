"""OpenAICompatibleModel — a dependency-light ``ModelPlugin`` (SPEC §4.3).

Speaks the OpenAI ``/chat/completions`` wire format over the *stdlib* (urllib), so it
drives OpenAI gpt-5.x, DeepSeek, vLLM, Ollama — any compatible endpoint — with the same
``ModelRequest → ModelResult`` contract and cost-tiered ``Tier`` routing as
``LiteLLMModel``, but **without the litellm dependency**. That matters because the slim
agent containers install only the base ``context-runtime`` (no ``[litellm]`` extra): this
adapter lets those apps run their conversational agent *on the Context Runtime model
plane* rather than reaching around it to a raw provider SDK. Degrade to ``StubModel``
when no key is configured (``from_env`` returns ``None``).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from ..types import ModelCapabilities, ModelRequest, ModelResult, PluginInfo
from .model_litellm import Tier  # dataclass only; importing it does not pull in litellm


class OpenAICompatibleModel:
    """ModelPlugin over an OpenAI-compatible endpoint using only the stdlib."""

    def __init__(self, tiers: list[Tier], default_tier: str = "chat", timeout: float = 40.0):
        self.tiers = {t.name: t for t in tiers}
        self.default_tier = default_tier
        self.timeout = timeout

    @classmethod
    def from_env(
        cls,
        *,
        model_env: str = "AGENT_LLM_MODEL",
        default_model: str = "gpt-5.5",
        key_envs: tuple[str, ...] = ("AGENT_LLM_KEY", "OPENAI_API_KEY"),
        base_envs: tuple[str, ...] = ("AGENT_LLM_BASE_URL", "OPENAI_BASE_URL"),
        cost_per_1k: float = 0.0,
    ) -> "OpenAICompatibleModel | None":
        """Build from environment, or ``None`` when nothing is configured (offline → StubModel).

        Priority: an explicit OpenAI/agent key, else the self-hosted OpenAI-compatible endpoint
        the agentic-os apps already point at (``REDEVOPS_LLM_BASE_URL`` / ``REDEVOPS_LLM_MODEL`` —
        typically a keyless on-prem DeepSeek). That keeps the agent on our own model plane without
        spreading a provider key across every container.
        """
        key = next((os.environ[k] for k in key_envs if os.environ.get(k)), None)
        if key:
            base = next((os.environ[b] for b in base_envs if os.environ.get(b)), "https://api.openai.com/v1")
            model = os.environ.get(model_env, default_model)
            return cls([Tier(name="chat", model=model, base_url=base.rstrip("/"), api_key=key, cost_per_1k=cost_per_1k)])
        rbase = os.environ.get("REDEVOPS_LLM_BASE_URL")
        if rbase:
            rmodel = os.environ.get("REDEVOPS_LLM_MODEL", "DeepSeek-V4-Flash")
            rkey = os.environ.get("REDEVOPS_LLM_KEY") or "sk-noauth"  # vLLM ignores it; header must exist
            return cls([Tier(name="chat", model=rmodel, base_url=rbase.rstrip("/"), api_key=rkey, cost_per_1k=cost_per_1k)])
        return None

    def _tier_for(self, capability: str) -> Tier:
        return self.tiers.get(self.default_tier) or next(iter(self.tiers.values()))

    def complete(self, req: ModelRequest) -> ModelResult:
        tier = self._tier_for(req.capability)
        messages: list[dict] = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        messages.extend(dict(m) for m in req.messages)
        # gpt-5.x wants max_completion_tokens; self-hosted vLLM/DeepSeek want max_tokens. Branch on host.
        tok_key = "max_completion_tokens" if "openai.com" in (tier.base_url or "") else "max_tokens"
        payload: dict = {"model": tier.model, "messages": messages, tok_key: req.max_tokens}
        if req.tools:
            payload["tools"] = list(req.tools)
        request = urllib.request.Request(
            tier.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + (tier.api_key or ""), "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                d = json.load(resp)
        except urllib.error.HTTPError as e:  # surface provider errors so the caller can fall back
            detail = e.read()[:300].decode("utf-8", "ignore")
            raise RuntimeError(f"model http {e.code}: {detail}") from e
        choice = (d.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = d.get("usage") or {}
        ptoks = int(usage.get("prompt_tokens") or self.count_tokens(json.dumps(messages), tier.model))
        ctoks = int(usage.get("completion_tokens") or self.count_tokens(text, tier.model))
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
        return ModelCapabilities(max_context_tokens=128000, tool_calling=True, structured_outputs=True)

    def count_tokens(self, text: str, model: str) -> int:
        return max(1, len(text) // 4)  # ~4 chars/token, matches StubModel's estimate

    def info(self) -> PluginInfo:
        return PluginInfo(name="openai_compatible", kind="model", capabilities=frozenset(self.tiers))
