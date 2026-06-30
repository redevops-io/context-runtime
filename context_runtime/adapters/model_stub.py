"""StubModel — a deterministic, offline ModelPlugin (SPEC §4.3).

Lets the whole runtime — run/explain/simulate — work with zero external deps or API
keys, which is what makes the conformance tests hermetic. It echoes a grounded answer
that cites the context it was given, so the citation verifier has something real to
check. Swap in ``LiteLLMModel`` for actual inference.
"""
from __future__ import annotations

import re

from ..types import ModelCapabilities, ModelRequest, ModelResult, PluginInfo

# per-tier token cost (cheap fictional numbers; only ordering matters offline)
_TIER_COST_PER_1K = {"local": 0.0, "cheap": 0.0005, "premium": 0.004}


class StubModel:
    def __init__(self, tier: str = "local", model: str = "stub-qwen"):
        self.tier = tier
        self.model = model

    def complete(self, req: ModelRequest) -> ModelResult:
        user = next((m["content"] for m in req.messages if m["role"] == "user"), "")
        # pull the first couple of [n] citation markers present in the context
        cites = re.findall(r"\[(\d+)\]", user)[:3] or ["1"]
        question = user.split("Question:")[-1].strip()[:200]
        answer = (
            f"Based on the retrieved context {', '.join('['+c+']' for c in dict.fromkeys(cites))}, "
            f"here is the grounded answer to: {question}"
        )
        ptoks = self.count_tokens(user, self.model)
        ctoks = self.count_tokens(answer, self.model)
        cost = (ptoks + ctoks) / 1000.0 * _TIER_COST_PER_1K.get(self.tier, 0.0)
        return ModelResult(
            text=answer, model=self.model, tier=self.tier,
            prompt_tokens=ptoks, completion_tokens=ctoks,
            est_cost_usd=round(cost, 6),
        )

    def capabilities(self, model: str) -> ModelCapabilities:
        return ModelCapabilities(max_context_tokens=32768, prompt_cache=False, tool_calling=False)

    def count_tokens(self, text: str, model: str) -> int:
        return max(1, len(text) // 4)   # ~4 chars/token

    def info(self) -> PluginInfo:
        return PluginInfo(name="stub_model", kind="model", capabilities=frozenset({self.tier}))
