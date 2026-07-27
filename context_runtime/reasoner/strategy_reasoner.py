"""StrategyReasoner — a parametric Reasoner that executes a chosen generation strategy.

Generalizes :class:`SingleShotReasoner`: instead of one fixed prompt/budget, it looks up the
strategy (terse · reason · decompose · mapreduce · single_shot) and threads its system prompt,
thinking flag, and token budget into the ModelRequest. For a reasoning strategy that ends with an
``Answer:`` line, it extracts the final answer from the trace so downstream (verify, the panel) sees
the answer, not the scratchpad. Same ReasonerPlugin seam as SingleShotReasoner, so the runtime's
reason node dispatches to it by the plan's reason-step ``strategy`` param with nothing else changing.
"""
from __future__ import annotations

from dataclasses import replace

from ..plugins.base import ModelPlugin
from ..types import ModelRequest, ModelResult, PluginInfo, ReasonRequest
from . import strategies


class StrategyReasoner:
    def __init__(self, model: ModelPlugin, strategy: str = "reason", *, verify: bool = False,
                 samples: int = 0, refine_depth: int | None = None):
        self.model = model
        self.strategy = strategies.get(strategy)
        self.verify = verify
        self.samples = samples   # ≥2 → self-consistency (+sc): sample K, return the consensus
        self.refine_depth = strategies.refine_depth() if refine_depth is None else refine_depth

    def _one(self, req: ReasonRequest, *, temperature: float | None = None) -> ModelResult:
        s = self.strategy
        ctx = req.context
        question = ctx.plan.intent.normalized or ctx.plan.id
        prompt = f"Context:\n{ctx.assembled_text}\n\nQuestion: {question}"
        mreq = ModelRequest(
            messages=({"role": "user", "content": prompt},),
            capability=req.capability,
            system=s.system,
            max_tokens=s.max_tokens,
            thinking=s.thinking,
            temperature=temperature,
        )
        res = self.model.complete(mreq)
        if s.extractive:
            res = replace(res, text=strategies.extract_final(res.text))
        if not res.models_used:
            res = replace(res, models_used=(res.model,))
        return res

    def reason(self, req: ReasonRequest) -> ModelResult:
        res = self._self_consistent(req) if self.samples >= 2 else self._one(req)
        if self.verify:
            res = self._refine(req, res)
        return res

    def _refine(self, req: ReasonRequest, res: ModelResult) -> ModelResult:
        """Self-refinement (C): up to ``refine_depth`` self-check→retry rounds, keeping the most-grounded
        attempt. Depth 1 reproduces the prior single retry."""
        from .verify import faithfulness
        ctx_text = req.context.assembled_text
        best, best_f = res, faithfulness(res.text, ctx_text)
        rounds = 0
        while best_f < strategies.VERIFY_FAITHFULNESS_MIN and rounds < max(1, self.refine_depth):
            retry = self._one(req)
            f = faithfulness(retry.text, ctx_text)
            if f > best_f:
                best, best_f = retry, f
            rounds += 1
        return best

    def _self_consistent(self, req: ReasonRequest) -> ModelResult:
        """Self-consistency (A): sample K traces at temperature > 0, return the consensus answer (largest
        agreement cluster, faithfulness tiebreak), rolling up the K-sample cost so the arm's cost prior
        reflects the extra compute — the bandit still learns whether Best@k pays for this class."""
        from .verify import consensus_index
        temp = strategies.self_consistency_temp()
        results = [self._one(req, temperature=temp) for _ in range(self.samples)]
        if not results:
            return self._one(req)
        chosen = results[consensus_index([r.text for r in results], req.context.assembled_text)]
        return replace(chosen,
                       prompt_tokens=sum(r.prompt_tokens for r in results),
                       completion_tokens=sum(r.completion_tokens for r in results),
                       est_cost_usd=sum(r.est_cost_usd for r in results))

    def info(self) -> PluginInfo:
        caps = {"single_shot", self.strategy.name}
        suffix = ""
        if self.samples >= 2:
            caps.add("self_consistency"); suffix += "+sc"
        if self.verify:
            caps.add("verify"); suffix += "+v"
        return PluginInfo(name=f"strategy:{self.strategy.name}{suffix}",
                          kind="reasoner", capabilities=frozenset(caps))
