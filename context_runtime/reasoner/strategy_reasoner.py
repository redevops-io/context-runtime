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
    def __init__(self, model: ModelPlugin, strategy: str = "reason", *, verify: bool = False):
        self.model = model
        self.strategy = strategies.get(strategy)
        self.verify = verify

    def _one(self, req: ReasonRequest) -> ModelResult:
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
        )
        res = self.model.complete(mreq)
        if s.extractive:
            res = replace(res, text=strategies.extract_final(res.text))
        if not res.models_used:
            res = replace(res, models_used=(res.model,))
        return res

    def reason(self, req: ReasonRequest) -> ModelResult:
        res = self._one(req)
        # Verification Optimizer: self-check the answer's grounding; if it's weak, retry once and keep
        # the better attempt. This is the "self-check · retry" the runtime learns to spend on per class.
        if self.verify:
            from .verify import faithfulness
            ctx_text = req.context.assembled_text
            f0 = faithfulness(res.text, ctx_text)
            if f0 < strategies.VERIFY_FAITHFULNESS_MIN:
                retry = self._one(req)
                if faithfulness(retry.text, ctx_text) > f0:
                    res = retry
        return res

    def info(self) -> PluginInfo:
        caps = {"single_shot", self.strategy.name} | ({"verify"} if self.verify else set())
        return PluginInfo(name=f"strategy:{self.strategy.name}{'+v' if self.verify else ''}",
                          kind="reasoner", capabilities=frozenset(caps))
