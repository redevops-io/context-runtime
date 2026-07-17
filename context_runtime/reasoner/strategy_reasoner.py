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
    def __init__(self, model: ModelPlugin, strategy: str = "reason"):
        self.model = model
        self.strategy = strategies.get(strategy)

    def reason(self, req: ReasonRequest) -> ModelResult:
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

    def info(self) -> PluginInfo:
        return PluginInfo(name=f"strategy:{self.strategy.name}", kind="reasoner",
                          capabilities=frozenset({"single_shot", self.strategy.name}))
