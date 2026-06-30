"""Reasoner — strategy over ≥1 ModelPlugin (SPEC §4.4).

v0.1 ships ``SingleShotReasoner``: one model call. The point of the seam is that the
``reason`` node always goes through a Reasoner, so mixture strategies
(plan_worker_critic, debate) drop in at v0.3–v0.4 without the graph changing.
Layering: Reasoner → Router → Model.
"""
from __future__ import annotations

from ..plugins.base import ModelPlugin
from ..types import ModelRequest, ModelResult, PluginInfo, ReasonRequest

_SYSTEM = (
    "Answer the question using ONLY the provided context. Cite sources inline like "
    "[1], [2]. If the context is insufficient, say so plainly — do not invent facts."
)


class SingleShotReasoner:
    def __init__(self, model: ModelPlugin):
        self.model = model

    def reason(self, req: ReasonRequest) -> ModelResult:
        ctx = req.context
        prompt = f"Context:\n{ctx.assembled_text}\n\nQuestion: {ctx.plan.intent.normalized or ctx.plan.id}"
        mreq = ModelRequest(
            messages=({"role": "user", "content": prompt},),
            capability=req.capability,
            system=_SYSTEM,
            max_tokens=1024,
        )
        res = self.model.complete(mreq)
        # roll up which model(s) were used (one, here)
        if not res.models_used:
            from dataclasses import replace
            res = replace(res, models_used=(res.model,))
        return res

    def info(self) -> PluginInfo:
        return PluginInfo(name="single_shot", kind="reasoner", capabilities=frozenset({"single_shot"}))
