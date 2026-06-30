"""Planner behavior: intent classification, candidate generation, constraint feasibility."""
from __future__ import annotations

from context_runtime import ContextRuntime
from context_runtime.constraints.hard import feasible
from context_runtime.planner.intent import RuleIntentAnalyzer
from context_runtime.types import Constraints, Goal


def _goal(text, **c):
    return Goal(text=text, constraints=Constraints(**c))


def test_intent_buckets():
    an = RuleIntentAnalyzer()
    assert an.analyze(_goal("ERR-500 keeps showing in logs")).bucket == "exact_lookup"
    assert an.analyze(_goal("Explain why the deploy incident happened")).bucket == "incident"
    assert an.analyze(_goal("run terraform apply on production")).bucket == "high_risk"
    assert an.analyze(_goal("rotate the api_key secret")).bucket == "sensitive"


def test_restricted_forces_local_tier():
    rt = ContextRuntime.default([{"chunk_id": "a::0", "filename": "a.md", "text": "secret material", "created_at": None}])
    plan = rt.plan("summarize the restricted document", constraints={"sensitivity": "restricted"})
    assert plan.chosen.model_tier == "local"


def test_require_citations_forces_verify_step():
    rt = ContextRuntime.default([{"chunk_id": "a::0", "filename": "a.md", "text": "alpha beta gamma", "created_at": None}])
    plan = rt.plan("what is alpha", constraints={"require_citations": True})
    assert any(s.type == "verify" for s in plan.chosen.steps)


def test_infeasible_cost_is_marked():
    rt = ContextRuntime.default([{"chunk_id": "a::0", "filename": "a.md", "text": "alpha beta", "created_at": None}])
    g = _goal("synthesize an overview of alpha", max_cost_usd=0.0001)
    # premium candidates should be infeasible under a near-zero budget
    cands = rt.candidates.generate(rt.intent.analyze(g), g)
    scored = [(c, rt.optimizer.score(c, g)) for c in cands]
    assert any(not s.feasible for _, s in scored)


def test_normalize_is_deterministic():
    an = RuleIntentAnalyzer()
    a = an.analyze(_goal("Why did the deployment fail?")).normalized
    b = an.analyze(_goal("the deployment fail did why")).normalized
    assert a == b  # order-independent canonical form
