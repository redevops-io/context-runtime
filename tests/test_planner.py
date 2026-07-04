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


# ── intent-table + feasibility branch coverage added for the v2 release audit ──

def test_classify_covers_all_intent_buckets():
    from context_runtime.planner.rules import classify
    assert classify("ERR-500 in logs")[0] == "exact_lookup"
    assert classify("trace the root cause chain across services")[0] == "multi_hop"
    assert classify("the deploy incident postmortem")[0] == "incident"
    assert classify("run terraform apply on production")[0] == "high_risk"
    assert classify("rotate the api_key secret")[0] == "sensitive"
    assert classify("refactor this function")[0] == "code_reasoning"
    assert classify("summarize the design decisions")[0] == "synthesis"
    assert classify("what does idempotent mean")[0] == "conceptual"
    assert classify("florp glimbo wibble")[0] == "unknown"


def test_classify_first_match_wins_precedence():
    from context_runtime.planner.rules import classify
    # high_risk (production/delete) precedes sensitive (api_key/secret) in the ordered table
    assert classify("delete the production api_key secret")[0] == "high_risk"
    assert classify("rotate api_key")[0] == "sensitive"


def test_normalize_drops_stopwords_shorts_and_dedups():
    from context_runtime.planner.rules import normalize
    toks = normalize("The the a of Foo Foo bar").split()
    assert toks == sorted(set(toks))                         # canonical: sorted + de-duplicated
    assert not ({"the", "a", "of"} & set(toks))              # stopwords dropped
    assert "foo" in toks and "bar" in toks


def test_feasible_latency_ceiling_and_require_verification():
    from context_runtime.types import Candidate, PlanScore, StepSpec
    retrieve_only = Candidate(steps=(StepSpec(type="retrieve"),), model_tier="local")
    with_verify = Candidate(steps=(StepSpec(type="retrieve"), StepSpec(type="verify")), model_tier="local")
    # latency ceiling branch
    ok, reason = feasible(retrieve_only, PlanScore(latency_seconds=9.0), Constraints(max_latency_seconds=2.0))
    assert not ok and "latency" in reason
    ok, _ = feasible(retrieve_only, PlanScore(latency_seconds=1.0), Constraints(max_latency_seconds=2.0))
    assert ok
    # require_verification branch (independent of require_citations)
    ok, reason = feasible(retrieve_only, PlanScore(), Constraints(require_verification=True))
    assert not ok and "verification" in reason
    ok, _ = feasible(with_verify, PlanScore(), Constraints(require_verification=True))
    assert ok
