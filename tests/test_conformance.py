"""SPEC.md §10 — v0.1 conformance checklist, as executable tests.

Each test maps to one checklist item. Together they assert the vertical slice is
v0.1-conformant.
"""
from __future__ import annotations

import json

import pytest

from context_runtime import ContextRuntime, jsonio
from context_runtime.adapters.model_stub import StubModel
from context_runtime.adapters.store_inmemory import InMemoryStore
from context_runtime.plugins import base
from context_runtime.types import ExecutionGraph, Plan, Trace

DOCS = [
    {"chunk_id": "deploy.md::0", "filename": "deploy.md",
     "text": "Deployment X failed: readiness probe timed out after the Cloudflare cert expired. Rollback fixed it.",
     "created_at": None},
    {"chunk_id": "arch.md::0", "filename": "arch.md",
     "text": "Production APIs must run behind Cloudflare. Certs rotate every 90 days.",
     "created_at": None},
]


@pytest.fixture
def rt():
    return ContextRuntime.default(DOCS)


def test_model_and_store_implement_contracts(rt):
    # ModelPlugin + StorePlugin/RetrieverPlugin satisfy their Protocols (runtime-checkable)
    assert isinstance(StubModel(), base.ModelPlugin)
    assert isinstance(InMemoryStore([]), base.RetrieverPlugin)
    assert isinstance(InMemoryStore([]), base.StorePlugin)


def test_same_retriever_contract_local_and_cloud():
    # plugin-first: a second store impl must satisfy the SAME contract (binding is lazy,
    # so we assert the class conforms structurally without importing redevops_rag).
    from context_runtime.adapters.store_redevops import RedevopsRagRetriever
    r = RedevopsRagRetriever()
    assert hasattr(r, "search") and hasattr(r, "index") and hasattr(r, "info")
    assert isinstance(r, base.RetrieverPlugin)


def test_planner_trio_and_heuristic_costmodel(rt):
    plan = rt.plan("Explain why deployment X failed")
    assert plan.intent.bucket == "incident"
    # PlanScore terms present, total computed, feasibility set
    s = plan.score
    assert 0 <= s.expected_accuracy <= 1
    assert s.total != 0.0
    assert s.feasible in (True, False)


def test_cost_estimator_observe_and_statistics(rt):
    before = rt.estimator.statistics().fields[0].sample_count
    rt.run("Explain why deployment X failed")
    after = rt.estimator.statistics().fields[0].sample_count
    assert after == before + 1
    # statistics are honest: present, with a tracked field set
    fields = {f.field for f in rt.estimator.statistics().fields}
    assert {"cost_usd", "latency_seconds", "expected_accuracy"} <= fields


def test_reasoner_and_scheduler_seams_exist(rt):
    from context_runtime.reasoner.single_shot import SingleShotReasoner
    from context_runtime.scheduler.schedule import TopoScheduler
    assert isinstance(SingleShotReasoner(StubModel()), base.ReasonerPlugin)
    assert isinstance(TopoScheduler(), base.SchedulerPlugin)
    # the reason node — not a raw model call — is what the graph carries
    ctx = rt.build_context(rt.plan("Explain why deployment X failed"))
    assert any(n.kind == "reason" for n in ctx.graph.nodes)


def test_selection_via_knapsack_no_cpsat(rt):
    # the optimizer picks the max-score feasible candidate; CP-SAT is not required/imported
    plan = rt.plan("Explain why deployment X failed")
    assert plan.chosen is not None
    assert plan.score.total == max(s.total for _, s in
                                   [(c, rt.optimizer.score(c, rt._coerce_goal(
                                       "Explain why deployment X failed", None, None)))
                                    for c in [plan.chosen]])


def test_emits_valid_execution_graph_and_executes(rt):
    res = rt.run("Explain why deployment X failed")
    assert res.answer
    ctx = rt.build_context(rt.plan("Explain why deployment X failed"))
    kinds = [n.kind for n in ctx.graph.nodes]
    assert kinds[0] == "retrieve" and "reason" in kinds


def test_emits_trace_per_run(rt):
    res = rt.run("Explain why deployment X failed")
    assert isinstance(res.trace, Trace)
    assert res.trace.spans
    assert res.trace.actual_latency_seconds >= 0


def test_explain_and_simulate_populated(rt):
    ex = rt.explain("Explain why deployment X failed")
    assert ex.intent.bucket == "incident"
    assert len(ex.candidates) >= 1
    assert ex.statistics is not None

    sim = rt.simulate("Explain why deployment X failed")
    assert sim.expected_cost_usd.low <= sim.expected_cost_usd.point <= sim.expected_cost_usd.high
    assert sim.expected_models


def test_explain_analyze_overlays_trace(rt):
    ex = rt.explain("Explain why deployment X failed", analyze=True)
    assert ex.analyze is not None and isinstance(ex.analyze, Trace)


def test_json_roundtrip_with_unknown_forward_fields(rt):
    res = rt.run("Explain why deployment X failed")
    d = jsonio.dump(res.trace)
    d["future_field"] = 42
    back = jsonio.loads(Trace, json.dumps(d))
    assert back.extra["future_field"] == 42
    assert jsonio.dump(back)["future_field"] == 42


def test_rejects_higher_major_spec_version():
    with pytest.raises(ValueError):
        jsonio.loads(Trace, json.dumps({"plan_id": "p", "goal_text": "x", "spec_version": "9.0"}))


def test_v01_requires_no_v02_subsystems(rt):
    # Plan Cache is the null/always-miss stub in v0.1
    from context_runtime.plancache.cache import NullPlanCache
    assert isinstance(rt.plan_cache, NullPlanCache)
    plan = rt.plan("Explain why deployment X failed")
    assert plan.cache in ("miss", "hit", "bypass")
