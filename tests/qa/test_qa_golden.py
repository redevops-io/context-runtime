"""A golden plan: a small hand-built instance whose plan, explanation and hashes are pinned.

A change to the planner's algorithm, the plan's canonical form or the explanation record changes these values and
fails this test. If the change is deliberate, bump PLANNER_VERSION and update the pinned values together.
"""
from __future__ import annotations

from qa_fixtures import fixture_node, tnode

from context_runtime.qa import (
    CostBasedQAPlanner, CostModel, Estimate, PlanBudget, PlanRequest, QAGraph, RunCaps, plan_with_explanation,
)

GOLDEN_PLAN_HASH = "rcv1:e21db77fac1e7713db0a15d7aa98beaa5d37f56e375ee8097d837ae2ecbc1ab0"
GOLDEN_EXPLANATION_HASH = "rcv1:2a9027ac69bf9d427a6f7aa6cc0fac9943f13b23d0b948a2225dbfa3d00b7ae9"


def golden_request() -> PlanRequest:
    g = QAGraph([fixture_node("db", exclusive=True), tnode("a", fixtures=["db"]), tnode("b", fixtures=["db"]),
                 tnode("c", after=["a"]), tnode("d", after=["b", "c"]), tnode("e"), tnode("f")])
    est = {"a": Estimate(300, risk_bp=8000), "b": Estimate(200), "c": Estimate(400), "d": Estimate(100, risk_bp=9000),
           "e": Estimate(500), "f": Estimate(150)}
    return PlanRequest(g, tuple("abcdef"), CostModel(est), PlanBudget(max_parallel=3),
                       RunCaps(deadline_ms=600000, max_retries=1, max_recovery_attempts=2, recovery_ceiling=2))


def test_the_golden_plan_layout():
    ex = plan_with_explanation(CostBasedQAPlanner(), golden_request())
    plan = ex.plan
    layout = sorted((i.lane, i.order, i.spec_ref.split("@")[0], i.budget.deadline_ms, i.evidence_level.value)
                    for i in plan.items)
    assert layout == [
        ("lane-0", 0, "a", 900, "FULL"), ("lane-0", 1, "c", 1200, "STANDARD"), ("lane-0", 2, "d", 300, "FULL"),
        ("lane-1", 0, "e", 1500, "STANDARD"), ("lane-1", 1, "f", 450, "STANDARD"),
        ("lane-2", 0, "b", 600, "STANDARD"),
    ]
    gates = sorted((g.item.split("@")[0], sorted(a.split("@")[0] for a in g.after)) for g in plan.gates)
    assert gates == [("b", ["a"]), ("d", ["b"])]          # b waits for a (shared lock); d waits for b (c is lane-ordered)
    assert plan.max_parallel == 3 and plan.lanes == ("lane-0", "lane-1", "lane-2")
    assert [(c.resource, c.amount) for c in plan.resource_claims] == [("fixture:db", 1)]
    assert ex.result.makespan_ms == 800 == ex.result.critical_path_ms
    assert [t.split("@")[0] for t in ex.explanation.record["critical_path"]["items"]] == ["a", "c", "d"]


def test_the_golden_hashes():
    ex = plan_with_explanation(CostBasedQAPlanner(), golden_request())
    assert ex.plan.plan_hash == GOLDEN_PLAN_HASH
    assert ex.explanation.hash == GOLDEN_EXPLANATION_HASH
