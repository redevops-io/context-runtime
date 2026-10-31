"""Plan explanation: complete, deterministic, hashed, attachable."""
from __future__ import annotations

import copy
import json
import random

import pytest
from qa_fixtures import fixture_node, random_dag, tnode
from runtime_contracts import qa as contracts

from context_runtime.qa import (
    Claim, CostBasedQAPlanner, CostModel, Estimate, PlanBudget, PlanExplanation, PlanRequest, QAGraph, RunCaps,
    explain_plan, plan_with_explanation,
)

CAPS = RunCaps(deadline_ms=3_600_000, max_recovery_attempts=1, recovery_ceiling=1)


def run(graph, targets, est=None, *, budget=None):
    est = CostModel({}, default=Estimate(1000)) if est is None else CostModel(est)
    req = PlanRequest(graph, tuple(targets), est, budget or PlanBudget(max_parallel=4), CAPS)
    return CostBasedQAPlanner().plan_detailed(req)


def explained(graph, targets, est=None, **kw):
    r = run(graph, targets, est, **kw)
    return r, explain_plan(r).record


def refs(result):
    return {i.spec_ref.split("@")[0]: i.spec_ref for i in result.plan.items}


def item(rec, ref):
    return next(i for i in rec["items"] if i["spec_ref"] == ref)


def test_every_item_says_why_it_is_included_and_why_it_sits_where_it_does():
    g = QAGraph([tnode("top", after=["l", "r"], fixtures=["db"]), tnode("l", after=["base"]), tnode("r", after=["base"]),
                 tnode("base"), fixture_node("db", exclusive=True), tnode("unrelated")])
    r, rec = explained(g, ["top"], budget=PlanBudget(max_parallel=2))
    ids = {i["test_id"] for i in rec["items"]}
    assert ids == {"top", "l", "r", "base"}                         # the unrelated test is not pulled in
    for it in rec["items"]:
        assert it["included"] and all(x["text"] for x in it["included"])
        la = it["lane_assignment"]
        assert la["chosen_lane"] == it["lane"] and la["rule"] and la["decided_by"]
        assert la["options"] and min(o["earliest_start_ms"] for o in la["options"]) == it["estimated_start_ms"]
        assert la["start_bound_by"] in {"none", "precedence", "conflict", "lane"}
        assert it["estimate"]["basis"] == "prior" and it["budget"]["deadline_ms"] >= 1
    assert [x["text"] for x in item(rec, refs(r)["top"])["included"]] == ["selected"]
    assert [x["text"] for x in item(rec, refs(r)["base"])["included"]] == ["required by test:l (depends_on)",
                                                                           "required by test:r (depends_on)"]
    closure = {c["key"]: c["reasons"] for c in rec["closure"]}
    assert closure["fixture:db"] == ["required by test:top (fixture)"]


def test_a_lane_choice_names_the_options_and_the_tie_break():
    g = QAGraph([tnode("a"), tnode("b")])
    r, rec = explained(g, ["a", "b"], budget=PlanBudget(max_parallel=2))
    first, second = rec["items"]
    assert first["lane"] == "lane-0" and first["lane_assignment"]["tied_lanes"] == ["lane-0", "lane-1"]
    assert first["lane_assignment"]["decided_by"] == "earliest start; tie broken by lowest lane index"
    assert second["lane"] == "lane-1" and second["lane_assignment"]["tied_lanes"] == ["lane-1"]
    assert second["lane_assignment"]["decided_by"] == "earliest start"
    only, = explained(QAGraph([tnode("a")]), ["a"])[1]["items"]
    assert only["lane_assignment"]["decided_by"] == "the only lane"


def test_gates_say_whether_they_are_precedence_or_a_conflict_on_a_named_resource():
    g = QAGraph([fixture_node("db", exclusive=True), tnode("a", fixtures=["db"]), tnode("b", fixtures=["db"]),
                 tnode("c", after=["a"]), tnode("x")])
    est = {"a": Estimate(100), "b": Estimate(100), "c": Estimate(100), "x": Estimate(100)}
    r, rec = explained(g, "abcx", est, budget=PlanBudget(max_parallel=3))
    by_item = {}
    for gate in rec["gates"]:
        by_item.setdefault(gate["item"].split("@")[0], []).append((gate["after"].split("@")[0], tuple(gate["reasons"])))
    reasons = {why for pairs in by_item.values() for _, why in pairs}
    assert reasons <= {("precedence",), ("conflict:fixture:db",), ("precedence", "conflict:fixture:db")}
    # every required ordering is either a gate or recorded as implied by lane order
    for it in rec["items"]:
        t = it["test_id"]
        want = {refs(r)[p] for p in r.closure.predecessors(t)}
        assert want <= set(it["gate_after"]) | set(it["ordered_without_gate"])
    b = item(rec, refs(r)["b"])
    a_ref = refs(r)["a"]
    assert "fixture:db" in b["conflict_keys"] or "fixture:db" in item(rec, a_ref)["conflict_keys"]
    assert any(a_ref in i["gate_after"] + i["ordered_without_gate"] for i in rec["items"]
               if i["test_id"] in ("b", "c"))


def test_the_critical_path_is_correct_on_hand_built_graphs():
    g = QAGraph([tnode("a"), tnode("b", after=["a"]), tnode("c", after=["b"]), tnode("d")])
    r, rec = explained(g, ["c", "d"], {"a": Estimate(100), "b": Estimate(200), "c": Estimate(300), "d": Estimate(550)})
    assert rec["critical_path"]["length_ms"] == 600
    assert [x.split("@")[0] for x in rec["critical_path"]["items"]] == ["a", "b", "c"]
    # diamond with unequal branches: the heavier branch is the critical one
    g2 = QAGraph([tnode("top", after=["l", "r"]), tnode("l", after=["base"]), tnode("r", after=["base"]), tnode("base")])
    r2, rec2 = explained(g2, ["top"], {"top": Estimate(50), "l": Estimate(400), "r": Estimate(100), "base": Estimate(10)})
    assert rec2["critical_path"]["length_ms"] == 460
    assert [x.split("@")[0] for x in rec2["critical_path"]["items"]] == ["base", "l", "top"]


def test_the_makespan_carries_its_basis_and_a_lower_bound():
    g = QAGraph([tnode("a"), tnode("b")])
    est = {"a": Estimate(100, basis="history"), "b": Estimate(300, basis="prior")}
    r, rec = explained(g, "ab", est, budget=PlanBudget(max_parallel=2))
    m = rec["makespan"]
    assert m["basis"] == ["history", "prior"] and m["estimated_ms"] == 300 == m["lower_bound_ms"]
    assert m["total_duration_ms"] == 400


def test_binding_constraints_are_named_for_a_parallelism_cap():
    g = QAGraph([tnode(f"t{i}") for i in range(8)])
    r, rec = explained(g, [f"t{i}" for i in range(8)], budget=PlanBudget(max_parallel=2))
    by = {c["constraint"]: c for c in rec["binding_constraints"]}
    assert by["max_parallel"]["binding"] is True and by["max_parallel"]["limit"] == 2
    assert by["critical_path"]["binding"] is False
    cand = {c["lanes"]: c for c in rec["selection"]["candidates"]}
    assert cand[3]["status"] == "unavailable" and cand[3]["reason"] == "MAX_PARALLEL" and cand[3]["makespan_ms"] == 3000


def test_binding_constraints_are_named_for_a_pool_capacity():
    g = QAGraph([tnode(f"t{i}", claims=[Claim("memory_mb", 700)]) for i in range(6)])
    r, rec = explained(g, [f"t{i}" for i in range(6)], budget=PlanBudget(max_parallel=6, capacity=(("memory_mb", 1500),)))
    by = {c["constraint"]: c for c in rec["binding_constraints"]}
    assert by["capacity:memory_mb"]["binding"] is True and by["capacity:memory_mb"]["limit"] == 2
    assert by["max_parallel"]["binding"] is False
    assert [c for c in rec["selection"]["rejected"] if c["reason"] == "CAPACITY_BOUND"]


def test_the_critical_path_is_the_binding_constraint_when_more_lanes_cannot_help():
    g = QAGraph([tnode("a"), tnode("b", after=["a"]), tnode("c", after=["b"])])
    r, rec = explained(g, ["c"], budget=PlanBudget(max_parallel=3))
    by = {c["constraint"]: c for c in rec["binding_constraints"]}
    assert by["critical_path"]["binding"] is True and by["max_parallel"]["binding"] is False


def test_wall_time_is_binding_when_it_ruled_out_a_narrower_plan():
    g = QAGraph([tnode(f"t{i}") for i in range(4)])
    r, rec = explained(g, [f"t{i}" for i in range(4)], budget=PlanBudget(max_parallel=4, wall_time_ms=2000))
    by = {c["constraint"]: c for c in rec["binding_constraints"]}
    assert by["wall_time_ms"]["binding"] is True and by["wall_time_ms"]["limit"] == 2000
    rejected = {c["lanes"]: c["reason"] for c in rec["selection"]["rejected"]}
    assert rejected[1] == "WALL_TIME_EXCEEDED"


def test_a_conflict_key_is_reported_as_binding_when_it_delayed_a_start():
    g = QAGraph([fixture_node("db", exclusive=True), tnode("a", fixtures=["db"]), tnode("b", fixtures=["db"])])
    r, rec = explained(g, "ab", budget=PlanBudget(max_parallel=2))
    by = {c["constraint"]: c for c in rec["binding_constraints"]}
    assert by["conflict:fixture:db"]["observed"] == 2
    assert any(i["lane_assignment"]["start_bound_by"] in ("conflict", "precedence", "lane") for i in rec["items"])


def test_every_rejected_candidate_has_a_reason_and_a_detail():
    g = random_dag(4, 14)
    est = {f"t{i:03d}": Estimate(80 + 13 * i) for i in range(14)}
    r, rec = explained(g, [f"t{i:03d}" for i in range(14)], est, budget=PlanBudget(max_parallel=5, wall_time_ms=1400))
    rej = rec["selection"]["rejected"]
    assert rej and all(c["reason"] and c["detail"] and c["status"] in ("rejected", "unavailable") for c in rej)
    assert sum(1 for c in rec["selection"]["candidates"] if c["status"] == "selected") == 1
    codes = {c["reason"] for c in rej}
    assert codes <= {"SLOWER", "NO_FASTER_MORE_LANES", "WALL_TIME_EXCEEDED", "CAPACITY_BOUND", "MAX_PARALLEL"}


def test_the_explanation_is_deterministic_and_hash_verifies():
    g = random_dag(8, 20)
    est = {f"t{i:03d}": Estimate(70 + 31 * i, i, (i * 733) % 10001) for i in range(20)}
    targets = [f"t{i:03d}" for i in range(20)]
    one = explain_plan(run(g, targets, est, budget=PlanBudget(max_parallel=4)))
    two = explain_plan(run(g, targets, est, budget=PlanBudget(max_parallel=4)))
    assert one.record == two.record and one.hash == two.hash
    assert one.canonical_text() == two.canonical_text()
    assert PlanExplanation.verify(one.to_json())
    tampered = copy.deepcopy(one.to_json())
    tampered["items"][0]["lane"] = "lane-9"
    assert not PlanExplanation.verify(tampered)
    assert not PlanExplanation.verify({**one.to_json(), "explanation_hash": "rcv1:" + "0" * 64})
    assert not PlanExplanation.verify({k: v for k, v in one.to_json().items() if k != "explanation_hash"})


def test_input_order_does_not_change_the_explanation():
    g = random_dag(8, 12)
    nodes = list(g.nodes())
    random.Random(5).shuffle(nodes)
    est = {f"t{i:03d}": Estimate(70 + 31 * i) for i in range(12)}
    targets = [f"t{i:03d}" for i in range(12)]
    a = explain_plan(run(g, targets, est))
    b = explain_plan(run(QAGraph(nodes), list(reversed(targets)), dict(reversed(list(est.items())))))
    assert a.hash == b.hash


def test_the_record_holds_only_integers_strings_and_structure():
    g = random_dag(2, 10)
    rec = explain_plan(run(g, [f"t{i:03d}" for i in range(10)])).record

    def walk(v):
        if isinstance(v, dict):
            assert all(isinstance(k, str) for k in v)
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        else:
            assert v is None or isinstance(v, (str, int, bool)) and not isinstance(v, float), v

    walk(rec)
    json.dumps(rec)


def test_a_changed_input_changes_the_explanation_hash():
    g = random_dag(8, 10)
    targets = [f"t{i:03d}" for i in range(10)]
    a = explain_plan(run(g, targets, budget=PlanBudget(max_parallel=2)))
    b = explain_plan(run(g, targets, budget=PlanBudget(max_parallel=3)))
    c = explain_plan(run(g, targets, {t: Estimate(1000, basis="history") for t in targets}, budget=PlanBudget(max_parallel=2)))
    assert len({a.hash, b.hash, c.hash}) == 3


def test_the_explanation_binds_to_the_plan_and_attaches_as_the_rationale_without_changing_the_plan_hash():
    g = random_dag(8, 10)
    targets = [f"t{i:03d}" for i in range(10)]
    est = CostModel({}, default=Estimate(500))
    req = PlanRequest(g, tuple(targets), est, PlanBudget(max_parallel=3), CAPS)
    bare = CostBasedQAPlanner().plan(req)
    ex = plan_with_explanation(CostBasedQAPlanner(), req)
    assert ex.plan.rationale_ref == ex.explanation.hash and bare.rationale_ref is None
    assert ex.plan.plan_hash == bare.plan_hash                                  # custody: identity unchanged
    assert ex.explanation.record["plan_hash"] == bare.plan_hash
    assert contracts.QAPlan.from_json(ex.plan.to_json()) == ex.plan
    assert ex.plan.to_json()["custody"]["rationale_ref"] == ex.explanation.hash


def test_the_explanation_lists_what_is_structural_and_what_is_only_estimated():
    rec = explain_plan(run(QAGraph([tnode("a")]), ["a"])).record
    assert "every test of the closure is planned exactly once (FULL coverage)" in rec["guarantees"]["structural"]
    assert "makespan" in rec["guarantees"]["estimated"]
    assert rec["coverage_contract"] == "FULL"
