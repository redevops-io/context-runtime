"""Plan selection: FULL coverage, bounded parallelism, conflict keys, budgets, refusals, contracts acceptance."""
from __future__ import annotations

import dataclasses
import random

import pytest
from qa_fixtures import env_node, fixture_node, make_spec, random_dag, tnode
from qa_sim import locks_of, max_overlap, overlaps, random_durations, run_plan
from runtime_contracts import qa as contracts
from runtime_contracts.canonical import content_hash

from context_runtime.qa import (
    Claim, CostBasedQAPlanner, CostModel, Estimate, InfeasibleBudget, Node, NodeKind, PlanBudget, PlanningRefusal,
    PlanRequest, QAGraph, QAPlanner, Ref, RunCaps, UnsupportedCoverage, graph_from_specs, resolve, verify_plan,
)
from context_runtime.qa import Code
from context_runtime.qa import planner as planner_module

CAPS = RunCaps(deadline_ms=3_600_000, max_retries=1, max_recovery_attempts=2, recovery_ceiling=2)


def request(graph, targets, est=None, *, budget=None, caps=CAPS, **kw):
    if est is None:
        est = CostModel({}, default=Estimate(1000))
    elif isinstance(est, dict):
        est = CostModel(est)
    return PlanRequest(graph, tuple(targets), est, budget or PlanBudget(max_parallel=4), caps, **kw)


def plan_of(req):
    return CostBasedQAPlanner().plan_detailed(req)


def lane_map(plan):
    out = {}
    for it in plan.items:
        out.setdefault(it.lane, []).append((it.order, it.spec_ref.split("@")[0]))
    return {k: [n for _, n in sorted(v)] for k, v in sorted(out.items())}


# ── hand-built cases ───────────────────────────────────────────────────────────────────────────────────────

def test_independent_tests_are_spread_over_lanes_up_to_the_cap():
    g = QAGraph([tnode(n) for n in "abcd"])
    r = plan_of(request(g, "abcd", budget=PlanBudget(max_parallel=2)))
    assert r.chosen_lanes == 2 and r.makespan_ms == 2000
    assert r.plan.max_parallel == 2 and set(r.plan.lanes) == {"lane-0", "lane-1"}
    assert r.plan.gates == ()
    assert all(len(v) == 2 for v in lane_map(r.plan).values())


def test_a_chain_uses_one_lane_even_when_more_are_allowed():
    g = QAGraph([tnode("c", after=["b"]), tnode("b", after=["a"]), tnode("a")])
    r = plan_of(request(g, ["c"], budget=PlanBudget(max_parallel=4)))
    assert r.chosen_lanes == 1 and lane_map(r.plan) == {"lane-0": ["a", "b", "c"]}
    assert r.plan.gates == ()          # lane order already implies every start condition
    reasons = {c.lanes: c.reason for c in r.candidates}
    assert reasons[1] == "SELECTED" and reasons[2] == "NO_FASTER_MORE_LANES"


def test_diamond_gates_the_join_after_both_branches():
    g = QAGraph([tnode("top", after=["l", "r"]), tnode("l", after=["base"]), tnode("r", after=["base"]),
                 tnode("base")])
    r = plan_of(request(g, ["top"], budget=PlanBudget(max_parallel=2)))
    assert r.makespan_ms == 3000 and r.critical_path_ms == 3000
    gates = {g.item.split("@")[0]: sorted(a.split("@")[0] for a in g.after) for g in r.plan.gates}
    # whichever lane the branches took, 'top' is ordered after both and 'l'/'r' after 'base'
    assert verify_plan(r.plan, r.closure) == []
    done = run_plan(r.plan, {i.spec_ref: 1000 for i in r.plan.items})
    ref = {i.spec_ref.split("@")[0]: i.spec_ref for i in r.plan.items}
    assert done[ref["top"]][0] >= max(done[ref["l"]][1], done[ref["r"]][1])
    assert gates or lane_map(r.plan)  # structure exists


def test_longest_remaining_path_goes_first():
    g = QAGraph([tnode("short"), tnode("long1"), tnode("long2", after=["long1"])])
    est = {"short": Estimate(100), "long1": Estimate(100), "long2": Estimate(500)}
    r = plan_of(request(g, ["short", "long2"], est, budget=PlanBudget(max_parallel=1)))
    assert [s.test_id for s in r.schedule] == ["long1", "long2", "short"]


def test_exclusive_fixture_users_are_ordered_never_concurrent():
    g = QAGraph([fixture_node("db", exclusive=True), tnode("a", fixtures=["db"]), tnode("b", fixtures=["db"]),
                 tnode("c")])
    r = plan_of(request(g, "abc", budget=PlanBudget(max_parallel=3)))
    ref = {i.spec_ref.split("@")[0]: i.spec_ref for i in r.plan.items}
    assert verify_plan(r.plan, r.closure) == []
    for seed in range(20):
        done = run_plan(r.plan, random_durations(r.plan, seed))
        assert not overlaps(done[ref["a"]], done[ref["b"]])
    assert r.chosen_lanes == 2         # c runs beside the a -> b pair


def test_conflicting_items_in_different_lanes_are_joined_by_a_gate():
    g = QAGraph([fixture_node("db", exclusive=True), tnode("a", fixtures=["db"]), tnode("b", fixtures=["db"]),
                 tnode("x"), tnode("y")])
    est = {"a": Estimate(100), "b": Estimate(100), "x": Estimate(100), "y": Estimate(100)}
    r = plan_of(request(g, ["a", "b", "x", "y"], est, budget=PlanBudget(max_parallel=2)))
    # four equal tests over two lanes, a and b share a lock: the planner either serialises them in one lane or gates
    la = next(i.lane for i in r.plan.items if i.spec_ref.startswith("a@"))
    lb = next(i.lane for i in r.plan.items if i.spec_ref.startswith("b@"))
    gated = any(g_.item.startswith(("a@", "b@")) and any(a.startswith(("a@", "b@")) for a in g_.after)
                for g_ in r.plan.gates)
    assert la == lb or gated


def test_a_shared_but_not_exclusive_fixture_does_not_serialise():
    g = QAGraph([fixture_node("cache"), tnode("a", fixtures=["cache"]), tnode("b", fixtures=["cache"])])
    r = plan_of(request(g, "ab", budget=PlanBudget(max_parallel=2)))
    assert r.chosen_lanes == 2 and r.plan.gates == ()


def test_max_parallel_is_honoured():
    g = QAGraph([tnode(f"t{i}") for i in range(10)])
    for cap in (1, 2, 3, 7):
        r = plan_of(request(g, [f"t{i}" for i in range(10)], budget=PlanBudget(max_parallel=cap)))
        assert len(r.plan.lanes) == r.plan.max_parallel <= cap
        if cap <= 3:                       # the cap binds: one more lane would be faster
            assert len(r.plan.lanes) == cap


def test_pool_capacity_caps_the_lane_count_for_any_real_durations():
    claims = [Claim("memory_mb", 600)]
    g = QAGraph([tnode(f"t{i}", claims=claims) for i in range(6)])
    budget = PlanBudget(max_parallel=6, capacity=(("memory_mb", 1500),))      # two 600s fit, three do not
    r = plan_of(request(g, [f"t{i}" for i in range(6)], budget=budget))
    assert r.chosen_lanes == 2 and r.limits["capacity:memory_mb"] == 2
    kinds = {c.lanes: c.reason for c in r.candidates}
    assert kinds[3] == "CAPACITY_BOUND" and max(kinds) == 3         # the first count the capacity rules out
    for seed in range(15):
        done = run_plan(r.plan, random_durations(r.plan, seed))
        assert max_overlap(done, lambda ref: 600) <= 1500


def test_pool_bound_uses_the_largest_claims_not_the_average():
    g = QAGraph([tnode("big", claims=[Claim("memory_mb", 900)]), tnode("s1", claims=[Claim("memory_mb", 100)]),
                 tnode("s2", claims=[Claim("memory_mb", 100)]), tnode("s3", claims=[Claim("memory_mb", 100)])])
    r = plan_of(request(g, ["big", "s1", "s2", "s3"], budget=PlanBudget(max_parallel=4, capacity=(("memory_mb", 1000),))))
    assert r.limits["capacity:memory_mb"] == 2     # big + one small fits; big + two small does not


def test_cost_model_estimates_set_the_item_deadline_within_the_caps():
    g = QAGraph([tnode("a"), tnode("b")])
    est = {"a": Estimate(1000, risk_bp=9000), "b": Estimate(2000, risk_bp=100)}
    caps = RunCaps(deadline_ms=5000, max_retries=1, max_recovery_attempts=1, recovery_ceiling=2, max_evidence="STANDARD")
    r = plan_of(request(g, "ab", est, caps=caps, budget=PlanBudget(max_parallel=2, deadline_multiplier_pct=300)))
    by = {i.spec_ref.split("@")[0]: i for i in r.plan.items}
    assert by["a"].budget.deadline_ms == 3000 and by["b"].budget.deadline_ms == 5000      # 6000 clipped to the cap
    assert by["a"].evidence_level is contracts.EvidenceLevel.STANDARD                      # FULL clipped to the cap
    assert all(i.recovery_ceiling == 2 and i.budget.max_recovery_attempts == 1 for i in r.plan.items)
    assert all(i.budget.max_model_calls == 0 and i.budget.max_tokens == 0 for i in r.plan.items)


def test_high_risk_gets_full_evidence_when_the_cap_allows():
    g = QAGraph([tnode("a"), tnode("b")])
    r = plan_of(request(g, "ab", {"a": Estimate(10, risk_bp=7000), "b": Estimate(10, risk_bp=6999)}))
    by = {i.spec_ref.split("@")[0]: i.evidence_level.value for i in r.plan.items}
    assert by == {"a": "FULL", "b": "STANDARD"}


def test_spec_budget_bounds_the_item_budget_when_specs_are_supplied():
    a = make_spec("spec-a")
    g = graph_from_specs([a], environments=[env_node("env-a")])
    caps = RunCaps(deadline_ms=3_600_000, max_recovery_attempts=5, recovery_ceiling=5)
    r = plan_of(request(g, ["spec-a"], budget=PlanBudget(max_parallel=1), caps=caps, specs={"spec-a": a}))
    item = r.plan.items[0]
    assert item.budget.max_recovery_attempts == a.budget.max_recovery_attempts == 2


def test_supplied_spec_that_does_not_match_the_graph_is_refused():
    a = make_spec("spec-a")
    g = graph_from_specs([a], environments=[env_node("env-a")])
    other = make_spec("spec-a", expected="Different")
    with pytest.raises(PlanningRefusal) as e:
        plan_of(request(g, ["spec-a"], specs={"spec-a": other}))
    assert e.value.code == Code.INVALID_REQUEST


def test_plan_from_real_specs_carries_their_refs_and_dependencies():
    a = make_spec("spec-a")
    b = make_spec("spec-b", depends_on=[a.spec_ref], fixtures=["db"])
    c = make_spec("spec-c", fixtures=["db"])
    g = graph_from_specs([a, b, c], fixtures=[fixture_node("db", exclusive=True)], environments=[env_node("env-a")])
    r = plan_of(request(g, ["spec-b", "spec-c"], budget=PlanBudget(max_parallel=3)))
    refs = {i.spec_ref for i in r.plan.items}
    assert refs == {a.spec_ref, b.spec_ref, c.spec_ref}
    assert verify_plan(r.plan, r.closure) == []


# ── refusals ───────────────────────────────────────────────────────────────────────────────────────────────

def test_a_wall_time_budget_that_cannot_cover_every_test_is_refused_not_trimmed():
    g = QAGraph([tnode(f"t{i}") for i in range(4)])
    with pytest.raises(InfeasibleBudget) as e:
        plan_of(request(g, [f"t{i}" for i in range(4)], budget=PlanBudget(max_parallel=2, wall_time_ms=1500)))
    assert e.value.code == Code.BUDGET_INFEASIBLE and e.value.binding == "wall_time_ms"
    assert dict(e.value.candidates) == {1: 4000, 2: 2000}


def test_wall_time_exactly_met_is_feasible():
    g = QAGraph([tnode(f"t{i}") for i in range(4)])
    r = plan_of(request(g, [f"t{i}" for i in range(4)], budget=PlanBudget(max_parallel=2, wall_time_ms=2000)))
    assert r.makespan_ms == 2000


def test_wall_time_can_force_a_wider_plan_and_rejects_the_narrower_with_a_reason():
    g = QAGraph([tnode(f"t{i}") for i in range(4)])
    r = plan_of(request(g, [f"t{i}" for i in range(4)], budget=PlanBudget(max_parallel=4, wall_time_ms=1000)))
    assert r.chosen_lanes == 4
    assert {c.lanes: c.reason for c in r.candidates}[2] == "WALL_TIME_EXCEEDED"


def test_a_single_test_above_a_pool_capacity_is_refused():
    g = QAGraph([tnode("huge", claims=[Claim("memory_mb", 4096)]), tnode("ok")])
    with pytest.raises(InfeasibleBudget) as e:
        plan_of(request(g, ["huge", "ok"], budget=PlanBudget(max_parallel=2, capacity=(("memory_mb", 2048),))))
    assert e.value.binding == "capacity:memory_mb"


def test_a_cost_cap_below_the_total_is_refused():
    g = QAGraph([tnode("a"), tnode("b")])
    est = {"a": Estimate(10, cost_micros=600), "b": Estimate(10, cost_micros=600)}
    with pytest.raises(InfeasibleBudget) as e:
        plan_of(request(g, "ab", est, budget=PlanBudget(max_parallel=2, max_cost_micros=1000)))
    assert e.value.binding == "max_cost_micros"
    assert plan_of(request(g, "ab", est, budget=PlanBudget(max_parallel=2, max_cost_micros=1200)))


def test_a_test_that_cannot_finish_inside_the_deadline_cap_is_refused():
    g = QAGraph([tnode("a")])
    with pytest.raises(InfeasibleBudget) as e:
        plan_of(request(g, ["a"], {"a": Estimate(5000)}, caps=RunCaps(deadline_ms=4000)))
    assert e.value.binding == "deadline_ms"


def test_a_test_without_an_estimate_is_refused_and_all_are_named():
    g = QAGraph([tnode("a"), tnode("b"), tnode("c")])
    with pytest.raises(PlanningRefusal) as e:
        plan_of(request(g, "abc", {"a": Estimate(10)}))
    assert e.value.code == Code.MISSING_ESTIMATE
    assert [p.subject for p in e.value.problems] == ["test:b", "test:c"]


def test_a_claim_with_no_capacity_and_no_exclusivity_is_refused():
    g = QAGraph([tnode("a", claims=[Claim("memory_mb", 100)])])
    with pytest.raises(PlanningRefusal) as e:
        plan_of(request(g, ["a"]))
    assert e.value.code == Code.UNDECLARED_RESOURCE


def test_a_test_without_a_spec_hash_cannot_be_planned():
    g = QAGraph([Node(NodeKind.TEST, "a")])
    with pytest.raises(PlanningRefusal) as e:
        plan_of(request(g, ["a"]))
    assert e.value.code == Code.MISSING_PIN


@pytest.mark.parametrize("value", ["SUBSET", contracts.CoverageContract.SUBSET, "PARTIAL", "", None])
def test_only_full_coverage_is_accepted(value):
    g = QAGraph([tnode("a")])
    with pytest.raises(UnsupportedCoverage) as e:
        plan_of(request(g, ["a"], coverage_contract=value))
    assert e.value.code == Code.UNSUPPORTED_COVERAGE_CONTRACT


def test_full_coverage_enum_is_accepted():
    g = QAGraph([tnode("a")])
    assert plan_of(request(g, ["a"], coverage_contract=contracts.CoverageContract.FULL)).plan


def test_closure_errors_pass_through_unchanged():
    g = QAGraph([tnode("a", after=["b"]), tnode("b", after=["a"])])
    with pytest.raises(Exception) as e:
        plan_of(request(g, ["a"]))
    assert getattr(e.value, "code", None) == Code.CYCLE


@pytest.mark.parametrize("bad", [dict(max_parallel=0), dict(max_parallel=True), dict(wall_time_ms=0),
                                 dict(capacity=(("bad name", 1),)), dict(capacity=(("r", 0),)),
                                 dict(capacity=(("r", 1), ("r", 2))), dict(deadline_multiplier_pct=50)])
def test_malformed_budgets_are_refused(bad):
    with pytest.raises(PlanningRefusal):
        PlanBudget(**bad)


def test_malformed_caps_and_estimates_are_refused():
    with pytest.raises(PlanningRefusal):
        RunCaps(deadline_ms=0)
    with pytest.raises(PlanningRefusal):
        RunCaps(deadline_ms=10, max_recovery_attempts=3, recovery_ceiling=2)
    with pytest.raises(PlanningRefusal):
        RunCaps(deadline_ms=10, max_evidence="HUGE")
    with pytest.raises(PlanningRefusal):
        Estimate(0)
    with pytest.raises(PlanningRefusal):
        Estimate(10, risk_bp=10001)
    with pytest.raises(PlanningRefusal):
        Estimate(1.5)  # type: ignore[arg-type]


# ── contracts acceptance and hash reproducibility ──────────────────────────────────────────────────────────

def sample_result(seed=5, n=14, parallel=3):
    g = random_dag(seed, n)
    est = {f"t{i:03d}": Estimate(100 + 37 * i, i, (i * 911) % 10001) for i in range(n)}
    return plan_of(request(g, [f"t{i:03d}" for i in range(n)], est, budget=PlanBudget(max_parallel=parallel)))


def test_the_plan_is_a_contracts_qaplan_and_round_trips():
    r = sample_result()
    assert type(r.plan) is contracts.QAPlan
    again = contracts.QAPlan.from_json(r.plan.to_json())     # strict codec: unknown keys, bad hash and enums refused
    assert again == r.plan and again.plan_hash == r.plan.plan_hash
    assert contracts.verify_stored_hash(r.plan.to_json())


def test_the_contracts_validators_accept_the_plan_and_the_planner_refuses_what_they_refuse():
    r = sample_result()
    r.plan.validate()
    with pytest.raises(contracts.QAContractError) as e:
        dataclasses.replace(r.plan, coverage_contract=contracts.CoverageContract.SUBSET)
    assert e.value.code == contracts.Code.UNSUPPORTED_COVERAGE_CONTRACT
    assert r.plan.coverage_contract is contracts.CoverageContract.FULL


def test_hash_is_the_contracts_rcv1_hash_of_the_canonical_form():
    r = sample_result()
    assert r.plan.plan_hash == content_hash(r.plan.canonical_form())
    assert r.plan.plan_hash.startswith("rcv1:") and len(r.plan.plan_hash) == 5 + 64
    assert r.plan_hash == r.plan.plan_hash


def test_canonical_ordering_rules_hold_in_the_emitted_plan():
    r = sample_result()
    j = r.plan.to_json()
    refs = [i["spec_ref"] for i in j["items"]]
    assert refs == sorted(refs)                                           # items by spec_ref, code point order
    assert [g["item"] for g in j["gates"]] == sorted(g["item"] for g in j["gates"])
    for g in j["gates"]:
        assert g["after"] == sorted(g["after"])
    assert j["lanes"] == sorted(j["lanes"])
    assert [c["resource"] for c in j["resource_claims"]] == sorted(c["resource"] for c in j["resource_claims"])
    assert j["coverage_contract"] == "FULL" and j["planner"] == {"id": "cr-qa-planner", "version": "0.1.0"}


def test_listing_order_of_the_inputs_to_the_contracts_does_not_change_the_hash():
    r = sample_result()
    p = r.plan
    rebuilt = contracts.QAPlan(
        items=tuple(reversed(p.items)), lanes=tuple(reversed(p.lanes)), max_parallel=p.max_parallel,
        coverage_contract=p.coverage_contract, planner=p.planner, gates=tuple(reversed(p.gates)),
        resource_claims=tuple(reversed(p.resource_claims)))
    assert rebuilt.plan_hash == p.plan_hash


def test_the_plan_hash_is_reproducible_within_and_across_input_orderings():
    ref = sample_result().plan.plan_hash
    for _ in range(3):
        assert sample_result().plan.plan_hash == ref
    g = random_dag(5, 14)
    nodes = list(g.nodes())
    random.Random(2).shuffle(nodes)
    est = {f"t{i:03d}": Estimate(100 + 37 * i, i, (i * 911) % 10001) for i in reversed(range(14))}
    targets = [f"t{i:03d}" for i in reversed(range(14))]
    shuffled = plan_of(request(QAGraph(nodes), targets, est, budget=PlanBudget(max_parallel=3)))
    assert shuffled.plan.plan_hash == ref


def test_a_changed_input_changes_the_hash_and_a_changed_planner_identity_would():
    base = sample_result().plan.plan_hash
    assert sample_result(parallel=2).plan.plan_hash != base or sample_result(parallel=2).chosen_lanes != 3
    assert sample_result(seed=6).plan.plan_hash != base


def test_action_policy_ref_is_hashed_into_the_plan_when_given():
    g = QAGraph([tnode("a")])
    ref = "rcv1:" + "0" * 64
    without = plan_of(request(g, ["a"])).plan
    with_ref = plan_of(request(g, ["a"], action_policy_ref=ref)).plan
    assert with_ref.action_policy_ref == ref and without.action_policy_ref is None
    assert with_ref.plan_hash != without.plan_hash


# ── the planner's own validator ────────────────────────────────────────────────────────────────────────────

def _rebuild(plan, **kw):
    fields = dict(items=plan.items, lanes=plan.lanes, max_parallel=plan.max_parallel,
                  coverage_contract=plan.coverage_contract, planner=plan.planner, gates=plan.gates,
                  resource_claims=plan.resource_claims, action_policy_ref=plan.action_policy_ref)
    fields.update(kw)
    return contracts.QAPlan(**fields)


def test_validator_rejects_a_plan_with_an_omission():
    r = sample_result()
    dropped = r.plan.items[-1].spec_ref
    items = tuple(i for i in r.plan.items if i.spec_ref != dropped)
    gates = tuple(g for g in r.plan.gates if g.item != dropped)
    gates = tuple(contracts.PlanGate(item=g.item, after=tuple(a for a in g.after if a != dropped))
                  for g in gates if [a for a in g.after if a != dropped])
    short = _rebuild(r.plan, items=items, gates=gates)
    problems = verify_plan(short, r.closure)
    assert problems and any(dropped in p.subject for p in problems)


def test_validator_rejects_a_plan_with_an_extra_item():
    r = sample_result()
    extra = contracts.PlanItem(spec_ref=contracts.make_spec_ref("stray", "rcv1:" + "1" * 64), lane=r.plan.lanes[0],
                               order=99, budget=contracts.Budget(deadline_ms=1000), recovery_ceiling=0)
    bigger = _rebuild(r.plan, items=(*r.plan.items, extra))
    assert any("not in the closure" in p.detail for p in verify_plan(bigger, r.closure))


def test_validator_rejects_unordered_conflicts_and_missing_precedence():
    g = QAGraph([fixture_node("db", exclusive=True), tnode("a", fixtures=["db"]), tnode("b", fixtures=["db"]),
                 tnode("c", after=["a"])])
    r = plan_of(request(g, "abc", budget=PlanBudget(max_parallel=3)))
    # two lanes, no gates at all: a and b share a lock, c follows a
    refs = {i.spec_ref.split("@")[0]: i.spec_ref for i in r.plan.items}
    items = (
        dataclasses.replace(next(i for i in r.plan.items if i.spec_ref == refs["a"]), lane="lane-0", order=0),
        dataclasses.replace(next(i for i in r.plan.items if i.spec_ref == refs["b"]), lane="lane-1", order=0),
        dataclasses.replace(next(i for i in r.plan.items if i.spec_ref == refs["c"]), lane="lane-2", order=0),
    )
    bad = _rebuild(r.plan, items=items, lanes=("lane-0", "lane-1", "lane-2"), max_parallel=3, gates=())
    details = " ".join(p.detail for p in verify_plan(bad, r.closure))
    assert "same exclusive resource" in details and "does not wait for" in details


def test_validator_rejects_budgets_above_caps_and_lane_count_above_max_parallel():
    r = sample_result()
    tight = RunCaps(deadline_ms=10, max_recovery_attempts=0, recovery_ceiling=0)
    assert verify_plan(r.plan, r.closure, caps=tight)
    assert verify_plan(r.plan, r.closure, budget=PlanBudget(max_parallel=1))
    assert verify_plan(r.plan, r.closure, budget=r.request.budget, caps=r.request.caps) == []


def test_validator_flags_a_pool_that_the_lane_count_could_overcommit():
    claims = [Claim("memory_mb", 600)]
    g = QAGraph([tnode(f"t{i}", claims=claims) for i in range(4)])
    r = plan_of(request(g, [f"t{i}" for i in range(4)], budget=PlanBudget(max_parallel=4, capacity=(("memory_mb", 1300),))))
    assert verify_plan(r.plan, r.closure, budget=r.request.budget) == []
    assert verify_plan(r.plan, r.closure, budget=PlanBudget(max_parallel=4, capacity=(("memory_mb", 1000),)))


# ── property tests ─────────────────────────────────────────────────────────────────────────────────────────

def _random_case(seed):
    rng = random.Random(seed)
    n = rng.randint(4, 36)
    graph = random_dag(seed, n, p_edge=rng.choice([0.05, 0.15, 0.3]), n_fixtures=rng.randint(1, 4),
                       p_fixture=rng.choice([0.2, 0.4]), p_exclusive=rng.choice([0.3, 0.7, 1.0]))
    names = [f"t{i:03d}" for i in range(n)]
    targets = rng.sample(names, rng.randint(1, n))
    est = {t: Estimate(rng.randint(10, 900), rng.randint(0, 50), rng.randint(0, 10000)) for t in names}
    budget = PlanBudget(max_parallel=rng.randint(1, 6))
    return graph, targets, est, budget


@pytest.mark.parametrize("seed", range(40))
def test_property_every_selected_test_is_planned_exactly_once_with_full_coverage(seed):
    graph, targets, est, budget = _random_case(seed)
    r = plan_of(request(graph, targets, est, budget=budget))
    want = {i.split("@")[0] for i in [x.spec_ref for x in r.plan.items]}
    assert want == set(r.closure.test_ids)
    assert len(r.plan.items) == len(set(i.spec_ref for i in r.plan.items)) == len(r.closure.test_ids)
    assert set(targets) <= want
    assert r.plan.coverage_contract is contracts.CoverageContract.FULL
    assert len(r.plan.lanes) <= budget.max_parallel and r.plan.max_parallel == len(r.plan.lanes)
    assert verify_plan(r.plan, r.closure, budget=budget, caps=CAPS) == []


@pytest.mark.parametrize("seed", range(40))
def test_property_exclusive_claimants_are_never_concurrent_whatever_the_durations(seed):
    graph, targets, est, budget = _random_case(seed)
    r = plan_of(request(graph, targets, est, budget=budget))
    locks = locks_of(graph)
    ref = {i.spec_ref.split("@")[0]: i.spec_ref for i in r.plan.items}
    ids = sorted(ref)
    for trial in range(6):
        done = run_plan(r.plan, random_durations(r.plan, seed * 100 + trial))
        assert max_overlap(done) <= budget.max_parallel
        for ai, a in enumerate(ids):
            for b in ids[ai + 1:]:
                if locks[a] & locks[b]:
                    assert not overlaps(done[ref[a]], done[ref[b]]), (a, b, locks[a] & locks[b])
        for t in ids:                                   # precedence holds at run time as well
            for p in r.closure.predecessors(t):
                assert done[ref[p]][1] <= done[ref[t]][0]


@pytest.mark.parametrize("seed", range(15))
def test_property_pool_capacity_is_never_exceeded_whatever_the_durations(seed):
    rng = random.Random(seed)
    n = rng.randint(4, 20)
    amounts = {f"t{i:03d}": rng.choice([100, 200, 300, 500]) for i in range(n)}
    graph = QAGraph([tnode(t, claims=[Claim("memory_mb", a)]) for t, a in amounts.items()])
    cap = rng.choice([600, 900, 1200, 2000])
    budget = PlanBudget(max_parallel=6, capacity=(("memory_mb", cap),))
    r = plan_of(request(graph, list(amounts), {t: Estimate(rng.randint(10, 500)) for t in amounts}, budget=budget))
    ref = {i.spec_ref.split("@")[0]: i.spec_ref for i in r.plan.items}
    weight = {ref[t]: a for t, a in amounts.items()}
    for trial in range(8):
        done = run_plan(r.plan, random_durations(r.plan, seed * 31 + trial))
        assert max_overlap(done, lambda x: weight[x]) <= cap


def test_a_larger_instance_plans_and_verifies():
    graph = random_dag(99, 120, p_edge=0.04, n_fixtures=5)
    names = [f"t{i:03d}" for i in range(120)]
    r = plan_of(request(graph, names, CostModel({}, default=Estimate(250)), budget=PlanBudget(max_parallel=8)))
    assert len(r.plan.items) == 120 and verify_plan(r.plan, r.closure) == []


# ── the planner interface ──────────────────────────────────────────────────────────────────────────────────

class SingleLaneInOrder:
    """A minimal planner that does no planning: the order given, one lane. It lives here only to show that a planner
    with another strategy implements the same interface and returns the same type."""

    id = "single-lane-test"
    version = "0.0.1"

    def plan(self, request: PlanRequest):
        closure = resolve(request.graph, request.targets)
        pins = {n.id: n.pin for n in closure.tests}
        order = [t for t in request.targets if t in pins] + [t for t in closure.test_ids if t not in request.targets]
        ref = {t: contracts.make_spec_ref(t, pins[t]) for t in order}
        items = tuple(contracts.PlanItem(
            spec_ref=ref[t], lane="lane-0", order=k,
            budget=contracts.Budget(deadline_ms=request.caps.deadline_ms), recovery_ceiling=request.caps.recovery_ceiling)
            for k, t in enumerate(order))
        return contracts.QAPlan(items=items, lanes=("lane-0",), max_parallel=1,
                                coverage_contract=contracts.CoverageContract.FULL,
                                planner=contracts.ComponentRef(id=self.id, version=self.version))


def test_two_planners_with_different_strategies_return_the_same_plan_type():
    g = QAGraph([tnode(n) for n in "abc"])
    req = request(g, ["c", "a", "b"], budget=PlanBudget(max_parallel=3))
    cost, simple = CostBasedQAPlanner(), SingleLaneInOrder()
    assert isinstance(cost, QAPlanner) and isinstance(simple, QAPlanner)
    a, b = cost.plan(req), simple.plan(req)
    assert type(a) is type(b) is contracts.QAPlan
    assert a.planner != b.planner and a.plan_hash != b.plan_hash
    assert [i.spec_ref for i in sorted(b.items, key=lambda i: i.order)][0].startswith("c@")   # the given order is kept
    closure = resolve(g, ["a", "b", "c"])
    assert verify_plan(a, closure) == [] and verify_plan(b, closure) == []


def test_planner_module_performs_no_io_and_imports_no_runner():
    import ast
    import inspect

    for mod in (planner_module, __import__("context_runtime.qa.closure", fromlist=["x"]),
                __import__("context_runtime.qa.cost", fromlist=["x"])):
        tree = ast.parse(inspect.getsource(mod))
        calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert not calls & {"open", "print", "input", "exec", "eval", "compile", "__import__"}, mod.__name__
        imported = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imported.update(a.name.split(".")[0] for a in n.names)
            elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
                imported.add(n.module.split(".")[0])
        assert not imported & {"os", "subprocess", "socket", "pathlib", "shutil", "urllib", "http", "playwright",
                               "time", "datetime", "random", "agent_harness"}, (mod.__name__, imported)
