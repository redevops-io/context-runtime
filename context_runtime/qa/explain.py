"""A machine-readable explanation of a plan, hashed so it can be attached as evidence.

``explain_plan`` turns a ``PlanResult`` into a record that says, for the plan the planner chose:

* why each test is in the plan (its closure reasons) and what else the closure pulled in;
* why each test sits in its lane and position: the earliest start on every lane, the rule that picked one, whether a
  tie was broken by lane index, what bound its start (precedence, a conflict, or the lane being busy), and which gate
  it carries and why (precedence, or a conflict on a named exclusive resource);
* which constraints bound the choice: the lane limits (``max_parallel``, pool capacities), the critical path, the wall
  time, the conflict keys;
* the critical path and the estimated makespan with the basis of the estimates behind it;
* the candidates that were considered and, for each rejected one, the reason.

The record holds integers and strings only and no clock, so the same inputs give the same record and the same hash.
The hash is ``rcv1`` (the contracts' canonical hashing). A plan carries it as ``rationale_ref``, which is custody and
does not change the plan hash.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from ._contracts import canonical_json, content_hash
from .closure import NodeKind
from .cost import bottom_levels
from .planner import PlanResult

EXPLANATION_SCHEMA_ID = "context-runtime/qa-plan-explanation"
EXPLANATION_SCHEMA_VERSION = "1"

_SELECTION_RULE = ("candidates are lane counts from 1 up to the lane limit; each is list-scheduled by priority "
                   "(longest remaining path, then risk, then duration, then identifier); the lowest estimated "
                   "makespan wins, then the fewest lanes, then the smallest lane count")
_LANE_RULE = "the lane with the earliest possible start; the lowest lane index on a tie"


@dataclass(frozen=True)
class PlanExplanation:
    record: dict
    """The explanation: JSON-compatible, integers and strings only."""
    hash: str
    """``rcv1`` hash of ``record``."""

    def to_json(self) -> dict:
        return {**self.record, "explanation_hash": self.hash}

    def canonical_text(self) -> str:
        return canonical_json(self.record)

    @staticmethod
    def verify(payload: dict) -> bool:
        """True when a received explanation's stored hash equals the hash of its own record."""
        stored = payload.get("explanation_hash")
        body = {k: v for k, v in payload.items() if k != "explanation_hash"}
        return isinstance(stored, str) and content_hash(body) == stored


@dataclass(frozen=True)
class ExplainedPlan:
    plan: object
    """The ``QAPlan``, with ``rationale_ref`` set to the explanation hash (its ``plan_hash`` is unchanged)."""
    explanation: PlanExplanation
    result: PlanResult


def explain_plan(result: PlanResult) -> PlanExplanation:
    closure, plan = result.closure, result.plan
    levels = bottom_levels(closure, result.estimates)
    ref = {s.test_id: s.spec_ref for s in result.schedule}
    used = result.chosen_lanes
    budgets = {i.spec_ref: i for i in plan.items}
    schedule = {s.test_id: s for s in result.schedule}
    position = {s.test_id: s.position for s in result.schedule}

    items = []
    gate_records = []
    for s in result.schedule:
        t = s.test_id
        node = closure.node(NodeKind.TEST, t)
        starts = s.lane_starts[:used]
        best = min(starts)
        tied = [k for k, v in enumerate(starts) if v == best]
        chosen_index = int(s.lane.rsplit("-", 1)[1])
        if len(starts) == 1:
            decided = "the only lane"
        elif len(tied) == 1:
            decided = "earliest start"
        else:
            decided = "earliest start; tie broken by lowest lane index"
        mine = {c.resource for c in closure.claims(t) if c.exclusive}
        required_conflicts = sorted(j for j in closure.test_ids
                                    if position[j] < s.position and closure.conflicts(t, j))
        required = sorted(set(closure.predecessors(t)) | set(required_conflicts))
        gated = list(s.gate_after)
        for a in gated:
            why = []
            if a in closure.predecessors(t):
                why.append("precedence")
            shared = sorted(mine & {c.resource for c in closure.claims(a) if c.exclusive})
            why.extend(f"conflict:{r}" for r in shared)
            gate_records.append({"item": s.spec_ref, "after": ref[a], "reasons": why})
        est = result.estimates[t]
        b = budgets[s.spec_ref]
        items.append({
            "position": s.position, "spec_ref": s.spec_ref, "test_id": t,
            "included": [r.to_dict() | {"text": r.text()} for r in node.reasons],
            "lane": s.lane, "order": s.order,
            "lane_assignment": {
                "rule": _LANE_RULE, "decided_by": decided, "chosen_lane": s.lane,
                "options": [{"lane": f"lane-{k}", "earliest_start_ms": v} for k, v in enumerate(starts)],
                "tied_lanes": [f"lane-{k}" for k in tied], "start_bound_by": s.bound_by,
                "ready_ms": s.ready_ms, "conflict_ms": s.conflict_ms,
            },
            "estimated_start_ms": s.start_ms, "estimated_finish_ms": s.finish_ms,
            "gate_after": [ref[a] for a in gated],
            "ordered_without_gate": [ref[a] for a in required if a not in gated],
            "conflict_keys": sorted(mine),
            "priority": {"remaining_path_ms": levels[t], "risk_bp": est.risk_bp, "duration_ms": est.duration_ms},
            "estimate": est.to_dict(),
            "budget": {"deadline_ms": b.budget.deadline_ms, "max_retries": b.budget.max_retries,
                       "max_recovery_attempts": b.budget.max_recovery_attempts,
                       "max_model_calls": b.budget.max_model_calls, "max_tokens": b.budget.max_tokens,
                       "recovery_ceiling": b.recovery_ceiling},
            "evidence_level": b.evidence_level.value,
        })

    wall = result.request.budget.wall_time_ms
    candidates = [{"lanes": c.lanes, "status": c.status, "reason": c.reason, "makespan_ms": c.makespan_ms,
                   "lanes_used": c.lanes_used, "detail": c.detail} for c in result.candidates]
    ruled_out = next((c for c in result.candidates if c.status == "unavailable"), None)
    constraints = []
    for name in sorted(result.limits):
        limit = result.limits[name]
        # binding: it is the limit that stopped the search, and one more lane would have been faster
        binding = (name != "tests" and limit == used and ruled_out is not None
                   and ruled_out.makespan_ms is not None and ruled_out.makespan_ms < result.makespan_ms)
        constraints.append({"constraint": name, "limit": limit, "observed": used, "binding": binding,
                            "effect": "a larger lane count is not available"})
    constraints.append({"constraint": "critical_path", "limit": result.critical_path_ms,
                        "observed": result.makespan_ms, "binding": result.makespan_ms == result.critical_path_ms,
                        "effect": "no lane count shortens the makespan below the longest dependency chain"})
    if wall is not None:
        constraints.append({"constraint": "wall_time_ms", "limit": wall, "observed": result.makespan_ms,
                            "binding": any(c.reason == "WALL_TIME_EXCEEDED" for c in result.candidates),
                            "effect": "narrower candidates exceed the wall time"})
    cost_cap = result.request.budget.max_cost_micros
    if cost_cap is not None:
        constraints.append({"constraint": "max_cost_micros", "limit": cost_cap,
                            "observed": result.total_cost_micros, "binding": False,
                            "effect": "feasibility only: the plan is refused when the total is above the cap"})
    locks = sorted({r for t in closure.test_ids for r in closure.exclusive_keys(t)})
    for r in locks:
        users = [t for t in closure.test_ids if r in closure.exclusive_keys(t)]
        constraints.append({"constraint": f"conflict:{r}", "limit": 1, "observed": len(users),
                            "binding": any(schedule[t].bound_by == "conflict" for t in users),
                            "effect": "claimants of this exclusive resource are ordered, never concurrent"})

    lower = max(result.critical_path_ms, -(-result.total_duration_ms // used))
    bases = sorted({e.basis for e in result.estimates.values()})
    record = {
        "schema_id": EXPLANATION_SCHEMA_ID, "schema_version": EXPLANATION_SCHEMA_VERSION,
        "plan_hash": plan.plan_hash,
        "planner": {"id": plan.planner.id, "version": plan.planner.version},
        "inputs": {"request_digest": result.request_digest, "closure_digest": closure.digest(),
                   "estimates_digest": result.request.estimates.digest(),
                   "budget": result.request.budget.to_dict(), "caps": result.request.caps.to_dict()},
        "coverage_contract": plan.coverage_contract.value,
        "closure": [{"key": n.key, "reasons": [r.text() for r in n.reasons]} for n in closure.nodes],
        "items": items,
        "gates": gate_records,
        "selection": {"rule": _SELECTION_RULE, "lanes_chosen": used, "limits": dict(sorted(result.limits.items())),
                      "candidates": candidates,
                      "rejected": [c for c in candidates if c["status"] != "selected"]},
        "binding_constraints": constraints,
        "critical_path": {"length_ms": result.critical_path_ms,
                          "items": [ref[t] for t in result.critical_path],
                          "basis": "precedence only, by estimated duration"},
        "makespan": {"estimated_ms": result.makespan_ms, "basis": bases, "lower_bound_ms": lower,
                     "total_duration_ms": result.total_duration_ms, "total_cost_micros": result.total_cost_micros,
                     "note": "estimated from the declared figures only; not a measurement and not a promise"},
        "guarantees": {
            "structural": ["every test of the closure is planned exactly once (FULL coverage)",
                           "at most max_parallel tests run at once",
                           "tests that share an exclusive resource are ordered, never concurrent",
                           "the largest claimants together fit every pool capacity"],
            "estimated": ["start and finish times", "makespan"],
        },
    }
    return PlanExplanation(record, content_hash(record))


def plan_with_explanation(planner, request) -> ExplainedPlan:
    """Plan, explain, and attach the explanation's hash to the plan as ``rationale_ref``."""
    result = planner.plan_detailed(request)
    explanation = explain_plan(result)
    plan = dataclasses.replace(result.plan, rationale_ref=explanation.hash)
    assert plan.plan_hash == result.plan.plan_hash      # custody only: the plan's identity is unchanged
    return ExplainedPlan(plan, explanation, result)
