"""Hard constraints — feasibility, separate from the soft PlanScore (SPEC §3).

A candidate that violates any hard ceiling/requirement is infeasible and excluded
*regardless* of its score. v0.1 enforces the numeric ceilings + boolean requirements;
CP-SAT (v0.2) handles the case where several interact across a multi-step plan.
"""
from __future__ import annotations

from ..types import Candidate, Constraints, PlanScore


def feasible(candidate: Candidate, score: PlanScore, c: Constraints) -> tuple[bool, str | None]:
    """Return (is_feasible, reason_if_not)."""
    if c.max_cost_usd is not None and score.cost_usd > c.max_cost_usd:
        return False, f"cost ${score.cost_usd:.2f} > max ${c.max_cost_usd:.2f}"
    if c.max_latency_seconds is not None and score.latency_seconds > c.max_latency_seconds:
        return False, f"latency {score.latency_seconds:.0f}s > max {c.max_latency_seconds:.0f}s"
    if c.require_citations and not any(s.type == "verify" for s in candidate.steps):
        return False, "require_citations but no verify step"
    if c.require_verification and not any(s.type == "verify" for s in candidate.steps):
        return False, "require_verification but no verify step"
    if c.sensitivity == "restricted" and candidate.model_tier != "local":
        return False, f"restricted data cannot use tier '{candidate.model_tier}'"
    return True, None
