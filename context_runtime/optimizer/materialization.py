"""Adaptive Context Materialization (F2) — make how much context to materialize a planner decision.

The audit (§2.4) proposes four materialization depths, cheapest → most expensive:

    STATE_ONLY    ContextState → model                                   (no retrieval)
    STATE_SPARSE  ContextState → region retrieval (F4) → evidence → model (sparse)
    STATE_DEEP    ContextState → multi-method retrieval → rerank → model  (the current default)
    FULL_CONTEXT  large raw/reconstructed history → model                 (everything)

Today the runtime always materializes at one fixed depth. F2 turns depth into a decision: escalate only as
far as the evidence requires. The value is on the frontier — a query answerable from state or a sparse
region should not pay for a deep multi-method assembly, and only the rare query that needs it pays for
FULL. "Context escalation occurs when sparse evidence is insufficient" (plan §Slice-2).

**Acceptance criterion — preserve the default bandit `plan_key` identity.** The depth is an arm axis, and
like the generation-strategy fold in the `plan_key`, it is folded into the key **only when the depth is
non-default**. So with F2 present but choosing the default depth (`STATE_DEEP`), every arm keys byte-for-byte
as it does today — existing learned bandit values and plan-cache entries are untouched. This module never
modifies the `plan_key`; it composes on top of it.

The depth→pipeline execution (what STATE_SPARSE actually retrieves) lives in the caller/deployment, so this
module stays pure and dependency-light: it decides the depth and computes the identity-preserving arm key;
the caller runs the matching pipeline (STATE_SPARSE → the F4 `SparseRegionRetriever`, etc.). It uses only
the `plan_key`/`Candidate` types from the optimizer, and composes on `plan_key` — it never modifies it.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable

from context_runtime.optimizer.online import plan_key
from context_runtime.types import Candidate


class Depth(IntEnum):
    """Materialization depth, ordered by cost (cheapest first) so escalation is a simple increase."""
    STATE_ONLY = 0
    STATE_SPARSE = 1
    STATE_DEEP = 2
    FULL_CONTEXT = 3

    @property
    def label(self) -> str:
        return self.name.lower()


# The default depth is the runtime's current behaviour: multi-method retrieval + rerank. Choosing it must
# leave the arm key identical to today's, so it is the one depth that is NOT folded into the key.
DEFAULT_DEPTH = Depth.STATE_DEEP


def materialization_arm(base_arm: str, depth: Depth) -> str:
    """Fold the materialization depth into a bandit arm key — but only when it is non-default, exactly like
    the `plan_key` folds a non-`single_shot` generation strategy. Default depth → base arm unchanged."""
    if depth == DEFAULT_DEPTH:
        return base_arm
    return f"{base_arm}:m={depth.label}"


def plan_key_with_materialization(candidate: Candidate, depth: Depth) -> str:
    """The full arm for a (candidate, depth): the `plan_key` with the depth folded in when non-default.
    Invariant (tested): ``plan_key_with_materialization(c, DEFAULT_DEPTH) == plan_key(c)`` for every c."""
    return materialization_arm(plan_key(candidate), depth)


# A depth probe answers "is the evidence materialized at this depth sufficient to answer?" It is a cheap,
# deterministic signal supplied by the caller — e.g. STATE_SPARSE's probe is whether F4's region routing
# cleared its confidence floor. No probe means "cannot decide at this depth" → escalate.
Probe = Callable[[], bool]


@dataclass(frozen=True)
class MaterializationChoice:
    depth: Depth
    arm: str                       # identity-preserving arm key for this (base arm, depth)
    reason: str
    escalations: tuple[Depth, ...]  # depths tried and found insufficient before this one


class MaterializationLadder:
    """Deterministic cost-minimising escalation: try depths cheapest-first and stop at the first whose probe
    reports sufficient; if none do, materialize FULL_CONTEXT (the last resort that always has the evidence).

    This is a planner decision, not an LLM call. It composes with F4: the STATE_SPARSE probe is naturally
    "did the sparse region retriever clear its confidence floor" — so F2 escalates past sparse exactly when
    F4 would itself have fallen back.
    """

    def __init__(self, *, floor: Depth = Depth.STATE_ONLY, ceiling: Depth = Depth.FULL_CONTEXT):
        # A deployment can pin a minimum/maximum depth (e.g. never STATE_ONLY for high-risk intents).
        self.floor = floor
        self.ceiling = ceiling

    def select(self, base_arm: str, probes: dict[Depth, Probe]) -> MaterializationChoice:
        tried: list[Depth] = []
        for depth in Depth:
            if depth < self.floor or depth > self.ceiling:
                continue
            probe = probes.get(depth)
            if depth == self.ceiling:                       # last resort: always sufficient by construction
                reason = (f"escalated to {depth.label} (last resort)" if tried
                          else f"{depth.label} (only depth in range)")
                return MaterializationChoice(depth, materialization_arm(base_arm, depth), reason, tuple(tried))
            if probe is not None and probe():
                reason = (f"{depth.label} sufficient" if not tried
                          else f"escalated to {depth.label} after {[d.label for d in tried]}")
                return MaterializationChoice(depth, materialization_arm(base_arm, depth), reason, tuple(tried))
            tried.append(depth)
        # ceiling was below the highest Depth and nothing matched → materialize at the ceiling.
        return MaterializationChoice(self.ceiling, materialization_arm(base_arm, self.ceiling),
                                     f"escalated to {self.ceiling.label} (ceiling)", tuple(tried))
