"""Contract tests for Adaptive Context Materialization (F2).

The load-bearing test is the acceptance criterion: choosing the DEFAULT depth leaves the bandit arm key
byte-identical to the OSS `plan_key`, so existing learned values and plan-cache entries are untouched. Plus:
the depth folds into the key only when non-default, and the escalation ladder is deterministic and
cost-minimising. Requires the public `context_runtime` package; skipped cleanly when absent.
"""
from __future__ import annotations

import pytest


from context_runtime.optimizer.online import plan_key  # noqa: E402
from context_runtime.types import Candidate, StepSpec  # noqa: E402

from context_runtime.optimizer.materialization import (  # noqa: E402
    DEFAULT_DEPTH, Depth, MaterializationLadder, materialization_arm, plan_key_with_materialization,
)


def _cand(method="hybrid", strat="single_shot", tier="local"):
    return Candidate(steps=(StepSpec(type="retrieve", params={"method": method}),
                            StepSpec(type="reason", params={"strategy": strat})), model_tier=tier)


# ── the acceptance criterion ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("cand", [
    _cand("hybrid", "single_shot", "local"),          # base key: "hybrid:local"
    _cand("bm25", "single_shot", "premium"),          # base key: "bm25:premium"
    _cand("hybrid", "best_of_n", "local"),            # base key folds genstrat: "hybrid:best_of_n:local"
])
def test_default_depth_preserves_plan_key_identity(cand):
    # F2 present but choosing the default depth must not change ANY arm's key.
    assert plan_key_with_materialization(cand, DEFAULT_DEPTH) == plan_key(cand)
    assert materialization_arm(plan_key(cand), DEFAULT_DEPTH) == plan_key(cand)


def test_non_default_depth_folds_into_the_key():
    base = plan_key(_cand("hybrid", "single_shot", "local"))   # "hybrid:local"
    assert materialization_arm(base, Depth.STATE_SPARSE) == "hybrid:local:m=state_sparse"
    assert materialization_arm(base, Depth.STATE_ONLY) == "hybrid:local:m=state_only"
    assert materialization_arm(base, Depth.FULL_CONTEXT) == "hybrid:local:m=full_context"
    # …and every non-default depth yields a key distinct from the default (so the bandit learns per depth)
    keys = {materialization_arm(base, d) for d in Depth}
    assert len(keys) == len(Depth)


# ── the escalation ladder ─────────────────────────────────────────────────────────────────────────

def test_stops_at_cheapest_sufficient_depth():
    # state can't answer, but sparse can → pick STATE_SPARSE (never pay for deep/full).
    probes = {Depth.STATE_ONLY: lambda: False, Depth.STATE_SPARSE: lambda: True, Depth.STATE_DEEP: lambda: True}
    choice = MaterializationLadder().select("hybrid:local", probes)
    assert choice.depth == Depth.STATE_SPARSE
    assert choice.arm == "hybrid:local:m=state_sparse"
    assert choice.escalations == (Depth.STATE_ONLY,)


def test_state_only_when_sufficient():
    probes = {Depth.STATE_ONLY: lambda: True, Depth.STATE_SPARSE: lambda: True}
    choice = MaterializationLadder().select("hybrid:local", probes)
    assert choice.depth == Depth.STATE_ONLY and choice.escalations == ()


def test_escalates_to_full_when_nothing_suffices():
    # every cheaper probe fails → FULL_CONTEXT is the last resort (always has the evidence).
    probes = {d: (lambda: False) for d in (Depth.STATE_ONLY, Depth.STATE_SPARSE, Depth.STATE_DEEP)}
    choice = MaterializationLadder().select("hybrid:local", probes)
    assert choice.depth == Depth.FULL_CONTEXT
    assert choice.escalations == (Depth.STATE_ONLY, Depth.STATE_SPARSE, Depth.STATE_DEEP)
    assert choice.arm == "hybrid:local:m=full_context"


def test_floor_pins_a_minimum_depth():
    # a high-risk intent may forbid STATE_ONLY; the ladder starts at the floor even if a cheaper probe passes.
    probes = {Depth.STATE_ONLY: lambda: True, Depth.STATE_SPARSE: lambda: True}
    choice = MaterializationLadder(floor=Depth.STATE_SPARSE).select("hybrid:local", probes)
    assert choice.depth == Depth.STATE_SPARSE


def test_deterministic():
    probes = {Depth.STATE_ONLY: lambda: False, Depth.STATE_SPARSE: lambda: True}
    a = MaterializationLadder().select("k", probes)
    b = MaterializationLadder().select("k", probes)
    assert a == b
