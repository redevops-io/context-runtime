"""Resolution Closure × Context Runtime tenant: composition arms (Gate 2) + ensemble verifier (Gate 4)."""
from __future__ import annotations

from context_runtime.integrations.resolution_closure import (
    CompositionArm,
    DEFAULT_ARMS,
    EnsembleVerdict,
    ResolutionClosureTenant,
    _closure_bandit,
    budget_fill,
    compose_order,
    ensemble_verdict,
    reward_closure,
)


# ── portable seam 1: composition (must match parity golden.json) ─────────────────────────────
def test_compose_order_seed_first_then_total_then_id():
    order = compose_order(["s"], {"s": 0.5, "a": 1.0, "b": 0.8, "c": 0.3},
                          {"s": 1.0, "b": 0.9, "d": 0.7}, alpha=0.6)
    assert order == ["s", "b", "a", "d", "c"]      # seed, then b(1.34), a(1.0), d(0.42), c(0.3)


def test_compose_order_tiebreak_is_unit_id():
    # two units with identical total → deterministic ascending-id tiebreak
    order = compose_order([], {"y": 1.0, "x": 1.0}, {}, alpha=0.0)
    assert order == ["x", "y"]


def test_budget_fill_breaks_at_first_overflow():
    kept = budget_fill(["s", "b", "a", "d", "c"], {k: 100 for k in "sabdc"}, budget=300)
    assert kept == {"s", "b", "a"}


def test_structure_first_and_content_only_arms():
    seed, cs, ss = ["s"], {"s": 0.2, "a": 0.9, "e": 0.1}, {"s": 1.0, "e": 1.0, "f": 0.8}
    assert CompositionArm("content_only").order(seed, cs, ss) == ["a", "s", "e"]
    assert CompositionArm("structure_first").order(seed, cs, ss)[0] in {"s", "e"}   # struct leads


# ── portable seam 2: ensemble verifier (must match parity golden.json) ────────────────────────
def test_ensemble_verdicts_and_confidence():
    assert ensemble_verdict("x", {"a": True, "b": True, "c": True}).verdict == "VERIFIED"
    assert ensemble_verdict("x", {"a": False, "b": False, "c": False}).verdict == "EXCLUDED"
    mid = ensemble_verdict("x", {"a": True, "b": True, "c": False})
    assert mid.verdict == "VERIFIED" and mid.disputed and abs(mid.confidence - 1 / 3) < 1e-9
    split = ensemble_verdict("x", {"a": True, "b": True, "c": False, "d": False})
    assert split.verdict == "DISPUTED" and split.confidence == 0.0


def test_ensemble_verdict_is_frozen_dataclass():
    ev = ensemble_verdict("dep", {"a": True, "b": False, "c": True})
    assert isinstance(ev, EnsembleVerdict) and ev.votes_yes == 2 and ev.n == 3


# ── reward + arms ─────────────────────────────────────────────────────────────────────────────
def test_reward_charges_structural_promotion_cost():
    a0 = CompositionArm("content_led", 0.0)
    a6 = CompositionArm("content_led", 0.6)
    # equal recall → the cheaper (α=0) composition earns more
    assert reward_closure(0.6, a0) > reward_closure(0.6, a6)
    # α=0 has no structural cost → reward equals recall
    assert reward_closure(0.589, a0) == 0.589


def test_arm_keys_unique_and_stable():
    keys = [a.key for a in DEFAULT_ARMS]
    assert len(keys) == len(set(keys))
    assert CompositionArm("content_led", 0.4).key == "content_led:0.4"
    assert CompositionArm("content_only").key == "content_only"


# ── tenant surface + loop closure ─────────────────────────────────────────────────────────────
def _task(cap, req, votes=None):
    return {"request": req, "capability": cap, "seed": ["u0"],
            "content_score": {"u0": 0.5, "u1": 1.0, "u2": 0.4},
            "struct_score": {"u0": 1.0, "u2": 0.8}, "budget": 200,
            "tokens": {"u0": 60, "u1": 60, "u2": 60}, "votes": votes or {}}


def test_resolve_stashes_pending_and_returns_closure():
    t = ResolutionClosureTenant()
    res = t.resolve(_task("code", "signature change"))
    assert res.arm in DEFAULT_ARMS
    assert res.closure and t._key("signature change") in t._pending


def test_resolve_routes_disputed_to_review():
    t = ResolutionClosureTenant()
    votes = {"u1": {"a": True, "b": True, "c": False}}   # split on a kept member
    res = t.resolve(_task("legal", "carve-out", votes=votes))
    assert "u1" in res.routed and res.status in {"DISPUTED", "REQUIRE_REVIEW"}


def test_record_outcome_closes_loop_and_calibrates():
    t = ResolutionClosureTenant()
    before = t.runtime.estimator.statistics().fields[0].sample_count
    t.resolve(_task("code", "add field"))
    r = t.record_outcome("add field", 0.9)
    after = t.runtime.estimator.statistics().fields[0].sample_count
    assert after == before + 1 and 0 < r <= 1


def test_tenant_learns_per_capability_composition():
    """The core Gate-2 claim: the bandit learns DIFFERENT optima per capability from the frozen recall surface."""
    cell = {"code":  {"content_led:0.6": 0.96, "content_led:0": 0.86},
            "legal": {"content_led:0.6": 0.582, "content_led:0": 0.589}}
    t = ResolutionClosureTenant(bandit=_closure_bandit(0.1))
    for _ in range(80):
        for cap, req in (("code", "code task"), ("legal", "legal task")):
            res = t.resolve(_task(cap, req))
            recall = cell[cap].get(res.arm.key, 0.4)     # only these two arms are strong; others weak
            t.record_outcome(req, recall)
    pol = t.policy()
    # code prefers structural promotion; legal prefers pure content — DIFFERENT arms
    assert pol["code"] == "content_led:0.6"
    assert pol["legal"] == "content_led:0"
    assert pol["code"] != pol["legal"]
