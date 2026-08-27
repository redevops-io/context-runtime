"""Contract tests for Execution Pattern Memory (F3).

The load-bearing properties: F3 warm-starts the bandit through its OWN learn path (additive, never a second
store); it never overwrites a learned value; a new-but-similar context inherits priors across contexts; and
priors never cross the tenant boundary. Requires the public ``context_runtime`` package; skipped if absent.
"""
from __future__ import annotations

import pytest


from context_runtime.integrations.bandit import EpsilonGreedyBandit  # noqa: E402
from context_runtime.optimizer.online import BanditOptimizer         # noqa: E402
from context_runtime.types import Candidate, Goal, PlanScore, StepSpec  # noqa: E402

from context_runtime.optimizer.pattern_memory import PatternMemory, signature  # noqa: E402


class _Est:
    """A trivial cost estimator: every candidate is feasible with a low, uniform prior total — so any
    warm-start prior from F3 is what actually distinguishes the arms at cold start."""
    def estimate(self, candidate, goal):
        return PlanScore(total=0.1, feasible=True)


def _opt():
    return BanditOptimizer(_Est(), bandit=EpsilonGreedyBandit(arms=(), epsilon=0.0))


def _cand(method, tier="local"):
    return Candidate(steps=(StepSpec(type="retrieve", params={"method": method}),), model_tier=tier)


# ── the mean matches the bandit's scale ─────────────────────────────────────────────────────────────

def test_prior_is_the_running_mean():
    pm = PatternMemory()
    for r in (1.0, 0.0, 1.0, 1.0):        # mean = 0.75
        pm.record("sig", "hybrid:local", r)
    n, mean = pm.prior("sig", "hybrid:local")
    assert n == 4 and abs(mean - 0.75) < 1e-9
    assert pm.prior("sig", "never-seen") is None


# ── additive priming through the bandit's own path ──────────────────────────────────────────────────

def test_prime_warm_starts_cold_arms_via_the_bandit():
    pm = PatternMemory()
    pm.record("sigA", "bm25:local", 0.9)
    opt = _opt()
    ctx = "ctx-1"
    with pytest.raises(KeyError):                             # cold: arm not registered in this context yet
        opt.bandit.value(ctx, "bm25:local")
    primed = pm.prime(opt, ctx, "sigA", ["bm25:local", "hybrid:local"])
    assert primed == ["bm25:local"]                           # only the arm with a prior
    n, mean = opt.bandit.value(ctx, "bm25:local")
    assert n == 1 and abs(mean - 0.9) < 1e-9                  # seeded into the EXISTING bandit


def test_prime_never_overwrites_a_learned_value():
    pm = PatternMemory()
    pm.record("sigA", "bm25:local", 0.2)                      # a (stale/low) prior
    opt = _opt()
    ctx = "ctx-1"
    opt.learn(ctx, "bm25:local", 0.95)                        # real observation already exists
    primed = pm.prime(opt, ctx, "sigA", ["bm25:local"])
    assert primed == []                                       # strictly additive: cold arms only
    _, mean = opt.bandit.value(ctx, "bm25:local")
    assert abs(mean - 0.95) < 1e-9                            # learned value untouched


def test_priming_biases_selection_toward_the_prior_best():
    # Two arms, no learned data. F3 primes the historically-better arm → greedy select picks it cold.
    pm = PatternMemory()
    pm.record("sigA", "hybrid:local", 0.9)
    pm.record("sigA", "bm25:local", 0.1)
    opt = _opt()
    ctx, goal = "ctx-1", Goal(text="q")
    pm.prime(opt, ctx, "sigA", ["hybrid:local", "bm25:local"])
    scored = [(_cand("hybrid"), opt.score(_cand("hybrid"), goal)),
              (_cand("bm25"), opt.score(_cand("bm25"), goal))]
    plan = opt.select(scored, goal, context=ctx)
    assert plan.extra["bandit"]["arm"] == "hybrid:local"


# ── cross-context generalization ────────────────────────────────────────────────────────────────────

def test_a_new_context_inherits_priors_by_signature():
    # The pattern was learned under one context; a DIFFERENT context with the same signature is warmed.
    pm = PatternMemory()
    sig = signature(goal_class="lookup", capability="retrieve", tenant="acme")
    pm.record(sig, "hybrid:local", 0.8)
    opt = _opt()
    primed = pm.prime(opt, "a-brand-new-context-string", sig, ["hybrid:local"])
    assert primed == ["hybrid:local"]
    assert abs(opt.bandit.value("a-brand-new-context-string", "hybrid:local")[1] - 0.8) < 1e-9


# ── tenant isolation ────────────────────────────────────────────────────────────────────────────────

def test_priors_never_cross_the_tenant_boundary():
    pm = PatternMemory()
    sig_acme = signature(goal_class="lookup", capability="retrieve", tenant="acme")
    sig_globex = signature(goal_class="lookup", capability="retrieve", tenant="globex")
    assert sig_acme != sig_globex
    pm.record(sig_acme, "hybrid:local", 0.9)                  # only acme has this prior
    assert pm.prior(sig_globex, "hybrid:local") is None       # globex sees nothing
    opt = _opt()
    assert pm.prime(opt, "ctx", sig_globex, ["hybrid:local"]) == []


# ── build from logs ─────────────────────────────────────────────────────────────────────────────────

def test_build_from_bandit_logs():
    logs = [{"context": "c", "arm": "hybrid:local", "reward": 1.0},
            {"context": "c", "arm": "hybrid:local", "reward": 0.0},
            {"context": "c", "arm": "bm25:local", "reward": 0.5}]
    pm = PatternMemory().build_from_logs(logs, sig_of=lambda row: signature(goal_class="g", tenant="t"))
    sig = signature(goal_class="g", tenant="t")
    assert pm.prior(sig, "hybrid:local") == (2, 0.5)
    assert pm.prior(sig, "bm25:local") == (1, 0.5)
