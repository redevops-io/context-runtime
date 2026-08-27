"""F3 earns its place: does Pattern Memory reduce cold-start regret vs an un-primed bandit?

A synthetic multi-context task where each context belongs to a *family* (goal class) with a known best arm.
The bandit keys on the exact context string, so every new context starts cold and must explore. Pattern
Memory, pre-trained on past executions of the SAME families, primes each new context's arms before its first
decision — so a new-but-familiar context skips the exploration the cold bandit pays for.

We measure cumulative regret (expected reward of the best arm minus the chosen arm) over a stream of new
contexts. F3 is justified only if primed regret is materially lower — and only in the cold-start regime; as
each context accumulates its own data, the prior washes out (that is the point — priors seed, data decides).

    PYTHONPATH=<contextos>:<CR-enterprise/py> python examples/pattern_memory_coldstart.py
"""
from __future__ import annotations

import random

from context_runtime.integrations.bandit import EpsilonGreedyBandit
from context_runtime.optimizer.online import BanditOptimizer
from context_runtime.types import Candidate, Goal, PlanScore, StepSpec

from context_runtime.optimizer.pattern_memory import PatternMemory, signature

FAMILIES = ["lookup", "synthesis", "incident"]
BEST = {"lookup": "bm25:local", "synthesis": "hybrid:local", "incident": "graph:premium"}
ARMS = {"bm25:local": ("bm25", "local"), "hybrid:local": ("hybrid", "local"), "graph:premium": ("graph", "premium")}
GOOD, BAD = 1.0, 0.3            # expected reward of the best vs any other arm


class _Est:
    def estimate(self, candidate, goal):
        return PlanScore(total=0.5, feasible=True)


def _cand(arm):
    method, tier = ARMS[arm]
    return Candidate(steps=(StepSpec(type="retrieve", params={"method": method}),), model_tier=tier)


def _expected(family, arm):
    return GOOD if arm == BEST[family] else BAD


def _reward(family, arm, rng):
    return _expected(family, arm) + rng.uniform(-0.05, 0.05)


def _train_memory(rng, rounds=40):
    """Priors from simulated past executions of each family (honest — derived from reward, not an oracle)."""
    pm = PatternMemory()
    for family in FAMILIES:
        sig = signature(goal_class=family, tenant="acme")
        for _ in range(rounds):
            for arm in ARMS:
                pm.record(sig, arm, _reward(family, arm, rng))
    return pm


def _run(primed: bool, pm, *, contexts=30, episodes=8, epsilon=0.15, seed=7):
    rng = random.Random(seed)
    opt = BanditOptimizer(_Est(), bandit=EpsilonGreedyBandit(arms=(), epsilon=epsilon, seed=0xABCDEF))
    scored_cache = {a: (_cand(a), None) for a in ARMS}
    goal = Goal(text="q")
    regret = 0.0
    for c in range(contexts):
        family = FAMILIES[c % len(FAMILIES)]
        ctx = f"{family}-ctx-{c}"
        if primed:
            pm.prime(opt, ctx, signature(goal_class=family, tenant="acme"), list(ARMS))
        for _ in range(episodes):
            scored = [(cand, opt.score(cand, goal)) for cand, _ in scored_cache.values()]
            plan = opt.select(scored, goal, context=ctx)
            arm = plan.extra["bandit"]["arm"]
            regret += GOOD - _expected(family, arm)
            opt.learn(ctx, arm, _reward(family, arm, rng))
    return regret


def main():
    pm = _train_memory(random.Random(1))
    cold = _run(False, pm)
    primed = _run(True, pm)
    print(f"cumulative regret over 30 new contexts × 8 episodes (lower = better):")
    print(f"  cold   (no pattern memory): {cold:6.2f}")
    print(f"  primed (F3 pattern memory): {primed:6.2f}")
    drop = 100 * (cold - primed) / cold if cold else 0.0
    print(f"  regret reduction:           {drop:5.1f}%")
    print("F3 seeds the historically-best arm per family, so a new-but-familiar context skips cold exploration.")


if __name__ == "__main__":
    main()
