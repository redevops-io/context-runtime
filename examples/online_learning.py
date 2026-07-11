"""Generation 4 — online learning: the planner explores, measures, and adapts.

A static planner always serves the plan with the best *estimate*. A Gen-4 planner treats plan
selection as a contextual bandit: it mostly exploits the best-known plan, occasionally explores an
alternative, learns from the measured reward, and can evaluate plans it never served using the logged
selection probabilities (off-policy). This script shows all three, deterministically.

    python examples/online_learning.py
"""
from __future__ import annotations

from context_runtime.optimizer.online import BanditOptimizer, offpolicy_values
from context_runtime.types import Candidate, Goal, PlanScore, StepSpec


def cand(tier, method):
    return Candidate(steps=(StepSpec(type="retrieve", params={"method": method}),), model_tier=tier)


def score(total):
    return PlanScore(total=total, feasible=True)


class _Est:
    def estimate(self, c, g): return score(0.5)
    def statistics(self, b=None): raise NotImplementedError
    def observe(self, p, t): raise NotImplementedError


def main():
    goal = Goal(text="how does the auth change relate to the billing outage")
    graph = cand("cheap", "graph")     # cost model under-rates this one (prior 0.55)
    hybrid = cand("cheap", "hybrid")   # cost model prefers it (prior 0.80)
    scored = [(hybrid, score(0.80)), (graph, score(0.55))]

    print("Estimated priors:  hybrid=0.80  graph=0.55  → a static planner always serves hybrid.\n")

    # 1) Exploitation only (ε=0): the estimate wins.
    opt = BanditOptimizer(_Est(), epsilon=0.0)
    served = opt.select(scored, goal, context="multi_hop").chosen
    print(f"ε=0 (exploit): serves {served.model_tier}:{_m(served)}")

    # 2) Reality disagrees — graph consistently earns higher reward. Feed it back.
    print("\nProduction reward comes back: graph=0.9, hybrid=0.3 (repeatedly). Learning...")
    for _ in range(4):
        opt.learn("multi_hop", graph, 0.9)
        opt.learn("multi_hop", hybrid, 0.3)
    served = opt.select(scored, goal, context="multi_hop").chosen
    print(f"After learning (exploit): serves {served.model_tier}:{_m(served)}  "
          f"← the planner adapted past its stale estimate")

    # 3) Off-policy (Tier 1): evaluate arms purely from logged executions, no live run.
    logs = [
        {"context": "multi_hop", "arm": "hybrid:cheap", "reward": 0.3, "p": 0.85},
        {"context": "multi_hop", "arm": "hybrid:cheap", "reward": 0.3, "p": 0.85},
        {"context": "multi_hop", "arm": "graph:cheap", "reward": 0.9, "p": 0.15},   # rarely explored
    ]
    vals = offpolicy_values(logs)["multi_hop"]
    print("\nOff-policy value estimate (IPS over logged selection probabilities):")
    for arm, (w, v) in sorted(vals.items(), key=lambda kv: -kv[1][1]):
        print(f"  {arm:14} value≈{v:.3f}")
    print("  → graph scores highest even though it was served rarely — surfaced without running it.")


def _m(c):
    return next(s.params.get("method") for s in c.steps if s.type == "retrieve")


if __name__ == "__main__":
    main()
