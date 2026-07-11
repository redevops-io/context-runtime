"""Generation 5 — Trust-Aware Execution: optimize for trust, and abstain rather than guess.

Two behaviors a trust-aware planner adds on top of cost/quality:
  1. trust is a weighted OBJECTIVE — a more-relied-upon plan can win over a nominally cheaper one;
  2. honest abstention — when the best plan's calibrated confidence is below the bar, decline (or
     escalate) instead of serving a confident-wrong answer, which is what actually burns operator trust.

    python examples/trust_aware.py
"""
from __future__ import annotations

from context_runtime.abstention import AbstentionGate
from context_runtime.optimizer.online import BanditOptimizer
from context_runtime.types import Candidate, Goal, PlanScore, StepSpec


def cand(tier, method):
    return Candidate(steps=(StepSpec(type="retrieve", params={"method": method}),), model_tier=tier)


def score(total, acc):
    return PlanScore(total=total, expected_accuracy=acc, feasible=True)


class Trust:
    def __init__(self, scores):
        self.scores = scores

    def score(self, c, g):
        return self.scores.get(c.model_tier, 0.5)


def main():
    goal = Goal(text="summarize the incident and recommend a fix")
    premium, local = cand("premium", "hybrid"), cand("local", "graph")
    scored = [(premium, score(0.90, acc=0.55)), (local, score(0.60, acc=0.80))]
    # operators have relied on the local graph plan; the premium plan they keep overriding
    trust = Trust({"premium": 0.2, "local": 0.9})

    print("Plans:  premium (cost-total 0.90, trust 0.2)   local (cost-total 0.60, trust 0.9)\n")

    tw0 = BanditOptimizer(None, epsilon=0.0, trust=trust, trust_weight=0.0)
    print(f"trust_weight=0.0 (cost only):  serves {tw0.select(scored, goal, context='incident').chosen.model_tier}")

    tw = BanditOptimizer(None, epsilon=0.0, trust=trust, trust_weight=0.6)
    print(f"trust_weight=0.6 (trust folded in):  serves {tw.select(scored, goal, context='incident').chosen.model_tier}"
          f"  ← the relied-upon plan wins")

    print("\nHonest abstention — the best plan's calibrated confidence must clear the bar (0.7):")
    gate = AbstentionGate(min_confidence=0.7)
    opt = BanditOptimizer(None, epsilon=0.0, trust=trust, trust_weight=0.6, abstain_gate=gate)
    plan = opt.select(scored, goal, context="incident")
    ab = plan.extra["abstention"]
    print(f"  chosen={plan.chosen.model_tier}  confidence={ab['confidence']}  → {ab['action'].upper()}")
    print(f"  reason: {ab['reason']}")
    print("\n  The chosen plan's expected accuracy (0.80) clears 0.7 → SERVE. Had it not, the planner")
    print("  would abstain (or escalate) — an honest 'not sure' instead of a confident-wrong answer.")


if __name__ == "__main__":
    main()
