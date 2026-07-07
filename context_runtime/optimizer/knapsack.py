"""Cost Optimizer — selection over the feasible set (SPEC §4.2, ARCHITECTURE §6).

v0.1: score every candidate, drop the infeasible (hard constraints), pick the max
PlanScore. The "knapsack" proper is the token-budget packing inside context assembly
(``runtime``); here selection is greedy-by-utility, which is the right move when the
candidate set is small. CP-SAT replaces this in v0.2 when constraints interact.

Governance seam: an optional ``policy`` (PolicyProvider) narrows the feasible space *before*
cost ranking — a policy-rejected candidate is infeasible regardless of its score, and its reason
is recorded in ``Plan.rejected`` (observable in EXPLAIN). An optional ``trust`` (TrustProvider)
breaks ties between equally cost-ranked feasible plans. Both default to None (OSS runs standalone);
the enterprise layer injects concrete PolicyEngine / TrustLedger adapters.
"""
from __future__ import annotations

from ..constraints.hard import feasible
from ..plugins.base import CostEstimator, PolicyProvider, TrustProvider
from ..types import Candidate, Goal, Plan, PlanScore
from dataclasses import replace


class KnapsackOptimizer:
    def __init__(
        self,
        estimator: CostEstimator,
        policy: PolicyProvider | None = None,
        trust: TrustProvider | None = None,
    ):
        self.estimator = estimator
        self.policy = policy
        self.trust = trust

    def _policy_reject(self, candidate: Candidate, goal: Goal, score: PlanScore) -> str | None:
        """None if policy-feasible (or no policy injected); else the rejection reason."""
        if self.policy is None:
            return None
        return self.policy.feasible(candidate, goal, score)

    def score(self, candidate: Candidate, goal: Goal) -> PlanScore:
        s = self.estimator.estimate(candidate, goal)
        ok, _ = feasible(candidate, s, goal.constraints)
        if ok and self._policy_reject(candidate, goal, s) is not None:
            ok = False
        return replace(s, feasible=ok)

    def _utility(self, goal: Goal):
        """Rank key: cost/quality total first, trust score as the tie-breaker. Trust defaults to a
        constant when no provider is injected, so selection is identical to the pre-seam behavior."""
        trust = self.trust
        def key(cs: tuple[Candidate, PlanScore]) -> tuple[float, float]:
            cand, sc = cs
            t = trust.score(cand, goal) if trust is not None else 0.0
            return (sc.total, t)
        return key

    def select(self, scored: list[tuple[Candidate, PlanScore]], goal: Goal) -> Plan:
        rejected: list[tuple[Candidate, str]] = []
        feasible_set: list[tuple[Candidate, PlanScore]] = []
        for cand, sc in scored:
            ok, reason = feasible(cand, sc, goal.constraints)
            if ok:
                preason = self._policy_reject(cand, goal, sc)
                if preason is not None:
                    ok, reason = False, f"policy: {preason}"
            if ok:
                feasible_set.append((cand, sc))
            else:
                rejected.append((cand, reason or "infeasible"))

        pool = feasible_set or scored   # never fail to produce a plan; mark infeasible
        chosen, chosen_score = max(pool, key=self._utility(goal))
        for cand, sc in pool:
            if cand is not chosen:
                rejected.append((cand, f"lower score {sc.total:.3f} < {chosen_score.total:.3f}"))

        # intent is attached by the runtime planner; placeholder kept minimal here
        from ..types import Intent
        return Plan(
            intent=Intent(bucket="unknown"),
            chosen=chosen,
            score=chosen_score,
            rejected=tuple(rejected),
        )
