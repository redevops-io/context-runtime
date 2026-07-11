"""Online-learning optimizer — Generation 4 (Whitepaper v3).

Generations 1-2 pick the plan with the best *estimated* cost/quality. But estimates go stale, so a
Gen-4 planner must **explore**: it treats plan selection as a contextual bandit keyed by intent bucket,
seeded with the cost model's estimate as an optimistic prior and refined by measured reward. Selection
balances exploitation (the best learned plan) against exploration (trying alternatives), and every
choice records its selection probability so alternatives can be evaluated off-policy — the cheapest
exploration tier.

Layered exploration (paper's "cheapest form that answers the question"):
  • Tier 1 — off-policy evaluation: score alternative plans from executions already observed, using the
    logged selection probabilities (``offpolicy_values``). Zero user-facing latency.
  • Tier 2 — shadow execution: run a candidate off the serving path (``select(..., shadow=True)`` marks
    a shadow pick; the caller executes it asynchronously and feeds the reward back via ``learn``).
  • Tier 3 — live exploration: the ε-greedy branch actually serves an exploratory plan.

Reuses the fleet's ``EpsilonGreedyBandit`` as the shared per-(context, arm) value store, and composes
with the Phase-0 governance seam: a PolicyProvider still filters the feasible space *before* any
exploration (you never explore an infeasible plan), and a TrustProvider breaks ties.
"""
from __future__ import annotations

from dataclasses import replace

from ..constraints.hard import feasible as _hard_feasible
from ..integrations.bandit import EpsilonGreedyBandit
from ..plugins.base import CostEstimator, PolicyProvider, TrustProvider
from ..types import Candidate, Goal, Intent, Plan, PlanScore


class _Arm:
    """Minimal Arm (exposes ``.key``) so plan shapes can be registered into the shared bandit."""
    __slots__ = ("key",)

    def __init__(self, key: str):
        self.key = key


def plan_key(candidate: Candidate) -> str:
    """The bandit arm for a candidate: its retrieval method × model tier — the decision that matters."""
    method = next((s.params.get("method", "") for s in candidate.steps if s.type == "retrieve"), "")
    return f"{method}:{candidate.model_tier}"


class BanditOptimizer:
    """A CostOptimizer that selects online. ``score`` is identical to the cost optimizer (estimate +
    hard/policy feasibility); ``select`` runs a subset-aware ε-greedy over the *feasible* candidates."""

    def __init__(
        self,
        estimator: CostEstimator,
        *,
        bandit: EpsilonGreedyBandit | None = None,
        epsilon: float = 0.15,
        discount: float = 0.0,
        policy: PolicyProvider | None = None,
        trust: TrustProvider | None = None,
        trust_weight: float = 0.0,
        abstain_gate=None,
        context_of=None,
        seed: int = 0x1234567,
    ):
        self.estimator = estimator
        # discount > 0 → recency-weighted learning (tracks a drifting best arm); see EpsilonGreedyBandit.
        self.bandit = bandit or EpsilonGreedyBandit(arms=(), epsilon=epsilon, discount=discount)
        self.policy = policy
        self.trust = trust
        # Gen-5: when > 0, trust is a weighted OBJECTIVE term in ranking, not just a tie-breaker.
        self.trust_weight = trust_weight
        # Gen-5: an optional AbstentionGate — records serve/escalate/abstain in Plan.extra.
        self.abstain_gate = abstain_gate
        self.context_of = context_of          # goal -> context str; else the caller passes context=
        self._rng = seed & 0xFFFFFFFF
        self._known: set[str] = {a.key for a in self.bandit.arms}

    # ── deterministic rng (xorshift), independent of the bandit's own ──
    def _rand(self) -> float:
        x = self._rng
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self._rng = x & 0xFFFFFFFF
        return self._rng / 0x100000000

    def _ensure_arm(self, key: str) -> None:
        """Register a newly-seen plan shape into the shared bandit so value()/update() don't KeyError."""
        if key not in self._known:
            self.bandit.arms = tuple(self.bandit.arms) + (_Arm(key),)
            self._known.add(key)

    def _context(self, goal: Goal, context: str) -> str:
        if context:
            return context
        if self.context_of:
            return self.context_of(goal)
        return "default"

    def _policy_reject(self, cand: Candidate, goal: Goal, sc: PlanScore) -> str | None:
        return self.policy.feasible(cand, goal, sc) if self.policy is not None else None

    def score(self, candidate: Candidate, goal: Goal) -> PlanScore:
        s = self.estimator.estimate(candidate, goal)
        ok, _ = _hard_feasible(candidate, s, goal.constraints)
        if ok and self._policy_reject(candidate, goal, s) is not None:
            ok = False
        return replace(s, feasible=ok)

    def _arm_value(self, ctx: str, cand: Candidate, sc: PlanScore, goal: Goal) -> tuple[float, float]:
        """Value to rank by: the learned mean once observed, else the cost-model total as an optimistic
        prior. With ``trust_weight`` (Gen 5) trust is folded into the objective; it is also the secondary
        tie-breaker (the Phase-0 seam shape)."""
        key = plan_key(cand)
        self._ensure_arm(key)
        n, mean = self.bandit.value(ctx, key)
        base = mean if n > 0 else sc.total
        t = self.trust.score(cand, goal) if self.trust is not None else 0.0
        if self.trust_weight > 0.0 and self.trust is not None:
            base = (1.0 - self.trust_weight) * base + self.trust_weight * t
        return (base, t)

    def select(self, scored, goal: Goal, context: str = "", *, shadow: bool = False) -> Plan:
        ctx = self._context(goal, context)

        rejected: list[tuple[Candidate, str]] = []
        pool: list[tuple[Candidate, PlanScore]] = []
        for cand, sc in scored:
            ok, reason = _hard_feasible(cand, sc, goal.constraints)
            if ok:
                pr = self._policy_reject(cand, goal, sc)
                if pr is not None:
                    ok, reason = False, f"policy: {pr}"
            (pool if ok else rejected).append((cand, sc) if ok else (cand, reason or "infeasible"))

        eff_pool = pool or [(c, s) for c, s in scored]   # never fail to produce a plan
        m = len(eff_pool)

        # greedy (exploit) reference — argmax learned/estimated value
        greedy_i = max(range(m), key=lambda i: self._arm_value(ctx, eff_pool[i][0], eff_pool[i][1], goal))

        explore = m > 1 and self._rand() < self.bandit.epsilon
        if explore:
            # Tier 3 (or Tier 2 shadow): pick a NON-greedy feasible arm uniformly.
            others = [i for i in range(m) if i != greedy_i]
            chosen_i = others[int(self._rand() * len(others)) % len(others)]
            mode = "shadow" if shadow else "explore"
        else:
            chosen_i, mode = greedy_i, "exploit"

        chosen, chosen_sc = eff_pool[chosen_i]

        # selection probability (propensity) under this ε-greedy policy — logged for off-policy eval
        eps = self.bandit.epsilon
        p = (eps / m) + ((1.0 - eps) if chosen_i == greedy_i else 0.0)

        for i, (cand, sc) in enumerate(eff_pool):
            if i != chosen_i:
                rejected.append((cand, f"not selected (value rank; mode={mode})"))

        extra = {"bandit": {"context": ctx, "arm": plan_key(chosen), "mode": mode,
                            "p": round(p, 6), "epsilon": eps}}
        # Gen-5 honest abstention: is the chosen plan confident enough to serve?
        if self.abstain_gate is not None:
            v = self.abstain_gate.evaluate(chosen_sc, goal)
            extra["abstention"] = {"action": v.action, "confidence": round(v.confidence, 6), "reason": v.reason}

        return Plan(
            intent=Intent(bucket="unknown"),   # runtime attaches the real intent
            chosen=chosen,
            score=chosen_sc,
            rejected=tuple(rejected),
            extra=extra,
        )

    # ── learning ──
    def learn(self, context: str, candidate_or_key, reward: float) -> None:
        """Fold a measured reward into the bandit for (context, arm). Accepts a Candidate or an arm key."""
        key = candidate_or_key if isinstance(candidate_or_key, str) else plan_key(candidate_or_key)
        self._ensure_arm(key)
        self.bandit.update(context, _Arm(key), reward)

    def learn_from_plan(self, plan: Plan, reward: float) -> None:
        """Convenience: learn using the (context, arm) recorded in ``plan.extra['bandit']``."""
        b = (plan.extra or {}).get("bandit")
        if b:
            self.learn(b["context"], b["arm"], reward)


def offpolicy_values(logs) -> dict[str, dict[str, tuple[float, float]]]:
    """Tier-1 off-policy evaluation: estimate each arm's value per context from already-observed
    executions, WITHOUT running anything. Self-normalized inverse-propensity weighting over the logged
    selection probabilities corrects for the fact that exploratory data isn't uniformly sampled.

    ``logs``: iterable of dicts ``{"context", "arm", "reward", "p"}`` (as recorded in Plan.extra).
    Returns ``{context: {arm: (effective_weight, estimated_value)}}`` — higher value = better plan,
    letting the planner rank alternatives it never served on the live path.
    """
    acc: dict[str, dict[str, list[float]]] = {}
    for row in logs:
        ctx, arm = row["context"], row["arm"]
        p = max(float(row.get("p", 1.0)), 1e-6)     # guard divide-by-zero
        w = 1.0 / p
        cell = acc.setdefault(ctx, {}).setdefault(arm, [0.0, 0.0])   # [sum_w, sum_w*reward]
        cell[0] += w
        cell[1] += w * float(row["reward"])
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for ctx, arms in acc.items():
        out[ctx] = {a: (sw, (swr / sw if sw else 0.0)) for a, (sw, swr) in arms.items()}
    return out
