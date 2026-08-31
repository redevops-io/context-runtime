"""Resolution Closure × Context Runtime — the composition-selection + ensemble-verification tenant.

Wires the two portable seams of the Resolution Closure capability (redevops-benchmarks/closure_resolution,
Gates 2 & 4) into the fleet pattern:

  • CompositionPolicy (Gate 2) is the DECISION POINT — which composition wins the scarce token budget for a
    resolution task: pure content, structure-first, or the content-led additive union at some structural-
    promotion weight α. The frozen experiments showed there is no single global winner — code rewards structural
    promotion (α≳0.4), legal is content-saturated (α=0) — but the Runtime *dispatches by capability*, so α is a
    per-capability choice. This tenant makes that choice a learned bandit arm keyed on the capability, and the
    reward is *closure recall at the cheapest structural cost*. The bandit should rediscover the per-capability
    optimum (code → content-led α≈0.6, legal → content-led α=0) that the benchmark established.

  • Verifier (Gate 4) is an ENSEMBLE — several independent verifiers vote per dependency; disagreement is carried
    as state (VERIFIED / EXCLUDED / DISPUTED) with a confidence = vote margin. That confidence is what lets the
    tenant resolve the confident decisions autonomously and ROUTE the disputed tail to review (selective accuracy),
    rather than collapse uncertainty into a false RESOLVED.

Both seams are pure, deterministic, ML-free functions held at cross-language parity with the Go and Kotlin ports
(fixtures: redevops-benchmarks/closure_resolution/parity/golden.json). Everything runs offline against a supplied
per-task score/oracle payload; a live deployment feeds real retrieval scores + real verifier votes in the same
shape.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from ..runtime.runtime import ContextRuntime
from ..types import Goal, Plan, Trace
from .bandit import EpsilonGreedyBandit

# ─────────────────────── portable seam 1: CompositionPolicy (Gate 2) ───────────────────────


def compose_order(seed, content_score: dict, struct_score: dict, alpha: float) -> list:
    """Canonical content-led composition ordering (PARITY: Go/Kotlin reproduce this exactly).
    total(u) = content_score[u] + alpha*struct_score[u]; order by (u not in seed, -total, u) — seed ids first,
    then descending total, then unit-id ascending as the deterministic tiebreak."""
    seedset = set(seed)
    units = set(content_score) | set(struct_score)
    tot = {u: content_score.get(u, 0.0) + alpha * struct_score.get(u, 0.0) for u in units}
    return sorted(units, key=lambda u: (u not in seedset, -tot[u], u))


def budget_fill(order, tokens: dict, budget: int) -> set:
    """Break at the first item that would overflow, once >=1 item is kept (mirrors the benchmark arms)."""
    kept, spent = set(), 0
    for u in order:
        c = tokens.get(u, 0)
        if spent + c > budget and kept:
            break
        kept.add(u)
        spent += c
    return kept


# ─────────────────────── portable seam 2: EnsembleVerifier (Gate 4) ───────────────────────


@dataclass(frozen=True)
class EnsembleVerdict:
    dependency: str
    votes_yes: int
    n: int
    verdict: str                              # VERIFIED | EXCLUDED | DISPUTED
    votes: dict = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        return abs(self.votes_yes - (self.n - self.votes_yes)) / self.n if self.n else 0.0

    @property
    def disputed(self) -> bool:
        return 0 < self.votes_yes < self.n


def ensemble_verdict(dependency: str, votes: dict, majority: float = 0.5) -> EnsembleVerdict:
    """Combine independent YES/NO verifier votes for one dependency. `majority` is the YES threshold (2-of-3)."""
    n = len(votes)
    yes = sum(1 for v in votes.values() if v)
    if n and yes == n:
        verdict = "VERIFIED"
    elif yes == 0:
        verdict = "EXCLUDED"
    elif yes / n > majority:
        verdict = "VERIFIED"
    elif yes / n < majority:
        verdict = "EXCLUDED"
    else:
        verdict = "DISPUTED"
    return EnsembleVerdict(dependency=dependency, votes_yes=yes, n=n, verdict=verdict, votes=dict(votes))


# ─────────────────────── the bandit arm: a composition ───────────────────────


@dataclass(frozen=True)
class CompositionArm:
    """A Context-Runtime composition choice. `policy` ∈ {content_led, structure_first, content_only}; `alpha`
    weights structural promotion for content_led (ignored otherwise)."""

    policy: str
    alpha: float = 0.6

    @property
    def key(self) -> str:
        return f"content_led:{self.alpha:g}" if self.policy == "content_led" else self.policy

    def order(self, seed, content_score: dict, struct_score: dict) -> list:
        if self.policy == "content_only":
            return sorted(content_score, key=lambda u: (-content_score[u], u))
        if self.policy == "structure_first":
            struct = sorted((u for u in struct_score if struct_score[u] > 0),
                            key=lambda u: (-struct_score[u], u))
            content = sorted(content_score, key=lambda u: (-content_score[u], u))
            seen, out = set(), []
            for u in [*struct, *content]:
                if u not in seen:
                    seen.add(u)
                    out.append(u)
            return out
        return compose_order(seed, content_score, struct_score, self.alpha)   # content_led


DEFAULT_ARMS: tuple[CompositionArm, ...] = (
    CompositionArm("content_only"),
    CompositionArm("structure_first"),
    CompositionArm("content_led", 0.0),
    CompositionArm("content_led", 0.4),
    CompositionArm("content_led", 0.6),
)
COST_LAMBDA = 0.05   # structural promotion is not free (graph traversal) — a small cost on α in the reward


def reward_closure(recall: float, arm: CompositionArm) -> float:
    """Closure recall minus the structural-promotion cost. The efficiency frontier: recover the closure with the
    least structural work. Content-saturated domains keep their recall at α=0 (no cost); domains that *need*
    structure pay a little to reach a much higher recall — and still win."""
    max_alpha = max((a.alpha for a in DEFAULT_ARMS if a.policy == "content_led"), default=1.0) or 1.0
    cost = (arm.alpha / max_alpha) if arm.policy in ("content_led", "structure_first") else 0.0
    return round(max(0.0, recall - COST_LAMBDA * cost), 4)


def _closure_bandit(epsilon: float = 0.15) -> EpsilonGreedyBandit:
    return EpsilonGreedyBandit(DEFAULT_ARMS, epsilon=epsilon)


# ─────────────────────── the tenant ───────────────────────


@dataclass
class ResolveResult:
    request: str
    capability: str                 # the bandit context (e.g. "code" / "legal")
    arm: CompositionArm
    closure: tuple                  # the reconciled unit ids kept under budget
    order: tuple
    status: str                     # RESOLVED | DISPUTED | REQUIRE_REVIEW
    verdicts: tuple                 # EnsembleVerdict per verified dependency
    routed: tuple                   # dependencies routed to review (disputed)
    plan: Plan


class ResolutionClosureTenant:
    """Context Runtime resolves a closure task: pick the composition that best recovers the dependency closure
    for this capability under budget, verify the members with an ensemble, resolve the confident decisions and
    route the disputed tail to review, then learn from the achieved closure recall."""

    def __init__(self, runtime: ContextRuntime | None = None,
                 bandit: EpsilonGreedyBandit | None = None):
        self.runtime = runtime or ContextRuntime.default([])
        self.bandit = bandit or _closure_bandit()
        self._pending: dict[str, tuple[Plan, CompositionArm, str]] = {}

    def resolve(self, task: dict) -> ResolveResult:
        """`task` = {request, capability, seed[], content_score{}, struct_score{}, tokens{}, budget,
        votes?{unit: {verifier: bool}}}. Composition is chosen by the bandit keyed on `capability`."""
        capability = task["capability"]
        request = task.get("request", capability)
        plan = self.runtime.plan(Goal(text=request))
        arm = self.bandit.select(capability)

        order = arm.order(task["seed"], task["content_score"], task.get("struct_score", {}))
        closure = budget_fill(order, task.get("tokens", {}), task.get("budget", 0))

        # Gate-4 ensemble verification over whatever votes are supplied for kept members
        votes = task.get("votes", {})
        verdicts, routed = [], []
        for u in closure:
            if u in votes:
                ev = ensemble_verdict(str(u), votes[u])
                verdicts.append(ev)
                if ev.disputed:
                    routed.append(str(u))
        status = "RESOLVED" if not routed else ("DISPUTED" if verdicts else "REQUIRE_REVIEW")

        self._pending[self._key(request)] = (plan, arm, capability)
        return ResolveResult(request, capability, arm, tuple(sorted(closure)), tuple(order),
                             status, tuple(verdicts), tuple(routed), plan)

    def record_outcome(self, request: str, closure_recall: float) -> float:
        """Feed back the achieved closure recall (vs the oracle). Updates bandit + cost model."""
        key = self._key(request)
        if key not in self._pending:
            return 0.0
        plan, arm, capability = self._pending.pop(key)
        reward = reward_closure(closure_recall, arm)
        self.bandit.update(capability, arm, reward)
        trace = Trace(plan_id=plan.id, goal_text=request,
                      actual_tokens=int(1000 * (1 + arm.alpha)),
                      verification_passed=closure_recall >= 0.5)
        self.runtime.estimator.observe(plan, trace)
        return reward

    def policy(self) -> dict[str, str]:
        """Learned best composition per capability — should converge to the per-capability α optimum."""
        return self.bandit.policy()

    @staticmethod
    def _key(request: str) -> str:
        return hashlib.sha256(request.encode()).hexdigest()[:16]
