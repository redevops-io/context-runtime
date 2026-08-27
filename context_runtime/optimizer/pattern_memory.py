"""Execution Pattern Memory (F3) — a deterministic associative prior for the planner (audit §2.5).

The online bandit learns the best plan per *context*, but it keys on the exact context string: a genuinely
new context starts cold and must explore from the cost-model prior. Yet most new missions rhyme with old
ones — same goal class, capability shape, evidence shape, policy/tenant scope. F3 is a cheap deterministic
lookup that captures those regularities and supplies a **prior** so a new-but-familiar context starts warm
instead of exploring from scratch.

**This is not a second value store.** It is an index over the bandit's OWN reward signal, aggregated by a
richer *pattern signature* than the bandit's context key, and it feeds the prior back through the existing
learning path: ``prime`` seeds unseen arms via ``optimizer.learn(ctx, arm, prior_mean)`` — one pseudo-
observation the bandit then refines with real reward (its incremental mean down-weights the prior as data
arrives). The bandit stays the single source of truth at decision time; F3 only warm-starts its cold cells.
This is exactly the "additive prior into the existing bandit" the acceptance criterion requires — the bandit
already *is* the associative-priors store (with discount decay + persisted snapshots); F3 generalizes it
across contexts, it does not duplicate it.

Two invariants:
  • **Strictly additive.** ``prime`` seeds an arm only when the bandit has *no* observation for it in this
    context (n == 0). It never overwrites a learned value — priors fill cold-start gaps, nothing else.
  • **No cross-tenant leakage.** The tenant is part of the pattern signature, so a prior is only ever
    matched within its own tenant scope. Pattern Memory cannot reveal one tenant's execution to another
    (plan §Threats). A test asserts this directly.

It touches the optimizer only through injection (duck-typed: needs ``.learn`` and ``.bandit.value``), so
the module itself stays dependency-light and is unit-tested without constructing a full runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable


def signature(*, tenant: str | None = None, **facets) -> str:
    """A deterministic pattern key from the facets that make two missions 'the same kind': goal class,
    capability requirements, evidence shape, policy scope, context characteristics — and always the tenant,
    so priors never cross the isolation boundary. Canonical (sorted) so equal facet sets hash equal."""
    items = dict(facets)
    items["tenant"] = tenant if tenant is not None else "∅"
    return "|".join(f"{k}={items[k]}" for k in sorted(items))


@dataclass
class _Stat:
    n: int = 0
    mean: float = 0.0

    def observe(self, reward: float, discount: float = 0.0) -> None:
        self.n += 1
        alpha = discount if discount > 0.0 else 1.0 / self.n
        self.mean += alpha * (reward - self.mean)   # same incremental/discounted mean as the bandit


@dataclass
class PatternMemory:
    """Associative (pattern_signature, arm) → outcome statistics, mirroring the bandit's own mean update so
    a prior is on the same scale as a learned value. ``discount`` (>0) gives recency-weighted priors."""
    discount: float = 0.0
    _stats: dict[str, dict[str, _Stat]] = field(default_factory=dict)

    def record(self, sig: str, arm: str, reward: float) -> None:
        self._stats.setdefault(sig, {}).setdefault(arm, _Stat()).observe(reward, self.discount)

    def prior(self, sig: str, arm: str) -> tuple[int, float] | None:
        s = self._stats.get(sig, {}).get(arm)
        return (s.n, s.mean) if s and s.n > 0 else None

    def priors_for(self, sig: str) -> dict[str, tuple[int, float]]:
        return {arm: (s.n, s.mean) for arm, s in self._stats.get(sig, {}).items() if s.n > 0}

    def build_from_logs(self, logs: Iterable[dict], sig_of: Callable[[dict], str]) -> "PatternMemory":
        """Populate from bandit execution logs (``{context, arm, reward, …}`` as in Plan.extra['bandit']),
        mapping each log to its pattern signature. Same reward signal the bandit learns from — an index,
        not a parallel ground truth."""
        for row in logs:
            self.record(sig_of(row), row["arm"], float(row["reward"]))
        return self

    def prime(self, optimizer, ctx: str, sig: str, arms: Iterable[str], *, weight: int = 1) -> list[str]:
        """Warm-start ``ctx``'s cold arms from the priors for ``sig``. Seeds an arm only when the bandit has
        NO observation for it in this context (strictly additive), via the optimizer's own ``learn`` path.
        ``weight`` seeds N pseudo-observations (a stronger, slower-to-wash-out prior). Returns primed arms."""
        primed: list[str] = []
        for arm in arms:
            p = self.prior(sig, arm)
            if p is None:
                continue
            try:
                n, _ = optimizer.bandit.value(ctx, arm)
            except KeyError:
                n = 0                     # arm not yet registered in this context → cold
            if n > 0:                     # already has real data → never overwrite a learned value
                continue
            for _ in range(max(1, weight)):
                optimizer.learn(ctx, arm, p[1])
            primed.append(arm)
        return primed
