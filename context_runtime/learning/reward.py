"""Shared reward contract — the default signal that closes the learning loop on the runtime's OWN
serving path.

The AWS-fit audit found the bandit machinery complete but wired only inside tenant integrations, each
hand-writing its own ``reward_from_*``. This module is the shared, validated default so the *core*
runtime learns out of the box: ``reward = quality − λ·cost``, with ``quality`` a pluggable proxy. A
deployment with a real judge overrides ``quality`` (or supplies its own ``RewardContract``); the
default gives the online optimizer a sensible, bounded signal with no bespoke code.

Bounded to [0, 1] so it composes with the bandit's running means regardless of the quality source.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class RewardContract(Protocol):
    """quality − λ·cost, returned in [0, 1]. Implement to plug a real judge in for ``quality``."""

    def reward(self, trace, verdict) -> float: ...


@dataclass(frozen=True)
class DefaultReward:
    """A cost-aware default reward from signals the runtime already produces — verification outcome,
    grounding (citations), and measured cost. Deliberately simple: it exists so learning is *on*, not
    to be the final word on quality. Override ``quality`` with an LLM/passage judge for production."""

    lam: float = 0.3               # cost aversion (weight on the cost penalty)
    cost_ref_usd: float = 0.05     # cost mapping to a full (1.0) penalty
    base: float = 0.5              # a produced, grounded answer starts here
    verified_bonus: float = 0.3    # verification passed
    citation_bonus: float = 0.2    # answer is grounded in retrieved sources

    def quality(self, trace, verdict) -> float:
        q = self.base
        passed = getattr(verdict, "passed", None)
        if passed is True or (passed is None and getattr(trace, "verified", False)):
            q += self.verified_bonus
        if getattr(trace, "citations", ()):        # grounded answer
            q += self.citation_bonus
        return min(1.0, q)

    def cost_penalty(self, trace) -> float:
        cost = float(getattr(trace, "actual_cost_usd", 0.0) or 0.0)
        if self.cost_ref_usd <= 0:
            return 0.0
        return min(1.0, cost / self.cost_ref_usd)

    def reward(self, trace, verdict) -> float:
        r = self.quality(trace, verdict) - self.lam * self.cost_penalty(trace)
        return max(0.0, min(1.0, r))
