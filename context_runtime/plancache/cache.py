"""Plan Cache — v0.2 subsystem, stubbed in v0.1 (SPEC §7).

The contract and key are defined now so the runtime can call it; v0.1 is a no-op that
always misses. It can only be made *correct* once the v0.2 Knowledge graph supplies
versioned sources to key/invalidate on (deterministic replay, principle #7).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..types import Goal, Intent, Plan


@dataclass(frozen=True)
class PlanCacheKey:
    intent_normalized: str
    source_fingerprint: str
    policy_fingerprint: str
    constraint_envelope: str
    analyzer_version: str = "rule_intent-0.1"
    planner_version: str = "knapsack-0.1"


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def build_key(intent: Intent, goal: Goal) -> PlanCacheKey:
    sources = sorted(f"{s.name}:{s.version or '∅'}" for s in goal.sources)
    c = goal.constraints
    envelope = f"{c.max_cost_usd}|{c.max_latency_seconds}|{c.max_tokens}|{c.require_citations}|{c.require_verification}|{c.sensitivity}"
    return PlanCacheKey(
        intent_normalized=intent.normalized,
        source_fingerprint=_h("|".join(sources)),
        policy_fingerprint=_h(c.sensitivity),
        constraint_envelope=_h(envelope),
    )


class NullPlanCache:
    """v0.1: always misses. v0.2 replaces this with a semantic-match store."""

    def get(self, key: PlanCacheKey) -> Plan | None:
        return None

    def put(self, key: PlanCacheKey, plan: Plan) -> None:
        return None
