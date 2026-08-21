"""Plan Cache — deterministic replay keyed on versioned evidence (SPEC §7, principle #7).

The key includes ``source_fingerprint`` — the hash of the pinned source *versions* (each source's
content fingerprint, ``SourceRef.version``). So a plan is reused only for the SAME intent against the
SAME pinned evidence identity under the SAME policy/constraints (exact replay); mutating a source's
version changes ``source_fingerprint`` and misses (re-plan = re-evaluation). ``NullPlanCache`` is the
v0.1 always-miss stub; :class:`SnapshotPlanCache` is the v0.2 store that makes the key load-bearing.
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
    """v0.1: always misses. Kept for the v0.1 conformance profile and for callers that opt out of
    caching (e.g. an online optimizer that must re-select every call)."""

    def get(self, key: PlanCacheKey) -> Plan | None:
        return None

    def put(self, key: PlanCacheKey, plan: Plan) -> None:
        return None


class SnapshotPlanCache:
    """v0.2 deterministic-replay cache: an exact-match store keyed on :class:`PlanCacheKey` (whose
    ``source_fingerprint`` pins the evidence identity). Same intent + same pinned sources + same
    policy/constraints ⇒ HIT (replay reuses the sealed plan); a mutated source version ⇒ new
    ``source_fingerprint`` ⇒ MISS (re-plan). Principle #7 made load-bearing rather than a stub.

    In-memory and process-local; a durable backend (keyed identically) can drop in behind this contract.
    """

    def __init__(self) -> None:
        self._store: dict[PlanCacheKey, Plan] = {}

    def get(self, key: PlanCacheKey) -> Plan | None:
        return self._store.get(key)

    def put(self, key: PlanCacheKey, plan: Plan) -> None:
        self._store[key] = plan
