"""Learning aggregator + state snapshot — the async policy loop (Whitepaper v3, Phase 4).

    execution → OutcomeEvent → [bus] → aggregator (off hot path) → LearnedStateSnapshot → [bus] → replicas

The aggregator is the ONE writer of learned state: it drains outcome events, folds each into the shared
bandit (and, via optional sinks, the trust ledger / calibration), and publishes a versioned snapshot.
Stateless planner replicas never learn on the serving path — they select against a local snapshot and
reconcile when a newer one is published. "The learning loop is asynchronous — and that is what makes it
scale."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..integrations.bandit import EpsilonGreedyBandit
from .bus import EventBus
from .events import OutcomeEvent


class _Arm:
    __slots__ = ("key",)

    def __init__(self, key: str):
        self.key = key


@dataclass
class LearnedStateSnapshot:
    """A versioned, serializable snapshot of learned state that a replica reconciles against."""
    version: int
    bandit_stats: dict = field(default_factory=dict)   # ctx -> arm.key -> [n, mean]

    def to_dict(self) -> dict:
        return {"version": self.version, "bandit_stats": self.bandit_stats}

    @classmethod
    def from_dict(cls, d: dict) -> "LearnedStateSnapshot":
        return cls(int(d["version"]), d.get("bandit_stats", {}))

    def apply_to(self, bandit: EpsilonGreedyBandit) -> None:
        """Reconcile a replica's bandit with this snapshot (replace its learned means)."""
        bandit.stats = {ctx: {k: list(v) for k, v in arms.items()}
                        for ctx, arms in self.bandit_stats.items()}


class LearningAggregator:
    """Consumes OutcomeEvents off the serving path and folds them into the shared learned state."""

    def __init__(
        self,
        bandit: EpsilonGreedyBandit,
        *,
        topic: str = "outcomes",
        on_trust: Callable[[OutcomeEvent], None] | None = None,
        on_calibration: Callable[[OutcomeEvent], None] | None = None,
    ):
        self.bandit = bandit
        self.topic = topic
        self.on_trust = on_trust                 # e.g. the enterprise TrustLedger sink
        self.on_calibration = on_calibration
        self.version = 0
        self.processed = 0
        self._last_seq = -1
        self._known: set[str] = {a.key for a in self.bandit.arms}

    def _ensure_arm(self, key: str) -> None:
        """Register a plan shape into the shared bandit so update()/value() don't miss the key."""
        if key not in self._known:
            self.bandit.arms = tuple(self.bandit.arms) + (_Arm(key),)
            self._known.add(key)

    def apply(self, event: OutcomeEvent) -> bool:
        """Fold one event into learned state. Ignores stale/duplicate seqs (idempotent replay)."""
        if event.seq and event.seq <= self._last_seq:
            return False
        self._last_seq = max(self._last_seq, event.seq)
        # An abstention produced no served answer to reward the arm with — it still carries trust signal.
        if not event.abstained:
            self._ensure_arm(event.arm)
            self.bandit.update(event.context, _Arm(event.arm), event.reward)
        if self.on_trust is not None:
            self.on_trust(event)
        if self.on_calibration is not None:
            self.on_calibration(event)
        self.processed += 1
        return True

    def drain(self, bus: EventBus) -> int:
        """Consume all pending outcome events (off the hot path). Bumps the version if anything applied."""
        applied = 0
        for ev in bus.poll(self.topic):
            if self.apply(ev):
                applied += 1
        if applied:
            self.version += 1
        return applied

    def snapshot(self) -> LearnedStateSnapshot:
        stats = {ctx: {k: list(v) for k, v in arms.items()} for ctx, arms in self.bandit.stats.items()}
        return LearnedStateSnapshot(self.version, stats)

    def publish(self, bus: EventBus, topic: str = "snapshots") -> LearnedStateSnapshot:
        snap = self.snapshot()
        bus.publish(topic, snap)
        return snap
