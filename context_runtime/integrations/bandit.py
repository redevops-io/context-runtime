"""The shared learning core for Context Runtime app integrations (the fleet pattern).

Every tenant (sidekick, redevops-rag, …) makes a discrete choice keyed by intent
bucket and gets a measurable reward back. That is a contextual bandit. This module is
that bandit, generic over any *arm* object exposing a ``.key: str``. App-specific
arms and reward functions live in each tenant's module; the learning is shared here.

This is the v0.1-achievable stand-in for the v0.3 River contextual bandit — same
select/update/reward seam, so swapping in River later is a drop-in.
"""
from __future__ import annotations

from typing import Protocol


class Arm(Protocol):
    @property
    def key(self) -> str: ...


class EpsilonGreedyBandit:
    """Contextual ε-greedy over arms, keyed by a context string (the intent bucket).

    Optimistic initialization makes every unseen arm look maximal, so each is tried at
    least once before the policy commits — cheap exploration without tuning a schedule.
    Deterministic xorshift rng (no global ``random``) keeps runs reproducible/testable.
    """

    def __init__(self, arms: tuple, epsilon: float = 0.15, optimistic: float = 1.0,
                 seed: int = 0x9E3779B9, persist_path: str | None = None, discount: float = 0.0):
        import threading
        self.arms = arms
        self.epsilon = epsilon
        self.optimistic = optimistic
        # discount ∈ (0,1] = constant step size (exponential recency weighting) so stale evidence
        # fades and the policy tracks a drifting best arm — the Whitepaper-v3 non-stationarity property.
        # 0 (default) = sample-average (1/n), the stationary estimator; behavior is unchanged.
        self.discount = discount
        self.stats: dict[str, dict[str, list[float]]] = {}   # ctx → arm.key → [n, mean]
        self._rng = seed & 0xFFFFFFFF
        self.persist_path = persist_path   # learned policy survives restarts if set
        # Guards stats + _rng. FastAPI runs sync endpoints in a threadpool (~40 threads)
        # over one shared bandit; without this, concurrent select/update on the shared
        # stats dict lose updates or raise "dict changed size during iteration".
        self._lock = threading.Lock()
        self._save_lock = threading.Lock()   # serializes disk writes, off the stats lock
        if persist_path:
            self._load()

    def _rand(self) -> float:
        x = self._rng
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        self._rng = x & 0xFFFFFFFF
        return self._rng / 0x100000000

    def _ctx(self, ctx: str) -> dict[str, list[float]]:
        # backfill any arms added since this context was first seen / persisted, so select,
        # update, and value() never KeyError on a new arm against an old (persisted) context.
        d = self.stats.setdefault(ctx, {})
        for a in self.arms:
            d.setdefault(a.key, [0.0, self.optimistic])
        return d

    def select(self, ctx: str):
        with self._lock:
            arms = self._ctx(ctx)
            if self._rand() < self.epsilon:
                return self.arms[int(self._rand() * len(self.arms)) % len(self.arms)]
            best = max(arms, key=lambda k: arms[k][1])
            return next(a for a in self.arms if a.key == best)

    def update(self, ctx: str, arm, reward: float) -> None:
        import json
        with self._lock:
            arms = self._ctx(ctx)
            n, mean = arms[arm.key]
            n += 1
            alpha = self.discount if self.discount > 0.0 else 1.0 / n
            arms[arm.key] = [n, mean + alpha * (reward - mean)]
            # serialize a consistent snapshot under the lock; write it to disk outside so
            # the fsync/rename never blocks concurrent select/update.
            snapshot = json.dumps(self.stats) if self.persist_path else None
        if snapshot is not None:
            self._write_snapshot(snapshot)

    # ── persistence (so learning survives restarts) ──
    def _load(self) -> None:
        import json
        import os
        if self.persist_path and os.path.exists(self.persist_path):
            try:
                raw = json.load(open(self.persist_path, encoding="utf-8"))
                with self._lock:
                    self.stats = {ctx: {k: list(v) for k, v in arms.items()} for ctx, arms in raw.items()}
            except Exception:
                pass

    def save(self) -> None:
        import json
        with self._lock:
            snapshot = json.dumps(self.stats) if self.persist_path else None
        if snapshot is not None:
            self._write_snapshot(snapshot)

    def _write_snapshot(self, data: str) -> None:
        """Atomically write an already-serialized stats snapshot. Serialized by its own
        lock (not the stats lock) so persistence I/O stays off the learning hot path."""
        import os
        with self._save_lock:
            os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
            tmp = self.persist_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(data)
            os.replace(tmp, self.persist_path)   # atomic

    def policy(self) -> dict[str, str]:
        """Current best arm key per context — the learned policy, for inspection."""
        with self._lock:
            return {ctx: max(a, key=lambda k: a[k][1]) for ctx, a in self.stats.items()}

    def value(self, ctx: str, arm_key: str) -> tuple[int, float]:
        with self._lock:
            a = self._ctx(ctx)[arm_key]
            return int(a[0]), a[1]
