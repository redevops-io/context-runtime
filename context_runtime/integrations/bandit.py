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
                 seed: int = 0x9E3779B9, persist_path: str | None = None):
        self.arms = arms
        self.epsilon = epsilon
        self.optimistic = optimistic
        self.stats: dict[str, dict[str, list[float]]] = {}   # ctx → arm.key → [n, mean]
        self._rng = seed & 0xFFFFFFFF
        self.persist_path = persist_path   # learned policy survives restarts if set
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
        return self.stats.setdefault(ctx, {a.key: [0.0, self.optimistic] for a in self.arms})

    def select(self, ctx: str):
        arms = self._ctx(ctx)
        if self._rand() < self.epsilon:
            return self.arms[int(self._rand() * len(self.arms)) % len(self.arms)]
        best = max(arms, key=lambda k: arms[k][1])
        return next(a for a in self.arms if a.key == best)

    def update(self, ctx: str, arm, reward: float) -> None:
        arms = self._ctx(ctx)
        n, mean = arms[arm.key]
        n += 1
        arms[arm.key] = [n, mean + (reward - mean) / n]
        self.save()

    # ── persistence (so learning survives restarts) ──
    def _load(self) -> None:
        import json
        import os
        if self.persist_path and os.path.exists(self.persist_path):
            try:
                raw = json.load(open(self.persist_path, encoding="utf-8"))
                self.stats = {ctx: {k: list(v) for k, v in arms.items()} for ctx, arms in raw.items()}
            except Exception:
                pass

    def save(self) -> None:
        if not self.persist_path:
            return
        import json
        import os
        os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
        tmp = self.persist_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.stats, f)
        os.replace(tmp, self.persist_path)   # atomic

    def policy(self) -> dict[str, str]:
        """Current best arm key per context — the learned policy, for inspection."""
        return {ctx: max(a, key=lambda k: a[k][1]) for ctx, a in self.stats.items()}

    def value(self, ctx: str, arm_key: str) -> tuple[int, float]:
        a = self._ctx(ctx)[arm_key]
        return int(a[0]), a[1]
