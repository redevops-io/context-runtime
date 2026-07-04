"""Learned QUALITY per choice — routing on error-character, not just latency/$.

The contextual bandit learns a single scalar reward = quality − λ·cost per (arm, context). That is
enough to pick the cheapest-good-enough arm, but it collapses two very different things into one
number: it cannot say *which* choice is actually better when cost is equal, or better *enough* to
justify more cost — and a genuinely superior provider can be locked out by a noisy scalar reward.

This ledger keeps the two terms apart. Per (context, choice) it tracks a running QUALITY (the
judged / served-relevance signal, with **no** cost penalty) and a running COST, each with a sample
count. Two uses:

  • routing — ``route()`` explores under-sampled choices, then exploits the best *blended* quality
    (quality − w·cost). A higher-quality choice is preferred at equal cost, and a cheap-but-worse
    one is not locked in. The ``choice`` key is arbitrary — a retrieval arm *or* a model/provider —
    so the same ledger routes both axes. This is the concrete answer to "provider-agnostic ≠
    provider-equal": you plug provider keys in and it learns their quality, not just their price.

  • EXPLAIN — ``stats()`` exposes, per candidate, *why* it won (quality X, cost Y, n samples), so
    the planner's choice is inspectable instead of a black-box argmax.

Opt-in and side-effect-free where unused; JSON-persistable; thread-safe (the control plane serves
concurrently). Deterministic ⇒ trivially unit-testable.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

DEFAULT_COST_WEIGHT = 0.15   # same trade-off as the reward's COST_LAMBDA, kept comparable


@dataclass(frozen=True)
class QualityStat:
    """A choice's learned quality/cost in one context."""

    choice: str
    n: int
    quality: float   # mean pure-quality signal in [0,1] (no cost penalty)
    cost: float      # mean normalized cost in [0,1]

    def blended(self, cost_weight: float = DEFAULT_COST_WEIGHT) -> float:
        return round(self.quality - cost_weight * self.cost, 6)


class QualityLedger:
    """Per-(context, choice) running quality + cost. ``observe`` after each outcome; ``route`` to
    pick; ``stats`` to explain."""

    def __init__(self, path: str | Path | None = None, cost_weight: float = DEFAULT_COST_WEIGHT):
        self.path = Path(path) if path else None
        self.cost_weight = float(cost_weight)
        # ctx -> choice -> [n, mean_quality, mean_cost]
        self._m: dict[str, dict[str, list[float]]] = {}
        self._lock = threading.Lock()
        if self.path:
            self._load()

    # ── learning ──
    def observe(self, ctx: str, choice: str, quality: float, cost: float) -> None:
        q = max(0.0, min(1.0, float(quality)))
        c = max(0.0, min(1.0, float(cost)))
        with self._lock:
            row = self._m.setdefault(ctx, {}).setdefault(choice, [0.0, 0.0, 0.0])
            n = row[0] + 1
            row[0] = n
            row[1] += (q - row[1]) / n     # running mean quality
            row[2] += (c - row[2]) / n     # running mean cost
            snapshot = json.dumps(self._m) if self.path else None
        if snapshot is not None:
            self._write(snapshot)

    # ── inspection ──
    def stat(self, ctx: str, choice: str) -> QualityStat | None:
        with self._lock:
            row = self._m.get(ctx, {}).get(choice)
            if not row or row[0] <= 0:
                return None
            return QualityStat(choice, int(row[0]), round(row[1], 6), round(row[2], 6))

    def stats(self, ctx: str, choices: list[str] | None = None,
              *, cost_weight: float | None = None) -> list[QualityStat]:
        """Every seen choice in ``ctx`` (or the given ``choices``), sorted by blended quality desc."""
        w = self.cost_weight if cost_weight is None else cost_weight
        keys = choices if choices is not None else list(self._m.get(ctx, {}).keys())
        out = [s for c in keys if (s := self.stat(ctx, c)) is not None]
        out.sort(key=lambda s: -s.blended(w))
        return out

    # ── routing: explore the under-sampled, then exploit the best blended quality ──
    def route(self, ctx: str, choices: list[str], *, min_samples: int = 3,
              cost_weight: float | None = None) -> str | None:
        """Pick a choice: any choice with < ``min_samples`` observations is explored first (the
        least-sampled), otherwise the best blended quality is exploited. Deterministic. Returns
        None only for an empty ``choices`` (so callers can fall back to the bandit on cold start)."""
        if not choices:
            return None
        w = self.cost_weight if cost_weight is None else cost_weight
        counts = {c: (self.stat(ctx, c).n if self.stat(ctx, c) else 0) for c in choices}
        under = [c for c in choices if counts[c] < min_samples]
        if under:
            # least-sampled first; ties broken by input order for determinism
            return min(under, key=lambda c: (counts[c], choices.index(c)))
        return max(choices, key=lambda c: (self.stat(ctx, c).blended(w), -choices.index(c)))

    def best(self, ctx: str, choices: list[str], *, min_samples: int = 1,
             cost_weight: float | None = None) -> str | None:
        """The best blended-quality choice among those with ≥ ``min_samples`` (None if none qualify)."""
        w = self.cost_weight if cost_weight is None else cost_weight
        ranked = [s for s in self.stats(ctx, choices, cost_weight=w) if s.n >= min_samples]
        return ranked[0].choice if ranked else None

    # ── persistence ──
    def _load(self) -> None:
        if self.path and self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                with self._lock:
                    self._m = {c: {k: list(v) for k, v in rows.items()} for c, rows in raw.items()}
            except Exception:
                pass

    def _write(self, data: str) -> None:
        os.makedirs(self.path.parent or ".", exist_ok=True)
        tmp = str(self.path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, self.path)   # atomic

    def to_dict(self) -> dict:
        with self._lock:
            return {c: {k: list(v) for k, v in rows.items()} for c, rows in self._m.items()}
