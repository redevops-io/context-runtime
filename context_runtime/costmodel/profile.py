"""Profiled cost table — measured stage latency, replacing hardcoded priors.

The planner's cost model is pure heuristic: `TIER_LATENCY`, `METHOD_RECALL` etc. are
hand-set constants that `estimate()` reads and `observe()` never feeds back (the learner
is 'v0.3'). DSpark profiles its engine's throughput curve `SPS(B)` **once** into a
lightweight lookup table and schedules off measured numbers, not guesses. This is that
table for us: per (stage, size-bucket) measured seconds, accumulated online and persisted
as JSON. `HeuristicEstimator` and the expensive-stage sizer consult it when present and
fall back to the priors when a cell has no samples — so it strictly improves the estimate
as real observations arrive, and changes nothing until they do.

A "stage" is a keyed unit of work whose cost we want to know, e.g. "retrieve:hybrid",
"rerank", "verify", "synthesis:premium". A "size" is a small integer knob (candidate
count, k) bucketed coarsely so the table stays tiny.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path


def _bucket(size: int) -> int:
    """Coarse size buckets so the table stays small: 0,1,2,4,8,16,32,...(cap 64)."""
    if size <= 0:
        return 0
    b = 1
    while b < size and b < 64:
        b *= 2
    return b


class CostProfile:
    """Online mean latency per (stage, size-bucket), JSON-persisted."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        # key "stage@bucket" -> [n, mean_seconds]
        self._cells: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        if self.path and self.path.exists():
            self._load()

    @staticmethod
    def _key(stage: str, size: int) -> str:
        return f"{stage}@{_bucket(size)}"

    def observe(self, stage: str, size: int, seconds: float) -> None:
        """Fold one measured latency sample into the (stage, size) cell."""
        key = self._key(stage, size)
        with self._lock:
            n, mean = self._cells.get(key, [0.0, 0.0])
            n += 1
            self._cells[key] = [n, mean + (seconds - mean) / n]
            if self.path:
                self._save_locked()

    def latency(self, stage: str, size: int) -> float | None:
        """Measured mean seconds for (stage, size), or None if never observed.

        Falls back to the nearest smaller populated size-bucket for the same stage, so a
        cell profiled at k=8 informs a k=6 query rather than reverting to the prior.
        """
        with self._lock:
            b = _bucket(size)
            while b >= 0:
                cell = self._cells.get(f"{stage}@{b}")
                if cell and cell[0] > 0:
                    return round(cell[1], 4)
                if b == 0:
                    break
                b //= 2
        return None

    def samples(self, stage: str, size: int) -> int:
        with self._lock:
            cell = self._cells.get(self._key(stage, size))
            return int(cell[0]) if cell else 0

    # ── persistence ──
    def _save_locked(self) -> None:
        assert self.path is not None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(self.path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._cells, f)
        os.replace(tmp, self.path)

    def _load(self) -> None:
        try:
            raw = json.loads(Path(self.path).read_text(encoding="utf-8"))
            self._cells = {k: [float(v[0]), float(v[1])] for k, v in raw.items()}
        except Exception:
            self._cells = {}
