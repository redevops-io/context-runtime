"""Cost-model statistics — the trust layer (SPEC §3.1).

Accumulates estimate-vs-actual error per estimated field, like PostgreSQL's
``pg_statistic``. Collection starts v0.1; the numbers are honest but low-confidence
until enough samples land. Persisted as JSON so calibration survives restarts.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ..types import CostModelStatistics, FieldStatistics

# the PlanScore fields we track calibration for (raw-valued ones most useful to callers)
TRACKED = ("cost_usd", "latency_seconds", "expected_accuracy")


@dataclass
class _Accum:
    """Online accumulator for one (field) — Welford-style for mean/variance of error."""

    n: int = 0
    abs_err_sum: float = 0.0
    within_ci: int = 0
    actual_mean: float = 0.0
    actual_m2: float = 0.0           # for std of actuals → CI half-width
    last_updated: str | None = None

    def update(self, estimate: float, actual: float, ci_half: float) -> None:
        self.n += 1
        self.abs_err_sum += abs(estimate - actual)
        if abs(estimate - actual) <= ci_half:
            self.within_ci += 1
        delta = actual - self.actual_mean
        self.actual_mean += delta / self.n
        self.actual_m2 += delta * (actual - self.actual_mean)
        self.last_updated = datetime.now(timezone.utc).isoformat()

    def std(self) -> float:
        return math.sqrt(self.actual_m2 / self.n) if self.n > 1 else 0.0

    def to_field(self, name: str) -> FieldStatistics:
        mae = self.abs_err_sum / self.n if self.n else 0.0
        # p≈0.9 interval half-width ≈ 1.645·std for a fresh estimate (wide when n small)
        half = 1.645 * self.std() if self.n > 1 else max(mae, self.actual_mean * 0.5)
        cal = self.within_ci / self.n if self.n else 0.0
        return FieldStatistics(
            field=name,
            mean_absolute_error=round(mae, 6),
            calibration=round(cal, 4),
            ci_low=round(max(0.0, self.actual_mean - half), 6),
            ci_high=round(self.actual_mean + half, 6),
            sample_count=self.n,
            last_updated=self.last_updated,
        )


@dataclass
class StatisticsStore:
    estimator_version: str = "heuristic-0.1"
    path: Path | None = None
    _accum: dict[str, _Accum] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for f in TRACKED:
            self._accum.setdefault(f, _Accum())
        if self.path and Path(self.path).exists():
            self._load()

    def observe(self, estimates: dict[str, float], actuals: dict[str, float]) -> None:
        for f in TRACKED:
            if f in estimates and f in actuals:
                acc = self._accum[f]
                ci_half = max(acc.to_field(f).ci_high - acc.actual_mean, 1e-9)
                acc.update(estimates[f], actuals[f], ci_half)
        if self.path:
            self._save()

    def snapshot(self, bucket: str | None = None) -> CostModelStatistics:
        return CostModelStatistics(
            estimator_version=self.estimator_version,
            fields=tuple(self._accum[f].to_field(f) for f in TRACKED),
            bucket=bucket,
        )

    def interval(self, field_name: str) -> tuple[float, float, float]:
        """(point, low, high) prediction for a fresh estimate of ``field_name``."""
        fs = self._accum[field_name].to_field(field_name)
        point = (fs.ci_low + fs.ci_high) / 2 if fs.sample_count else 0.0
        return point, fs.ci_low, fs.ci_high

    def samples(self) -> int:
        return min((a.n for a in self._accum.values()), default=0)

    # ── persistence ──
    def _save(self) -> None:
        data = {
            f: {
                "n": a.n, "abs_err_sum": a.abs_err_sum, "within_ci": a.within_ci,
                "actual_mean": a.actual_mean, "actual_m2": a.actual_m2,
                "last_updated": a.last_updated,
            }
            for f, a in self._accum.items()
        }
        Path(self.path).write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        data = json.loads(Path(self.path).read_text())
        for f, d in data.items():
            self._accum[f] = _Accum(**d)
