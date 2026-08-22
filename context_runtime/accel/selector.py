"""Accelerator selection — the costed-capability decision the roadmap calls for.

``decide(kind, n)`` answers one question: for an operation of this kind over ``n`` items, is the GPU backend
worth choosing right now? It is **fail-closed to the CPU**: unless the master switch is on, the library and
a GPU are actually present, and ``n`` is past the measured crossover, the answer is "use the CPU" — so the
default install, a CPU-only host, and a small workload all behave exactly as before.

Crossovers come from the real benchmark (FINDINGS_v0.3.0_accelerators): ANN retrieval pays above ~12k
vectors, temporal/evidence dataframe processing above ~10^6 rows. Both are overridable by env for tuning.
Every decision carries a human ``reason`` so EXPLAIN can show *why* a backend was (not) selected.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# Measured crossover defaults (rows/vectors) — below these the CPU wins, so we do not accelerate.
_ANN_MIN = int(os.getenv("CR_ACCEL_ANN_MIN", "12000"))
_TEMPORAL_MIN = int(os.getenv("CR_ACCEL_TEMPORAL_MIN", "1000000"))
_CROSSOVER = {"ann": _ANN_MIN, "temporal": _TEMPORAL_MIN}
_LIB = {"ann": "cuvs", "temporal": "cudf"}

_gpu_cache: bool | None = None


@dataclass(frozen=True)
class AccelDecision:
    """The outcome of a selection. ``use_gpu`` is the only thing a caller must honour; ``backend`` and
    ``reason`` are for EXPLAIN/observability."""
    use_gpu: bool
    kind: str
    n: int
    backend: str          # "cpu" | "cuvs" | "cudf"
    reason: str

    def __bool__(self) -> bool:
        return self.use_gpu


def _flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def _enabled(kind: str) -> bool:
    """Master switch ``CR_ACCEL`` turns the whole thing on; ``CR_ACCEL_ANN`` / ``CR_ACCEL_TEMPORAL`` can
    enable one kind on their own. Off by default → nothing changes."""
    return _flag("CR_ACCEL") or _flag(f"CR_ACCEL_{kind.upper()}")


def gpu_available() -> bool:
    """True iff a CUDA device is visible via cupy. Cached; never raises (a missing lib → False → CPU)."""
    global _gpu_cache
    if _gpu_cache is None:
        try:
            import cupy as cp
            _gpu_cache = cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            _gpu_cache = False
    return _gpu_cache


def _lib_importable(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


def decide(kind: str, n: int) -> AccelDecision:
    """Select CPU vs GPU for an operation of ``kind`` ("ann"|"temporal") over ``n`` items."""
    backend = _LIB.get(kind, "cpu")
    cpu = lambda why: AccelDecision(False, kind, n, "cpu", why)  # noqa: E731

    if not _enabled(kind):
        return cpu("accelerator disabled (set CR_ACCEL=1)")
    if not gpu_available():
        return cpu("no CUDA device — CPU is the fallback")
    if not _lib_importable(backend):
        return cpu(f"{backend} not installed — CPU is the fallback")
    threshold = _CROSSOVER.get(kind, 0)
    if n < threshold:
        return cpu(f"n={n} below {kind} crossover {threshold} — CPU is faster here")
    return AccelDecision(True, kind, n, backend, f"n={n} ≥ crossover {threshold} — {backend} selected")
