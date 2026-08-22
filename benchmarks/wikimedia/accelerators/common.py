"""Shared utilities for the accelerator crossover arms: GPU detection and honest timing.

Timing methodology (roadmap §2.2 "total accelerator decision latency"): a GPU number is only fair if it
includes the host<->device transfer and result copy, not just the kernel. Each timed callable therefore
does its own transfer inside the measured region. We warm up once (to pay one-time CUDA context / kernel
JIT / cuVS+cuOpt init outside steady state), then take the **median** of N repeats. The one-time cold
cost is reported separately so a reader can see both "first call" and "amortised" behaviour.
"""
from __future__ import annotations

import statistics
import time
from typing import Callable


def gpu_available() -> bool:
    try:
        import cupy  # noqa: F401
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def gpu_sync() -> None:
    """Block until all queued GPU work finishes — required before stopping a GPU timer."""
    import cupy as cp
    cp.cuda.Stream.null.synchronize()


def gpu_info() -> dict:
    try:
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        cc = f"{props['major']}.{props['minor']}"
        free, total = cp.cuda.runtime.memGetInfo()
        return {"name": name, "compute_capability": cc,
                "vram_free_gb": round(free / 1e9, 1), "vram_total_gb": round(total / 1e9, 1)}
    except Exception as e:  # pragma: no cover - only on non-GPU hosts
        return {"error": f"{type(e).__name__}: {e}"}


def time_median(fn: Callable[[], object], *, repeats: int = 5, warmup: int = 1,
                sync: bool = False) -> tuple[float, float, float, object]:
    """Run ``fn`` ``warmup``+``repeats`` times. Return (median_ms, min_ms, cold_ms, last_result).

    ``cold_ms`` is the very first (warmup) call — the one that pays CUDA init / JIT for GPU work.
    ``median_ms``/``min_ms`` are over the post-warmup repeats (steady state). ``sync=True`` calls
    ``gpu_sync()`` inside the timed region so GPU kernels are actually complete when the clock stops.
    """
    def _one() -> tuple[float, object]:
        t0 = time.perf_counter()
        r = fn()
        if sync:
            gpu_sync()
        return (time.perf_counter() - t0) * 1000.0, r

    cold_ms, result = _one()
    for _ in range(max(0, warmup - 1)):
        _, result = _one()
    samples: list[float] = []
    for _ in range(repeats):
        ms, result = _one()
        samples.append(ms)
    return statistics.median(samples), min(samples), cold_ms, result
