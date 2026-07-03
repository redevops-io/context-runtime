"""In-process load meter — the missing 'system load' signal for load-aware retrieval.

DSpark's scheduler routes verification budget by *current engine load* (verify more when
idle, prune hard when saturated). We have no such signal: the control plane serves each
request in Starlette's threadpool with no cross-request view. This is the cheapest thing
that recovers one: a shared counter of in-flight retrieval calls plus a coarse band.

It intentionally does NOT build an admission queue or batch window (DSpark's Algorithm 1
schedules over a batch of concurrent requests — we don't have one yet). It only exposes
"how busy are we right now?" so the bandit context and the expensive-stage sizer can
condition on load. Thread-safe; process-local (one control-plane process).
"""
from __future__ import annotations

import threading


class LoadMeter:
    """Counts in-flight work and maps the count to a coarse band (lo / mid / hi)."""

    def __init__(self, mid: int = 4, hi: int = 12):
        # thresholds: < mid → "lo", < hi → "mid", else "hi". Defaults suit a single
        # threadpool control plane; override for bigger deployments.
        self._mid = mid
        self._hi = hi
        self._inflight = 0
        self._peak = 0
        self._lock = threading.Lock()

    def enter(self) -> None:
        with self._lock:
            self._inflight += 1
            self._peak = max(self._peak, self._inflight)

    def leave(self) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)

    def inflight(self) -> int:
        with self._lock:
            return self._inflight

    def band(self) -> str:
        n = self.inflight()
        if n < self._mid:
            return "lo"
        if n < self._hi:
            return "mid"
        return "hi"

    class _Scope:
        def __init__(self, meter: "LoadMeter"):
            self._m = meter

        def __enter__(self):
            self._m.enter()
            return self._m

        def __exit__(self, *exc):
            self._m.leave()
            return False

    def track(self) -> "LoadMeter._Scope":
        """`with meter.track():` — count this block as one in-flight request."""
        return LoadMeter._Scope(self)
