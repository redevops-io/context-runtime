"""A deterministic cost model: declared per-test estimates, and what follows from them.

Inputs only. Nothing here measures anything, reads a clock or consults history: an estimate is a number somebody
declared (a prior, or a figure taken from earlier runs and written down), and ``basis`` says which. All quantities
are integers (milliseconds, micro-units of cost, basis points of risk), so a result is exactly reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ._contracts import content_hash
from .closure import Closure
from .errors import Code, PlanningRefusal, Problem


@dataclass(frozen=True)
class Estimate:
    """What is declared about one test."""

    duration_ms: int
    """Expected duration, at least 1."""
    cost_micros: int = 0
    """Expected cost in millionths of a currency unit (or any consistent unit); 0 when free."""
    risk_bp: int = 0
    """Risk in basis points, 0 to 10000: how much a failure here matters, for ordering and evidence level."""
    basis: str = "prior"
    """Where the figures come from (``prior`` or ``history``). Reported in the explanation, not interpreted."""

    def __post_init__(self) -> None:
        for name in ("duration_ms", "cost_micros", "risk_bp"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int):
                raise _bad(f"{name} is an integer")
        if self.duration_ms < 1:
            raise _bad("duration_ms is at least 1")
        if self.cost_micros < 0:
            raise _bad("cost_micros is not negative")
        if not 0 <= self.risk_bp <= 10000:
            raise _bad("risk_bp is between 0 and 10000")
        if not isinstance(self.basis, str) or not self.basis:
            raise _bad("basis is a non-empty string")

    def to_dict(self) -> dict:
        return {"duration_ms": self.duration_ms, "cost_micros": self.cost_micros, "risk_bp": self.risk_bp,
                "basis": self.basis}


def _bad(detail: str) -> PlanningRefusal:
    return PlanningRefusal(Code.INVALID_REQUEST, detail, (Problem(Code.INVALID_REQUEST, "estimate", detail),))


class CostModel:
    """Declared estimates by test id. A test without an estimate is refused unless a default was declared
    explicitly; the model never invents a figure."""

    def __init__(self, estimates: Mapping[str, Estimate], default: Estimate | None = None):
        for k, v in estimates.items():
            if not isinstance(v, Estimate):
                raise _bad(f"the estimate for {k!r} is not an Estimate")
        self._estimates = dict(estimates)
        self._default = default

    def estimate(self, test_id: str) -> Estimate:
        est = self._estimates.get(test_id, self._default)
        if est is None:
            raise PlanningRefusal(Code.MISSING_ESTIMATE, f"no estimate declared for test {test_id}",
                                  (Problem(Code.MISSING_ESTIMATE, f"test:{test_id}", "no estimate declared"),))
        return est

    def require(self, closure: Closure) -> dict[str, Estimate]:
        """Estimates for every test of the closure, or a refusal naming every test that has none."""
        out: dict[str, Estimate] = {}
        missing: list[Problem] = []
        for tid in closure.test_ids:
            est = self._estimates.get(tid, self._default)
            if est is None:
                missing.append(Problem(Code.MISSING_ESTIMATE, f"test:{tid}", "no estimate declared"))
            else:
                out[tid] = est
        if missing:
            raise PlanningRefusal(Code.MISSING_ESTIMATE, f"{len(missing)} test(s) have no declared estimate",
                                  tuple(missing))
        return out

    def to_dict(self) -> dict:
        return {"estimates": {k: self._estimates[k].to_dict() for k in sorted(self._estimates)},
                "default": self._default.to_dict() if self._default else None}

    def digest(self) -> str:
        return content_hash(self.to_dict())


def successors(closure: Closure) -> dict[str, tuple[str, ...]]:
    """Direct successor tests, from the closure's direct predecessors."""
    succ: dict[str, list[str]] = {t: [] for t in closure.test_ids}
    for t in closure.test_ids:
        for p in closure.predecessors(t):
            succ[p].append(t)
    return {t: tuple(sorted(v)) for t, v in succ.items()}


def bottom_levels(closure: Closure, estimates: Mapping[str, Estimate]) -> dict[str, int]:
    """Longest path (by duration) from each test to the end of the dependency graph, the test itself included."""
    succ = successors(closure)
    level: dict[str, int] = {}
    for t in reversed(closure.test_ids):
        level[t] = estimates[t].duration_ms + max((level[s] for s in succ[t]), default=0)
    return level


def critical_path(closure: Closure, estimates: Mapping[str, Estimate]) -> tuple[int, tuple[str, ...]]:
    """The longest dependency chain by duration and its length. Ties break by the smallest identifier."""
    level = bottom_levels(closure, estimates)
    succ = successors(closure)
    start = min(level, key=lambda t: (-level[t], t))
    path = [start]
    while succ[path[-1]]:
        path.append(min(succ[path[-1]], key=lambda t: (-level[t], t)))
    return level[start], tuple(path)
