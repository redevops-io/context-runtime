"""Observability — the Trace builder (SPEC §6).

Every run emits one Trace: the observability record, the EXPLAIN ANALYZE data, the
replay input, and the learning loop's training row. v0.1 builds it in-process and can
persist as JSON; OpenLLMetry → Langfuse export is the optional exporter.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .. import jsonio
from ..types import Span, Trace


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TraceBuilder:
    """Accumulates spans during a run, then finalizes into a Trace."""

    def __init__(self, plan_id: str, goal_text: str):
        self.plan_id = plan_id
        self.goal_text = goal_text
        self.spans: list[Span] = []
        self._cost = 0.0
        self._tokens = 0
        self._cite: tuple[str, ...] = ()
        self._verified: bool | None = None
        self._cache = "miss"
        self._t0 = datetime.now(timezone.utc)

    def span(self, name: str, kind: str, attrs: dict, start: str, end: str) -> None:
        self.spans.append(Span(name=name, kind=kind, start=start, end=end, attrs=attrs))  # type: ignore[arg-type]

    def add_cost(self, usd: float, tokens: int) -> None:
        self._cost += usd
        self._tokens += tokens

    def set_citations(self, cites: tuple[str, ...]) -> None:
        self._cite = cites

    def set_verified(self, ok: bool | None) -> None:
        self._verified = ok

    def finalize(self) -> Trace:
        latency = (datetime.now(timezone.utc) - self._t0).total_seconds()
        return Trace(
            plan_id=self.plan_id, goal_text=self.goal_text, spans=tuple(self.spans),
            actual_cost_usd=round(self._cost, 6), actual_latency_seconds=round(latency, 4),
            actual_tokens=self._tokens, citations=self._cite,
            verification_passed=self._verified, cache=self._cache,  # type: ignore[arg-type]
        )


def now() -> str:
    return _now()


def save_trace(trace: Trace, dir_path: str) -> str:
    d = Path(dir_path).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    fp = d / f"{trace.id}.json"
    fp.write_text(jsonio.dumps(trace, indent=2))
    return str(fp)
