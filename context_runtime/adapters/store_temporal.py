"""Temporal / bi-temporal retrieval — "what changed, and when?" (Whitepaper v3, forthcoming retrieval).

The next retrieval method: a **bi-temporal** store (Graphiti/Zep lineage) where every fact carries two
time axes —

  • **valid time**  (``valid_from`` / ``valid_to``): when the fact is true in the world;
  • **transaction time** (``recorded_at``): when the system learned it.

That lets the planner answer point-in-time questions the other methods can't: the state *as of* a past
date, and *as it was known* at a past date (so a late-arriving correction doesn't rewrite history). It is
densest in conversational corpora (chat/tickets) where facts are constantly revised.

Dependency-free (stdlib + the ``RetrieverPlugin`` seam), so it drops into the planner as one more
routable capability (``method="temporal"``); the full Graphiti engine can back it as an optional binding.
Times are compared as plain ISO strings (``"2026-01-15"``): ``""`` = the beginning of time, ``None`` on
``valid_to`` = still valid.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..types import Hit, PluginInfo, Retrieval

_STOP = {"the", "a", "an", "of", "to", "in", "for", "is", "was", "did", "who", "what", "when", "how",
         "owns", "own"}
# keep hyphen/underscore inside a token so entity ids ("auth-service") stay whole and don't collide
# with a shared fragment ("service") on unrelated subjects
_WORD = re.compile(r"[a-z0-9][a-z0-9_-]*")


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if len(t) > 1 and t not in _STOP}


@dataclass
class TemporalFact:
    subject: str
    predicate: str
    obj: str
    valid_from: str = ""          # inclusive; "" = -inf
    valid_to: str | None = None   # exclusive; None = still valid (+inf)
    recorded_at: str = ""         # transaction time; "" = -inf
    meta: dict = field(default_factory=dict)

    def text(self) -> str:
        return f"{self.subject} {self.predicate} {self.obj}"

    def valid_at(self, at: str | None) -> bool:
        if at is None:                                   # "current" = not yet superseded
            return self.valid_to is None
        return self.valid_from <= at and (self.valid_to is None or at < self.valid_to)

    def known_at(self, known: str | None) -> bool:
        return known is None or self.recorded_at <= known


class TemporalStore:
    """An in-memory bi-temporal fact store exposing the RetrieverPlugin seam."""

    def __init__(self, facts: list[TemporalFact] | None = None):
        self._facts: list[TemporalFact] = list(facts or [])

    def add(self, subject: str, predicate: str, obj: str, *, valid_from: str = "",
            valid_to: str | None = None, recorded_at: str = "", meta: dict | None = None) -> "TemporalStore":
        self._facts.append(TemporalFact(subject, predicate, obj, valid_from, valid_to,
                                        recorded_at, meta or {}))
        return self

    def _hit(self, f: TemporalFact, score: float) -> Hit:
        return Hit(
            chunk_id=f"{f.subject}:{f.predicate}:{f.valid_from or '-'}",
            filename="temporal",
            text=f.text(),
            score=score,
            created_at=f.valid_from or None,
            source="temporal",
            meta={"subject": f.subject, "predicate": f.predicate, "object": f.obj,
                  "valid_from": f.valid_from, "valid_to": f.valid_to, "recorded_at": f.recorded_at,
                  **f.meta},
        )

    def _ranked(self, query: str, candidates: list[TemporalFact], k: int) -> list[Hit]:
        q = _tokens(query)
        scored: list[tuple[float, str, TemporalFact]] = []
        for f in candidates:
            overlap = len(q & _tokens(f.text()))
            if q and overlap == 0:
                continue
            score = overlap / (len(q) or 1)
            scored.append((score, f.valid_from, f))
        # best match first; ties → most-recently-valid first
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [self._hit(f, s) for s, _, f in scored[:k]]

    # ── the RetrieverPlugin seam ──
    def search(self, query: str, k: int = 5, method: Retrieval = "temporal") -> list[Hit]:
        """Current state: facts not yet superseded (``valid_to is None``), ranked by query match."""
        return self._ranked(query, [f for f in self._facts if f.valid_at(None)], k)

    def as_of(self, query: str, k: int = 5, *, at: str, known_at: str | None = None) -> list[Hit]:
        """Point-in-time query: facts valid at ``at`` (world time) and known at ``known_at`` (transaction
        time). ``known_at`` lets you ask "what did we believe on date X", ignoring later corrections."""
        return self._ranked(query, [f for f in self._facts
                                    if f.valid_at(at) and f.known_at(known_at)], k)

    def changes(self, query: str = "", *, since: str, until: str, k: int = 20) -> list[dict]:
        """"What changed, and when?" — facts that began or ended validity in ``[since, until)``."""
        q = _tokens(query)
        out: list[dict] = []
        for f in self._facts:
            if q and not (q & _tokens(f.text())):
                continue
            if since <= f.valid_from < until:
                out.append({"at": f.valid_from, "change": "began", "fact": f.text(),
                            "subject": f.subject, "predicate": f.predicate, "object": f.obj})
            if f.valid_to is not None and since <= f.valid_to < until:
                out.append({"at": f.valid_to, "change": "ended", "fact": f.text(),
                            "subject": f.subject, "predicate": f.predicate, "object": f.obj})
        out.sort(key=lambda c: c["at"])
        return out[:k]

    def info(self) -> PluginInfo:
        return PluginInfo(name="temporal", kind="retriever",
                          capabilities=frozenset({"temporal", "retrieval"}))
