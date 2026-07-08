"""Lean, self-contained retrieval — a compact BM25 with score-gating and a cheap
rerank, honoring the same knobs Context Runtime's bandit tunes on
``redevops_rag.hybrid_search`` (``pool`` / ``limit`` / ``vector_threshold`` / ``rerank``).

No torch, no vector DB: reproducible anywhere. BM25 is a fair substrate for the
pollution experiment precisely because cross-company distractors share line-item
vocabulary — lexical retrieval is exactly what struggles, and gating is exactly what
Context Runtime adds.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(s: str) -> list:
    return _TOKEN.findall(s.lower())


@dataclass
class Hit:
    passage: object      # data.Passage
    score: float         # normalized 0..1 within the query
    raw: float


class BM25Index:
    """BM25 over a fixed passage pool. Cheap enough to rebuild per (question, pollution
    level) since a scoped pool is small (one filing + a few distractors)."""

    def __init__(self, passages: list, *, k1: float = 1.5, b: float = 0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        self.docs = [tokenize(p.text) for p in passages]
        self.dl = [len(d) for d in self.docs]
        self.avgdl = (sum(self.dl) / len(self.dl)) if self.dl else 0.0
        self.tf = [Counter(d) for d in self.docs]
        df: Counter = Counter()
        for d in self.tf:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def _score(self, q_tokens: list, i: int) -> float:
        tf, dl = self.tf[i], self.dl[i]
        s = 0.0
        for t in q_tokens:
            if t not in tf:
                continue
            f = tf[t]
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            s += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / (denom or 1)
        return s

    def search(self, query: str, *, pool: int = 50, limit: int = 8,
               vector_threshold: float = 0.0, rerank: bool = False,
               keyword_boost_per_term: float = 0.0, **_ignored) -> list:
        """Return up to ``limit`` Hits. ``vector_threshold`` gates weak matches (the
        pollution filter); ``rerank`` runs a cheap lexical-overlap reorder of the pool.
        Extra kwargs (recency_*, boost caps) are accepted and ignored so a full
        ``RetrievalConfig.kwargs()`` can be splatted straight in."""
        qt = tokenize(query)
        scored = [(self._score(qt, i), i) for i in range(len(self.passages))]
        scored.sort(reverse=True)
        cand = scored[: max(pool, limit)]
        if not cand:
            return []
        top = cand[0][0] or 1.0
        norm = [(raw / top, i, raw) for raw, i in cand]      # 0..1 within this query

        if rerank:
            qset = set(qt)
            def overlap(i):
                dt = self.tf[i]
                return sum(dt[t] for t in qset if t in dt) / (self.dl[i] or 1)
            norm.sort(key=lambda r: (0.5 * r[0] + 0.5 * min(1.0, 20 * overlap(r[1]))), reverse=True)

        hits = [Hit(passage=self.passages[i], score=sc, raw=raw)
                for sc, i, raw in norm if sc >= vector_threshold]
        return hits[:limit]
