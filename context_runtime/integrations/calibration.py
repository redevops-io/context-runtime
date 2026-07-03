"""Score calibration — turn raw retrieval scores into P(relevant).

Retrieval scores (BM25, cosine, RRF) are uncalibrated and not comparable across
methods or queries: a hybrid score of 0.6 and a bm25 score of 0.6 mean different
things, and neither is a probability. That is fine for *ranking* (RRF only needs
order), but the moment we want to reason about *absolute* quality — abstain when the
best passage is probably irrelevant, size the expensive stage by expected accepted
relevance (the load-aware scheduler), or show an honest confidence in the LibreQB
panel — we need P(relevant | score).

This module is the DSpark "confidence head + Sequential Temperature Scaling" idea
ported to retrieval: a cheap, **order-preserving** map fit from (score, judged-relevance)
pairs, per method. We fit it with isotonic regression (pool-adjacent-violators) — the
non-parametric, monotone analogue of temperature scaling — so it corrects the absolute
magnitude without ever reordering hits within a method.

Two halves:
  • CalibrationLog  — append (method, per-hit scores, relevance label) rows as JSONL.
                      This is the training-data layer, which did not exist before.
  • CalibrationMap  — per-method fitted score→P(relevant), persisted as JSON, applied
                      at query time. fit_from_log() builds one from a log.

Everything is opt-in: no map ⇒ callers behave exactly as before.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path


# ──────────────────────────── isotonic regression (PAV) ────────────────────────────


def _isotonic_fit(pairs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Pool-Adjacent-Violators: fit a non-decreasing step function to (x, y) pairs.

    Returns a compact list of (x_threshold, y_value) breakpoints, sorted by x. y is the
    calibrated probability for scores >= that x (and < the next breakpoint's x). Pure
    Python, no numpy/sklearn — matches the repo's no-heavy-deps convention.
    """
    if not pairs:
        return []
    pts = sorted(pairs, key=lambda p: p[0])
    # each block: [sum_y, weight, x_left]; merge adjacent blocks that violate monotonicity
    blocks: list[list[float]] = []
    for x, y in pts:
        blocks.append([y, 1.0, x])
        while len(blocks) >= 2 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            ys, w, xl = blocks.pop()
            blocks[-1][0] += ys
            blocks[-1][1] += w
            # keep the left-most x of the merged block as its threshold
    return [(b[2], round(b[0] / b[1], 6)) for b in blocks]


@dataclass
class _MethodCal:
    """Fitted calibration for one retrieval method: monotone step breakpoints."""

    breakpoints: list[tuple[float, float]] = field(default_factory=list)
    n: int = 0

    def apply(self, score: float) -> float:
        """Calibrated P(relevant) for a raw score (piecewise-constant, monotone)."""
        if not self.breakpoints:
            return score  # identity fallback until fit — never worse than raw
        p = self.breakpoints[0][1]
        for x, y in self.breakpoints:
            if score >= x:
                p = y
            else:
                break
        return p


class CalibrationMap:
    """Per-method score→P(relevant). Load a fitted artifact, apply at query time."""

    def __init__(self, methods: dict[str, _MethodCal] | None = None):
        self._m: dict[str, _MethodCal] = methods or {}

    def apply(self, method: str, score: float) -> float:
        cal = self._m.get(method)
        return cal.apply(score) if cal else score

    def has(self, method: str) -> bool:
        return method in self._m and bool(self._m[method].breakpoints)

    def to_dict(self) -> dict:
        return {m: {"breakpoints": c.breakpoints, "n": c.n} for m, c in self._m.items()}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)
        os.replace(tmp, path)  # atomic

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationMap | None":
        p = Path(path)
        if not p.exists():
            return None
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        methods = {
            m: _MethodCal(breakpoints=[tuple(bp) for bp in d.get("breakpoints", [])],
                          n=int(d.get("n", 0)))
            for m, d in raw.items()
        }
        return cls(methods)


# ──────────────────────────── the training-data log ────────────────────────────


class CalibrationLog:
    """Append (method, per-hit score, relevance label) rows as JSONL — the data the
    calibration map is fit from. Thread-safe append (control plane serves concurrently).

    A row is one query outcome:
      {"method": "hybrid", "bucket": "lookup", "label": 0.9,
       "hits": [{"chunk_id": "...", "score": 0.71, "rel": 1.0}, ...]}
    where ``label`` is the per-query judge score and each hit's ``rel`` is a per-passage
    relevance label when available (else the query label is used as a weak proxy).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, method: str, bucket: str, label: float, hits: list[dict]) -> None:
        row = {"method": method, "bucket": bucket, "label": round(float(label), 4), "hits": hits}
        line = json.dumps(row, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def rows(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out


# ──────────────────────────── fit a map from a log ────────────────────────────


def fit_from_log(log: "CalibrationLog | str | Path", min_samples: int = 20) -> CalibrationMap:
    """Fit a per-method CalibrationMap from a CalibrationLog.

    Per hit, the training pair is (raw score, relevance label): prefer the per-passage
    ``rel`` when present, else fall back to the per-query ``label`` as a weak label. A
    method with fewer than ``min_samples`` pairs is left unfit (identity) rather than
    over-fit to noise.
    """
    if not isinstance(log, CalibrationLog):
        log = CalibrationLog(log)
    by_method: dict[str, list[tuple[float, float]]] = {}
    for row in log.rows():
        method = row.get("method", "")
        label = float(row.get("label", 0.0))
        for h in row.get("hits", []):
            try:
                score = float(h["score"])
            except (KeyError, TypeError, ValueError):
                continue
            rel = h.get("rel")
            y = float(rel) if rel is not None else label
            by_method.setdefault(method, []).append((score, max(0.0, min(1.0, y))))
    methods: dict[str, _MethodCal] = {}
    for method, pairs in by_method.items():
        if len(pairs) < min_samples:
            methods[method] = _MethodCal(breakpoints=[], n=len(pairs))
        else:
            methods[method] = _MethodCal(breakpoints=_isotonic_fit(pairs), n=len(pairs))
    return CalibrationMap(methods)
