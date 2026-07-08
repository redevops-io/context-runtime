"""Retrieval-quality + context-pollution metrics.

Scored against FinanceBench's gold ``evidence_text`` by TEXT OVERLAP, not chunk index —
because the corpus is chunked (1200 chars) and the gold is a PDF-page/table, the two
numbering schemes don't align. A retrieved chunk counts as relevant when it is from the
target filing AND most of its tokens appear in a gold evidence passage (containment) —
robust to how the corpus was chunked.
"""
from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")
# Fraction of a chunk's tokens that must appear in a gold evidence passage. A 1200-char
# chunk that contains the gold table shares only a modest fraction of ITS tokens with the
# shorter gold snippet, so 0.35 (probe-calibrated: gold ranks 1-4 at this level) is the
# right operating point; 0.55 undercounted true hits.
_REL_THRESH = 0.35


def _toks(s: str) -> set:
    return set(_TOKEN.findall(s.lower()))


def _relevant(chunk, question) -> bool:
    if chunk.doc_name != question.doc_name:
        return False
    ct = _toks(chunk.text)
    if not ct:
        return False
    for g in question.gold_evidences:
        gt = _toks(g)
        if gt and len(ct & gt) / len(ct) >= _REL_THRESH:
            return True
    return False


def retrieval_metrics(retrieved, question) -> dict:
    """Precision/recall/hit/MRR of ``retrieved`` chunks vs the gold evidence passages."""
    if not question.gold_evidences:
        return {"precision": None, "recall": None, "hit": None, "mrr": None, "n_ret": len(retrieved)}
    flags = [_relevant(c, question) for c in retrieved]
    n = len(retrieved)
    rr = 0.0
    for rank, f in enumerate(flags, 1):
        if f:
            rr = 1.0 / rank
            break
    # recall: gold evidence passages covered by ≥1 retrieved chunk
    covered = 0
    for g in question.gold_evidences:
        gt = _toks(g)
        if not gt:
            continue
        if any(c.doc_name == question.doc_name and _toks(c.text)
               and len(_toks(c.text) & gt) / len(_toks(c.text)) >= _REL_THRESH
               for c in retrieved):
            covered += 1
    return {
        "precision": (sum(flags) / n) if n else 0.0,
        "recall": covered / len(question.gold_evidences),
        "hit": 1.0 if any(flags) else 0.0,
        "mrr": rr,
        "n_ret": n,
    }


def pollution_fraction(retrieved, question) -> float:
    """Share of retrieved chunks NOT from the target filing (off-company/off-doc noise
    that actually reached the model)."""
    if not retrieved:
        return 0.0
    off = sum(1 for c in retrieved if c.doc_name != question.doc_name)
    return off / len(retrieved)
