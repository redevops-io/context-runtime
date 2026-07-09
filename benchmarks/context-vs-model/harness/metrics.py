"""Retrieval-quality + context-pollution metrics — dual mode.

- If a question has gold_evidences (FinanceBench / nutrients: gold is a specific PASSAGE),
  a chunk is relevant when it's from a gold doc AND overlaps the gold text (Unicode-aware,
  so Cyrillic works).
- Else (LiveRAG: gold is a whole doc), relevance is doc-id membership.
"""
from __future__ import annotations

import re

_TOKEN = re.compile(r"\w+", re.UNICODE)
_REL_THRESH = 0.35


def _toks(s: str) -> set:
    return set(_TOKEN.findall(s.lower()))


def _relevant(chunk, question) -> bool:
    if chunk.doc_id not in question.gold_docs:
        return False
    ev = getattr(question, "gold_evidences", ())
    if not ev:
        return True                      # doc-id mode (LiveRAG)
    ct = _toks(chunk.text)
    if not ct:
        return False
    return any(_toks(g) and len(ct & _toks(g)) / len(ct) >= _REL_THRESH for g in ev)


def retrieval_metrics(retrieved, question) -> dict:
    if not question.gold_docs:
        return {"precision": None, "recall": None, "hit": None, "mrr": None, "n_ret": len(retrieved)}
    flags = [_relevant(c, question) for c in retrieved]
    n = len(retrieved)
    rr = next((1.0 / i for i, f in enumerate(flags, 1) if f), 0.0)
    ev = getattr(question, "gold_evidences", ())
    if ev:   # recall over the gold passage(s)
        covered = sum(1 for g in ev if any(_relevant(c, question) for c in retrieved
                                           if _toks(c.text) and _toks(g)
                                           and len(_toks(c.text) & _toks(g)) / len(_toks(c.text)) >= _REL_THRESH))
        recall = covered / len(ev)
    else:    # recall over gold docs
        got = {c.doc_id for c in retrieved if c.doc_id in question.gold_docs}
        recall = len(got) / len(question.gold_docs)
    return {"precision": (sum(flags) / n) if n else 0.0, "recall": recall,
            "hit": 1.0 if any(flags) else 0.0, "mrr": rr, "n_ret": n}


def pollution_fraction(retrieved, question) -> float:
    if not retrieved:
        return 0.0
    off = sum(1 for c in retrieved if c.doc_id not in question.gold_docs)
    return off / len(retrieved)
