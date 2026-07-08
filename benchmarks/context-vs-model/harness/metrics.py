"""Retrieval-quality + context-pollution metrics, scored against FinanceBench gold
evidence pages."""
from __future__ import annotations


def _page_keys(passages) -> list:
    return [(p.doc_name, p.page) for p in passages]


def retrieval_metrics(retrieved, gold_pages) -> dict:
    """Precision/recall/hit/MRR of ``retrieved`` passages vs the gold (doc, page) set."""
    keys = _page_keys(retrieved)
    goldset = set(gold_pages)
    if not goldset:
        return {"precision": None, "recall": None, "hit": None, "mrr": None, "n_ret": len(keys)}
    inter = [k for k in keys if k in goldset]
    rr = 0.0
    for rank, k in enumerate(keys, 1):
        if k in goldset:
            rr = 1.0 / rank
            break
    return {
        "precision": (len(set(inter)) / len(keys)) if keys else 0.0,
        "recall": len(set(inter) & goldset) / len(goldset),
        "hit": 1.0 if inter else 0.0,
        "mrr": rr,
        "n_ret": len(keys),
    }


def pollution_fraction(retrieved, question) -> float:
    """Share of retrieved passages NOT from the target filing (off-company/off-doc noise
    that actually reached the model)."""
    if not retrieved:
        return 0.0
    off = sum(1 for p in retrieved if p.doc_name != question.doc_name)
    return off / len(retrieved)
