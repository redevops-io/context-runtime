"""Retrieval-quality + context-pollution metrics, scored at DOC-ID level against
LiveRAG's gold Supporting_Documents (clean — no text-overlap heuristics needed)."""
from __future__ import annotations


def retrieval_metrics(retrieved, question) -> dict:
    """Precision/recall/hit/MRR of ``retrieved`` chunks vs the gold doc set."""
    gold = question.gold_docs
    if not gold:
        return {"precision": None, "recall": None, "hit": None, "mrr": None, "n_ret": len(retrieved)}
    flags = [c.doc_id in gold for c in retrieved]
    n = len(retrieved)
    rr = 0.0
    for rank, f in enumerate(flags, 1):
        if f:
            rr = 1.0 / rank
            break
    retrieved_gold = {c.doc_id for c in retrieved if c.doc_id in gold}
    return {
        "precision": (sum(flags) / n) if n else 0.0,
        "recall": len(retrieved_gold) / len(gold),
        "hit": 1.0 if any(flags) else 0.0,
        "mrr": rr,
        "n_ret": n,
    }


def pollution_fraction(retrieved, question) -> float:
    """Share of retrieved chunks NOT from a gold doc (distractor noise that reached the model)."""
    if not retrieved:
        return 0.0
    off = sum(1 for c in retrieved if c.doc_id not in question.gold_docs)
    return off / len(retrieved)
