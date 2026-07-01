#!/usr/bin/env python3
"""Minimal retrieval harness for BM25 vs. hybrid semantic ranking.

This script builds a tiny multilingual-ish synthetic corpus and exercises two
retrievers provided by Context Runtime:

* ``InMemoryStore`` — a pure BM25 lexical scorer
* ``HybridRetriever`` — BM25 fused with semantic embeddings via reciprocal
  rank fusion (c=60)

It demonstrates that lexical retrieval excels on exact terms while the hybrid
stack (when ``fastembed`` is installed) surfaces the right passage for a synonym
query. The script is dependency-light and exits with status 0 whether or not
``fastembed`` is present.
"""
from __future__ import annotations

import pathlib
import sys

# Ensure the repo root is importable when the script is invoked directly.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from context_runtime.adapters.store_inmemory import InMemoryStore

EXACT_QUERY = "steroid hormone cortisol testosterone stress panel"
SYNONYM_QUERY = "How do androgens and glucocorticoid signals coordinate the human stress response?"
TARGET_CHUNK_ID = "steroid::0"


def _make_doc(name: str, text: str) -> dict[str, object]:
    return {"chunk_id": f"{name}::0", "filename": name, "text": text, "created_at": None}


CORPUS = (
    _make_doc(
        "steroid",
        (
            "Steroid hormones orchestrate adaptation: cortisol calibrates the stress "
            "response while testosterone sustains anabolic recovery. Clinicians order "
            "a steroid hormone panel when fatigue hints at endocrine imbalance."
        ),
    ),
    _make_doc(
        "lipid",
        (
            "Lipid clinicians analyze LDL cholesterol, HDL cholesterol, and triglycerides "
            "to refine cardiometabolic risk and adjust statin therapy or dietary fiber."
        ),
    ),
    _make_doc(
        "chat",
        (
            "User: hi, can you move my appointment to Friday? Assistant: sure, I have a slot "
            "after lunch — shall I book it?"
        ),
    ),
)


def run_bm25(corpus: tuple[dict[str, object], ...]) -> None:
    store = InMemoryStore(list(corpus))

    exact_hits = store.search(EXACT_QUERY, k=3, method="bm25")
    assert exact_hits, "BM25 returned no results for the exact-term query"
    assert (
        exact_hits[0].chunk_id == TARGET_CHUNK_ID
    ), f"Expected exact-term query to return {TARGET_CHUNK_ID}, got {exact_hits[0].chunk_id!r}"

    synonym_hits = store.search(SYNONYM_QUERY, k=3, method="bm25")
    print("BM25 exact-term top hit:", exact_hits[0].chunk_id)
    print("BM25 synonym top hit:", synonym_hits[0].chunk_id if synonym_hits else None)


def run_hybrid(corpus: tuple[dict[str, object], ...]) -> None:
    try:  # optional dependency — skip gracefully when missing
        import fastembed  # noqa: F401
    except ImportError:
        print("SKIP")
        return

    from context_runtime.adapters.store_semantic import HybridRetriever

    retriever = HybridRetriever(list(corpus))
    if not retriever.semantic.available:
        print("SKIP")
        return

    hybrid_hits = retriever.search(SYNONYM_QUERY, k=3, method="hybrid")
    assert hybrid_hits, "Hybrid retrieval returned no results for the synonym query"
    assert (
        hybrid_hits[0].chunk_id == TARGET_CHUNK_ID
    ), f"Expected hybrid synonym search to return {TARGET_CHUNK_ID}, got {hybrid_hits[0].chunk_id!r}"
    print("Hybrid synonym top hit:", hybrid_hits[0].chunk_id)


def main() -> int:
    run_bm25(CORPUS)
    run_hybrid(CORPUS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
