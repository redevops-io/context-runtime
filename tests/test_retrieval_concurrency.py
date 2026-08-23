"""Retrieval fan-out (LLM-parallelization audit, P1) — opt-in, order-preserving, result-identical.
Default (CR_RETRIEVAL_CONCURRENCY unset) is the historical serial path."""
from __future__ import annotations

import time

import pytest

from context_runtime._parallel import run_parallel


def test_run_parallel_serial_by_default(monkeypatch):
    monkeypatch.delenv("CR_RETRIEVAL_CONCURRENCY", raising=False)
    assert run_parallel([lambda: 1, lambda: 2, lambda: 3]) == [1, 2, 3]


def test_run_parallel_preserves_order_and_overlaps(monkeypatch):
    monkeypatch.setenv("CR_RETRIEVAL_CONCURRENCY", "3")

    def slow(v):
        return lambda: (time.sleep(0.05), v)[1]

    t0 = time.perf_counter()
    out = run_parallel([slow(1), slow(2), slow(3)])
    wall = time.perf_counter() - t0
    assert out == [1, 2, 3]                 # order preserved
    assert wall < 0.12                      # 3×0.05=0.15s serial → overlapped to ~0.05s


def test_hybrid_search_concurrent_equals_serial(monkeypatch):
    """HybridRetriever's BM25 + vector legs overlap under concurrency but return the identical fused set."""
    from context_runtime.adapters.store_semantic import HybridRetriever, embeddings_available
    if not embeddings_available():
        pytest.skip("no embedder (fastembed) available")
    docs = [{"chunk_id": f"c{i}", "filename": f"f{i}.md",
             "text": t, "created_at": None}
            for i, t in enumerate(["testosterone and androgens", "blood lipid profile panel",
                                   "the cat sat on the mat", "insulin and glucose metabolism",
                                   "quarterly revenue growth"])]
    r = HybridRetriever(list(docs))

    monkeypatch.delenv("CR_RETRIEVAL_CONCURRENCY", raising=False)
    serial = [(h.filename, h.chunk_id) for h in r.search("androgen hormone levels", k=3)]

    r2 = HybridRetriever(list(docs))
    monkeypatch.setenv("CR_RETRIEVAL_CONCURRENCY", "2")
    parallel = [(h.filename, h.chunk_id) for h in r2.search("androgen hormone levels", k=3)]

    assert serial == parallel and len(serial) > 0   # identical fused ranking, order and all
