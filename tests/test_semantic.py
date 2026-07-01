"""Hybrid BM25⊕semantic retrieval. The graceful-fallback path (no fastembed) is always
tested; the semantic path runs only when the [embeddings] extra is installed."""
from __future__ import annotations

import pytest

from context_runtime.adapters.store_semantic import HybridRetriever, embeddings_available


def _corpus(tmp_path):
    docs = {
        "steroid.txt": "Стероидный профиль: тестостерон, кортизол, ДГЭА методом ВЭЖХ-МС/МС.",
        "lipid.txt": "Липидный профиль: холестерин, ЛПНП, ЛПВП, триглицериды.",
        "hormone.txt": "Репродуктивные гормоны: ФСГ, ЛГ, пролактин.",
    }
    for name, text in docs.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return str(tmp_path)


def test_hybrid_falls_back_to_bm25_without_embeddings(tmp_path):
    # Without fastembed, HybridRetriever must still work (pure BM25) — the base install
    # is never broken by the optional semantic layer.
    r = HybridRetriever()
    r.index(_corpus(tmp_path))
    hits = r.search("тестостерон кортизол", k=3, method="bm25")
    assert hits and "steroid" in hits[0].filename
    # hybrid/vector requested but embeddings absent → still returns BM25 results, no crash.
    hits2 = r.search("тестостерон", k=3, method="hybrid")
    assert hits2, "hybrid must fall back to BM25 when embeddings are unavailable"


@pytest.mark.skipif(not embeddings_available(), reason="[embeddings] extra not installed")
def test_semantic_bridges_morphology(tmp_path):
    r = HybridRetriever()
    r.index(_corpus(tmp_path))
    # «триглицеридах» (a different morphological form than the doc's «триглицериды») —
    # BM25 would miss it; the embedding model should still rank the lipid doc top.
    hits = r.search("анализ на триглицеридах и холестерине", k=3, method="vector")
    assert hits and "lipid" in hits[0].filename
