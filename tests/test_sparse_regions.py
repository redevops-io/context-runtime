"""Contract tests for Evidence Sparse Retrieval (F4).

The three acceptance criteria are tested directly: deterministic corpus-statistical routing (no LLM),
confidence-floor → global fallback, and identity-transparent EvidenceRefs (the store's Hits pass through
unchanged). Requires the public ``context_runtime`` package; skipped cleanly when absent (Doris pattern).
"""
from __future__ import annotations

import pytest


from context_runtime.types import Hit  # noqa: E402

from context_runtime.adapters.sparse_regions import (  # noqa: E402
    RegionIndex, SparseRegionRetriever,
)

# A tiny corpus: two documents (regions), three chunks each. Region "alpha" is about turbines; region
# "beta" is about ledgers. Embeddings are 2-D and separate the topics cleanly.
CHUNKS = [
    {"id": "a0", "document_id": "a0", "filename": "alpha", "text": "wind turbine blade pitch control"},
    {"id": "a1", "document_id": "a1", "filename": "alpha", "text": "turbine gearbox lubrication schedule"},
    {"id": "a2", "document_id": "a2", "filename": "alpha", "text": "turbine yaw bearing inspection"},
    {"id": "b0", "document_id": "b0", "filename": "beta", "text": "general ledger reconciliation entries"},
    {"id": "b1", "document_id": "b1", "filename": "beta", "text": "accounts payable ledger posting"},
    {"id": "b2", "document_id": "b2", "filename": "beta", "text": "ledger trial balance closing"},
]
EMB = [[1.0, 0.0], [0.9, 0.1], [0.95, 0.05],      # alpha ≈ x-axis
       [0.0, 1.0], [0.1, 0.9], [0.05, 0.95]]       # beta  ≈ y-axis


def _embed_query(q: str):
    ql = q.lower()
    return [1.0, 0.0] if "turbine" in ql else [0.0, 1.0] if "ledger" in ql else [0.5, 0.5]


def _fake_search(calls):
    """A scoped_search that records the doc_ids it was scoped to, and returns store-identity Hits."""
    def search(query, k, method, doc_ids):
        calls.append(doc_ids)
        ids = doc_ids if doc_ids is not None else [c["document_id"] for c in CHUNKS]
        return [Hit(chunk_id=i, filename="f", text=f"chunk {i}", content_hash=f"h-{i}", version="v1")
                for i in ids][:k]
    return search


def _index():
    return RegionIndex.build(CHUNKS, EMB)


def test_regions_form_deterministically_from_documents():
    idx = _index()
    ids = [r.region_id for r in idx.regions]
    assert ids == ["alpha", "beta"]                       # sorted, one region per source document
    # rebuild → identical regions (no seed, no model, stable order)
    assert [r.region_id for r in RegionIndex.build(CHUNKS, EMB).regions] == ids
    assert idx.regions[0].doc_ids == ("a0", "a1", "a2")   # references to original evidence, not copies


def test_routing_narrows_to_the_relevant_region():
    calls = []
    r = SparseRegionRetriever(_index(), _fake_search(calls), _embed_query, top_regions=1)
    hits = r.search("turbine gearbox oil", k=3)
    assert calls[-1] == ["a0", "a1", "a2"]                # scoped to the turbine region only
    assert "alpha" in r.last_reason and "beta" not in r.last_reason
    assert [h.chunk_id for h in hits] == ["a0", "a1", "a2"]


def test_confidence_floor_falls_back_to_global():
    calls = []
    # An off-topic query embeds to [0.5,0.5] with no term overlap → no region clears a high floor.
    r = SparseRegionRetriever(_index(), _fake_search(calls), _embed_query, top_regions=1, floor=0.9)
    r.search("xyzzy plugh frobnicate", k=6)
    assert calls[-1] is None                              # global fallback: no doc scope
    assert "global fallback" in r.last_reason


def test_identity_is_transparent():
    r = SparseRegionRetriever(_index(), _fake_search([]), _embed_query, top_regions=1)
    hits = r.search("ledger trial balance", k=3)
    # F4 returns the store's Hits unchanged — provenance (content_hash/version) survives the region layer.
    assert all(h.content_hash == f"h-{h.chunk_id}" and h.version == "v1" for h in hits)
    assert [h.chunk_id for h in hits] == ["b0", "b1", "b2"]


def test_tenant_scope_filters_regions_before_ranking():
    idx = _index()
    # Tag the two regions with tenants; a caller scoped to "t-beta" must never route into alpha.
    idx.regions[0] = idx.regions[0].__class__(**{**idx.regions[0].__dict__, "tenant": "t-alpha"})
    idx.regions[1] = idx.regions[1].__class__(**{**idx.regions[1].__dict__, "tenant": "t-beta"})
    calls = []
    r = SparseRegionRetriever(idx, _fake_search(calls), _embed_query, top_regions=2, tenant="t-beta")
    r.search("turbine", k=6)                              # topically alpha, but tenant-scoped to beta
    assert calls[-1] == ["b0", "b1", "b2"]                # only beta's docs are reachable
