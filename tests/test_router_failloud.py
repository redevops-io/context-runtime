"""Phase 0: the router never silently substitutes BM25 for an analytical/multimodal query.

The one genuinely dangerous behaviour the AWS-fit audit found: planner routes method="sql", no
analytical backend is wired, and the query falls through to BM25 with no signal. Here we prove the
router fails loudly by default, abstains when asked, and dispatches to a bound backend.
"""
import pytest

from context_runtime.adapters.store_inmemory import InMemoryStore
from context_runtime.adapters.store_hipporag import SimGraphRetriever
from context_runtime.adapters.store_router import HopRouterRetriever, UnboundRepresentationError
from context_runtime.types import Hit

CORPUS = [{"chunk_id": "d1", "filename": "d1", "text": "revenue rows here", "created_at": None}]


def _router(**kw):
    return HopRouterRetriever(single_hop=InMemoryStore(list(CORPUS)),
                              graph=SimGraphRetriever(list(CORPUS)), **kw)


@pytest.mark.parametrize("method", ["sql", "mongo", "elastic", "logs", "api", "image", "colpali", "video"])
def test_unbound_nonlexical_method_raises_not_bm25(method):
    with pytest.raises(UnboundRepresentationError):
        _router().search("how many invoices last quarter", k=5, method=method)


@pytest.mark.parametrize("method", ["sql", "image"])
def test_abstain_mode_returns_empty_not_bm25(method):
    assert _router(on_unbound="abstain").search("how many invoices", k=5, method=method) == []


def test_bound_analytical_backend_is_used():
    class FakeAnalytical:
        def search(self, query, k, method):
            return [Hit(chunk_id="sql:1", filename="warehouse", text="count=42", score=1.0,
                        meta={"method": method})]
        def info(self):
            from context_runtime.types import PluginInfo
            return PluginInfo(name="fake_sql", kind="retriever", capabilities=frozenset({"sql"}))

    hits = _router(analytical=FakeAnalytical()).search("how many invoices", k=5, method="sql")
    assert hits and hits[0].text == "count=42" and hits[0].meta["method"] == "sql"


def test_document_methods_still_fall_through_gracefully():
    # bm25/hybrid/file/code are lexical — the safe fallback is preserved, no raise
    r = _router()
    for method in ("hybrid", "bm25", "file", "code"):
        assert isinstance(r.search("revenue", k=3, method=method), list)
