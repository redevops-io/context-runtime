"""Incremental insert (LightRAG-style live-corpus update) for the graph retrievers — extend, not rebuild."""
from context_runtime.adapters.store_hipporag import HippoRAGRetriever, SimGraphRetriever


def test_simgraph_insert_dedups_and_is_visible():
    g = SimGraphRetriever()
    g.insert([{"chunk_id": "a", "filename": "a", "text": "messi barcelona forward"}])
    r = g.insert([{"chunk_id": "b", "filename": "b", "text": "barcelona spain league"},
                  {"chunk_id": "a", "filename": "a", "text": "dup"}])   # 'a' already present
    assert r == {"inserted": 1, "skipped": 1, "total": 2}
    # 'b' has no direct overlap with the query but bridges via 'barcelona' — multi-hop, and it's live
    hits = g.search("messi league", k=2)
    assert any(h.chunk_id == "b" for h in hits), "inserted bridge doc not visible to the next search"


class _FakeHippo:
    """Stand-in for the real engine: records exactly which docs were sent to OpenIE/index()."""
    def __init__(self):
        self.indexed = []

    def index(self, docs):
        self.indexed += list(docs)


def test_hipporag_insert_only_extracts_new_docs():
    hr = HippoRAGRetriever()
    hr._hr = _FakeHippo()   # bypass the heavy lazy import
    assert hr.insert(["doc one", "doc two"]) == {"inserted": 2, "skipped": 0, "total": 2}
    assert hr.insert(["doc two", "doc three"]) == {"inserted": 1, "skipped": 1, "total": 3}
    # the whole point: a re-insert does NOT re-OpenIE the corpus — only genuinely new texts are extracted
    assert hr._hr.indexed == ["doc one", "doc two", "doc three"]
