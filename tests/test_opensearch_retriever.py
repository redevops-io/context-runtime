"""OpenSearchRetriever is a document RetrieverPlugin, exercised with a fake OS client (no network)."""
from context_runtime.plugins import base
from context_runtime.providers.aws.opensearch_retriever import OpenSearchRetriever


class FakeOS:
    def __init__(self):
        self.calls = []

    def search(self, index, body):
        self.calls.append((index, body))
        return {"hits": {"hits": [
            {"_id": "c1", "_score": 4.2, "_source": {"text": "revenue grew 20%", "filename": "q3.md",
                                                     "created_at": "2026-01-01", "tenant": "acme"}},
            {"_id": "c2", "_score": 1.1, "_source": {"text": "costs held flat", "filename": "q3.md"}},
        ]}}


def _r():
    return OpenSearchRetriever(client=FakeOS(), index="docs")


def test_satisfies_retriever_protocol():
    assert isinstance(_r(), base.RetrieverPlugin)


def test_search_maps_os_hits_to_hits():
    hits = _r().search("revenue", k=5, method="hybrid")
    assert [h.chunk_id for h in hits] == ["c1", "c2"]
    assert hits[0].text == "revenue grew 20%" and hits[0].filename == "q3.md"
    assert hits[0].score == 4.2 and hits[0].source == "opensearch"
    assert hits[0].meta.get("tenant") == "acme"       # non-text fields carried into meta
    assert "text" not in hits[0].meta                 # the text field is not duplicated into meta


def test_query_shape_and_k():
    fake = FakeOS()
    OpenSearchRetriever(client=fake, index="docs", text_field="body").search("q", k=7, method="bm25")
    index, body = fake.calls[0]
    assert index == "docs" and body["size"] == 7
    assert body["query"]["multi_match"]["fields"] == ["body"]


def test_routes_through_hop_router_as_document_arm():
    from context_runtime.adapters.store_router import HopRouterRetriever
    from context_runtime.adapters.store_inmemory import InMemoryStore
    from context_runtime.adapters.store_hipporag import SimGraphRetriever
    router = HopRouterRetriever(single_hop=_r(), graph=SimGraphRetriever([]))
    hits = router.search("revenue", k=3, method="hybrid")
    assert hits and hits[0].source == "opensearch"
