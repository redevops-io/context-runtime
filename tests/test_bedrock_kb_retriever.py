"""BedrockKBRetriever is a RetrieverPlugin over `retrieve`, exercised with a fake client."""
from context_runtime.plugins import base
from context_runtime.providers.aws.bedrock_kb_retriever import BedrockKBRetriever


class FakeKB:
    def __init__(self):
        self.calls = []

    def retrieve(self, **kw):
        self.calls.append(kw)
        return {"retrievalResults": [
            {"content": {"text": "policy says 30 days"}, "score": 0.91,
             "location": {"type": "S3", "s3Location": {"uri": "s3://kb/policy.pdf"}},
             "metadata": {"page": 4}},
            {"content": {"text": "exceptions require approval"}, "score": 0.55,
             "location": {"type": "WEB", "webLocation": {"url": "https://x/doc"}}},
        ]}


def _r():
    return BedrockKBRetriever(client=FakeKB(), knowledge_base_id="KB123")


def test_satisfies_retriever_protocol():
    assert isinstance(_r(), base.RetrieverPlugin)


def test_search_maps_results_and_locations():
    hits = _r().search("refund window", k=4, method="hybrid")
    assert hits[0].text == "policy says 30 days" and hits[0].score == 0.91
    assert hits[0].filename == "s3://kb/policy.pdf" and hits[0].source == "bedrock_kb"
    assert hits[0].meta.get("page") == 4
    assert hits[1].filename == "https://x/doc"       # web location surfaced too


def test_retrieve_payload():
    fake = FakeKB()
    BedrockKBRetriever(client=fake, knowledge_base_id="KB9").search("q", k=6, method="vector")
    call = fake.calls[0]
    assert call["knowledgeBaseId"] == "KB9"
    assert call["retrievalQuery"] == {"text": "q"}
    assert call["retrievalConfiguration"]["vectorSearchConfiguration"]["numberOfResults"] == 6


def test_kb_as_managed_arm_in_router():
    from context_runtime.adapters.store_router import HopRouterRetriever
    from context_runtime.adapters.store_hipporag import SimGraphRetriever
    router = HopRouterRetriever(single_hop=_r(), graph=SimGraphRetriever([]))
    hits = router.search("refund", k=3, method="hybrid")
    assert hits and hits[0].source == "bedrock_kb"
