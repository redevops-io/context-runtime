"""CommunityRetriever (#1, global/broad queries) and IterativeRetriever (#3, multi-hop).

Both are exercised on their deterministic, no-LLM paths so they always run in CI.
"""
from __future__ import annotations

from context_runtime.adapters.retriever_iterative import IterativeRetriever
from context_runtime.adapters.store_community import CommunityRetriever
from context_runtime.adapters.store_inmemory import InMemoryStore


def _docs():
    # two clear topics: a steroid/hormone cluster and a lipid cluster, plus an off-topic chat
    return [
        {"chunk_id": "s1", "filename": "s1", "text": "steroid hormone panel testosterone cortisol dhea androstenedione levels", "created_at": None},
        {"chunk_id": "s2", "filename": "s2", "text": "testosterone cortisol steroid results hormone dhea reference ranges", "created_at": None},
        {"chunk_id": "s3", "filename": "s3", "text": "cortisol steroid hormone dhea testosterone endocrine assessment", "created_at": None},
        {"chunk_id": "l1", "filename": "l1", "text": "lipid panel cholesterol ldl hdl triglycerides cardiometabolic", "created_at": None},
        {"chunk_id": "l2", "filename": "l2", "text": "cholesterol ldl hdl lipid triglycerides statin therapy results", "created_at": None},
        {"chunk_id": "c1", "filename": "c1", "text": "hello can we reschedule the meeting to friday afternoon please", "created_at": None},
    ]


def test_community_detection_groups_topics():
    cr = CommunityRetriever(_docs(), min_shared=2)
    comms = cr._build()
    # the three steroid docs should land in one community, the two lipid docs in another
    by_member = {}
    for c in comms:
        for m in c["members"]:
            by_member[cr.docs[m]["chunk_id"]] = c["id"]
    assert by_member["s1"] == by_member["s2"] == by_member["s3"]
    assert by_member["l1"] == by_member["l2"]
    assert by_member["s1"] != by_member["l1"]  # distinct communities


def test_community_search_returns_summary_spanning_members():
    cr = CommunityRetriever(_docs(), min_shared=2)
    hits = cr.search("steroid hormone testosterone results", k=2, method="community")
    assert hits, "expected a community hit"
    top = hits[0]
    assert top.chunk_id.startswith("community::")
    assert top.meta["size"] >= 2                      # a real community, not a singleton
    assert set(top.meta["members"]) >= {"s1", "s2"}   # spans the steroid passages
    assert "steroid" in top.text.lower()


def test_iterative_gathers_more_than_single_shot_deterministically():
    base = InMemoryStore(_docs())
    it = IterativeRetriever(base, max_rounds=2, expand_terms=4)  # no model → deterministic expansion
    single = base.search("steroid hormone", k=2, method="bm25")
    multi = it.search("steroid hormone", k=5, method="bm25")
    assert {h.chunk_id for h in single} <= {h.chunk_id for h in multi}
    # round-2 expansion pulls in sibling steroid passages the single shot may have missed
    assert len([h for h in multi if h.chunk_id.startswith("s")]) >= 2


def test_iterative_is_a_transparent_wrapper():
    base = InMemoryStore(_docs())
    it = IterativeRetriever(base, max_rounds=1)  # 1 round == plain base retrieval
    a = {h.chunk_id for h in base.search("lipid cholesterol", k=3, method="bm25")}
    b = {h.chunk_id for h in it.search("lipid cholesterol", k=3, method="bm25")}
    assert a == b
    assert "iterative" in it.info().capabilities


def test_hop_router_routes_community():
    from context_runtime.adapters.store_router import HopRouterRetriever
    base = InMemoryStore(_docs())
    cr = CommunityRetriever(_docs(), min_shared=2)
    router = HopRouterRetriever(single_hop=base, graph=base, community=cr)
    hits = router.search("steroid hormone panel", k=2, method="community")
    assert hits and hits[0].chunk_id.startswith("community::")


def test_community_search_scales_via_query_conditioned_clustering(monkeypatch):
    monkeypatch.setenv("CR_COMMUNITY_MAX_NODES", "2")   # below corpus size → global build is skipped
    docs = [{"chunk_id": c, "filename": c + ".md", "text": t, "created_at": None} for c, t in [
        ("s1", "steroid hormone testosterone cortisol dhea androstenedione"),
        ("s2", "testosterone cortisol steroid hormone dhea reference ranges"),
        ("s3", "cortisol steroid hormone dhea testosterone endocrine panel"),
        ("l1", "lipid cholesterol ldl hdl triglycerides"),
    ]]
    r = CommunityRetriever(docs)
    assert r._build() == []                             # corpus > cap → no global communities
    hits = r.search("steroid hormone testosterone", k=2)
    assert hits and hits[0].chunk_id.startswith("community::")   # query-conditioned community still returned
    members = hits[0].meta["members"]
    assert any(m in ("s1", "s2", "s3") for m in members)         # the steroid cluster, not the lipid doc
