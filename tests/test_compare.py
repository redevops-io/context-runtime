"""LibreChatTenant.compare — the retrieval-transparency view behind the LibreChat panel:
run every method side by side + report what the learned policy chose/served, read-only."""
from __future__ import annotations

from context_runtime.adapters.store_community import CommunityRetriever
from context_runtime.adapters.store_hipporag import SimGraphRetriever
from context_runtime.adapters.store_inmemory import InMemoryStore
from context_runtime.adapters.store_router import HopRouterRetriever
from context_runtime.integrations.librechat import COMPARE_METHODS, LibreChatTenant


def _tenant():
    docs = [{"chunk_id": i, "filename": i + ".md", "text": t, "created_at": None} for i, t in [
        ("s1", "steroid hormone testosterone cortisol dhea androstenedione"),
        ("s2", "testosterone cortisol steroid hormone dhea reference ranges"),
        ("s3", "cortisol steroid hormone dhea testosterone endocrine"),
        ("l1", "lipid cholesterol ldl hdl triglycerides"),
    ]]
    router = HopRouterRetriever(single_hop=InMemoryStore(list(docs)),
                               graph=SimGraphRetriever(list(docs)),
                               community=CommunityRetriever(list(docs)))
    return LibreChatTenant(retriever=router)


def test_compare_runs_every_method():
    out = _tenant().compare("steroid hormone testosterone results", k=3)
    assert set(out["methods"].keys()) == set(COMPARE_METHODS)
    bm25 = out["methods"]["bm25"]
    assert bm25 and bm25[0]["chunk_id"] in ("s1", "s2", "s3")
    community = out["methods"]["community"]
    assert community and community[0]["chunk_id"].startswith("community::")
    # each hit carries the fields the panel renders (incl. full text for click-to-expand)
    assert set(bm25[0]) >= {"chunk_id", "filename", "score", "snippet", "text"}


def test_compare_reports_chosen_and_served():
    out = _tenant().compare("steroid hormone testosterone", k=3)
    chosen = out["chosen"]
    assert chosen["method"] in COMPARE_METHODS and chosen["key"]
    assert "bucket" in chosen and isinstance(chosen["learned"], bool)
    assert isinstance(out["served"]["citations"], list) and "context" in out["served"]


def test_compare_is_read_only():
    t = _tenant()
    before = dict(t.policy())
    t.compare("steroid hormone testosterone", k=3)
    assert dict(t.policy()) == before  # transparency view must not perturb learning
