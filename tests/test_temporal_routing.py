"""Temporal (bi-temporal) routing: intent → method, the router slot + fallback, cost edge,
the point-in-time / what-changed semantics, and the representation taxonomy (Whitepaper v4)."""
from __future__ import annotations

from context_runtime import ContextRuntime, Goal
from context_runtime.adapters.store_inmemory import InMemoryStore
from context_runtime.adapters.store_router import HopRouterRetriever
from context_runtime.adapters.store_temporal import TemporalStore
from context_runtime.planner import representations
from context_runtime.plugins import base

# A tiny bi-temporal history: the client's vitamin-D dose was revised on 2026-03-01.
DOC = [{"chunk_id": "d1", "filename": "d1", "text": "The client takes vitamin D daily.", "created_at": None}]


def _history() -> TemporalStore:
    return (TemporalStore()
            .add("client", "takes", "vitamin D 5000 IU",
                 valid_from="2026-01-01", valid_to="2026-03-01", recorded_at="2026-01-01")
            .add("client", "takes", "vitamin D 2000 IU",
                 valid_from="2026-03-01", recorded_at="2026-03-01"))   # current (valid_to=None)


TEMPORAL_QS = [
    "What was the client's dose as of February 2026?",
    "What changed in the client's supplements?",
    "Which supplement did the client previously take?",
    "Is the client still on the higher dose?",
]


def test_planner_routes_temporal_queries_to_temporal_method():
    rt = ContextRuntime.default([])
    for q in TEMPORAL_QS:
        p = rt.plan(q)
        assert p.intent.bucket == "temporal", f"{q!r} → {p.intent.bucket}"
        method = next(s.params.get("method") for s in p.chosen.steps if s.type == "retrieve")
        assert method == "temporal", f"{q!r} chose {method}"


def test_non_temporal_queries_do_not_route_to_temporal():
    rt = ContextRuntime.default([])
    for q in ("What is reciprocal rank fusion?", "Find ERR-500 in the logs"):
        p = rt.plan(q)
        method = next(s.params.get("method") for s in p.chosen.steps if s.type == "retrieve")
        assert method != "temporal", f"{q!r} should not use temporal"


def test_temporal_candidate_beats_hybrid_on_a_temporal_query():
    # inverse of the graph case: for a temporal question, temporal must score HIGHER than hybrid
    rt = ContextRuntime.default([])
    g = Goal(text="What changed in the client's supplements?")
    from context_runtime.types import Candidate, StepSpec
    temporal = Candidate(steps=(StepSpec("retrieve", {"method": "temporal"}),), model_tier="cheap")
    hybrid = Candidate(steps=(StepSpec("retrieve", {"method": "hybrid"}),), model_tier="cheap")
    assert rt.optimizer.score(temporal, g).total > rt.optimizer.score(hybrid, g).total


def test_router_dispatches_temporal_to_the_bitemporal_store():
    router = HopRouterRetriever(single_hop=InMemoryStore(list(DOC)),
                                graph=InMemoryStore(list(DOC)), temporal=_history())
    assert isinstance(router, base.RetrieverPlugin)
    hits = router.search("client vitamin dose", k=5, method="temporal")
    assert hits and all(h.source == "temporal" for h in hits)
    # "current" search returns only the not-yet-superseded fact
    assert any("2000 IU" in h.text for h in hits)
    assert not any("5000 IU" in h.text for h in hits)


def test_router_falls_back_when_temporal_unwired_or_empty():
    # (a) no temporal slot → temporal method falls through to single-hop, no crash
    r1 = HopRouterRetriever(single_hop=InMemoryStore(list(DOC)), graph=InMemoryStore(list(DOC)))
    h1 = r1.search("client vitamin", k=3, method="temporal")
    assert all(h.source != "temporal" for h in h1)
    # (b) wired but EMPTY store → also falls back (never returns nothing when documents exist)
    r2 = HopRouterRetriever(single_hop=InMemoryStore(list(DOC)),
                            graph=InMemoryStore(list(DOC)), temporal=TemporalStore())
    h2 = r2.search("client vitamin", k=3, method="temporal")
    assert h2 and all(h.source != "temporal" for h in h2)


def test_bitemporal_point_in_time_and_changes():
    hist = _history()
    now = hist.search("client vitamin", k=5)                       # current state
    assert any("2000 IU" in h.text for h in now)
    past = hist.as_of("client vitamin", at="2026-02-01")           # as it was in February
    assert any("5000 IU" in h.text for h in past)
    assert not any("2000 IU" in h.text for h in past)
    changes = hist.changes("vitamin", since="2026-01-01", until="2026-06-01")
    kinds = {(c["at"], c["change"]) for c in changes}
    assert ("2026-03-01", "began") in kinds and ("2026-03-01", "ended") in kinds


def test_representation_taxonomy():
    assert representations.representation_for("temporal") == "temporal"
    assert representations.representation_for("graph") == "graph"
    assert representations.representation_for("hybrid") == "document"
    assert representations.representation_for("sql") == "analytical"
    assert representations.representation_for("video") == "multimodal"
    assert representations.representation_for("unknown-method") == "document"   # safe default
    assert "hybrid" in representations.methods_for("document")


def test_end_to_end_temporal_run_uses_the_history():
    router = HopRouterRetriever(single_hop=InMemoryStore(list(DOC)),
                                graph=InMemoryStore(list(DOC)), temporal=_history())
    from context_runtime.adapters.model_stub import StubModel
    rt = ContextRuntime(models={t: StubModel(tier=t) for t in ("local", "cheap", "premium")},
                        retriever=router)
    res = rt.run("What did the client previously take, and what is it now?")
    # the planner routed to temporal; the assembled context carries the bi-temporal facts
    assert any("client:takes" in c for c in res.citations)
