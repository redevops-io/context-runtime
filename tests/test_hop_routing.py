"""Single-hop vs multi-hop routing: intent → method, the router, and SimGraph multi-hop."""
from __future__ import annotations

from context_runtime import ContextRuntime, Goal
from context_runtime.adapters.store_hipporag import HippoRAGRetriever, SimGraphRetriever
from context_runtime.adapters.store_inmemory import InMemoryStore
from context_runtime.adapters.store_router import HopRouterRetriever
from context_runtime.plugins import base

CORPUS = [
    {"chunk_id": "d1", "filename": "d1", "text": "Mitochondrial dysfunction impairs ATP production in neurons.", "created_at": None},
    {"chunk_id": "d2", "filename": "d2", "text": "Reduced ATP production triggers alpha-synuclein aggregation.", "created_at": None},
    {"chunk_id": "d3", "filename": "d3", "text": "Alpha-synuclein aggregation is the hallmark of Parkinson disease.", "created_at": None},
]
MULTIHOP = "How is mitochondrial dysfunction linked to Parkinson disease?"


def test_planner_routes_multihop_to_graph():
    rt = ContextRuntime.default([])
    p = rt.plan(MULTIHOP)
    assert p.intent.bucket == "multi_hop"
    method = next(s.params.get("method") for s in p.chosen.steps if s.type == "retrieve")
    assert method == "graph"


def test_planner_keeps_single_hop_on_hybrid():
    rt = ContextRuntime.default([])
    for q in ("Find ERR-500 in the logs", "What is reciprocal rank fusion?"):
        p = rt.plan(q)
        method = next(s.params.get("method") for s in p.chosen.steps if s.type == "retrieve")
        assert method != "graph", f"{q} should not use graph"


def test_graph_costs_more_so_single_hop_queries_avoid_it():
    # for a non-multi-hop query, a graph candidate must score LOWER than hybrid
    rt = ContextRuntime.default([])
    g = Goal(text="What is reciprocal rank fusion?")
    from context_runtime.types import Candidate, StepSpec
    hybrid = Candidate(steps=(StepSpec("retrieve", {"method": "hybrid"}),), model_tier="cheap")
    graph = Candidate(steps=(StepSpec("retrieve", {"method": "graph"}),), model_tier="cheap")
    assert rt.optimizer.score(hybrid, g).total > rt.optimizer.score(graph, g).total


def test_simgraph_surfaces_the_bridge_document():
    # the multi-hop signal: d2 shares NO query term but bridges d1↔d3
    q = MULTIHOP
    single = InMemoryStore(list(CORPUS)).search(q, k=3, method="hybrid")
    graph = SimGraphRetriever(list(CORPUS)).search(q, k=3, method="graph")
    assert "d2" not in {h.filename for h in single}        # single-hop misses the bridge
    assert "d2" in {h.filename for h in graph}             # multi-hop finds it
    assert any(h.meta.get("hop", "").startswith("bridge") for h in graph)


def test_hop_router_dispatches_by_method():
    router = HopRouterRetriever(single_hop=InMemoryStore(list(CORPUS)),
                               graph=SimGraphRetriever(list(CORPUS)))
    assert isinstance(router, base.RetrieverPlugin)
    g = router.search(MULTIHOP, k=3, method="graph")
    h = router.search("hallmark of Parkinson", k=3, method="hybrid")
    assert all(hit.source == "graph" for hit in g)
    assert all(hit.source != "graph" for hit in h)


def test_hipporag_binding_is_lazy():
    # constructing must not require the heavy dep; it imports on first search/index
    r = HippoRAGRetriever()
    assert isinstance(r, base.RetrieverPlugin)
    assert r.info().capabilities == frozenset({"graph"})


def test_end_to_end_multihop_run_uses_bridge():
    router = HopRouterRetriever(single_hop=InMemoryStore(list(CORPUS)),
                               graph=SimGraphRetriever(list(CORPUS)))
    from context_runtime.adapters.model_stub import StubModel
    rt = ContextRuntime(models={t: StubModel(tier=t) for t in ("local", "cheap", "premium")},
                        retriever=router)
    res = rt.run(MULTIHOP)
    assert "d2" in res.citations    # the bridge doc made it into the assembled context


def test_simgraph_ranks_overlap_and_empty_on_nomatch():
    g = SimGraphRetriever(list(CORPUS))
    hits = g.search("ATP production in neurons", k=5)
    assert hits and hits[0].chunk_id in ("d1", "d2")   # term-overlap docs surface first
    assert g.search("zzzzz qqqqq wibble", k=5) == []   # no term overlap → empty, no bridge-only noise
