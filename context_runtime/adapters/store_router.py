"""HopRouterRetriever — dispatch single-hop vs multi-hop by retrieval method (SPEC §4.5).

The control plane's per-query decision made physical, routing by knowledge representation:
``method="graph"`` → the graph retriever (HippoRAG / SimGraph); ``method="temporal"`` → the
bi-temporal store (Graphiti / in-memory TemporalStore); ``method="community"`` → the community
retriever; everything else → the single-hop retriever (redevops-rag / in-memory). Itself a
RetrieverPlugin, so the runtime holds ONE retriever and the planner's method choice routes
transparently underneath.

The graph/community/temporal slots are OPTIONAL: an unwired slot (or, for temporal, a store
with no matching fact) falls back to single-hop retrieval, so adding a representation never
regresses default behavior — it only *adds* a route the planner can take once a backend is bound.
"""
from __future__ import annotations

from ..types import Hit, PluginInfo, Retrieval

_GRAPH_METHODS = {"graph"}
_COMMUNITY_METHODS = {"community"}
_TEMPORAL_METHODS = {"temporal"}


class HopRouterRetriever:
    def __init__(self, single_hop, graph, community=None, temporal=None):
        self.single_hop = single_hop      # RetrieverPlugin: bm25/vector/hybrid (document)
        self.graph = graph                # RetrieverPlugin: graph/multi-hop
        self.community = community          # RetrieverPlugin: community/global (optional)
        self.temporal = temporal            # RetrieverPlugin: bi-temporal facts (optional)

    def search(self, query: str, k: int, method: Retrieval = "hybrid") -> list[Hit]:
        if method in _COMMUNITY_METHODS and self.community is not None:
            return self.community.search(query, k, method)
        if method in _GRAPH_METHODS:
            return self.graph.search(query, k, method)
        if method in _TEMPORAL_METHODS and self.temporal is not None:
            hits = self.temporal.search(query, k, method)
            if hits:                        # a populated bi-temporal store answered
                return hits
            # unpopulated store / no temporal fact matched → fall back to single-hop
        return self.single_hop.search(query, k, method)

    def index(self, path: str) -> dict:
        a = self.single_hop.index(path) if hasattr(self.single_hop, "index") else {}
        b = self.graph.index(path) if hasattr(self.graph, "index") else {}
        out = {"single_hop": a, "graph": b}
        for name, r in (("community", self.community), ("temporal", self.temporal)):
            if r is not None and hasattr(r, "index"):
                out[name] = r.index(path)
        return out

    def info(self) -> PluginInfo:
        caps = set()
        for r in (self.single_hop, self.graph, self.community, self.temporal):
            if r is not None and hasattr(r, "info"):
                caps |= set(r.info().capabilities)
        return PluginInfo(name="hop_router", kind="retriever", capabilities=frozenset(caps))
