"""HopRouterRetriever — dispatch single-hop vs multi-hop by retrieval method (SPEC §4.5).

The control plane's per-query decision made physical: ``method="graph"`` → the graph
retriever (HippoRAG / SimGraph); everything else → the single-hop retriever
(redevops-rag / in-memory). Itself a RetrieverPlugin, so the runtime holds ONE
retriever and the planner's method choice routes transparently underneath.
"""
from __future__ import annotations

from ..types import Hit, PluginInfo, Retrieval

_GRAPH_METHODS = {"graph"}
_COMMUNITY_METHODS = {"community"}


class HopRouterRetriever:
    def __init__(self, single_hop, graph, community=None):
        self.single_hop = single_hop      # RetrieverPlugin: bm25/vector/hybrid
        self.graph = graph                # RetrieverPlugin: graph/multi-hop
        self.community = community          # RetrieverPlugin: community/global (optional)

    def search(self, query: str, k: int, method: Retrieval = "hybrid") -> list[Hit]:
        if method in _COMMUNITY_METHODS and self.community is not None:
            return self.community.search(query, k, method)
        if method in _GRAPH_METHODS:
            return self.graph.search(query, k, method)
        return self.single_hop.search(query, k, method)

    def index(self, path: str) -> dict:
        a = self.single_hop.index(path) if hasattr(self.single_hop, "index") else {}
        b = self.graph.index(path) if hasattr(self.graph, "index") else {}
        out = {"single_hop": a, "graph": b}
        if self.community is not None and hasattr(self.community, "index"):
            out["community"] = self.community.index(path)
        return out

    def info(self) -> PluginInfo:
        caps = set()
        for r in (self.single_hop, self.graph, self.community):
            if r is not None and hasattr(r, "info"):
                caps |= set(r.info().capabilities)
        return PluginInfo(name="hop_router", kind="retriever", capabilities=frozenset(caps))
