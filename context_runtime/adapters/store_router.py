"""HopRouterRetriever — dispatch by knowledge representation (SPEC §4.5).

The control plane's per-query decision made physical, routing by knowledge representation:
``method="graph"`` → the graph retriever (HippoRAG / SimGraph); ``method="temporal"`` → the
bi-temporal store; ``method="community"`` → the community retriever; ``analytical`` methods
(``sql``/``mongo``/``elastic``/``logs``/``api``) → the analytical retriever; ``multimodal`` methods
(``image``/``colpali``/``video``) → the multimodal retriever; document methods → the single-hop
retriever. Itself a RetrieverPlugin, so the runtime holds ONE retriever and the planner's method
choice routes transparently underneath.

**Graceful vs. dangerous fallback.** The graph/community/temporal slots fall back to single-hop when
unwired — safe, because they're still *document* retrieval over the same corpus. The analytical and
multimodal slots do NOT: silently answering "how many incidents last quarter" with a BM25 keyword
match is worse than an honest failure. So an unbound analytical/multimodal route **fails loudly**
(``UnboundRepresentationError``) by default, or explicitly abstains (empty result) when
``on_unbound="abstain"`` — it never substitutes lexical hits for an analytical answer.
"""
from __future__ import annotations

from ..types import Hit, PluginInfo, Retrieval

_GRAPH_METHODS = {"graph"}
_COMMUNITY_METHODS = {"community"}
_TEMPORAL_METHODS = {"temporal"}
# these two must never silently degrade to lexical retrieval:
_ANALYTICAL_METHODS = {"sql", "mongo", "elastic", "logs", "api"}
_MULTIMODAL_METHODS = {"image", "colpali", "video"}


class UnboundRepresentationError(RuntimeError):
    """Raised when the planner routes to a representation whose backend isn't wired and for which a
    lexical fallback would be a silent, wrong substitution (analytical / multimodal)."""


class HopRouterRetriever:
    def __init__(self, single_hop, graph, community=None, temporal=None,
                 analytical=None, multimodal=None, on_unbound: str = "raise"):
        self.single_hop = single_hop      # RetrieverPlugin: bm25/vector/hybrid (document)
        self.graph = graph                # RetrieverPlugin: graph/multi-hop
        self.community = community          # RetrieverPlugin: community/global (optional)
        self.temporal = temporal            # RetrieverPlugin: bi-temporal facts (optional)
        self.analytical = analytical        # RetrieverPlugin: text-to-SQL / structured (optional)
        self.multimodal = multimodal        # RetrieverPlugin: image/video/colpali (optional)
        if on_unbound not in ("raise", "abstain"):
            raise ValueError("on_unbound must be 'raise' or 'abstain'")
        self.on_unbound = on_unbound

    def _unbound(self, method: str) -> list[Hit]:
        """No backend for a non-lexical representation. Fail loudly, or abstain — never fall to BM25."""
        if self.on_unbound == "abstain":
            return []
        raise UnboundRepresentationError(
            f"retrieval method '{method}' needs an analytical/multimodal backend that isn't wired; "
            f"refusing to substitute lexical (BM25) results. Bind one on the HopRouterRetriever "
            f"(analytical=/multimodal=), or construct it with on_unbound='abstain' to return no hits."
        )

    def search(self, query: str, k: int, method: Retrieval = "hybrid") -> list[Hit]:
        if method in _COMMUNITY_METHODS and self.community is not None:
            return self.community.search(query, k, method)
        if method in _GRAPH_METHODS:
            return self.graph.search(query, k, method)
        if method in _TEMPORAL_METHODS and self.temporal is not None:
            hits = self.temporal.search(query, k, method)
            if hits:                        # a populated bi-temporal store answered
                return hits
            # unpopulated store / no temporal fact matched → fall back to single-hop (still document)
        if method in _ANALYTICAL_METHODS:
            if self.analytical is None:
                return self._unbound(method)
            return self.analytical.search(query, k, method)
        if method in _MULTIMODAL_METHODS:
            if self.multimodal is None:
                return self._unbound(method)
            return self.multimodal.search(query, k, method)
        return self.single_hop.search(query, k, method)

    def index(self, path: str) -> dict:
        a = self.single_hop.index(path) if hasattr(self.single_hop, "index") else {}
        b = self.graph.index(path) if hasattr(self.graph, "index") else {}
        out = {"single_hop": a, "graph": b}
        for name, r in (("community", self.community), ("temporal", self.temporal),
                        ("analytical", self.analytical), ("multimodal", self.multimodal)):
            if r is not None and hasattr(r, "index"):
                out[name] = r.index(path)
        return out

    def info(self) -> PluginInfo:
        caps = set()
        for r in (self.single_hop, self.graph, self.community, self.temporal,
                  self.analytical, self.multimodal):
            if r is not None and hasattr(r, "info"):
                caps |= set(r.info().capabilities)
        return PluginInfo(name="hop_router", kind="retriever", capabilities=frozenset(caps))
