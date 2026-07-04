"""MultimodalRetriever — routes a retrieval method to the right modality store.

Multimodal is a *routable capability*, not a separate architecture: each method goes to the
store that owns that modality, and every text method (bm25/vector/hybrid/graph/community) goes to
the existing text retriever:

    method="image"    → cross-modal single-vector image search   (store_image)
    method="colpali"  → late-interaction visual-document search   (store_multivector)
    method="video"    → timestamped video-segment search          (store_video)
    everything else   → the text retriever

A mixed corpus dir indexes into all wired stores (each ignores files it doesn't handle). The
planner/bandit picks the method per query exactly as before — image/colpali/video are just more
arms — so the cost-based planner decides when the expensive visual/temporal methods are worth
paying for. Same shape as HopRouterRetriever, several modalities up.
"""
from __future__ import annotations

from ..types import Hit, PluginInfo, Retrieval

# which modality store each non-text method dispatches to (attribute name on this router)
_METHOD_STORE: dict[str, str] = {"image": "image", "colpali": "colpali", "video": "video"}


class MultimodalRetriever:
    def __init__(self, text, image=None, colpali=None, video=None):
        self.text = text        # RetrieverPlugin: bm25/vector/hybrid/graph/community
        self.image = image      # ImageRetriever: cross-modal single-vector image search
        self.colpali = colpali  # MultiVectorRetriever: late-interaction visual docs
        self.video = video      # VideoRetriever: timestamped segments

    def _store_for(self, method: str):
        store = getattr(self, _METHOD_STORE.get(method, ""), None)
        return store

    def search(self, query: str, k: int, method: Retrieval = "hybrid") -> list[Hit]:
        store = self._store_for(method)
        if store is not None:
            return store.search(query, k, method)
        return self.text.search(query, k, method)

    def index(self, path: str) -> dict:
        out = {"text": self.text.index(path) if hasattr(self.text, "index") else {}}
        for name in ("image", "colpali", "video"):
            store = getattr(self, name, None)
            if store is not None and hasattr(store, "index"):
                out[name] = store.index(path)   # same dir; each picks up its own file types
        return out

    def info(self) -> PluginInfo:
        return PluginInfo(name="multimodal_router", kind="retriever",
                          capabilities=frozenset({"search", "multimodal", "image", "colpali",
                                                  "video", "routing"}))
