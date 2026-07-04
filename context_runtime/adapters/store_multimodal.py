"""MultimodalRetriever — routes a retrieval method to the right modality store.

Multimodal is a *routable capability*, not a separate architecture: ``method="image"`` goes
to the cross-modal image store; every text method (bm25/vector/hybrid/graph/community) goes to
the existing text retriever. A mixed corpus dir indexes into both (each store ignores files it
doesn't handle). The planner/bandit picks the method per query exactly as before — image is
just another arm — so the cost-based planner decides when visual retrieval is worth paying for.

Same shape as HopRouterRetriever, one modality up.
"""
from __future__ import annotations

from ..types import Hit, PluginInfo, Retrieval

_IMAGE_METHODS = {"image"}


class MultimodalRetriever:
    def __init__(self, text, image):
        self.text = text     # RetrieverPlugin: bm25/vector/hybrid/graph/community
        self.image = image   # ImageRetriever: cross-modal image search

    def search(self, query: str, k: int, method: Retrieval = "hybrid") -> list[Hit]:
        if method in _IMAGE_METHODS and self.image is not None:
            return self.image.search(query, k, method)
        return self.text.search(query, k, method)

    def index(self, path: str) -> dict:
        out = {"text": self.text.index(path) if hasattr(self.text, "index") else {}}
        if self.image is not None and hasattr(self.image, "index"):
            out["image"] = self.image.index(path)   # same dir; picks up the image files
        return out

    def info(self) -> PluginInfo:
        return PluginInfo(name="multimodal_router", kind="retriever",
                          capabilities=frozenset({"search", "multimodal", "image", "routing"}))
