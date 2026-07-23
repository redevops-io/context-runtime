"""GradientKBRetriever — a ``RetrieverPlugin`` over a DigitalOcean Gradient knowledge base.

DO exposes a standalone hybrid ``/retrieve`` on the knowledge base (kbaas.do-ai.run), so retrieval is a
first-class RetrieverPlugin the KR router and bandit can select - no need to route through an agent.
The response's chunk list is parsed defensively (field names vary), and the HTTP transport is injected
via the DoSession for tests.
"""
from __future__ import annotations

from ...types import Hit, PluginInfo, Retrieval


def _first(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


class GradientKBRetriever:
    def __init__(self, session, *, knowledge_base_id: str):
        self.session = session
        self.knowledge_base_id = knowledge_base_id

    def search(self, query: str, k: int, method: Retrieval = "hybrid") -> list[Hit]:
        url = f"{self.session.kb_base}/{self.knowledge_base_id}/retrieve"
        headers = {"Authorization": f"Bearer {self.session.api_token or ''}"}
        data = self.session.post(url, {"query": query, "k": k}, headers)
        results = _first(data, "results", "chunks", "data", default=[]) or []
        out: list[Hit] = []
        for i, r in enumerate(results):
            if not isinstance(r, dict):
                continue
            text = _first(r, "content", "text", "chunk", default="")
            meta = _first(r, "metadata", "meta", default={}) or {}
            out.append(Hit(
                chunk_id=str(_first(r, "id", "chunk_id", default=f"kb:{i}")),
                filename=str(_first(meta, "source", "filename", "document", default=self.knowledge_base_id)),
                text=str(text),
                score=float(_first(r, "score", "relevance", default=0.0) or 0.0),
                source="gradient_kb",
                meta=meta if isinstance(meta, dict) else {"meta": meta},
            ))
        return out

    def index(self, path: str) -> dict:
        return {"gradient_kb": "indexing managed on DigitalOcean", "knowledge_base_id": self.knowledge_base_id}

    def info(self) -> PluginInfo:
        return PluginInfo(name="gradient_kb", kind="retriever", capabilities=frozenset({"vector", "bm25", "hybrid"}))
