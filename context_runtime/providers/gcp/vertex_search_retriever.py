"""VertexSearchRetriever — a document ``RetrieverPlugin`` over Vertex AI Search (Discovery Engine).

Registers Vertex AI Search as a document-representation retriever the KR router and bandit can select
alongside the local engines - so Context Runtime learns when it beats the in-tree engines for a query
class. The Discovery Engine client is injectable (duck-typed ``.search(request) -> iterable`` of
results with a ``.document``); the real path builds a SigV4-free google client lazily. Text is pulled
from the document's derived struct data (snippets / extractive answers), defensively.
"""
from __future__ import annotations

from ...types import Hit, PluginInfo, Retrieval


def _as_dict(v):
    """Coerce a proto Struct / MapComposite (or a dict) to a plain dict."""
    if isinstance(v, dict):
        return v
    try:
        return dict(v)
    except Exception:  # noqa: BLE001
        return {}


def _text_of(derived: dict) -> str:
    for key in ("extractive_answers", "snippets"):
        items = derived.get(key) or []
        parts = [(_as_dict(it).get("content") or _as_dict(it).get("snippet") or "") for it in items]
        joined = " ".join(p for p in parts if p).strip()
        if joined:
            return joined
    return str(derived.get("content") or derived.get("title") or "")


class VertexSearchRetriever:
    def __init__(self, session=None, *, engine_id: str | None = None, data_store_id: str | None = None,
                 client=None, collection: str = "default_collection", serving_config: str = "default_search"):
        self._session = session
        self.engine_id = engine_id
        self.data_store_id = data_store_id
        self.collection = collection
        self.serving_config = serving_config
        self._client = client

    def _de(self):
        if self._client is None:
            self._client = self._session.discoveryengine_client()
        return self._client

    def _serving_config_path(self) -> str:
        p, loc = self._session.project, self._session.location
        base = f"projects/{p}/locations/{loc}/collections/{self.collection}"
        if self.engine_id:
            return f"{base}/engines/{self.engine_id}/servingConfigs/{self.serving_config}"
        return f"{base}/dataStores/{self.data_store_id}/servingConfigs/{self.serving_config}"

    def search(self, query: str, k: int, method: Retrieval = "hybrid") -> list[Hit]:
        # a dict request is coerced to the SearchRequest proto by the google client (test fakes get a dict)
        resp = self._de().search(request={"serving_config": self._serving_config_path(),
                                           "query": query, "page_size": k})
        out: list[Hit] = []
        for i, r in enumerate(resp):
            doc = getattr(r, "document", None)
            if doc is None:
                continue
            derived = _as_dict(getattr(doc, "derived_struct_data", {}) or {})
            out.append(Hit(
                chunk_id=str(getattr(doc, "id", f"vs:{i}")),
                filename=str(derived.get("link") or derived.get("title") or getattr(doc, "id", "vertex-search")),
                text=_text_of(derived),
                score=float(getattr(r, "relevance_score", 0.0) or 0.0),
                source="vertex_search",
                meta={"derived": derived},
            ))
        return out

    def index(self, path: str) -> dict:
        return {"vertex_search": "index managed in Google Cloud", "engine": self.engine_id or self.data_store_id}

    def info(self) -> PluginInfo:
        return PluginInfo(name="vertex_search", kind="retriever",
                          capabilities=frozenset({"vector", "bm25", "hybrid"}))
