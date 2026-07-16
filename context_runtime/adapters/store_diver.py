"""DIVER temporal-reasoning retriever — redevops-rag's ``diver_search`` exposed as a Context
Runtime temporal-slot RetrieverPlugin (``method="temporal"``).

Faithful to the merged redevops-rag DIVER (the ablation default: DIVER over a cheap **bge**
encoder, 0.448 NDCG@10 vs 0.337 for the DIVER+ReasonIR combo). The pipeline is
expand → hybrid-retrieve each sub-query → dedup union → (optional cross-encoder) → LLM
listwise rerank, driven by a ``reason_llm(system, user) -> str`` (the deployment's reasoning
upstream, e.g. KIMI). With ``reason_llm`` None it degrades to plain hybrid, so it is a safe
cold-start arm.

Opt-in only (``CR_DIVER=1``): it pulls redevops-rag's ``sentence-transformers`` + a bge model
and keeps its OWN bge index of the corpus, so it is heavier than the fastembed-native arms and
is wired only behind the flag. Same ``RetrieverPlugin`` seam (index / search / info) as
``TemporalDocumentRetriever``, so ``HopRouterRetriever`` can't tell them apart.
"""
from __future__ import annotations

from typing import Any, Callable

from ..types import Hit, PluginInfo, Retrieval

# redevops-rag ranks with different score keys depending on the stage that produced a hit
# (rerank vs fused vs raw); take the first present so the panel gets an honest number.
_SCORE_KEYS = ("score", "rerank_score", "diver_score", "rrf", "similarity", "bm25")


def _score(d: dict) -> float:
    for k in _SCORE_KEYS:
        v = d.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


class DiverTemporalRetriever:
    """redevops-rag DIVER as the Context Runtime temporal arm.

    ``reason_llm(system, user) -> str`` drives DIVER's query expansion + listwise rerank; pass
    the deployment's reasoning upstream. Keep the store's embedder **bge** (the default) — do
    NOT use ReasonIREmbedder, which is the losing combined arm (0.337 vs 0.448 in the ablation).
    """

    name = "diver_temporal"

    def __init__(self, reason_llm: Callable[[str, str], str] | None = None, *,
                 embed_model: str | None = None, pool: int = 25, n_subqueries: int = 3,
                 db_path: str = ":memory:") -> None:
        # Lazy imports: redevops-rag (sentence-transformers/torch) is an opt-in [rag] extra,
        # so importing this module must not require it — only constructing the arm does.
        from redevops_rag.embed import Embedder
        from redevops_rag.store import Store
        from redevops_rag.temporal import TemporalReasoningRetriever

        self._embedder = Embedder(embed_model)          # bge-small-en by default
        self._store = Store(self._embedder, db_path)
        self._plug = TemporalReasoningRetriever(reason_llm, pool=pool, n_subqueries=n_subqueries)

    # ── ingest ── (HopRouterRetriever.index() fans the corpus dir into this slot)
    def index(self, path: str) -> dict[str, Any]:
        from redevops_rag.ingest import ingest as _ingest

        # A persisted DuckDB store (db_path is a file) survives restarts: skip the expensive bge
        # embed when it is already populated, and only rebuild the (cheap) in-process FTS index.
        # This is what keeps a warm, pre-built index from re-embedding a large corpus every boot.
        if self._store.count() > 0:
            self._store.reindex_fts()
            return {"reused_chunks": self._store.count()}
        res = _ingest(self._store, self._embedder, path)
        self._store.reindex_fts()
        n = self._store.count()
        return res if isinstance(res, dict) else {"indexed_chunks": n}

    def _hit(self, d: dict) -> Hit:
        return Hit(
            chunk_id=str(d.get("chunk_id") or d.get("id") or ""),
            filename=d.get("filename") or "temporal",
            text=d.get("text") or "",
            score=round(_score(d), 4),
            meta={"source": "diver", "arm": "temporal", **(d.get("metadata") or {})},
        )

    def search(self, query: str, k: int = 5, method: Retrieval = "temporal") -> list[Hit]:
        # Empty store ⇒ return nothing so HopRouterRetriever falls back to single-hop.
        if self._store.count() == 0:
            return []
        rows = self._plug.search(self._store, query, limit=k)
        return [self._hit(d) for d in rows]

    def info(self) -> PluginInfo:
        return PluginInfo(name="diver_temporal", kind="retriever",
                          capabilities=frozenset({"temporal"}))
