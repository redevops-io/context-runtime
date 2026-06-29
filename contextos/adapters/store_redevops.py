"""RedevopsRagRetriever — the real hybrid-retrieval binding (SPEC §4.5).

Wraps ``redevops_rag.RAG`` (DuckDB dense + BM25 → RRF → recency/keyword priors →
optional bge-reranker). Lazy-imports so the core package runs without it.
``RAG.search`` returns dicts with chunk_id/filename/text/created_at/boosted_score,
mapped field-for-field into ``Hit`` (SPEC §4.5 binding note).

Install:  pip install "contextos[rag]"
"""
from __future__ import annotations

from ..types import Hit, PluginInfo, Retrieval


class RedevopsRagRetriever:
    def __init__(self, db_path: str = "./contextos_rag.duckdb", use_reranker: bool = False,
                 source: str = "docs"):
        self.db_path = db_path
        self.use_reranker = use_reranker
        self.source = source
        self._rag = None

    def _get(self):
        if self._rag is None:
            try:
                from redevops_rag import RAG  # type: ignore
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "RedevopsRagRetriever needs the 'rag' extra: pip install 'contextos[rag]'"
                ) from e
            self._rag = RAG(db_path=self.db_path, use_reranker=self.use_reranker)
        return self._rag

    def index(self, path: str) -> dict:
        return self._get().index(path)

    def search(self, query: str, k: int, method: Retrieval = "hybrid") -> list[Hit]:
        rows = self._get().search(query, k=k)
        out: list[Hit] = []
        for r in rows:
            score = r.get("rerank_score", r.get("boosted_score", r.get("rrf_score", 0.0)))
            out.append(Hit(
                chunk_id=r.get("chunk_id") or f"{r.get('filename')}::{r.get('chunk_index')}",
                filename=r.get("filename", "?"), text=r.get("text", ""),
                score=float(score), created_at=r.get("created_at"), source=self.source,
                meta={kk: r[kk] for kk in ("rrf_score", "boosted_score") if kk in r},
            ))
        return out

    def info(self) -> PluginInfo:
        caps = {"bm25", "vector", "hybrid"} | ({"rerank"} if self.use_reranker else set())
        return PluginInfo(name="redevops_rag", kind="store", capabilities=frozenset(caps))
