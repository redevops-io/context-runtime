"""The REAL retrieval stack — redevops-rag v2 (`hybrid_search`: DuckDB semantic + BM25 →
RRF → boosts → optional cross-encoder rerank), with the `document_ids` scoping we added
for the pollution axis.

We embed every FinanceBench corpus passage ONCE into a DuckDB store (document_id =
filing, so retrieval can be scoped to a subset of filings). All three arms then retrieve
from this store with different RetrievalConfigs — the difference between "Context Runtime
on vs off" is which config the planner's bandit picks, not a different retriever.
"""
from __future__ import annotations

import inspect
import os
from dataclasses import dataclass

from redevops_rag.rag import RAG
from redevops_rag.retrieve import hybrid_search

# Native document_ids scoping landed in redevops-rag (pinned in pyproject). Feature-detect so the
# harness still runs against an older/unpinned install, falling back to post-hoc filtering there.
_HYBRID_HAS_DOC_IDS = "document_ids" in inspect.signature(hybrid_search).parameters


@dataclass
class Chunk:
    doc_id: str
    page: int
    text: str
    score: float = 0.0


def _hit_to_chunk(h: dict) -> Chunk:
    return Chunk(doc_id=h.get("document_id") or "",
                 page=int(h.get("chunk_index") or 0), text=h.get("text") or "",
                 score=float(h.get("score") or h.get("similarity") or h.get("bm25_score") or 0.0))


class FinanceBenchStore:
    """Facade over a redevops-rag RAG whose DuckDB store holds the embedded corpus."""

    # NOTE: cross-encoder rerank is OFF by default — the shipped FlagEmbedding reranker is
    # incompatible with transformers>=5.13 (XLMRobertaTokenizer.prepare_for_model removed).
    # hybrid_search still runs semantic + BM25 + RRF + score boosts; CR keeps its
    # pool/limit/threshold levers. Re-enable once a compatible reranker is pinned.
    def __init__(self, db_path: str, use_reranker: bool = False, embed_model: str | None = None):
        self.rag = RAG(db_path=db_path, embed_model=embed_model, use_reranker=use_reranker)

    @property
    def count(self) -> int:
        return self.rag.store.count()

    def search(self, query: str, cfg, document_ids: list | None = None) -> list:
        """Real hybrid_search with the bandit-chosen RetrievalConfig, scoped to
        ``document_ids``. cfg.rerank decides whether the cross-encoder runs."""
        rr = self.rag.reranker if getattr(cfg, "rerank", False) else None
        kw = cfg.kwargs()
        if document_ids and _HYBRID_HAS_DOC_IDS:
            # Native pre-filtering: scope BEFORE RRF fusion + boosts, so the candidate pool is drawn
            # entirely from the allowed docs — the correct graduated-pollution pool.
            hits = hybrid_search(self.rag.store, query, reranker=rr,
                                 document_ids=list(document_ids), **kw)
        elif document_ids:
            # Fallback for an older/unpinned redevops-rag without document_ids: scope POST-HOC (widen
            # → retrieve → filter → top-k). Runs RRF/boosts over the GLOBAL pool then filters, so it
            # biases toward globally-high-ranking docs — kept only so the harness still runs on a
            # stale install. Pin redevops-rag to the document_ids commit to take the native path.
            allowed = set(document_ids)
            want = int(kw.get("limit", 8) or 8)
            wide = dict(kw, limit=max(want * 8, 60), pool=max(int(kw.get("pool", 100) or 100), 400))
            hits = hybrid_search(self.rag.store, query, reranker=rr, **wide)
            hits = [h for h in hits if (h.get("document_id") or "") in allowed][:want]
        else:
            hits = hybrid_search(self.rag.store, query, reranker=rr, **kw)
        return [_hit_to_chunk(h) for h in hits]

    def close(self):
        self.rag.close()


def build_store(corpus, db_path: str, *, use_reranker: bool = True, batch: int = 512,
                overwrite: bool = True, progress=None) -> FinanceBenchStore:
    """Embed every corpus passage into a fresh DuckDB store (document_id = filing)."""
    if overwrite and os.path.exists(db_path):
        os.remove(db_path)
    fb = FinanceBenchStore(db_path, use_reranker=use_reranker)
    store, emb = fb.rag.store, fb.rag.embedder
    passages = corpus.passages
    for i in range(0, len(passages), batch):
        part = passages[i:i + batch]
        vecs = emb.encode([p.text for p in part])
        store.add_chunks([
            {"document_id": p.doc_id, "filename": p.id, "chunk_index": p.page,
             "text": p.text, "embedding": v, "metadata": {"page": p.page}}
            for p, v in zip(part, vecs)
        ])
        if progress:
            progress(min(i + batch, len(passages)), len(passages))
    store.reindex_fts()
    return fb
