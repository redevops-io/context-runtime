"""Semantic (embedding) retrieval + hybrid BM25⊕semantic fusion — the deterministic
"smart" layer that bridges synonyms lexical BM25 cannot (e.g. «андрогены»↔«тестостерон»,
«жиры крови»↔«липидный профиль»). Embeddings are a fixed ONNX model (argmax cosine), so
retrieval stays deterministic — no generative LLM in the path.

Backend: fastembed (ONNX runtime, multilingual, no torch). Optional — everything degrades
to pure BM25 when fastembed / the model is unavailable, so the base install is unaffected.
Enable with the [embeddings] extra; pick the model via CR_EMBED_MODEL.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..types import Hit
from .store_inmemory import InMemoryStore, _token_list  # noqa: F401  (reuse tokenizer parity)

_MODEL_NAME = os.getenv("CR_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
_EMBED_EXTS = {".md", ".txt", ".rst"}
_embedder = None
_embedder_tried = False


def _nemotron_selected() -> bool:
    """Nemotron-3-Embed-8B chosen as the vector encoder — via CR_NEMOTRON=1 or
    REDEVOPS_RAG_EMBED_BACKEND=nemotron. Opt-in; the default stays the cheap ONNX model."""
    if os.getenv("CR_NEMOTRON", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    return os.getenv("REDEVOPS_RAG_EMBED_BACKEND", "").strip().lower() in ("nemotron", "nemo", "nemotron-embed")


class _NemotronONNXShim:
    """Adapts redevops-rag's :class:`NemotronEmbedder` (``.encode`` over an HTTP /v1/embeddings
    endpoint) to the fastembed ``.embed(texts) -> iterable[vector]`` interface this module expects, so
    the vector/hybrid arm can run on Nemotron with no other change. Queries get Nemotron's instruction
    prefix; documents are embedded raw (``_embed`` is called per-chunk at index time and per-query at
    search time — both symmetric here, matching how the ONNX path treats them)."""

    def __init__(self, inner):
        self._inner = inner

    def embed(self, texts):
        return self._inner.encode(list(texts))


def _get_embedder():
    """Lazily load the embedder once; None if unavailable (→ pure BM25). The Nemotron HTTP arm is
    picked when flagged, else the cheap fastembed ONNX model."""
    global _embedder, _embedder_tried
    if not _embedder_tried:
        _embedder_tried = True
        if _nemotron_selected():
            try:
                from redevops_rag.embed import NemotronEmbedder
                _embedder = _NemotronONNXShim(NemotronEmbedder())
            except Exception:
                _embedder = None
            return _embedder
        try:
            from fastembed import TextEmbedding
            _embedder = TextEmbedding(_MODEL_NAME)
        except Exception:
            _embedder = None
    return _embedder


def embeddings_available() -> bool:
    return _get_embedder() is not None


def _embed(texts: list[str]):
    import numpy as np
    model = _get_embedder()
    vecs = [np.asarray(v, dtype="float32") for v in model.embed(texts)]
    out = []
    for v in vecs:
        n = float((v @ v) ** 0.5) or 1.0
        out.append(v / n)  # L2-normalize so dot product == cosine
    return out


class SemanticRetriever:
    """Embeds every chunk once (cached) and ranks by cosine similarity to the query
    embedding — finds documents that MEAN the query even with no shared terms."""

    def __init__(self, docs: list[dict] | None = None, source: str = "semantic"):
        self.source = source
        self.docs: list[dict] = list(docs or [])
        self._emb = None      # numpy matrix [n_docs, dim]
        self._emb_n = -1

    @property
    def available(self) -> bool:
        return embeddings_available()

    def index(self, path: str) -> dict:
        p = Path(path).expanduser()
        n = 0
        for fp in sorted(p.rglob("*")) if p.is_dir() else [p]:
            if fp.is_file() and fp.suffix.lower() in _EMBED_EXTS:
                self.docs.append({"chunk_id": f"{fp.name}::0", "filename": fp.name,
                                  "text": fp.read_text(errors="ignore"), "created_at": None})
                n += 1
        self._emb = None  # invalidate
        return {"files": n, "chunks": n}

    def _matrix(self):
        import numpy as np
        if self._emb is not None and self._emb_n == len(self.docs):
            return self._emb
        if not self.docs or not self.available:
            self._emb, self._emb_n = None, len(self.docs)
            return None
        vecs = _embed([d["text"][:1200] for d in self.docs])  # cap per-chunk cost
        self._emb = np.vstack(vecs) if vecs else None
        self._emb_n = len(self.docs)
        return self._emb

    def search(self, query: str, k: int, method: str = "vector") -> list[Hit]:
        mat = self._matrix()
        if mat is None:
            return []
        import numpy as np
        qv = _embed([query])[0]
        sims = mat @ qv  # cosine (both normalized)
        order = np.argsort(-sims)[: max(k, 0) or len(sims)]
        out: list[Hit] = []
        for i in order:
            d = self.docs[int(i)]
            out.append(Hit(chunk_id=d["chunk_id"], filename=d["filename"], text=d["text"],
                           score=float(sims[int(i)]), created_at=d.get("created_at"), source=self.source))
        return out

    def info(self):
        from ..types import PluginInfo
        return PluginInfo(name="semantic_store", kind="retriever",
                          capabilities=frozenset({"search", "semantic", "embeddings"}))


def _rrf_fuse(*ranked_lists: list[Hit], k: int, c: int = 60) -> list[Hit]:
    """Reciprocal-rank fusion — deterministic combine of BM25 and semantic rankings."""
    scores: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for hits in ranked_lists:
        for rank, h in enumerate(hits):
            key = h.filename + "\x00" + h.chunk_id
            scores[key] = scores.get(key, 0.0) + 1.0 / (c + rank + 1)
            best.setdefault(key, h)
    fused = sorted(best.values(), key=lambda h: (-scores[h.filename + "\x00" + h.chunk_id], h.filename, h.chunk_id))
    return fused[:k] if k > 0 else fused


class HybridRetriever:
    """BM25 ⊕ semantic, fused by reciprocal-rank: BM25's precision on exact terms plus the
    embedding model's synonym bridging. Falls back to pure BM25 when embeddings are absent.
    Implements the RetrieverPlugin surface (search/index/info) so it drops into the runtime."""

    def __init__(self, docs: list[dict] | None = None):
        self.lexical = InMemoryStore(list(docs or []))
        self.semantic = SemanticRetriever(list(docs or []))

    def index(self, path: str) -> dict:
        report = self.lexical.index(path)
        self.semantic.index(path)
        return report

    @property
    def docs(self):
        return self.lexical.docs

    def search(self, query: str, k: int, method: str = "hybrid") -> list[Hit]:
        if method == "bm25" or not self.semantic.available:
            return self.lexical.search(query, k, "bm25")
        if method == "vector":
            return self.semantic.search(query, k)
        pool = max(k * 3, 30)  # graph is handled upstream by the HopRouter
        return _rrf_fuse(self.lexical.search(query, pool, "bm25"),
                         self.semantic.search(query, pool), k=k)

    def info(self):
        from ..types import PluginInfo
        return PluginInfo(name="hybrid_store", kind="retriever",
                          capabilities=frozenset({"search", "index", "bm25", "semantic", "rrf"}))
