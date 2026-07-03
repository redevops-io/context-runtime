"""InMemoryStore — offline Retriever+Store for the default/test path (SPEC §4.5).

A tiny keyword-overlap retriever over an in-memory corpus. Implements the SAME
``RetrieverPlugin``/``StorePlugin`` contracts as the redevops-rag binding, which is
the whole point of plugin-first: the runtime can't tell them apart.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..types import Hit, PluginInfo, Retrieval

_WORD = re.compile(r"\w+")


def _tokens(s: str) -> set[str]:
    return {t for t in _WORD.findall(s.lower()) if len(t) > 2}


def _token_list(s: str) -> list[str]:
    """Ordered tokens (with repeats) — BM25 needs term frequencies, not just presence."""
    return [t for t in _WORD.findall(s.lower()) if len(t) > 2]


class InMemoryStore:
    def __init__(self, docs: list[dict] | None = None, source: str = "memory"):
        # each doc: {"chunk_id","filename","text","created_at"?}
        self.docs = docs or []
        self.source = source

    def index(self, path: str) -> dict:
        """Index a corpus. Fast path: a `corpus.parquet` (or a .parquet file) is bulk-loaded
        columnar; otherwise a folder of text/markdown files (one chunk per file)."""
        from ..ingest.parquet_corpus import read_corpus_parquet, resolve_parquet
        pq = resolve_parquet(Path(path).expanduser())
        if pq is not None:
            rows = read_corpus_parquet(pq)
            for r in rows:
                self.docs.append({"chunk_id": r["chunk_id"], "filename": r["filename"],
                                  "text": r["text"], "created_at": None})
            return {"files": 1, "chunks": len(rows), "parquet": str(pq)}
        p = Path(path).expanduser()
        n = 0
        for fp in sorted(p.rglob("*")):
            if fp.suffix.lower() in (".md", ".txt", ".rst") and fp.is_file():
                self.docs.append({
                    "chunk_id": f"{fp.name}::0", "filename": fp.name,
                    "text": fp.read_text(errors="ignore"), "created_at": None,
                })
                n += 1
        return {"files": n, "chunks": n}

    def _bm25_index(self):
        """Cached (doc_token_lists, df, n_docs, avgdl). Rebuilt when the corpus changes."""
        key = (id(self.docs), len(self.docs))
        cache = getattr(self, "_bm25_cache", None)
        if cache is not None and cache[0] == key:
            return cache[1]
        doc_toks = [_token_list(d["text"]) for d in self.docs]
        df: dict[str, int] = {}
        for toks in doc_toks:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        n_docs = len(self.docs)
        avgdl = (sum(len(t) for t in doc_toks) / n_docs) if n_docs else 1.0
        index = (doc_toks, df, n_docs, avgdl or 1.0)
        self._bm25_cache = (key, index)
        return index

    def search(self, query: str, k: int, method: Retrieval = "hybrid") -> list[Hit]:
        """Deterministic BM25 ranking (IDF term-weighting + length normalization) — rare,
        specific terms (e.g. «тестостерон») dominate over common words, and short focused
        passages outrank long multi-panel dumps. `hybrid` adds a query-coverage bonus that
        favours chunks matching MORE distinct query terms."""
        import math

        q_terms = _tokens(query)
        if not q_terms or not self.docs:
            return []
        # corpus statistics (tokens/df/avgdl) are cached and rebuilt only when the corpus
        # changes — keyed on (identity, length) of self.docs, which catches both Add() and
        # a direct `.docs = [...]` reindex. Per-query cost is then just scoring, not
        # re-tokenizing the whole corpus (≈300ms→a few ms over 10k chunks).
        doc_toks, df, n_docs, avgdl = self._bm25_index()
        idf = {t: math.log((n_docs - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1.0) for t in q_terms}
        k1, b = 1.5, 0.75

        scored: list[tuple[float, dict]] = []
        for d, toks in zip(self.docs, doc_toks):
            if not toks:
                continue
            dl = len(toks)
            tf: dict[str, int] = {}
            for t in toks:
                if t in idf:
                    tf[t] = tf.get(t, 0) + 1
            if not tf:
                continue
            score = 0.0
            for t, f in tf.items():
                score += idf[t] * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
            if method in ("hybrid", "vector"):  # reward covering more of the query
                coverage = len(tf) / len(q_terms)
                score *= (1.0 + 0.5 * coverage)
            scored.append((score, d))
        scored.sort(key=lambda x: (-x[0], x[1]["filename"], x[1]["chunk_id"]))
        return [Hit(chunk_id=d["chunk_id"], filename=d["filename"], text=d["text"],
                    score=float(s), created_at=d.get("created_at"), source=self.source)
                for s, d in scored[:k]]

    def info(self) -> PluginInfo:
        return PluginInfo(name="inmemory", kind="store", capabilities=frozenset({"bm25", "vector", "hybrid"}))
