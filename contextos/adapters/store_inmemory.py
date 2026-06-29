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


class InMemoryStore:
    def __init__(self, docs: list[dict] | None = None, source: str = "memory"):
        # each doc: {"chunk_id","filename","text","created_at"?}
        self.docs = docs or []
        self.source = source

    def index(self, path: str) -> dict:
        """Index a folder of text/markdown files (one chunk per file for v0.1 simplicity)."""
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

    def search(self, query: str, k: int, method: Retrieval = "hybrid") -> list[Hit]:
        q = _tokens(query)
        scored: list[tuple[float, dict]] = []
        for d in self.docs:
            dt = _tokens(d["text"])
            if not dt:
                continue
            overlap = len(q & dt)
            if method == "bm25":
                score = overlap                                   # crude term-frequency proxy
            elif method == "vector":
                score = overlap / (len(q | dt) ** 0.5 or 1)       # crude cosine proxy
            else:  # hybrid / others → blend
                score = overlap + overlap / (len(q | dt) ** 0.5 or 1)
            if score > 0:
                scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[Hit] = []
        for score, d in scored[:k]:
            out.append(Hit(
                chunk_id=d["chunk_id"], filename=d["filename"], text=d["text"],
                score=float(score), created_at=d.get("created_at"), source=self.source,
            ))
        return out

    def info(self) -> PluginInfo:
        return PluginInfo(name="inmemory", kind="store", capabilities=frozenset({"bm25", "vector", "hybrid"}))
