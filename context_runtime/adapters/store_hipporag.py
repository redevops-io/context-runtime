"""HippoRAG × Context Runtime — the multi-hop (graph) RetrieverPlugin (SPEC §4.5).

redevops-rag answers single-hop questions (find the chunk similar to the query).
HippoRAG answers *multi-hop* questions (connect facts across documents via a knowledge
graph + Personalized PageRank). They are not competitors — they are two retrieval
methods the planner chooses between per query (ARCHITECTURE §6: "dependency chain →
graph traversal"). This module provides:

  * ``HippoRAGRetriever`` — wraps the real redevops-io/HippoRAG (lazy import). API:
    HippoRAG(save_dir, llm_model_name, embedding_model_name).index(docs) /
    .retrieve(queries, num_to_retrieve=k).
  * ``SimGraphRetriever`` — a dependency-free, in-process multi-hop retriever (2-hop
    term spreading ≈ Personalized PageRank). It is the offline fallback AND lets the
    routing/learning run without HippoRAG's heavy deps, the same way the CrowdSec and
    redevops-rag bindings degrade gracefully.

Install the real engine:  pip install "context_runtime[hipporag]"
"""
from __future__ import annotations

import re

from ..types import Hit, PluginInfo, Retrieval

_WORD = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{2,}")
_STOP = {"the", "and", "for", "are", "was", "how", "does", "what", "which", "into",
         "with", "from", "this", "that", "between", "linked", "related", "relate"}


def _terms(s: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(s)} - _STOP


class SimGraphRetriever:
    """Offline multi-hop retriever: spreads activation across documents that share
    entities, so a *bridge* document connecting query terms surfaces even when it is
    not itself lexically similar to the query. This is the behavior single-hop
    retrieval structurally cannot produce."""

    def __init__(self, docs: list[dict] | None = None, source: str = "graph", bridge_weight: float = 0.5):
        self.docs = docs or []
        self.source = source
        self.bridge_weight = bridge_weight

    def index(self, path: str) -> dict:
        from pathlib import Path
        p = Path(path).expanduser()
        n = 0
        for fp in sorted(p.rglob("*")):
            if fp.suffix.lower() in (".md", ".txt", ".rst") and fp.is_file():
                self.docs.append({"chunk_id": f"{fp.name}::0", "filename": fp.name,
                                  "text": fp.read_text(errors="ignore"), "created_at": None})
                n += 1
        return {"files": n, "chunks": n}

    def search(self, query: str, k: int, method: Retrieval = "graph") -> list[Hit]:
        q = _terms(query)
        doc_terms = [(_terms(d["text"]), d) for d in self.docs]

        # hop 0: documents with direct query-term overlap
        direct: dict[int, int] = {}
        for i, (dt, _d) in enumerate(doc_terms):
            ov = len(q & dt)
            if ov:
                direct[i] = ov

        # the "activated" entity set = all terms appearing in any hop-0 doc
        activated: set[str] = set()
        for i in direct:
            activated |= doc_terms[i][0]

        # hop 1: a bridge doc (no direct query overlap) connected via a shared entity
        scored: list[tuple[float, dict, str]] = []
        for i, (dt, d) in enumerate(doc_terms):
            base = float(direct.get(i, 0))
            bridge = 0.0
            if i not in direct:
                bridge = self.bridge_weight * len(activated & dt)
            score = base + bridge
            if score > 0:
                kind = "direct" if base else "bridge(multi-hop)"
                scored.append((score, d, kind))

        scored.sort(key=lambda t: t[0], reverse=True)
        out = []
        for score, d, kind in scored[:k]:
            out.append(Hit(chunk_id=d["chunk_id"], filename=d["filename"], text=d["text"],
                           score=round(score, 3), created_at=d.get("created_at"),
                           source=self.source, meta={"hop": kind}))
        return out

    def info(self) -> PluginInfo:
        return PluginInfo(name="sim_graph", kind="retriever", capabilities=frozenset({"graph"}))


class HippoRAGRetriever:
    """The real graph retriever — wraps redevops-io/HippoRAG. Lazy-imports so the core
    package and the offline path don't need its (heavy) deps."""

    def __init__(self, save_dir: str = ".context_runtime/hipporag", llm_model_name: str = "gpt-5-mini",
                 embedding_model_name: str = "nvidia/NV-Embed-v2", source: str = "graph"):
        self.save_dir = save_dir
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.source = source
        self._hr = None
        self._doc_meta: dict[str, dict] = {}   # doc text → {chunk_id, filename}

    def _get(self):
        if self._hr is None:
            try:
                from hipporag import HippoRAG  # type: ignore
            except ImportError as e:  # pragma: no cover
                raise RuntimeError("HippoRAGRetriever needs: pip install 'context_runtime[hipporag]'") from e
            self._hr = HippoRAG(save_dir=self.save_dir, llm_model_name=self.llm_model_name,
                                embedding_model_name=self.embedding_model_name)
        return self._hr

    def index(self, path_or_docs) -> dict:
        from pathlib import Path
        if isinstance(path_or_docs, (list, tuple)):
            docs = list(path_or_docs)
        else:
            p = Path(path_or_docs).expanduser()
            docs = [fp.read_text(errors="ignore") for fp in sorted(p.rglob("*"))
                    if fp.suffix.lower() in (".md", ".txt", ".rst") and fp.is_file()]
        for i, text in enumerate(docs):
            self._doc_meta[text] = {"chunk_id": f"hr::{i}", "filename": f"doc-{i}"}
        self._get().index(docs=docs)
        return {"docs": len(docs)}

    def search(self, query: str, k: int, method: Retrieval = "graph") -> list[Hit]:
        results = self._get().retrieve(queries=[query], num_to_retrieve=k)
        sol = results[0]
        docs = getattr(sol, "docs", []) or []
        scores = list(getattr(sol, "doc_scores", []) or [])
        out = []
        for i, text in enumerate(docs[:k]):
            meta = self._doc_meta.get(text, {"chunk_id": f"hr::{i}", "filename": f"doc-{i}"})
            out.append(Hit(chunk_id=meta["chunk_id"], filename=meta["filename"], text=text,
                           score=float(scores[i]) if i < len(scores) else 0.0,
                           source=self.source, meta={"hop": "graph(PPR)"}))
        return out

    def info(self) -> PluginInfo:
        return PluginInfo(name="hipporag", kind="retriever", capabilities=frozenset({"graph"}))
