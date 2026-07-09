"""LiveRAG loader — anti-parametric RAG QA (SIGIR'25).

895 questions machine-synthesized (DataMorgana) grounded in niche FineWeb-10BT web pages,
so answers REQUIRE the retrieved context — a model can't answer from parametric memory
(unlike famous-entity sets such as FinanceBench). Each question ships its gold
`Supporting_Documents` (doc_id + content); the union of all gold docs (~970) IS the
corpus, so every OTHER question's docs are natural distractors — a self-contained,
reproducible pollution pool with no 15M-doc FineWeb download.

Retrieval is scored at DOC-ID level (clean), answers via the gpt-5.5 judge against
`Answer` / `Answer_Claims.direct`.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

_HF_DATASET = "LiveRAG/Benchmark"


@dataclass(frozen=True)
class Passage:
    id: str            # <doc_id>#<chunk_index>
    doc_id: str
    page: int          # chunk index within the doc
    text: str

    @property
    def key(self) -> str:
        return self.id


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answer: str
    gold_docs: frozenset        # {doc_id, ...} — the Supporting_Documents
    gold_claims: tuple          # Answer_Claims.direct (gradeable atomic claims)
    qtype: str                  # DataMorgana answer-type (factoid, ...)
    difficulty: float           # IRT-diff (-6..6)
    is_numeric: bool


@dataclass
class Corpus:
    passages: list
    by_doc: dict = field(default_factory=dict)     # doc_id -> [Passage]
    all_docs: list = field(default_factory=list)   # every doc_id (distractor pool)
    docs: dict = field(default_factory=dict)       # doc_id -> {} (compat shim)
    companies: dict = field(default_factory=dict)  # unused (compat shim)

    def pages_for(self, doc_id: str) -> list:
        return self.by_doc.get(doc_id, [])


def _num_in(s: str) -> bool:
    return bool(re.search(r"\d", str(s)))


def _chunk(text: str, size: int = 1200, overlap: int = 150) -> list:
    text = text or ""
    if len(text) <= size:
        return [text] if text.strip() else []
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


@lru_cache(maxsize=1)
def _load_raw():
    """Load the 895 LiveRAG rows (HF cache, or a local parquet mirror)."""
    from datasets import load_dataset
    path = os.environ.get("LIVERAG_PARQUET")
    if path and os.path.isfile(path):
        return load_dataset("parquet", data_files=path, split="train")
    return load_dataset(_HF_DATASET, split="train")


def load_questions(root: str | None = None) -> list:
    ds = _load_raw()
    out = []
    for r in ds:
        cfg = r.get("DataMorgana_Config") or {}
        claims = (r.get("Answer_Claims") or {}).get("direct") or []
        gold = frozenset(d["doc_id"] for d in (r.get("Supporting_Documents") or []) if d.get("doc_id"))
        out.append(Question(
            id=str(r["Index"]), question=r["Question"], answer=str(r["Answer"]),
            gold_docs=gold, gold_claims=tuple(claims),
            qtype=cfg.get("answer-type-categorization", "?"),
            difficulty=float(r.get("IRT-diff [-6 : 6]") or 0.0),
            is_numeric=_num_in(r["Answer"]),
        ))
    return out


def load_corpus(root: str | None = None, *, limit_docs: set | None = None,
                chunk_size: int = 1200) -> Corpus:
    """Build the passage corpus from the union of all questions' Supporting_Documents."""
    ds = _load_raw()
    contents: dict = {}          # doc_id -> content (first seen)
    for r in ds:
        for d in (r.get("Supporting_Documents") or []):
            did = d.get("doc_id")
            if did and did not in contents and d.get("content"):
                if limit_docs is None or did in limit_docs:
                    contents[did] = d["content"]

    passages, by_doc = [], {}
    for did, content in contents.items():
        for ci, chunk in enumerate(_chunk(content, size=chunk_size)):
            p = Passage(id=f"{did}#{ci}", doc_id=did, page=ci, text=chunk)
            passages.append(p)
            by_doc.setdefault(did, []).append(p)
    return Corpus(passages=passages, by_doc=by_doc, all_docs=list(by_doc.keys()),
                  docs={d: {} for d in by_doc})


@lru_cache(maxsize=1)
def default_root() -> str:
    return os.environ.get("LIVERAG_PARQUET", _HF_DATASET)
