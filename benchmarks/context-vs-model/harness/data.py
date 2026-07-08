"""FinanceBench loader — real gold answers + gold evidence pages over a 361-doc,
32-company 10-K corpus.

Three record streams (all under ``.financebench/``):
  - ``qa.jsonl``           : 150 questions; gold ``answer`` + ``evidence[].evidence_page_num``
  - ``docs.jsonl``         : 361 filings; company / sector / period per ``doc_name``
  - ``corpus.manifest.jsonl``: 5028 passages; each id → source pdf + page; text in ``corpus/<id>.txt``

The multi-company corpus is the pollution axis: a question targets ONE filing, but the
corpus holds 360 others whose financial vocabulary collides (same line-items, different
numbers) — adversarial distractors by construction.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache


def _doc_from_source(source: str) -> str:
    """``3M_2022_10K.pdf`` → ``3M_2022_10K`` (matches qa.jsonl ``doc_name``)."""
    return re.sub(r"\.pdf$", "", source, flags=re.IGNORECASE)


@dataclass(frozen=True)
class Passage:
    id: str            # e.g. 0001_3m_2022_10k-pdf_p00
    doc_name: str      # 3M_2022_10K
    company: str       # 3M
    page: int          # 0-indexed passage/page number
    text: str

    @property
    def key(self) -> str:
        return f"{self.doc_name}#p{self.page}"


@dataclass(frozen=True)
class Question:
    id: str
    company: str
    doc_name: str
    qtype: str                       # metrics-generated | domain-relevant | novel-generated
    question: str
    answer: str
    gold_pages: frozenset            # {(doc_name, page), ...} from evidence[] (page = PDF page)
    gold_evidences: tuple            # the gold evidence_text strings — used for chunk-robust
                                     # retrieval scoring (text overlap, not chunk-index match)
    is_numeric: bool


@dataclass
class Corpus:
    passages: list                   # list[Passage]
    by_doc: dict = field(default_factory=dict)     # doc_name -> list[Passage]
    docs: dict = field(default_factory=dict)       # doc_name -> {company, sector, period}
    companies: dict = field(default_factory=dict)  # company -> [doc_name, ...]

    def pages_for(self, doc_name: str) -> list:
        return self.by_doc.get(doc_name, [])


def _num_in(s: str) -> bool:
    return bool(re.search(r"[-+]?\$?\d", str(s)))


def load_questions(root: str) -> list:
    out = []
    with open(os.path.join(root, "qa.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            gold = frozenset(
                (e.get("doc_name") or r["doc_name"], int(e["evidence_page_num"]))
                for e in r.get("evidence", [])
                if e.get("evidence_page_num") is not None
            )
            evidences = tuple(
                e["evidence_text"] for e in r.get("evidence", []) if e.get("evidence_text")
            )
            out.append(Question(
                id=r["financebench_id"], company=r["company"], doc_name=r["doc_name"],
                qtype=r.get("question_type", "?"), question=r["question"],
                answer=str(r["answer"]), gold_pages=gold, gold_evidences=evidences,
                is_numeric=_num_in(r["answer"]),
            ))
    return out


def load_corpus(root: str, *, limit_docs: set | None = None) -> Corpus:
    """Load the passage corpus. ``limit_docs`` (a set of doc_names) restricts the
    load for fast smoke runs; None loads all 5028 passages."""
    docs: dict = {}
    with open(os.path.join(root, "docs.jsonl")) as f:
        for line in f:
            d = json.loads(line)
            docs[d["doc_name"]] = {"company": d["company"], "sector": d.get("gics_sector"),
                                   "period": d.get("doc_period")}

    corpus_dir = os.path.join(root, "corpus")
    passages: list = []
    by_doc: dict = {}
    with open(os.path.join(root, "corpus.manifest.jsonl")) as f:
        for line in f:
            m = json.loads(line)
            doc_name = _doc_from_source(m["source"])
            if limit_docs is not None and doc_name not in limit_docs:
                continue
            company = docs.get(doc_name, {}).get("company", doc_name.split("_")[0])
            txt_path = os.path.join(corpus_dir, m["id"] + ".txt")
            try:
                with open(txt_path, encoding="utf-8", errors="ignore") as tf:
                    text = tf.read()
            except FileNotFoundError:
                continue
            p = Passage(id=m["id"], doc_name=doc_name, company=company,
                        page=int(m.get("passage", 0)), text=text)
            passages.append(p)
            by_doc.setdefault(doc_name, []).append(p)

    companies: dict = {}
    for dn in by_doc:
        companies.setdefault(docs.get(dn, {}).get("company", dn.split("_")[0]), []).append(dn)
    return Corpus(passages=passages, by_doc=by_doc, docs=docs, companies=companies)


@lru_cache(maxsize=1)
def default_root() -> str:
    """Locate the FinanceBench data dir. ``FINANCEBENCH_ROOT`` wins; else search up from
    the harness for a ``.financebench`` (symlinked in the repo)."""
    env = os.environ.get("FINANCEBENCH_ROOT")
    if env and os.path.isdir(env):
        return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))
    for up in (1, 2, 3, 4):
        cand = os.path.join(here, *([".."] * up), ".financebench")
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    raise FileNotFoundError("could not locate .financebench — set FINANCEBENCH_ROOT or "
                            "run scripts/download_data.sh")
