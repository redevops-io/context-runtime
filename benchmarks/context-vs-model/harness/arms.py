"""The three context-construction arms + the pollution axis, over the REAL redevops-rag
store.

  full_dump   — large-K retrieval, no gating: hand the model a big, noisy context (as much
                of the polluted candidate set as the token budget holds). Tests whether the
                model can find the answer in a large context it must manage itself.
  naive_rag   — fixed top-K hybrid_search (Context Runtime OFF; the library default arm).
  context_rt  — Context Runtime ON: the intent-keyed bandit picks a RetrievalConfig
                (gating threshold / limit / rerank) and prunes.

All three call the same `hybrid_search`; they differ only in the RetrievalConfig. The
pollution axis scopes the search to (target filing + `level` distractor filings) via
`document_ids`. Distractors are chosen deterministically per question.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from context_runtime.integrations.redevops_rag import RetrievalConfig, DEFAULT_ARMS

_CHARS_PER_TOK = 4
# CR-OFF baselines (fixed configs):
NAIVE_ARM = DEFAULT_ARMS[1]                                              # "the library default"
FULL_DUMP_ARM = RetrievalConfig(pool=80, limit=40, vector_threshold=0.0, rerank=False)


def _seeded_order(seed_key: str, items: list) -> list:
    return sorted(items, key=lambda x: hashlib.sha1(f"{seed_key}:{x}".encode()).hexdigest())


def scope_docs(corpus, question, level: int) -> tuple:
    """(scoped_doc_names, distractor_doc_names): the target filing + `level` distractor
    filings from OTHER companies, chosen deterministically per question."""
    others = [d for d in corpus.by_doc
              if corpus.docs.get(d, {}).get("company") != question.company]
    picked = _seeded_order(question.id, others)[:level]
    return [question.doc_name] + picked, picked


def _chunks_to_context(chunks, *, max_tokens: int) -> str:
    budget = max_tokens * _CHARS_PER_TOK
    parts, used = [], 0
    for c in chunks:
        block = f"[{c.company} — {c.doc_name}]\n{c.text}\n"
        if used + len(block) > budget:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


@dataclass
class ArmResult:
    context: str
    retrieved: list = field(default_factory=list)   # list[rag_store.Chunk]
    config: object = None
    decision: dict = field(default_factory=dict)


def run_arm(arm_name: str, store, corpus, question, level: int, *,
            max_tokens: int, tuner=None) -> ArmResult:
    docs, distractors = scope_docs(corpus, question, level)
    bucket = None
    if arm_name == "full_dump":
        cfg = FULL_DUMP_ARM
    elif arm_name == "naive_rag":
        cfg = NAIVE_ARM
    elif arm_name == "context_runtime":
        cfg, bucket = tuner.choose(question)
    else:
        raise ValueError(f"unknown arm {arm_name}")
    chunks = store.search(question.question, cfg, document_ids=docs)
    ctx = _chunks_to_context(chunks, max_tokens=max_tokens)
    decision = {"arm": arm_name, "config": cfg.key, "k": len(chunks), "scope_docs": len(docs),
                "n_distractors": len(distractors), "threshold": cfg.vector_threshold,
                "rerank": cfg.rerank}
    if bucket:
        decision["bucket"] = bucket
    return ArmResult(context=ctx, retrieved=chunks, config=cfg, decision=decision)
