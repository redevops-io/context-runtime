"""The three context-construction arms + the pollution axis.

  A0  full_dump   — stuff the whole scoped pool (target filing + distractors) up to a
                    token budget. No retrieval discipline: tests the model's NATIVE
                    long-context handling (the strong model's claimed edge).
  A1  naive_rag   — fixed top-K BM25 over the pool (Context Runtime OFF; the library
                    default arm). Distractors leak in unfiltered.
  A2  context_rt  — Context Runtime ON: an intent-keyed bandit picks a RetrievalConfig
                    (gating threshold / limit / rerank) and prunes the context.

Pollution axis: the candidate POOL = the target filing's pages + pages from ``level``
distractor filings (other companies). level 0 = clean (target only); higher = noisier.
Distractors are chosen deterministically per question so runs are reproducible.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from context_runtime.integrations.redevops_rag import RetrievalConfig, DEFAULT_ARMS

from .retriever import BM25Index

# characters-per-token rough proxy; keeps A0 within a model's context window.
_CHARS_PER_TOK = 4
NAIVE_ARM = DEFAULT_ARMS[1]        # "the library default" — CR-OFF baseline


def _seeded_order(seed_key: str, items: list) -> list:
    """Deterministic shuffle of ``items`` seeded by ``seed_key`` (no global RNG)."""
    return sorted(items, key=lambda x: hashlib.sha1(f"{seed_key}:{x}".encode()).hexdigest())


def build_pool(corpus, question, level: int) -> tuple:
    """Return (pool_passages, distractor_docs). ``level`` distractor filings from OTHER
    companies are mixed in with the target filing's pages."""
    target = corpus.pages_for(question.doc_name)
    others = [d for d in corpus.by_doc
              if corpus.docs.get(d, {}).get("company") != question.company]
    picked = _seeded_order(question.id, others)[:level]
    pool = list(target)
    for d in picked:
        pool.extend(corpus.pages_for(d))
    return pool, picked


def _passages_to_context(passages, *, max_tokens: int) -> str:
    budget = max_tokens * _CHARS_PER_TOK
    parts, used = [], 0
    for p in passages:
        block = f"[{p.company} — {p.doc_name} p.{p.page}]\n{p.text}\n"
        if used + len(block) > budget:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


@dataclass
class ArmResult:
    context: str
    retrieved: list = field(default_factory=list)   # list[Passage] actually handed to the model
    config: object = None                            # RetrievalConfig used (A2) / NAIVE_ARM (A1)
    decision: dict = field(default_factory=dict)     # execution-decision record


def arm_full_dump(pool, question, *, max_tokens: int) -> ArmResult:
    # order by a cheap query-lexical prefilter so the truncation keeps the plausible pages
    idx = BM25Index(pool)
    ranked = [h.passage for h in idx.search(question.question, pool=len(pool), limit=len(pool),
                                            vector_threshold=0.0)]
    ranked = ranked or pool
    ctx = _passages_to_context(ranked, max_tokens=max_tokens)
    return ArmResult(context=ctx, retrieved=ranked, config=None,
                     decision={"arm": "full_dump", "pool": len(pool)})


def arm_naive_rag(pool, question, *, max_tokens: int) -> ArmResult:
    idx = BM25Index(pool)
    hits = idx.search(question.question, **NAIVE_ARM.kwargs())
    ret = [h.passage for h in hits]
    ctx = _passages_to_context(ret, max_tokens=max_tokens)
    return ArmResult(context=ctx, retrieved=ret, config=NAIVE_ARM,
                     decision={"arm": "naive_rag", "config": NAIVE_ARM.key, "k": len(ret)})


def arm_context_runtime(pool, question, *, max_tokens: int, tuner) -> ArmResult:
    """CR ON — the tuner (intent-keyed bandit) picks a config; we gate + prune with it."""
    cfg, bucket = tuner.choose(question)
    idx = BM25Index(pool)
    hits = idx.search(question.question, **cfg.kwargs(), rerank=cfg.rerank)
    ret = [h.passage for h in hits]
    ctx = _passages_to_context(ret, max_tokens=max_tokens)
    return ArmResult(context=ctx, retrieved=ret, config=cfg,
                     decision={"arm": "context_runtime", "bucket": bucket, "config": cfg.key,
                               "k": len(ret), "threshold": cfg.vector_threshold, "rerank": cfg.rerank})
