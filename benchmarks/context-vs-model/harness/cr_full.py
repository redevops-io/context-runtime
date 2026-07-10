"""The REAL Context Runtime arm — the full ``ContextRuntime.run()`` pipeline.

Unlike ``tuner.ContextRuntimeTuner`` (a bandit over retrieval *config* only, which never
compresses or verifies), this drives
``context_runtime.runtime.runtime.ContextRuntime.run()`` end to end:

    plan → redevops-rag retrieve → StructuralCompressor.compress
         → SingleShotReasoner (the served GGUF, via OpenAICompatibleModel)
         → CitationVerifier.verify

Retrieval is scoped to the SAME (gold + ``level`` distractors) doc set the other arms use
(``arms.scope_docs``) via ``ScopedRetriever``, so the ONLY moving part vs ``naive_rag`` is
the context-management pipeline itself — which is exactly the thing under test.
"""
from __future__ import annotations

import time

from context_runtime.runtime.runtime import ContextRuntime
from context_runtime.types import Goal, Hit, PluginInfo
from context_runtime.integrations.redevops_rag import RetrievalConfig

from .arms import scope_docs
from .rag_store import Chunk


class ScopedRetriever:
    """Adapts ``FinanceBenchStore`` into a Context Runtime retriever scoped to a fixed doc
    set — so the CR arm sees the identical pollution scope as full_dump / naive_rag.

    The runtime calls ``search(query, k, method)`` (k + method chosen by the plan); we run
    the same ``hybrid_search`` (semantic ⊕ BM25 ⊕ RRF) the other arms use, filtered to the
    scoped ``document_ids``, and map ``Chunk → Hit``. ``last_chunks`` is stashed so the
    driver can compute retrieval metrics on exactly what the pipeline saw."""

    def __init__(self, store, doc_ids):
        self.store = store
        self.doc_ids = doc_ids
        self.last_chunks: list[Chunk] = []

    def search(self, query, k, method="hybrid"):
        cfg = RetrievalConfig(pool=max(k * 3, 40), limit=k, vector_threshold=0.0, rerank=False)
        chunks = self.store.search(query, cfg, document_ids=self.doc_ids)
        self.last_chunks = chunks
        return [Hit(chunk_id=f"{c.doc_id}::{c.page}", filename=c.doc_id, text=c.text,
                    score=c.score, source="corpus") for c in chunks]

    def index(self, path):            # already embedded in the shared DuckDB
        return {"indexed": 0}

    def info(self) -> PluginInfo:
        return PluginInfo(name="scoped_redevops", kind="store",
                          capabilities=frozenset({"bm25", "vector", "hybrid"}))


def run_cr(model, store, corpus, question, level):
    """Run one question through the full pipeline. Returns (resp, retrieved_chunks, decision)
    with the SAME shape the driver expects from the other arms."""
    docs, distractors = scope_docs(corpus, question, level)
    retr = ScopedRetriever(store, docs)
    # models=<single plugin> broadcasts to all tiers; compressor + verifier default on.
    cr = ContextRuntime(models=model, retriever=retr)

    t0 = time.time()
    verify_passed = None
    n_cit = 0
    try:
        res = cr.run(Goal(text=question.question))
        text = (res.answer or "").strip()
        verify_passed = (res.verdict.passed if res.verdict is not None else None)
        n_cit = len(res.citations)
        # best-effort token accounting off the trace (compression should shrink prompt tokens)
        comp_tokens = int(getattr(res.trace, "actual_tokens", 0) or 0)
    except Exception as e:  # noqa: BLE001
        text, comp_tokens = f"__ERROR__ {type(e).__name__}: {e}", 0
    dt = time.time() - t0

    decision = {"arm": "context_runtime", "scope_docs": len(docs),
                "n_distractors": len(distractors), "verify_passed": verify_passed,
                "n_citations": n_cit,
                "pipeline": "full: plan+retrieve+compress+reason+verify"}
    resp = {"text": text, "prompt_tokens": 0, "completion_tokens": comp_tokens,
            "latency_s": dt}
    return resp, retr.last_chunks, decision
