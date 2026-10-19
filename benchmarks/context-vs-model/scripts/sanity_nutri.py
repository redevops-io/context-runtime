#!/usr/bin/env python3
"""Retrieval-only sanity on the Russian nutrients store: does bge-m3 find the gold passage
(among ~600 chunks in a lecture), and does pollution reduce the hit? No model needed."""
import os
import sys

os.environ["DATASET"] = "nutri"
from harness import data_nutri as data, metrics
from harness.arms import run_arm
from harness.rag_store import FinanceBenchStore
from harness.tuner import ContextRuntimeTuner

STORE = sys.argv[1] if len(sys.argv) > 1 else "/cache/bench/nutri.duckdb"
qs = data.load_questions()
corpus = data.load_corpus()
store = FinanceBenchStore(STORE)
print(f"questions={len(qs)}  corpus={len(corpus.passages)} chunks / {len(corpus.by_doc)} docs")
tuner = ContextRuntimeTuner()
for lvl in (0, 20, 60):
    hits = {"full_dump": 0, "naive_rag": 0, "context_runtime": 0}
    n = min(12, len(qs))
    for q in qs[:n]:
        for arm in hits:
            ar = run_arm(arm, store, corpus, q, lvl, max_tokens=6000, tuner=tuner)
            hits[arm] += metrics.retrieval_metrics(ar.retrieved, q)["hit"] or 0
    print(f"pollution={lvl:3d}: " + "  ".join(f"{a}={hits[a]/n:.0%}" for a in hits))
