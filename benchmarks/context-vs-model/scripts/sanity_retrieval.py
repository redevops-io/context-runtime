#!/usr/bin/env python3
"""Quick sanity: does the real hybrid_search return gold-overlapping chunks, and does
CR gating cut pollution as distractors rise?"""
import sys
from harness import data, metrics
from harness.arms import run_arm
from harness.rag_store import FinanceBenchStore
from harness.tuner import ContextRuntimeTuner

STORE = sys.argv[1] if len(sys.argv) > 1 else "/cache/bench/financebench.duckdb"
root = data.default_root()
qs = data.load_questions(root)
corpus = data.load_corpus(root)
qs = [q for q in qs if q.doc_name in corpus.by_doc]
store = FinanceBenchStore(STORE)
print(f"store chunks={store.count}  covered_qs={len(qs)}")

q = next((x for x in qs if "capital expenditure" in x.question.lower() and x.company == "3M"), qs[0])
print(f"\nQ [{q.company}]: {q.question[:90]}")
print(f"gold_ans={q.answer} | gold_evidences={len(q.gold_evidences)}")

tuner = ContextRuntimeTuner()
for level in (0, 8):
    print(f"\n--- pollution level={level} ---")
    for arm in ("full_dump", "naive_rag", "context_runtime"):
        ar = run_arm(arm, store, corpus, q, level, max_tokens=6000, tuner=tuner)
        rm = metrics.retrieval_metrics(ar.retrieved, q)
        poll = metrics.pollution_fraction(ar.retrieved, q)
        print("  {:15s} k={:2d} hit={} recall={:.2f} prec={:.2f} pollution={:.2f} cfg={}".format(
            arm, len(ar.retrieved), rm["hit"], rm["recall"], rm["precision"], poll,
            ar.decision.get("config")))
print("\nCR policy (cold):", tuner.policy())

# --- learning check: does the bandit converge to a recall-preserving arm? ---
# Reward = retrieval hit (did we fetch a gold-overlapping chunk), across mixed pollution.
print("\n=== bandit learning check (reward = retrieval hit, mixed pollution) ===")
learn = ContextRuntimeTuner()
import itertools
levels_cycle = itertools.cycle([0, 2, 4, 8])
hits_first, hits_second = [], []
numeric = [x for x in qs if x.is_numeric][:80]
for i, qq in enumerate(numeric):
    lvl = next(levels_cycle)
    ar = run_arm("context_runtime", store, corpus, qq, lvl, max_tokens=6000, tuner=learn)
    hit = metrics.retrieval_metrics(ar.retrieved, qq)["hit"] or 0.0
    learn.record(qq, ar.config, hit)          # reward on retrieval hit
    (hits_first if i < len(numeric) // 2 else hits_second).append(hit)
fh = sum(hits_first) / max(1, len(hits_first))
sh = sum(hits_second) / max(1, len(hits_second))
print(f"CR retrieval hit-rate: first half={fh:.0%}  second half={sh:.0%}  (n={len(numeric)})")
print("CR learned policy:", learn.policy())
