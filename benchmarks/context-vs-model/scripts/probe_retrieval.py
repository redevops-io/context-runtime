#!/usr/bin/env python3
"""Is the gold evidence retrievable at all? Rank of the best gold-overlap chunk within a
top-100 search of the TARGET FILING ONLY (no pollution). If it's rank>20, retrieval/embed
is the bottleneck; if it's top-5 but metrics say miss, the overlap metric is too strict."""
import sys
from harness import data, metrics
from harness.rag_store import FinanceBenchStore
from context_runtime.integrations.redevops_rag import RetrievalConfig

STORE = sys.argv[1] if len(sys.argv) > 1 else "/cache/bench/financebench.duckdb"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
root = data.default_root()
qs = [q for q in data.load_questions(root) if q.is_numeric][:N]
store = FinanceBenchStore(STORE)
big = RetrievalConfig(pool=200, limit=100, vector_threshold=0.0, rerank=False)
print("embed:", store.rag.embedder.model_name, "dim", store.rag.embedder.dim)

# also try a looser relevance threshold to see if the metric is undercounting
def ranks_at(hits, q, thr):
    out = []
    for i, c in enumerate(hits, 1):
        if c.doc_name != q.doc_name:
            continue
        ct = metrics._toks(c.text)
        if ct and any(metrics._toks(g) and len(ct & metrics._toks(g)) / len(ct) >= thr
                      for g in q.gold_evidences):
            out.append(i)
    return out

found55 = found30 = 0
for q in qs:
    hits = store.search(q.question, big, document_ids=[q.doc_name])
    r55 = ranks_at(hits, q, 0.55)
    r30 = ranks_at(hits, q, 0.30)
    found55 += bool(r55); found30 += bool(r30)
    print(f"  [{q.company:14s}] top-100 hits @0.55={r55[:3] or 'NONE'} @0.30={r30[:3] or 'NONE'} | {q.question[:45]}")
print(f"\ngold found in top-100 (target doc): @0.55 thresh {found55}/{len(qs)}  @0.30 thresh {found30}/{len(qs)}")
