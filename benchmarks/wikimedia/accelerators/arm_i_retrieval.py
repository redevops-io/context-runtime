"""Arm I — cuVS retrieval (roadmap §5.2, §6.2; benchmark question I).

The runtime treats retrieval as a plannable capability: embed a query, find the nearest evidence, return
its EvidenceRefs. Arm C established that lineage identity is *which* refs come back. So the accelerator
question is: does GPU ANN return the **same** neighbours as the CPU exact search (identity preserved),
while improving throughput as the corpus grows — and from what corpus size on?

Three backends over the *same* real bge embeddings of real strategywiki revisions:

  * CPU-exact   — full cosine top-k (numpy matmul). The identity ground truth (recall 1.0 by definition);
                  O(N) per query, no index to build. The expensive path the accelerator must beat.
  * CPU-ANN     — hnswlib graph index (realistic CPU production alternative), recall measured vs exact.
  * GPU-cuVS    — cuVS CAGRA index, recall measured vs exact.

Retrieval indexes are built once and queried many times, so build and query are timed **separately**:
build cost amortises, query throughput is what a serving system actually pays per request. Vectors are
L2-normalised (cosine == inner product). recall@k = mean over queries of |ann_topk ∩ exact_topk| / k.
Correctness gate: cuVS recall@k >= a floor (identity essentially preserved) at every size, before any
timing is read for the crossover.

The real embedding corpus (~12k) sizes recall honestly; for the throughput sweep beyond it, the corpus is
extended with distribution-matched vectors (a real vector + small Gaussian jitter, re-normalised) so ANN
geometry stays realistic — these larger points are labelled ``scaled=True`` and used for timing only.
"""
from __future__ import annotations

import numpy as np

from .common import time_median

ARM = "I"
NAME = "cuVS RAG ANN retrieval (identity-preserving, query-throughput crossover)"
RECALL_FLOOR = 0.95      # cuVS must return essentially the same neighbours as CPU-exact


# --- corpus scaling (distribution-matched, for throughput beyond the real embeddings) ---------------

def extend_corpus(X: np.ndarray, n: int, seed: int = 0) -> tuple[np.ndarray, bool]:
    """Return an N-vector corpus. Up to len(X) it is the real embeddings; beyond, real vectors plus small
    Gaussian jitter (re-normalised) so the ANN problem stays realistic. Second value = whether scaled."""
    if n <= X.shape[0]:
        return np.ascontiguousarray(X[:n]), False
    rng = np.random.default_rng(seed)
    reps = -(-n // X.shape[0])                       # ceil
    base = np.repeat(X, reps, axis=0)[:n].astype(np.float32)
    jitter = rng.normal(0, 0.02, base.shape).astype(np.float32)
    out = base + jitter
    out /= (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)
    return np.ascontiguousarray(out), True


# --- CPU exact (identity ground truth, query-only) --------------------------------------------------

def cpu_exact_topk(X: np.ndarray, Q: np.ndarray, k: int) -> np.ndarray:
    sims = Q @ X.T
    part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(sims, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


# --- CPU ANN (hnswlib): build then query ------------------------------------------------------------

def hnsw_build(X: np.ndarray):
    import hnswlib
    n, d = X.shape
    idx = hnswlib.Index(space="ip", dim=d)
    idx.init_index(max_elements=n, ef_construction=200, M=16)
    idx.add_items(X, np.arange(n))
    return idx


def hnsw_query(idx, Q: np.ndarray, k: int):
    idx.set_ef(max(64, k * 4))
    labels, _ = idx.knn_query(Q, k=k)
    return labels


# --- GPU cuVS CAGRA: build then query ---------------------------------------------------------------

def cuvs_build(X: np.ndarray):
    import cupy as cp
    from cuvs.neighbors import cagra
    Xd = cp.asarray(X)                               # host->device (part of build cost)
    params = cagra.IndexParams(graph_degree=64, intermediate_graph_degree=128)
    return cagra.build(params, Xd)


def cuvs_query(idx, Q: np.ndarray, k: int):
    import cupy as cp
    from cuvs.neighbors import cagra
    Qd = cp.asarray(Q)
    # wider internal top-k for high recall (identity preservation) at scale
    sp = cagra.SearchParams(itopk_size=256, search_width=8)
    _, I = cagra.search(sp, idx, Qd, k)
    return cp.asarray(I).get()                        # device->host result copy (counted)


def recall_at_k(ann: np.ndarray, exact: np.ndarray) -> float:
    hits = [len(set(a.tolist()) & set(e.tolist())) for a, e in zip(ann, exact)]
    return float(np.mean(hits)) / exact.shape[1]


# --- crossover sweep --------------------------------------------------------------------------------

def sweep(X_real: np.ndarray, Q: np.ndarray, sizes, k: int = 10, *, repeats: int = 3,
          have_hnsw: bool = True) -> list[dict]:
    rows = []
    for n in sizes:
        Xn, scaled = extend_corpus(X_real, n)
        exact = cpu_exact_topk(Xn, Q, k)             # ground-truth neighbours + timing
        ex_med, *_ = time_median(lambda: cpu_exact_topk(Xn, Q, k), repeats=repeats)
        nq = Q.shape[0]
        row = {"n": int(Xn.shape[0]), "k": k, "queries": nq, "dim": X_real.shape[1], "scaled": bool(scaled),
               "cpu_exact_query_ms": round(ex_med, 3),
               "cpu_exact_qps": round(nq / (ex_med / 1000.0), 1) if ex_med else None}

        if have_hnsw:
            try:
                hidx = hnsw_build(Xn)
                hb, *_ = time_median(lambda: hnsw_build(Xn), repeats=1, warmup=0)
                hlabels = hnsw_query(hidx, Q, k)
                hq, *_ = time_median(lambda: hnsw_query(hidx, Q, k), repeats=repeats)
                row.update(hnsw_build_ms=round(hb, 3), hnsw_query_ms=round(hq, 3),
                           hnsw_recall=round(recall_at_k(hlabels, exact), 4))
            except Exception as e:  # pragma: no cover
                row["hnsw_error"] = f"{type(e).__name__}: {e}"

        gidx = cuvs_build(Xn)
        gb_med, gb_min, gb_cold, _ = time_median(lambda: cuvs_build(Xn), repeats=repeats, sync=True)
        glabels = cuvs_query(gidx, Q, k)
        gq_med, gq_min, gq_cold, _ = time_median(lambda: cuvs_query(gidx, Q, k), repeats=repeats, sync=True)
        recall = recall_at_k(glabels, exact)
        row.update(
            cuvs_build_ms=round(gb_med, 3), cuvs_build_ms_cold=round(gb_cold, 3),
            cuvs_query_ms=round(gq_med, 3), cuvs_recall=round(recall, 4),
            cuvs_qps=round(nq / (gq_med / 1000.0), 1) if gq_med else None,
            # query-only crossover (amortised index): the number a serving system pays per request
            query_speedup_vs_exact=round(ex_med / gq_med, 2) if gq_med else None,
            correct=bool(recall >= RECALL_FLOOR),
        )
        if "hnsw_query_ms" in row:
            row["query_speedup_vs_hnsw"] = round(row["hnsw_query_ms"] / gq_med, 2) if gq_med else None
        rows.append(row)
    return rows
