"""cuVS CAGRA backend for semantic retrieval (the winning arm-I backend).

Wraps a matrix of L2-normalised embeddings in a cuVS CAGRA graph index and answers top-k cosine queries on
the GPU. The neighbour *set* is approximate (recall ~0.96 vs exact — hence opt-in, for corpora past the
~12k crossover where the 36× query speedup is worth it); the returned **scores are exact**, recomputed as
the true inner product of each returned neighbour, so ranking among the returned set is faithful.

All GPU imports are lazy and every entry point raises cleanly on any failure, so the caller can fall back
to its CPU path. This module imports fine on a host with no GPU / no cuVS.
"""
from __future__ import annotations

import numpy as np


class CuvsAnnIndex:
    """A built CAGRA index over one embedding matrix. Rebuild by constructing a new instance."""

    def __init__(self, mat: np.ndarray):
        import cupy as cp
        from cuvs.neighbors import cagra
        self._cp = cp
        self._cagra = cagra
        self._n = int(mat.shape[0])
        self._mat_dev = cp.asarray(np.ascontiguousarray(mat, dtype=np.float32))   # kept for exact rescoring
        params = cagra.IndexParams(graph_degree=64, intermediate_graph_degree=128)
        self._index = cagra.build(params, self._mat_dev)

    @property
    def size(self) -> int:
        return self._n

    def query(self, qv: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, scores) for the top-``k`` neighbours of ``qv``. Scores are exact cosine
        (inner product on normalised vectors); indices are the approximate CAGRA neighbours."""
        cp, cagra = self._cp, self._cagra
        k = max(1, min(int(k), self._n))
        qd = cp.asarray(np.ascontiguousarray(qv, dtype=np.float32).reshape(1, -1))
        sp = cagra.SearchParams(itopk_size=256, search_width=8)
        _, idx = cagra.search(sp, self._index, qd, k)
        idx = cp.asarray(idx).reshape(-1)
        exact_scores = self._mat_dev[idx] @ qd.reshape(-1)          # true cosine for the returned set
        order = cp.argsort(-exact_scores)                            # rank by exact score
        idx, exact_scores = idx[order], exact_scores[order]
        return cp.asnumpy(idx).astype(int), cp.asnumpy(exact_scores).astype(float)
