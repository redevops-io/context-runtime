"""A deterministic, dependency-free embedder for the benchmark.

The evidence-native invariants under test (identity, replay, lineage, freshness, incremental
equivalence) do not depend on embedding *quality* — only on determinism and reproducibility. A hashed
bag-of-words vector gives byte-identical results across runs and machines with no torch/network, which
is exactly what a correctness benchmark and its ≥3 clean reruns need. Retrieval-quality claims are out
of scope here (and would use the real bge encoder).
"""
from __future__ import annotations

import hashlib


class StubEmbedder:
    backend = "stub-hash"
    model_name = "stub-hash-64"

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.sim_floor = 0.0

    def encode(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in str(t).lower().split():
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0
            n = sum(x * x for x in v) ** 0.5 or 1.0
            out.append([x / n for x in v])
        return out
