"""Vector-ranking parity across operating systems.

What is compared and how (the tolerances are deliberate, not tuned until green):

* **Ranks** of the top-k are compared for exact equality. The corpus is constructed so that no two documents
  inside the top-k are close: consecutive reference cosines differ by at least ``MIN_GAP`` (0.05), which is
  more than twice the largest score error any store shows (below), so a correct implementation cannot
  swap them on any CPU. A test asserts the gap, so the corpus cannot silently degrade into near-ties.
* **Scores**: ``SemanticRetriever`` computes float32 ``matrix @ vector`` with whatever BLAS numpy ships for the
  platform; ``SCORE_TOL_EXACT`` = 1e-5 absolute. ``TurboVecRetriever`` stores 4-bit quantised vectors, so its
  scores are approximate by design; ``SCORE_TOL_QUANTISED`` = 0.05 absolute (the largest deviation from the
  exact cosine seen while developing this was about 0.015).
* The golden JSON under ``tests/golden`` was generated on one machine (macOS arm64) and is compared on
  Linux, macOS and Windows by the CI lanes. ``CR_UPDATE_GOLDEN=1`` regenerates it.

The embedder is injected (fixed vectors), so these tests need no model download and measure the retrieval
mathematics (BLAS, quantisation, SIMD), not an embedding model. ``test_real_model_ranking`` adds a real
fastembed ONNX model when ``CR_TEST_PARITY_MODEL=1``.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from context_runtime.adapters import store_semantic as ss  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "golden"
UPDATE = os.environ.get("CR_UPDATE_GOLDEN") == "1"
DIM = 32
K = 5
MIN_GAP = 0.05
SCORE_TOL_EXACT = 1e-5
SCORE_TOL_QUANTISED = 0.05

# cosine of document j to the axis query e0; spaced 0.12 apart so the ranking is unambiguous
COSINES = [0.96, 0.84, 0.72, 0.60, 0.48, 0.36, 0.24, 0.12, 0.0, -0.12, -0.24, -0.36]


def _unit(v):
    v = np.asarray(v, dtype="float64")
    return (v / np.linalg.norm(v)).astype("float32")


def _vectors() -> dict[str, "np.ndarray"]:
    vec: dict[str, np.ndarray] = {}
    for j, c in enumerate(COSINES):
        d = np.zeros(DIM)
        d[0] = c
        d[j + 1] = math.sqrt(1 - c * c)
        vec[f"doc {j:02d}"] = _unit(d)
    q0 = np.zeros(DIM)
    q0[0] = 1.0
    qm = np.zeros(DIM)                       # a second query: between axis 0 and axis 3
    qm[0], qm[3] = 1.0, 1.0
    vec["query axis"], vec["query mix"] = _unit(q0), _unit(qm)
    return vec


class _Embedder:
    def __init__(self, vec):
        self.vec = vec

    def embed(self, texts):
        return [self.vec[t] for t in texts]


@pytest.fixture()
def injected(monkeypatch):
    vec = _vectors()
    monkeypatch.setattr(ss, "_embedder", _Embedder(vec))
    monkeypatch.setattr(ss, "_embedder_tried", True)
    return vec


def _docs():
    return [{"chunk_id": f"d{j:02d}", "filename": f"d{j:02d}.md", "text": f"doc {j:02d}"}
            for j in range(len(COSINES))]


def _reference(vec, query):
    """Exact float64 cosine ranking: the ground truth the stores are compared with."""
    q = vec[query].astype("float64")
    scored = sorted(((float(vec[f"doc {j:02d}"].astype("float64") @ q), f"d{j:02d}") for j in range(len(COSINES))),
                    reverse=True)
    return [(cid, s) for s, cid in scored]


def _golden(name, obj):
    path = GOLDEN / name
    data = (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("ascii")
    if UPDATE:
        path.write_bytes(data)
    return json.loads(path.read_bytes().decode("ascii"))


def _compare(got, want, tol):
    assert [c for c, _ in got] == [c for c, _ in want], f"ranks differ: {got} vs {want}"
    for (c, s), (_, s0) in zip(got, want):
        assert abs(s - s0) <= tol, f"{c}: score {s} vs golden {s0} (tolerance {tol})"


def test_the_corpus_has_no_near_ties_in_the_top_k(injected):
    for q in ("query axis", "query mix"):
        ref = [s for _, s in _reference(injected, q)][: K + 1]
        gaps = [a - b for a, b in zip(ref, ref[1:])]
        assert min(gaps) >= MIN_GAP, (q, gaps)


def test_semantic_store_matches_the_golden_ranking_and_scores(injected):
    store = ss.SemanticRetriever(_docs())
    out = {q: [[h.chunk_id, round(h.score, 6)] for h in store.search(q, K)] for q in ("query axis", "query mix")}
    golden = _golden("vector_semantic.json", out)
    for q, hits in out.items():
        _compare([(c, s) for c, s in hits], [(c, s) for c, s in golden[q]], SCORE_TOL_EXACT)
        _compare([(c, s) for c, s in hits], _reference(injected, q)[:K], SCORE_TOL_EXACT)   # and the exact maths


def test_hybrid_vector_method_agrees_with_the_semantic_store(injected):
    hy = ss.HybridRetriever(_docs())
    sem = ss.SemanticRetriever(_docs())
    for q in ("query axis", "query mix"):
        assert [h.chunk_id for h in hy.search(q, K, "vector")] == [h.chunk_id for h in sem.search(q, K)]


def test_turbovec_store_matches_the_golden_ranking_within_quantisation_tolerance(injected):
    pytest.importorskip("turbovec", reason="UNVERIFIED on this platform: turbovec is not installed")
    pytest.importorskip("fastembed", reason="UNVERIFIED on this platform: fastembed is not installed")
    from context_runtime.adapters.store_turbovec import TurboVecRetriever
    store = TurboVecRetriever(_docs())
    out = {q: [[h.chunk_id, round(h.score, 6)] for h in store.search(q, K)] for q in ("query axis", "query mix")}
    golden = _golden("vector_turbovec.json", out)
    for q, hits in out.items():
        _compare([(c, s) for c, s in hits], [(c, s) for c, s in golden[q]], SCORE_TOL_QUANTISED)
        _compare([(c, s) for c, s in hits], _reference(injected, q)[:K], SCORE_TOL_QUANTISED)


# A small natural-language corpus for the real ONNX model. Only the top-1 rank is asserted exactly (the margin
# to the runner-up is asserted too); positions 2 and 3 are close in cosine, so only their scores are compared (0.01).
REAL_DOCS = [
    "The invoice is overdue and payment must be made within thirty days.",
    "Quarterly revenue grew twelve percent driven by subscription renewals.",
    "The server crashed overnight because the disk ran out of space.",
    "Our hiring plan adds three backend engineers in the spring.",
    "The recipe calls for flour, butter and two cups of sugar.",
    "A new vaccine trial reported strong antibody responses in adults.",
]
REAL_QUERIES = ["unpaid bill reminder", "company sales growth", "why did the machine go down"]


@pytest.mark.skipif(os.environ.get("CR_TEST_PARITY_MODEL") != "1",
                    reason="UNVERIFIED here: set CR_TEST_PARITY_MODEL=1 to download and run the real ONNX embedding model")
def test_real_model_ranking(monkeypatch):
    pytest.importorskip("fastembed")
    monkeypatch.setattr(ss, "_embedder", None)
    monkeypatch.setattr(ss, "_embedder_tried", False)
    monkeypatch.setattr(ss, "_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    docs = [{"chunk_id": f"r{i}", "filename": f"r{i}.md", "text": t} for i, t in enumerate(REAL_DOCS)]
    store = ss.SemanticRetriever(docs)
    out = {q: [[h.chunk_id, round(h.score, 4)] for h in store.search(q, 3)] for q in REAL_QUERIES}
    golden = _golden("vector_real_model.json", out)
    for q in REAL_QUERIES:
        got = out[q]
        assert got[0][0] == golden[q][0][0], f"top-1 differs for {q!r}"
        top2_gap = got[0][1] - got[1][1]
        assert top2_gap > 0.05, f"corpus must keep a clear top-1 margin, got {top2_gap}"
        for (c, s), (c0, s0) in zip(got, golden[q]):
            assert abs(s - s0) <= 0.01, f"{q!r}/{c}: {s} vs {s0}"
