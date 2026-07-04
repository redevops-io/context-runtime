"""Phase 2b — multi-vector late-interaction (MaxSim) retrieval. Embedders are injected, so no
ColPali VLM / Qdrant is needed."""
from __future__ import annotations

from context_runtime.adapters.store_multivector import MultiVectorRetriever, maxsim

# a tiny concept space; each page is a SET of patch vectors (one hot axis per patch)
AXES = ["lease", "revenue", "risk"]


def _onehot(axis):
    import numpy as np
    return np.array([1.0 if a == axis else 0.0 for a in AXES], dtype="float32")


def _page_mat(*axes):
    import numpy as np
    return np.vstack([_onehot(a) for a in axes])


def test_maxsim_late_interaction_math():
    import numpy as np
    q = _page_mat("lease", "revenue")               # 2 query tokens
    d_good = _page_mat("lease", "revenue", "risk")   # contains both → score ~2
    d_bad = _page_mat("risk", "risk")                # contains neither → score ~0
    assert maxsim(q, d_good) > maxsim(q, d_bad)
    assert abs(maxsim(q, d_good) - 2.0) < 1e-5       # each query token maxes to its patch


def test_multivector_retrieval_picks_right_page(tmp_path):
    for n in ("lease-note.png", "revenue-note.png"):
        (tmp_path / n).write_bytes(b"\x89PNG\r\n")

    def doc_embed(paths):
        out = []
        for p in paths:
            out.append(_page_mat("lease", "risk") if "lease" in p else _page_mat("revenue", "risk"))
        return out

    def query_embed(text):
        # a query about lease liabilities → a lease token (+ a generic risk token)
        return _page_mat("lease", "risk") if "lease" in text else _page_mat("revenue", "risk")

    ret = MultiVectorRetriever(doc_embed=doc_embed, query_embed=query_embed)
    rep = ret.index(str(tmp_path))
    assert rep["pages"] == 2
    hits = ret.search("total lease liability in 2023", k=2)
    assert hits[0].filename == "lease-note.png"
    assert hits[0].meta["type"] == "page_image" and hits[0].meta["late_interaction"] is True
    assert ret.path_for(hits[0].chunk_id).endswith("lease-note.png")
    assert ret.path_for("nope::page") is None       # only indexed pages resolve


def test_degrades_without_backend(tmp_path):
    from context_runtime.adapters.store_multivector import colpali_available
    if colpali_available():
        return
    (tmp_path / "x.png").write_bytes(b"\x89PNG\r\n")
    ret = MultiVectorRetriever()                     # no injected embedder, no colpali installed
    assert ret.index(str(tmp_path))["pages"] == 0
    assert ret.search("anything", 3) == []


def test_maxsim_empty_and_zero_vectors_are_finite():
    import numpy as np
    from context_runtime.adapters.store_multivector import _l2_rows, maxsim
    assert maxsim([], [[1.0, 0.0]]) == 0.0          # empty query → 0, not a crash
    assert maxsim([[1.0, 0.0]], []) == 0.0          # empty doc → 0
    out = _l2_rows(np.array([[0.0, 0.0], [3.0, 4.0]], dtype="float32"))
    assert np.isfinite(out).all()                   # zero-norm row → 0/1, never NaN (would poison MaxSim)
    assert abs(out[1][0] - 0.6) < 1e-6 and abs(out[1][1] - 0.8) < 1e-6
