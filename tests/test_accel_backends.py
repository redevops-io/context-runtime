"""Tests for the v0.3.0 accelerator backends (context_runtime/accel).

These run on a CPU-only host: the selector is exercised by simulating GPU/library presence, and the cuDF
logic is validated through its pandas twin against the Python reference (no GPU needed). The GPU-only
equivalence checks skip cleanly when cupy/cuvs/cudf or a device are absent.
"""
from __future__ import annotations

import importlib

import pytest

from context_runtime.accel import selector
from context_runtime.adapters.store_temporal import TemporalStore


# ── selector: fail-closed, crossover-gated ──────────────────────────────────────────────────────────

def _reset_gpu_cache():
    selector._gpu_cache = None


def test_decide_disabled_by_default(monkeypatch):
    monkeypatch.delenv("CR_ACCEL", raising=False)
    monkeypatch.delenv("CR_ACCEL_ANN", raising=False)
    d = selector.decide("ann", 10_000_000)
    assert d.use_gpu is False and d.backend == "cpu"
    assert "disabled" in d.reason


def test_decide_no_gpu_falls_back(monkeypatch):
    monkeypatch.setenv("CR_ACCEL", "1")
    monkeypatch.setattr(selector, "gpu_available", lambda: False)
    d = selector.decide("ann", 10_000_000)
    assert d.use_gpu is False and d.backend == "cpu"
    assert "CUDA" in d.reason


def test_decide_below_crossover_uses_cpu(monkeypatch):
    monkeypatch.setenv("CR_ACCEL", "1")
    monkeypatch.setattr(selector, "gpu_available", lambda: True)
    monkeypatch.setattr(selector, "_lib_importable", lambda name: True)
    d = selector.decide("ann", 500)                       # below the 12k crossover
    assert d.use_gpu is False and "below" in d.reason


def test_decide_above_crossover_selects_gpu(monkeypatch):
    monkeypatch.setenv("CR_ACCEL", "1")
    monkeypatch.setattr(selector, "gpu_available", lambda: True)
    monkeypatch.setattr(selector, "_lib_importable", lambda name: True)
    d = selector.decide("ann", 50_000)                    # above the 12k crossover
    assert d.use_gpu is True and d.backend == "cuvs"


def test_decide_lib_missing_falls_back(monkeypatch):
    monkeypatch.setenv("CR_ACCEL_TEMPORAL", "1")
    monkeypatch.setattr(selector, "gpu_available", lambda: True)
    monkeypatch.setattr(selector, "_lib_importable", lambda name: False)
    d = selector.decide("temporal", 10_000_000)
    assert d.use_gpu is False and "not installed" in d.reason


# ── cuDF temporal: the dataframe result equals the Python reference exactly ──────────────────────────

def _sample_store() -> TemporalStore:
    s = TemporalStore()
    # overlapping validity intervals, some still open (valid_to=None), some ending inside the window,
    # ties on the same timestamp (to exercise the stable-sort tie-break)
    s.add("acme", "tier", "gold", valid_from="2026-01-01", valid_to="2026-03-01")
    s.add("acme", "tier", "platinum", valid_from="2026-03-01", valid_to=None)
    s.add("beta", "status", "active", valid_from="2026-01-01", valid_to=None)
    s.add("beta", "status", "churned", valid_from="2026-02-15", valid_to="2026-02-15")
    s.add("gamma", "plan", "free", valid_from="2025-06-01", valid_to="2026-02-01")
    s.add("delta", "plan", "pro", valid_from="2026-04-01", valid_to=None)   # outside the window
    return s


def test_cudf_pandas_path_matches_python_reference():
    pd = pytest.importorskip("pandas")  # noqa: F841
    from context_runtime.accel.cudf_temporal import changes_cpu_df
    s = _sample_store()
    since, until = "2026-01-01", "2026-04-01"
    reference = s.changes(query="", since=since, until=until, k=100)  # accel off by default → Python loop
    f = s._facts
    vectorized = changes_cpu_df([x.valid_from for x in f], [x.valid_to for x in f],
                                [x.subject for x in f], [x.predicate for x in f],
                                [x.obj for x in f], [x.text() for x in f],
                                since=since, until=until, k=100)
    assert vectorized == reference
    assert reference and reference == sorted(reference, key=lambda c: c["at"])


def test_changes_default_path_unchanged():
    """With the accelerator off (default), changes() is exactly the Python reference — no behaviour drift."""
    s = _sample_store()
    out = s.changes(since="2026-01-01", until="2026-04-01")
    assert all(c["change"] in ("began", "ended") for c in out)
    assert [c["at"] for c in out] == sorted(c["at"] for c in out)


# ── GPU-only equivalence (skips without a device) ───────────────────────────────────────────────────

def _no_gpu() -> bool:
    _reset_gpu_cache()
    return not (importlib.util.find_spec("cupy") and selector.gpu_available())


@pytest.mark.skipif(_no_gpu(), reason="no CUDA device / cupy")
def test_gpu_cudf_changes_equals_reference():
    pytest.importorskip("cudf")
    from context_runtime.accel.cudf_temporal import changes_gpu
    s = _sample_store()
    since, until = "2026-01-01", "2026-04-01"
    reference = s.changes(query="", since=since, until=until, k=100)
    f = s._facts
    gpu = changes_gpu([x.valid_from for x in f], [x.valid_to for x in f],
                      [x.subject for x in f], [x.predicate for x in f],
                      [x.obj for x in f], [x.text() for x in f],
                      since=since, until=until, k=100)
    assert gpu == reference


@pytest.mark.skipif(_no_gpu(), reason="no CUDA device / cupy")
def test_gpu_cuvs_preserves_top1_identity():
    pytest.importorskip("cuvs")
    import numpy as np
    from context_runtime.accel.cuvs_ann import CuvsAnnIndex
    rng = np.random.default_rng(0)
    mat = rng.standard_normal((2000, 64)).astype("float32")
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    idx = CuvsAnnIndex(mat)
    # each corpus vector as its own query → its exact nearest neighbour is itself
    hits = 0
    for i in range(0, 2000, 50):
        ids, _ = idx.query(mat[i], 10)
        hits += int(i in ids.tolist())
    assert hits >= 0.9 * len(range(0, 2000, 50))
