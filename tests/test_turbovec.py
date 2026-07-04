"""TurboVecRetriever — quantized ANN index for the `vector` method at scale.

Exercised only when the [turbovec] extra is installed (turbovec + fastembed), so default
CI stays light. It must satisfy the same RetrieverPlugin contract as SemanticRetriever.
"""
from __future__ import annotations

import importlib.util

import pytest

from context_runtime.adapters.store_turbovec import TurboVecRetriever

_HAVE = importlib.util.find_spec("turbovec") and importlib.util.find_spec("fastembed")


def _docs():
    return [
        {"chunk_id": "s1", "filename": "s1", "text": "steroid hormone panel testosterone cortisol dhea", "created_at": None},
        {"chunk_id": "l1", "filename": "l1", "text": "lipid panel cholesterol ldl hdl triglycerides", "created_at": None},
        {"chunk_id": "c1", "filename": "c1", "text": "reschedule the meeting to friday afternoon", "created_at": None},
    ]


def test_turbovec_reports_availability():
    r = TurboVecRetriever(_docs())
    assert r.info().kind == "retriever"
    assert "quantized" in r.info().capabilities
    assert isinstance(r.available, bool)


@pytest.mark.skipif(not _HAVE, reason="turbovec/fastembed extra not installed")
def test_turbovec_ranks_semantically():
    r = TurboVecRetriever(_docs(), bit_width=4)
    hits = r.search("testosterone hormone results", k=2, method="vector")
    assert hits, "expected quantized ANN hits"
    assert hits[0].chunk_id == "s1"  # steroid passage ranks first


def test_turbovec_degrades_to_empty_without_extra():
    if importlib.util.find_spec("turbovec") and importlib.util.find_spec("fastembed"):
        pytest.skip("[turbovec] extra installed; the degradation path is not exercised here")
    r = TurboVecRetriever(_docs())
    assert r.search("alpha", k=2) == []              # no backend → empty, not a crash (the only default-CI path)
    r.index("/nonexistent/path")                     # index must not raise either
