"""Phase 4 — the modality-intent classifier + its bandit-context wiring."""
from __future__ import annotations

from context_runtime.integrations.multimodal_classifier import (
    CHART, DIAGRAM, TABLE, TEXT, TIMESTAMP, VISUAL, classify_query,
)


def test_classify_query_labels():
    assert classify_query("what did revenue do after Q2 on the chart") == CHART
    assert classify_query("show me the architecture diagram") == DIAGRAM
    assert classify_query("which row of the balance sheet has the lease liability") == TABLE
    assert classify_query("the scene where she mentions the merger") == TIMESTAMP
    assert classify_query("what does the company logo look like") == VISUAL
    assert classify_query("who is the current CFO") == TEXT
    assert classify_query("") == TEXT              # safe on empty
    # specificity: a chart cue beats a generic visual cue in the same query
    assert classify_query("the picture of the revenue chart") == CHART


def test_classifier_folds_into_bandit_context():
    from context_runtime.integrations.librechat import LibreChatTenant

    class _Stub:
        def search(self, q, k, method):
            return []
        def index(self, p):
            return {}

    t = LibreChatTenant(retriever=_Stub(), query_classifier=classify_query)
    # the query-type appears in the context key, so chart vs text queries learn separately
    p = t.runtime.plan.__self__ if hasattr(t.runtime.plan, "__self__") else None  # noqa: F841
    plan = t.runtime.plan(__import__("context_runtime.types", fromlist=["Goal"]).Goal(text="x"))
    chart_ctx = t._select_ctx(plan, "what did revenue do on the chart")
    text_ctx = t._select_ctx(plan, "who is the CFO")
    assert chart_ctx.endswith(f":{CHART}") and text_ctx.endswith(f":{TEXT}")
    assert chart_ctx != text_ctx                   # different arms can be learned per modality
    # no classifier → context is the bare bucket (byte-for-byte legacy behaviour)
    t2 = LibreChatTenant(retriever=_Stub())
    assert ":" not in t2._select_ctx(plan, "what did revenue do on the chart")
