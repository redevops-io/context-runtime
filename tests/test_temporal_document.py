"""TemporalDocumentRetriever — the default (non-lossy) temporal binding: document recall + a
bi-temporal time layer.

A LongMemEval-oracle-shaped check: the retriever surfaces the gold *raw* session (where Graphiti's
lossy LLM extraction capped out — hydration can only recover turns behind edges it *found*), and
answers point-in-time queries plain document retrieval can't. Dependency-free; satisfies the
RetrieverPlugin seam."""
from __future__ import annotations

from context_runtime.adapters.store_temporal import TemporalDocumentRetriever
from context_runtime.plugins.base import RetrieverPlugin


def _oracle() -> TemporalDocumentRetriever:
    r = TemporalDocumentRetriever()
    r.index([
        {"id": "s1", "reference_time": "2023-03-02T09:00", "body":
            "user: planning a trip next month. assistant: exciting! "
            "user: also, keep in mind I'm severely allergic to peanuts."},
        {"id": "s2", "reference_time": "2023-03-05T14:00", "body":
            "user: what's the weather in Berlin. assistant: mild and rainy this week."},
        {"id": "s3", "reference_time": "2023-04-10T17:50", "body":
            "user: recommend a restaurant. assistant: any cuisine? user: thai, and no shellfish for me."},
        {"id": "s4", "reference_time": "2023-05-01T11:00", "body":
            "user: draft an email to my manager about the Q2 roadmap. assistant: here is a draft."},
    ])
    return r


def test_conforms_to_retriever_seam():
    assert isinstance(_oracle(), RetrieverPlugin)  # structural: search / as_of / changes / info


def test_recall_surfaces_the_gold_raw_session():
    # the LongMemEval question — answered from the RAW turn, not an LLM-extracted fact
    hits = _oracle().search("what am I allergic to?", k=3)
    assert hits, "temporal retriever returned nothing"
    assert hits[0].chunk_id == "s1", f"gold session not top-ranked: {[h.chunk_id for h in hits]}"
    assert "peanuts" in hits[0].text.lower()               # non-lossy: the raw turn is present
    assert hits[0].meta["valid_at"].startswith("2023-03-02")  # carries the time axis


def test_point_in_time_view():
    # a role revised across sessions — the bi-temporal view document retrieval can't give
    r = TemporalDocumentRetriever()
    r.index([
        {"id": "mar", "reference_time": "2023-03-01", "body": "user: I just started as a data analyst at Acme."},
        {"id": "jun", "reference_time": "2023-06-01", "body": "user: update: promoted, I'm now the data engineering lead."},
    ])
    # as of April: only the March session is in view
    assert [h.chunk_id for h in r.as_of("what is my data role", at="2023-04-15", k=5)] == ["mar"]
    # as of July: the later session is now available
    assert "jun" in {h.chunk_id for h in r.as_of("what is my data role", at="2023-07-01", k=5)}


def test_known_at_ignores_late_corrections():
    r = TemporalDocumentRetriever()
    # a correction filed late: valid in June but only RECORDED in August
    r.index([
        {"id": "early", "valid_at": "2023-03-01", "recorded_at": "2023-03-01",
         "body": "user: my shipping address is 1 Old Street."},
        {"id": "late", "valid_at": "2023-06-01", "recorded_at": "2023-08-01",
         "body": "user: correction, my address has always been 2 New Road."},
    ])
    # what did we KNOW as of July (before the August correction was filed)? only the early record.
    known_july = [h.chunk_id for h in r.as_of("shipping address", at="2023-09-01", known_at="2023-07-01", k=5)]
    assert known_july == ["early"]


def test_changes_scan():
    ids = [c["id"] for c in _oracle().changes(since="2023-03-01", until="2023-04-01")]
    assert ids == ["s1", "s2"]  # only the March sessions entered the record in the window, in order
