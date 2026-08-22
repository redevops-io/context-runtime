"""Arm C — evidence lineage integrity (plan §7).

Every retrieval hit must trace back to the exact real revision it was derived from: the source
page id, the source revision (version), and the strict-rcv1 content hash of the authoritative source
— and after a source update, the prior revision stays resolvable with its own correct identity.

Hard gates (plan §7): authoritative source/hash mismatch = 0, wrong-revision lineage = 0, silent
cross-version substitution = 0.
"""
from __future__ import annotations

from harness.embedder import StubEmbedder
from harness.evidence_corpus import RevisionPair


def run(pairs: list[RevisionPair]) -> dict:
    from redevops_rag.store import Store
    from redevops_rag.evidence import (
        EvidenceRevision, ingest_revision, evidence_ref_from_hit, source_evidence_ref,
    )
    from runtime_contracts.canonical import content_hash as rcv1

    hits_checked = 0
    hash_mismatch = 0
    missing_lineage = 0
    wrong_revision_lineage = 0
    stale_served_as_current = 0
    prior_resolvable = 0
    n = 0

    for pair in pairs:
        ref = f"strategywiki/page/{pair.page_id}"
        rag = Store(StubEmbedder(), ":memory:")
        # ingest revision A, then advance to B (retention keeps A)
        ingest_revision(rag, rag.embedder, EvidenceRevision(
            ref=ref, version=pair.a_revid, content=pair.a_text, observed_at=pair.a_ts, source="wikimedia"))
        ingest_revision(rag, rag.embedder, EvidenceRevision(
            ref=ref, version=pair.b_revid, content=pair.b_text, observed_at=pair.b_ts, source="wikimedia"))
        rag.reindex_fts()
        n += 1

        # every current hit traces to revision B with a correct, matching identity
        for h in rag.semantic_search(pair.b_text, top_k=5, threshold=0.0, current_only=True):
            hits_checked += 1
            if not (h.get("source_ref") and h.get("source_version") and h.get("source_content_hash")):
                missing_lineage += 1
                continue
            if h["source_ref"] != ref or h["source_version"] != pair.b_revid:
                wrong_revision_lineage += 1
            if h["source_content_hash"] != rcv1(pair.b_text):
                hash_mismatch += 1
            er = evidence_ref_from_hit(h)
            if er is None or er.pin() != source_evidence_ref(EvidenceRevision(
                    ref=ref, version=pair.b_revid, content=pair.b_text)).pin():
                wrong_revision_lineage += 1
            if h.get("superseded_by") is not None:
                stale_served_as_current += 1   # a superseded revision leaked into the current view

        # the prior revision A remains resolvable with ITS own correct identity (point-in-time)
        a_hits = rag.semantic_search(pair.a_text, top_k=5, threshold=0.0, source_version=pair.a_revid)
        if a_hits and a_hits[0]["source_version"] == pair.a_revid \
                and a_hits[0]["source_content_hash"] == rcv1(pair.a_text) \
                and a_hits[0]["superseded_by"] == pair.b_revid:
            prior_resolvable += 1

    passed = (n > 0 and hash_mismatch == 0 and wrong_revision_lineage == 0
              and missing_lineage == 0 and stale_served_as_current == 0 and prior_resolvable == n)
    return {
        "arm": "C", "name": "evidence lineage integrity", "passed": passed,
        "n_cases": n,
        "metrics": {
            "hits_checked": hits_checked,
            "authoritative_hash_mismatch": hash_mismatch,        # HARD GATE = 0
            "wrong_revision_lineage": wrong_revision_lineage,    # HARD GATE = 0
            "stale_served_as_current": stale_served_as_current,  # HARD GATE = 0
            "missing_lineage_rate": missing_lineage,
            "prior_revision_resolvable": prior_resolvable,       # == n_cases
        },
    }
