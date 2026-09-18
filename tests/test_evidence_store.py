"""Durable evidence store — immutable observations + versioned opportunity lineage that survives restarts.

The restart/replay test is the point: persist a run, drop the in-memory store entirely, construct a fresh
store from the SAME file, and re-run the collector+qualification pipeline — a rediscovered tender must
classify UNCHANGED (idempotent), and a materially amended one UPDATED with the exact fields that changed.
This is what makes discovery a longitudinal dataset rather than a per-process experiment.
"""
from __future__ import annotations

from context_runtime.integrations.evidence_store import ChangeKind, EvidenceStore
from context_runtime.integrations.local_gov import BusinessProfile
from context_runtime.integrations.procurement_sources import (
    FetchResult, FixtureFetcher, ProcurementSource, SourceMethod, run_region,
)

_INFORMS_HTML = """<html><body>
<span class='ps_box-value' id='SCP_PUB_AUC_VW_AUC_NAME$0' >E26SP01: Bond Engineering Services</span>
<span class='ps_box-value' id='SCP_PUB_AUC_VW_AUC_ID$0' >E26SP01</span>
<span class='ps_box-value' id='SCP_PUB_AUC_VW_AUC_FORMAT$0' >RFP</span>
</body></html>"""

_DETAIL_V1 = """<html><body>
<span id='SCP_P_AUCDTL_VW_AUC_NAME$0' >E26SP01: Bond Engineering Services</span>
<span id='SCP_P_AUCDTL_VW_AUC_ID$0' >E26SP01</span>
<span id='SCP_P_AUCDTL_VW_AUC_STATUS$0' >Posted</span>
<span id='SCP_P_AUCDTL_VW_AUC_FORMAT$0' >RFP</span>
<span id='SCP_P_AUCDTL_VW_SCP_END_DATE_CHAR$0' >09/21/2026 02:00 PM EST</span>
</body></html>"""

# same solicitation, deadline moved up + an addendum posted → a material EvidenceChange
_DETAIL_V2 = _DETAIL_V1.replace("09/21/2026", "09/14/2026").replace(
    "</body>", "<a>Download</a></body>")

_SRC = ProcurementSource("src-miami-dade-informs", "fl-miami-dade-county",
                         "Miami-Dade County — INFORMS Public Bidding", "https://fixture/informs",
                         SourceMethod.INFORMS)
_PROFILE = BusinessProfile(name="Metro Eng & Tech", service_zip="33180", service_radius_miles=50,
                           services=("engineering", "professional services"))


def _fetcher(detail_html: str) -> FixtureFetcher:
    return FixtureFetcher(
        {"https://fixture/informs": FetchResult(200, _INFORMS_HTML, "text/html")},
        detail_responses={"SCP_COSP_WK_FL_DESCR$0": FetchResult(200, detail_html, "text/html")})


def test_store_records_observations_and_new_opportunity(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.jsonl")
    res = run_region("33180", 50, _PROFILE, fetcher=_fetcher(_DETAIL_V1), sources=[_SRC],
                     enrich=True, store=store)
    assert store.observation_count() == 2                 # one list + one detail observation persisted
    assert [c.kind for c in res.changes] == [ChangeKind.NEW]
    assert store.path.exists() and store.path.read_text().count("\n") >= 3


def test_restart_reload_is_idempotent_then_detects_amendment(tmp_path):
    path = tmp_path / "evidence.jsonl"

    # ── run 1: first discovery persists everything ──
    r1 = run_region("33180", 50, _PROFILE, fetcher=_fetcher(_DETAIL_V1), sources=[_SRC],
                    enrich=True, store=EvidenceStore(path))
    assert r1.changes[0].kind is ChangeKind.NEW

    # ── restart: a brand-new store process reloads the SAME durable file ──
    store2 = EvidenceStore(path)
    assert store2.known_opportunity_ids() == {"src-miami-dade-informs:E26SP01"}
    assert store2.observation_count() == 2                # observations reloaded, not re-derived

    # rediscovering the identical tender is a no-op (idempotent across the restart)
    r2 = run_region("33180", 50, _PROFILE, fetcher=_fetcher(_DETAIL_V1), sources=[_SRC],
                    enrich=True, store=store2)
    assert r2.changes[0].kind is ChangeKind.UNCHANGED
    assert store2.observation_count() == 2                # immutable: nothing re-written

    # ── the deadline moves up + an addendum appears → UPDATED with the exact changed fields ──
    store3 = EvidenceStore(path)
    r3 = run_region("33180", 50, _PROFILE, fetcher=_fetcher(_DETAIL_V2), sources=[_SRC],
                    enrich=True, store=store3)
    chg = r3.changes[0]
    assert chg.kind is ChangeKind.UPDATED
    assert "response_due_at" in chg.changed_fields        # the deadline change is captured
    assert store3.observation_count() == 3                # the new detail observation was appended
    # the updated deadline is what the qualification/handoff now carries
    bond = next(h for h in r3.handoffs if h["opportunity_id"].endswith("E26SP01"))
    assert bond["response_due_at"] == "2026-09-14"


def test_reload_survives_across_three_independent_store_objects(tmp_path):
    # digests must round-trip through JSONL: three separate store instances, same file, stable classification
    path = tmp_path / "e.jsonl"
    run_region("33180", 50, _PROFILE, fetcher=_fetcher(_DETAIL_V1), sources=[_SRC], enrich=True,
               store=EvidenceStore(path))
    for _ in range(2):
        r = run_region("33180", 50, _PROFILE, fetcher=_fetcher(_DETAIL_V1), sources=[_SRC], enrich=True,
                       store=EvidenceStore(path))
        assert r.changes[0].kind is ChangeKind.UNCHANGED
