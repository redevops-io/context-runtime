"""Longitudinal soak — a CollectionRun is recorded every pass, so a zero-change day is distinguishable
from a silently-broken collector, and the report separates source reliability from procurement activity.
"""
from __future__ import annotations

from context_runtime.integrations.evidence_store import EvidenceStore
from context_runtime.integrations.local_gov import BusinessProfile
from context_runtime.integrations.procurement_sources import (
    FetchResult, FixtureFetcher, ProcurementSource, SourceMethod,
)
from context_runtime.integrations.soak import (
    CollectionRunStore, discovery_report, run_collection, soak_report,
)

_INFORMS_HTML = """<html><body>
<span class='ps_box-value' id='SCP_PUB_AUC_VW_AUC_NAME$0' >E26SP01: Bond Engineering Services</span>
<span class='ps_box-value' id='SCP_PUB_AUC_VW_AUC_ID$0' >E26SP01</span>
<span class='ps_box-value' id='SCP_PUB_AUC_VW_AUC_FORMAT$0' >RFP</span>
</body></html>"""

_DETAIL_V1 = """<html><body>
<span id='SCP_P_AUCDTL_VW_AUC_ID$0' >E26SP01</span>
<span id='SCP_P_AUCDTL_VW_AUC_STATUS$0' >Posted</span>
<span id='SCP_P_AUCDTL_VW_AUC_FORMAT$0' >RFP</span>
<span id='SCP_P_AUCDTL_VW_SCP_END_DATE_CHAR$0' >09/21/2026 02:00 PM EST</span>
</body></html>"""

_FUTURE_JSON = """[{"webPostingCounter":8960,"releaseDate":"9/11/2026 12:00:00 AM",
 "removalDate":"9/25/2026 12:00:00 AM","documentTitle":"Bond Engineering Services",
 "sendFeedBack":"Julie Whiteside","emailAddress":"julie@miamidade.gov","attachmentCount":1}]"""

_INFORMS = ProcurementSource("src-miami-dade-informs", "fl-miami-dade-county",
                             "Miami-Dade County — INFORMS Public Bidding", "https://fixture/informs",
                             SourceMethod.INFORMS)
_FUTURE = ProcurementSource("src-miami-dade-future", "fl-miami-dade-county",
                            "Miami-Dade County — Future Solicitations (forecast)", "https://fixture/future",
                            SourceMethod.MDC_FUTURE)
_PROFILE = BusinessProfile(name="Metro Eng & Tech", service_zip="33180", service_radius_miles=50,
                           services=("engineering", "professional services"))


def _fetcher(detail=_DETAIL_V1, informs_ok=True, future_ok=True) -> FixtureFetcher:
    resp = {}
    if informs_ok:
        resp["https://fixture/informs"] = FetchResult(200, _INFORMS_HTML, "text/html")
    else:
        resp["https://fixture/informs"] = FetchResult(0, "", "error:timeout")
    if future_ok:
        resp["https://fixture/future"] = FetchResult(200, _FUTURE_JSON, "application/json")
    return FixtureFetcher(resp, detail_responses={
        "SCP_COSP_WK_FL_DESCR$0": FetchResult(200, detail, "text/html")})


def _run(tmp_path, fetcher, run_id):
    return run_collection(
        zip_code="33180", radius_miles=50, profile=_PROFILE, fetcher=fetcher, run_id=run_id,
        sources=[_INFORMS, _FUTURE], enrich=True,
        evidence_store=EvidenceStore(tmp_path / "evidence.jsonl"),
        run_store=CollectionRunStore(tmp_path / "runs.jsonl"),
        outbox=tmp_path / "handoffs.jsonl")


def test_first_run_records_new_and_writes_handoffs(tmp_path):
    res = _run(tmp_path, _fetcher(), "run-1")
    assert res.run.new_count >= 2 and res.run.unchanged_count == 0    # INFORMS + forecast are new
    assert res.run.sources_ok == 2 and all(s.ok for s in res.run.sources)
    assert res.run.handoff_count >= 1
    # the INFORMS source enriched a detail page (real deadline reached)
    informs = next(s for s in res.run.sources if s.source_id == "src-miami-dade-informs")
    assert informs.detail_fetches == 1
    # handoffs are queued for the Mission side (decoupled outbox)
    assert (tmp_path / "handoffs.jsonl").read_text().count("\n") == res.run.handoff_count


def test_zero_change_day_is_recorded_and_distinct_from_a_broken_collector(tmp_path):
    _run(tmp_path, _fetcher(), "run-1")                               # seed
    quiet = _run(tmp_path, _fetcher(), "run-2")                       # nothing changed
    assert quiet.run.new_count == 0 and quiet.run.updated_count == 0
    assert quiet.run.unchanged_count >= 2                             # rediscovered, unchanged
    assert quiet.run.sources_ok == 2                                  # …but the sources WERE reached
    assert quiet.run.had_change is False

    # a broken collector: same quiet activity picture, but the source was NOT reached
    broken = _run(tmp_path, _fetcher(informs_ok=False), "run-3")
    informs = next(s for s in broken.run.sources if s.source_id == "src-miami-dade-informs")
    assert informs.ok is False and informs.error                     # reliability failure is recorded
    assert broken.run.sources_ok == 1                                 # distinguishable from the quiet day


def test_amendment_is_updated_not_new(tmp_path):
    _run(tmp_path, _fetcher(), "run-1")
    detail_v2 = _DETAIL_V1.replace("09/21/2026", "09/14/2026")        # deadline moved up
    res = _run(tmp_path, _fetcher(detail=detail_v2), "run-2")
    assert res.run.updated_count == 1 and res.run.new_count == 0


def test_report_separates_reliability_from_activity(tmp_path):
    _run(tmp_path, _fetcher(), "run-1")
    _run(tmp_path, _fetcher(), "run-2")                               # quiet
    _run(tmp_path, _fetcher(informs_ok=False), "run-3")              # INFORMS down, quiet otherwise
    rep = soak_report(CollectionRunStore(tmp_path / "runs.jsonl").runs())
    assert rep["runs"] == 3
    # reliability: INFORMS reached 2 of 3 runs; the forecast source every run
    assert rep["reliability"]["src-miami-dade-informs"]["uptime"] == round(2 / 3, 3)
    assert rep["reliability"]["src-miami-dade-future"]["uptime"] == 1.0
    # activity: real change happened once (run-1); two runs were zero-change
    assert rep["activity"]["runs_with_change"] == 1
    assert rep["activity"]["zero_change_runs"] == 2
    assert rep["activity"]["handoffs_total"] >= 1
    # funnel is separated out and versioned
    assert rep["funnel"]["pursue_total"] >= 1 and rep["qualification_versions"]


def test_run_records_scheduler_and_qualification_version(tmp_path):
    res = run_collection(
        zip_code="33180", radius_miles=50, profile=_PROFILE, fetcher=_fetcher(), run_id="run-1",
        scheduled_at="2026-09-18T12:00:00Z", now="2026-09-18T12:00:07Z",
        sources=[_INFORMS, _FUTURE], enrich=True,
        evidence_store=EvidenceStore(tmp_path / "evidence.jsonl"),
        run_store=CollectionRunStore(tmp_path / "runs.jsonl"))
    assert res.run.scheduled_at == "2026-09-18T12:00:00Z"            # scheduler-health signal recorded
    assert res.run.qualification_version.startswith("qual-rules/v1+profile:")
    # per-source latency captured
    assert all(s.collect_ms >= 0 and s.fetch_ms >= 0 for s in res.run.sources)
    rep = soak_report(CollectionRunStore(tmp_path / "runs.jsonl").runs())
    assert rep["scheduler"]["max_lag_seconds"] == 7                  # 12:00:07 - 12:00:00


def test_source_onboarded_midway_is_scored_on_its_own_coverage(tmp_path):
    # INFORMS present from run-1; the forecast source is "onboarded" only at run-2
    run_collection(zip_code="33180", radius_miles=50, profile=_PROFILE, fetcher=_fetcher(future_ok=True),
                   run_id="run-1", sources=[_INFORMS], enrich=True,
                   evidence_store=EvidenceStore(tmp_path / "e.jsonl"),
                   run_store=CollectionRunStore(tmp_path / "r.jsonl"))
    for rid in ("run-2", "run-3"):
        run_collection(zip_code="33180", radius_miles=50, profile=_PROFILE, fetcher=_fetcher(),
                       run_id=rid, sources=[_INFORMS, _FUTURE], enrich=True,
                       evidence_store=EvidenceStore(tmp_path / "e.jsonl"),
                       run_store=CollectionRunStore(tmp_path / "r.jsonl"))
    rep = soak_report(CollectionRunStore(tmp_path / "r.jsonl").runs())
    # 3 runs total, but the forecast source only had 2 scheduled runs since onboarding → uptime over 2, not 3
    assert rep["reliability"]["src-miami-dade-informs"]["scheduled_runs_since_onboarding"] == 3
    fut = rep["reliability"]["src-miami-dade-future"]
    assert fut["scheduled_runs_since_onboarding"] == 2 and fut["uptime"] == 1.0


def test_discovery_report_measures_posting_to_discovery_latency(tmp_path):
    # forecast posted 2026-09-11; we "discover" it now → posting→discovery latency is measured, not inferred
    ev = EvidenceStore(tmp_path / "evidence.jsonl")
    run_collection(zip_code="33180", radius_miles=50, profile=_PROFILE, fetcher=_fetcher(),
                   run_id="run-1", now="2026-09-18T00:00:00Z", sources=[_INFORMS, _FUTURE], enrich=True,
                   evidence_store=ev, run_store=CollectionRunStore(tmp_path / "runs.jsonl"))
    dr = discovery_report(ev.opportunities())
    assert dr["opportunities"] >= 2
    assert dr["detail_observed"] >= 1 and dr["detail_coverage"] > 0
    # the forecast (posted 2026-09-11, discovered 2026-09-18) contributes a 7-day latency
    assert dr["posting_to_discovery_days"]["measured"] >= 1
    assert dr["posting_to_discovery_days"]["max"] >= 7


def test_first_seen_preserved_last_seen_advances(tmp_path):
    ev_path = tmp_path / "evidence.jsonl"
    _run(tmp_path, _fetcher(), "run-1")
    run_collection(zip_code="33180", radius_miles=50, profile=_PROFILE, fetcher=_fetcher(),
                   run_id="run-2", now="2026-09-19T00:00:00Z", sources=[_INFORMS, _FUTURE], enrich=True,
                   evidence_store=EvidenceStore(ev_path),
                   run_store=CollectionRunStore(tmp_path / "runs.jsonl"))
    opps = {o["opportunity_id"]: o for o in EvidenceStore(ev_path).opportunities()}
    bond = opps["src-miami-dade-informs:E26SP01"]
    assert bond["first_seen_at"] < bond["last_seen_at"]              # first_seen kept, last_seen advanced
