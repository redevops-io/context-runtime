"""Procurement collectors — fetch → parse → preserve observations → normalize → qualify → handoff,
offline with a FixtureFetcher (no live sites hit)."""
from __future__ import annotations

from context_runtime.integrations.local_gov import BusinessProfile
from context_runtime.integrations.procurement_sources import (
    FetchResult, FixtureFetcher, ProcurementSource, SourceMethod, UrllibFetcher,
    collect, parse_json_items, parse_rss, run_region,
)

_PROFILE = BusinessProfile(
    name="Coastal Mechanical", service_zip="33180", service_radius_miles=50,
    services=("hvac", "mechanical", "chillers"), naics=("238220",),
    licenses=("mechanical contractor",), certifications=("EPA 608",),
    min_contract_value=5_000, max_contract_value=2_000_000)

_RSS = """<?xml version="1.0"?><rss><channel>
  <item><title>ITB — HVAC replacement, community center</title>
        <description>Replace 12 rooftop units.</description>
        <link>https://example.gov/ave/itb-014</link><pubDate>2026-10-02</pubDate></item>
</channel></rss>"""

_JSON = """{"items": [
  {"title": "RFP Chiller preventive maintenance", "description": "County facilities.",
   "link": "https://example.gov/mdc/rfp-2231", "response_due_at": "2026-10-20"}
]}"""

_AVE = ProcurementSource("src-aventura", "fl-aventura", "City of Aventura — Bids",
                         "https://fixture/aventura.rss", SourceMethod.RSS)
_MDC = ProcurementSource("src-miami-dade", "fl-miami-dade-county", "Miami-Dade County — Procurement",
                         "https://fixture/mdc.json", SourceMethod.JSON)


def _fetcher() -> FixtureFetcher:
    return FixtureFetcher({
        "https://fixture/aventura.rss": FetchResult(200, _RSS, "application/rss+xml"),
        "https://fixture/mdc.json": FetchResult(200, _JSON, "application/json"),
    })


def test_parsers():
    assert parse_rss(_RSS)[0]["title"].startswith("ITB")
    assert parse_json_items(_JSON)[0]["title"].startswith("RFP")
    assert parse_rss("not xml") == [] and parse_json_items("not json") == []


def test_collect_preserves_observations_with_hashes():
    obs = collect([_AVE, _MDC], _fetcher())
    assert len(obs) == 2
    for o in obs:
        assert o.status == 200 and o.content_hash.startswith("sha256:") and o.raw   # evidence preserved
    # deterministic content hashes (same record → same hash)
    assert collect([_AVE], _fetcher())[0].content_hash == obs[0].content_hash


def test_run_region_end_to_end_offline():
    res = run_region("33180", 50, _PROFILE, fetcher=_fetcher(), sources=[_AVE, _MDC])
    assert "fl-aventura" in res.monitored_jurisdiction_ids            # geospatial resolved the region
    assert len(res.observations) == 2                                 # every observation preserved
    assert res.handoffs                                              # HVAC in-area solicitations qualified
    h = res.handoffs[0]
    assert h["contract_version"] == "revenue-handoff/v1"
    assert h["qualification"]["decision"] in ("PURSUE", "REVIEW")
    assert h["evidence_ids"] and h["source_url"].startswith("https://")


def test_missing_source_yields_nothing_not_fabrication():
    # a fetcher that returns 404 for everything → no observations, no invented opportunities
    assert collect([_AVE, _MDC], FixtureFetcher({})) == []


def test_urllib_fetcher_is_constructible_offline():
    # constructing the live fetcher must not touch the network; it only fetches when .fetch is called
    f = UrllibFetcher(min_interval_s=0.0)
    assert f.user_agent.startswith("ReDevOps")


# ── INFORMS (Miami-Dade PeopleSoft public bidding) portal extractor ──
_INFORMS_HTML = """<html><body>
<span class='ps_box-value' id='SCP_PUB_AUC_VW_AUC_NAME$0' >Cisco Software,Hardware,Maintenance, Profess Svcs</span>
<span class='ps_box-value' id='SCP_PUB_AUC_VW_AUC_ID$0' >EVN0052970</span>
<span class='ps_box-value' id='SCP_PUB_AUC_VW_AUC_FORMAT$0' >RFI</span>
<span class='ps_box-value' id='SCP_PUB_AUC_VW_AUC_NAME$1' >E26SP01: Bond Engineering Services</span>
<span class='ps_box-value' id='SCP_PUB_AUC_VW_AUC_ID$1' >E26SP01</span>
<span class='ps_box-value' id='SCP_PUB_AUC_VW_AUC_FORMAT$1' >RFI</span>
<span class='ps_box-value' id='SCP_PUB_AUC_VW_AUC_NAME$2' >Retirement Awards for MDC PROS</span>
<span class='ps_box-value' id='SCP_PUB_AUC_VW_AUC_ID$2' >EVN0061099</span>
</body></html>"""


def test_parse_informs_extracts_rows():
    from context_runtime.integrations.procurement_sources import parse_informs, tag_categories
    rows = parse_informs(_INFORMS_HTML)
    assert len(rows) == 3
    assert rows[0]["event_id"] == "EVN0052970" and rows[0]["title"].startswith("Cisco")
    assert "engineering" in tag_categories("E26SP01: Bond Engineering Services")
    assert "it" in tag_categories("Cisco Software,Hardware")


def test_informs_run_region_qualifies_real_shape():
    from context_runtime.integrations.procurement_sources import ProcurementSource, SourceMethod
    src = ProcurementSource("src-miami-dade-informs", "fl-miami-dade-county",
                            "Miami-Dade County — INFORMS Public Bidding", "https://fixture/informs",
                            SourceMethod.INFORMS)
    fetcher = FixtureFetcher({"https://fixture/informs": FetchResult(200, _INFORMS_HTML, "text/html")})
    profile = BusinessProfile(name="Metro Eng & Tech", service_zip="33180", service_radius_miles=50,
                              services=("engineering", "it", "professional services"))
    res = run_region("33180", 50, profile, fetcher=fetcher, sources=[src])
    assert len(res.observations) == 3                              # every INFORMS row preserved
    decisions = [h["qualification"]["decision"] for h in res.handoffs]
    assert "PURSUE" in decisions                                   # Cisco/Bond match an eng/IT firm, in-county
    # every handoff carries a Miami-Dade observation reference (replayable evidence)
    assert all(any(e.startswith("obs:src-miami-dade-informs") for e in h["evidence_ids"]) for h in res.handoffs)
