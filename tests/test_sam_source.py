"""SAM.gov (source #3) — a FEDERAL, region-agnostic source measured on its own terms.

Geography for federal notices is place-of-performance, not local-jurisdiction membership; the API key is
a secret sourced from the environment and never stored in the registry or evidence. Offline throughout —
the live path activates when SAM_API_KEY is set.
"""
from __future__ import annotations

from context_runtime.integrations.local_gov import (
    BusinessProfile, Decision, RevenueOpportunity, SolicitationType, default_registry, qualify,
)
from context_runtime.integrations.procurement_sources import (
    FetchResult, FixtureFetcher, ProcurementSource, SourceMethod, parse_sam, resolve_fetch_url, run_region,
)

# a realistic SAM.gov Opportunities v2 response (two notices: one FL, one CA)
_SAM_JSON = """{"totalRecords":2,"opportunitiesData":[
 {"noticeId":"abc123","title":"A-E Services for Coastal Resiliency","solicitationNumber":"W912EP26R0001",
  "fullParentPathName":"DEPT OF DEFENSE.DEPT OF THE ARMY.USACE","type":"Solicitation",
  "postedDate":"2026-09-15","responseDeadLine":"2026-10-15T14:00:00-04:00","naicsCode":"541330",
  "typeOfSetAsideDescription":"Total Small Business Set-Aside","uiLink":"https://sam.gov/opp/abc123/view",
  "pointOfContact":[{"fullName":"Dana Lee","email":"dana.lee@usace.army.mil"}],
  "placeOfPerformance":{"state":{"code":"FL","name":"Florida"},"city":{"name":"Miami"}}},
 {"noticeId":"def456","title":"Engineering Design Services, Sacramento District","naicsCode":"541330",
  "postedDate":"2026-09-16","responseDeadLine":"2026-10-20","type":"Solicitation",
  "uiLink":"https://sam.gov/opp/def456/view",
  "placeOfPerformance":{"state":{"code":"CA","name":"California"}}}
]}"""

_PROFILE = BusinessProfile(name="Metro Eng & Tech", service_zip="33180", service_radius_miles=50,
                           services=("engineering", "professional services"), naics=("541330",))


def test_parse_sam_maps_federal_fields():
    rows = parse_sam(_SAM_JSON)
    assert len(rows) == 2 and rows[0]["record_kind"] == "sam"
    fl = rows[0]
    assert fl["event_id"] == "abc123" and fl["naics"] == "541330"
    assert fl["response_due_at"] == "2026-10-15" and fl["posted_at"] == "2026-09-15"
    assert fl["place_of_performance_state"] == "FL"
    assert fl["department"] == "USACE" and fl["contact"].startswith("Dana Lee")
    assert fl["set_aside"] == "Total Small Business Set-Aside"


def test_qualify_federal_uses_place_of_performance():
    reg = default_registry()
    in_state = RevenueOpportunity("sam:1", "src-sam", "SAM.gov", "*", "A-E Services",
                                  SolicitationType.BID, "", categories=("541330",),
                                  place_of_performance_state="FL", response_due_at="2026-10-15")
    out_state = RevenueOpportunity("sam:2", "src-sam", "SAM.gov", "*", "Design Services",
                                   SolicitationType.BID, "", categories=("541330",),
                                   place_of_performance_state="CA", response_due_at="2026-10-20")
    q_in = qualify(in_state, _PROFILE, reg)
    q_out = qualify(out_state, _PROFILE, reg)
    assert q_in.decision is Decision.PURSUE                     # FL place of performance + NAICS match
    assert any("federal opportunity" in r for r in q_in.reasons)
    assert q_out.decision is Decision.REJECT                    # CA is out of the business's state
    assert any("outside your state" in r for r in q_out.reasons)


def test_resolve_fetch_url_requires_env_key(monkeypatch):
    sam = ProcurementSource("src-sam", "*", "SAM.gov federal opportunities",
                            "https://api.sam.gov/opportunities/v2/search", SourceMethod.SAM_API)
    monkeypatch.delenv("SAM_API_KEY", raising=False)
    url, err = resolve_fetch_url(sam, profile=_PROFILE, now="2026-09-18T00:00:00Z")
    assert url is None and err == "SAM_API_KEY not set"          # missing key → recorded failure, not a crash

    monkeypatch.setenv("SAM_API_KEY", "test-key-123")
    url, err = resolve_fetch_url(sam, profile=_PROFILE, now="2026-09-18T00:00:00Z")
    assert err == "" and "api_key=test-key-123" in url
    assert "ncode=541330" in url and "postedTo=09%2F18%2F2026" in url


def test_non_sam_url_is_unchanged():
    src = ProcurementSource("s", "fl-x", "X", "https://fixture/x.rss", SourceMethod.RSS)
    url, err = resolve_fetch_url(src, profile=_PROFILE)
    assert url == "https://fixture/x.rss" and err == ""


def test_sam_run_region_yields_federal_handoff():
    sam = ProcurementSource("src-sam", "*", "SAM.gov federal opportunities",
                            "https://fixture/sam", SourceMethod.SAM_API)
    fetcher = FixtureFetcher({"https://fixture/sam": FetchResult(200, _SAM_JSON, "application/json")})
    res = run_region("33180", 50, _PROFILE, fetcher=fetcher, sources=[sam])
    assert len(res.observations) == 2                            # both federal notices preserved
    decisions = {h["opportunity_id"]: h["qualification"]["decision"] for h in res.handoffs}
    # the FL notice qualifies (PURSUE); the CA notice is rejected by place-of-performance → no handoff
    assert decisions.get("src-sam:abc123") == "PURSUE"
    assert "src-sam:def456" not in decisions
    fl = next(h for h in res.handoffs if h["opportunity_id"] == "src-sam:abc123")
    assert fl["opportunity_kind"] == "gov_solicitation" and fl["place_of_performance_state"] == "FL"
    assert fl["evidence_ids"][0].startswith("obs:src-sam:")
