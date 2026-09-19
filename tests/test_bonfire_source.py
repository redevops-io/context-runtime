"""Bonfire (bonfirehub.com) open-opportunities portal (source #5) — a browser-rendered source.

Broward County (which blocks plain HTTP with a 403) publishes via Bonfire, a platform many agencies use,
so one parser serves them all. The fixture mirrors the REAL rendered Bonfire DOM captured from Broward's
live portal (Status / Ref# / Project / Department / Close Date / Days Left / a link to /opportunities/N),
including the underscore.js template row that must be skipped.
"""
from __future__ import annotations

from context_runtime.integrations.local_gov import BusinessProfile
from context_runtime.integrations.procurement_sources import (
    BrowserFetcher, FetchResult, FixtureFetcher, ProcurementSource, SourceMethod, _bonfire_date_to_iso,
    parse_bonfire, run_region,
)

# faithful to the real Broward Bonfire DOM: a template row (skipped) + two real data rows
_BONFIRE_HTML = """<table id="opportunities"><thead><tr>
<th>Status</th><th>Ref. #</th><th>Project</th><th>Department</th><th>Close Date</th><th>Days Left</th><th>Action</th>
</tr></thead><tbody>
<tr><td><div class="statusTag <%- auctionStatusData.statusClass %>"><%- auctionStatusData.statusLabel %></div></td>
<td><%- auction.Title %></td><td><%- auction.Description %></td><td><%- departments %></td>
<td><%- auction.EndDate %></td><td></td><td><a href="/opportunities/<%- auction.ProjectID %>">View</a></td></tr>
<tr><td data-order="1790013600" class="sorting_1"><div class="statusTag statusOpen"> Open </div></td>
<td>GEN2132859B1</td><td id="projectNameId-GEN2132859B1"><b>Removal and Transportation of Deceased Persons</b></td>
<td>FASD - Purchasing</td><td>Sep 21st 2026, 2:00 PM EDT</td>
<td><div class="alignCenter badgeDaysLeft warning">3</div></td>
<td><a class="btn" href="/opportunities/249240"> View Opportunity </a></td></tr>
<tr><td class="sorting_1"><div class="statusTag statusOpen"> Open </div></td>
<td>ENG2026-14</td><td id="projectNameId-ENG2026-14"><b>Engineering Design Services for Bridge Rehabilitation</b></td>
<td>Highway Construction &amp; Engineering</td><td>Oct 15th 2026, 2:00 PM EDT</td>
<td><div class="badgeDaysLeft">27</div></td>
<td><a class="btn" href="/opportunities/251880"> View Opportunity </a></td></tr>
</tbody></table>"""

_PROFILE = BusinessProfile(name="Metro Eng & Tech", service_zip="33180", service_radius_miles=50,
                           services=("engineering", "professional services"))


def test_bonfire_date_parsing():
    assert _bonfire_date_to_iso("Sep 21st 2026, 2:00 PM EDT") == "2026-09-21"
    assert _bonfire_date_to_iso("Oct 3rd 2026") == "2026-10-03"
    assert _bonfire_date_to_iso("March 1 2027") == "2027-03-01"
    assert _bonfire_date_to_iso("no date here") == ""                 # never inferred


def test_parse_bonfire_skips_template_and_maps_rows():
    rows = parse_bonfire(_BONFIRE_HTML)
    assert len(rows) == 2                                             # the <%- template row is skipped
    a, b = rows
    assert a["event_id"] == "GEN2132859B1" and a["status"] == "Open"
    assert a["title"].startswith("Removal and Transportation")
    assert a["department"] == "FASD - Purchasing"
    assert a["response_due_at"] == "2026-09-21" and a["link"] == "/opportunities/249240"
    assert b["event_id"] == "ENG2026-14" and b["response_due_at"] == "2026-10-15"


def test_bonfire_run_region_qualifies_broward():
    src = ProcurementSource("src-broward-bonfire", "fl-broward-county", "Broward County — Purchasing (Bonfire)",
                            "https://fixture/broward/portal/?tab=openOpportunities", SourceMethod.BONFIRE)
    fetcher = FixtureFetcher({"https://fixture/broward/portal/?tab=openOpportunities":
                              FetchResult(200, _BONFIRE_HTML, "text/html")})
    res = run_region("33180", 50, _PROFILE, fetcher=fetcher, sources=[src])
    assert len(res.observations) == 2                                 # both Broward opportunities preserved
    eng = next((h for h in res.handoffs if "Engineering" in h["title"]), None)
    assert eng is not None                                            # Broward is within the 50-mi radius
    assert eng["response_due_at"] == "2026-10-15"
    assert eng["source_url"].startswith("https://fixture/") and "opportunities/251880" in eng["source_url"]
    assert eng["evidence_ids"][0].startswith("obs:src-broward-bonfire:")


def test_browser_fetcher_without_chrome_reports_error_not_crash():
    bf = BrowserFetcher()
    bf.chrome = None                                                  # simulate no browser installed
    r = bf.fetch("https://example.gov/whatever")                      # returns before any network call
    assert r.status == 0 and "no-chrome" in r.content_type            # recorded failure, not an exception
