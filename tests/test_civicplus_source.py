"""CivicPlus municipal Bids module (source #4) — server-rendered HTML, the platform many FL cities run.

The fixture below mirrors the REAL CivicPlus markup (captured from a live city with active bids): a
``listItemsRow bid`` block with a title link to bids.aspx?bidID=N, an optional "Bid No.", and a status
block. A non-date closing ("Upon Contract") yields no deadline — we never infer one — and an empty list
is a legitimate "no open bids", distinguishable from a broken collector (which would fail to fetch).
"""
from __future__ import annotations

from context_runtime.integrations.local_gov import BusinessProfile
from context_runtime.integrations.procurement_sources import (
    FetchResult, FixtureFetcher, ProcurementSource, SourceMethod, parse_civicplus, run_region,
)

# faithful to real CivicPlus markup: bid 1 closes "Upon Contract" (no date), bid 2 has a real closing date
_CIVICPLUS_HTML = """<html><body>
<div class="bidItems listItems"><div class="bidsHeader listHeader"><span>Bids</span><span>2 Bids</span></div>
<div class="listItemsRow bid"><div class="bidTitle"><span><a href="bids.aspx?bidID=153">Parking Management Services</a></span>
<br><span style="font-size:0.75em;"><strong>Bid No.</strong> 2026-05</span>
<br><span>The purpose of this RFP is to select a qualified firm... [<a href="bids.aspx?bidID=153">Read&nbsp;on</a>]</span></div>
<div class="bidStatus"><div><span id="BidStatus1">Status:</span><br><span id="BidCloses1">Closes:</span></div>
<div><span>Open</span><br><span>Upon Contract</span></div></div></div>
<div class="listItemsRow bid alt"><div class="bidTitle"><span><a href="bids.aspx?bidID=200">Engineering Design Services for Roadway</a></span>
<br><span style="font-size:0.75em;"><strong>Bid No.</strong> 2026-20</span>
<br><span>Professional engineering services... [<a href="bids.aspx?bidID=200">Read&nbsp;on</a>]</span></div>
<div class="bidStatus"><div><span id="BidStatus2">Status:</span><br><span id="BidCloses2">Closes:</span></div>
<div><span>Open</span><br><span>12/15/2026 2:00 PM</span></div></div></div>
</div></body></html>"""

_EMPTY_HTML = """<html><body><div class="bidItems listItems"><span class="BidDetail">
There are no open bids at this time.</span></div></body></html>"""

_PROFILE = BusinessProfile(name="Metro Eng & Tech", service_zip="33180", service_radius_miles=50,
                           services=("engineering", "professional services"))


def test_parse_civicplus_real_markup():
    rows = parse_civicplus(_CIVICPLUS_HTML)
    assert len(rows) == 2 and all(r["record_kind"] == "civicplus" for r in rows)
    a, b = rows
    assert a["event_id"] == "2026-05" and a["title"] == "Parking Management Services"
    assert a["status"] == "Open" and a["response_due_at"] == ""        # "Upon Contract" → no inferred date
    assert a["link"] == "bids.aspx?bidID=153"
    assert b["event_id"] == "2026-20" and b["response_due_at"] == "2026-12-15"   # real closing date parsed


def test_parse_civicplus_empty_is_no_open_bids_not_error():
    assert parse_civicplus(_EMPTY_HTML) == []                          # legitimately zero, not a crash


def test_robots_honors_user_agent_groups():
    # CivicPlus robots.txt blanket-blocks Baiduspider/Yandex; that must NOT block our compliant bot
    from context_runtime.integrations.procurement_sources import _robots_disallows
    robots = ("User-agent: Baiduspider\nDisallow: /\n"
              "User-agent: Yandex\nDisallow: /\n"
              "User-agent: *\nDisallow: /admin\nDisallow: /RSS.aspx\n")
    dis = _robots_disallows(robots, "ReDevOps-ProcurementBot/0.1 (+https://redevops.io)")
    assert "/admin" in dis and "/RSS.aspx" in dis and "/" not in dis   # only the * group applies to us
    assert not any(d and "/bids.aspx".startswith(d) for d in dis)      # bids allowed
    assert any(d and "/admin/x".startswith(d) for d in dis)            # admin still disallowed


def test_civicplus_run_region_qualifies_and_resolves_links():
    src = ProcurementSource("src-aventura", "fl-aventura", "City of Aventura — Bids",
                            "https://fixture/aventura/bids.aspx", SourceMethod.CIVICPLUS)
    fetcher = FixtureFetcher({"https://fixture/aventura/bids.aspx": FetchResult(200, _CIVICPLUS_HTML, "text/html")})
    res = run_region("33180", 50, _PROFILE, fetcher=fetcher, sources=[src])
    assert len(res.observations) == 2                                  # both bids preserved as evidence
    # Aventura contains ZIP 33180 → in-area; the professional/engineering services bids qualify
    assert res.handoffs and all(h["place_of_performance_zip"] == "33180" for h in res.handoffs)
    eng = next((h for h in res.handoffs if "Engineering" in h["title"]), None)
    assert eng is not None and eng["response_due_at"] == "2026-12-15"
    # the relative bid link was resolved to an absolute URL against the source
    assert eng["source_url"].startswith("https://fixture/aventura/") and "bidID=200" in eng["source_url"]
    assert eng["evidence_ids"][0].startswith("obs:src-aventura:")
