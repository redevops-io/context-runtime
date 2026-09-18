"""Local-Government Opportunity Discovery — offline end-to-end, on real geography.

Turns a business ZIP + service radius into the real public entities to monitor, then qualifies real-
shaped solicitations against a specific business with evidence — the plan's *"3 opportunities worth
reviewing today, not 137 new bids."*

Everything geographic is REAL and computed on the deterministic geospatial engine (real jurisdictions,
real centroids, haversine distance, polygon containment). The solicitation *content* below is sample
fixture data standing in for whatever collector (portal scraper / SAM.gov / RSS) produced the
Observation — collection is a separate, pluggable concern in the plan; qualification is what this proves.

    python examples/local_gov_discovery.py
"""
from __future__ import annotations

from context_runtime.integrations.local_gov import (
    BusinessProfile, Decision, RevenueOpportunity, SolicitationType,
    default_registry, has_valid_next_state, qualify,
)

# The owner: a mechanical/HVAC contractor near ZIP 33180 (Aventura, FL), serving a 50-mile area.
PROFILE = BusinessProfile(
    name="Coastal Mechanical Services",
    service_zip="33180",
    service_radius_miles=50,
    services=("hvac", "mechanical", "chillers", "building maintenance"),
    naics=("238220",),                 # Plumbing, Heating, and Air-Conditioning Contractors
    licenses=("mechanical contractor",),
    certifications=("EPA 608",),
    min_contract_value=5_000,
    max_contract_value=2_000_000,
)

# Sample solicitations (fixture CONTENT; the issuing jurisdictions are real registry entries).
SOLICITATIONS = [
    RevenueOpportunity(
        opportunity_id="AVE-2026-014", source="city portal (sample)", issuing_entity="City of Aventura",
        jurisdiction_id="fl-aventura", title="HVAC replacement — community center (12 RTUs)",
        solicitation_type=SolicitationType.ITB, place_of_performance_zip="33180",
        categories=("hvac", "mechanical"), response_due_at="2026-10-02", estimated_value=180_000,
        source_url="https://www.cityofaventura.com/bids/AVE-2026-014"),
    RevenueOpportunity(
        opportunity_id="MDC-2026-2231", source="county portal (sample)", issuing_entity="Miami-Dade County",
        jurisdiction_id="fl-miami-dade-county", title="Chiller preventive maintenance — county facilities",
        solicitation_type=SolicitationType.RFP, place_of_performance_zip="33180",
        categories=("hvac", "chillers", "building maintenance"), response_due_at="2026-10-20",
        estimated_value=450_000, required_certifications=("DBE",),
        source_url="https://www.miamidade.gov/procurement/MDC-2026-2231"),
    RevenueOpportunity(
        opportunity_id="MDC-2026-2240", source="county portal (sample)", issuing_entity="Miami-Dade County",
        jurisdiction_id="fl-miami-dade-county", title="Emergency generator install (electrical)",
        solicitation_type=SolicitationType.ITB, place_of_performance_zip="33180",
        categories=("electrical",), required_licenses=("electrical contractor",),
        response_due_at="2026-10-09", estimated_value=95_000),
    RevenueOpportunity(
        opportunity_id="FTL-2026-051", source="city portal (sample)", issuing_entity="City of Fort Lauderdale",
        jurisdiction_id="fl-fort-lauderdale", title="HVAC maintenance — parks buildings",
        solicitation_type=SolicitationType.RFQ, place_of_performance_zip="33009",
        categories=("hvac", "building maintenance"), response_due_at="2026-11-01", estimated_value=60_000),
    RevenueOpportunity(
        opportunity_id="MDCPS-2026-77", source="school district portal (sample)",
        issuing_entity="Miami-Dade County Public Schools", jurisdiction_id="fl-mdcps",
        title="District-wide HVAC controls upgrade", solicitation_type=SolicitationType.RFP,
        place_of_performance_zip="33180", categories=("hvac", "controls"),
        response_due_at="2026-12-15", estimated_value=6_500_000),   # above max contract size → REVIEW
    RevenueOpportunity(
        opportunity_id="KW-2026-003", source="city portal (sample)", issuing_entity="City of Key West",
        jurisdiction_id="fl-key-west", title="HVAC service — city hall", solicitation_type=SolicitationType.RFQ,
        place_of_performance_zip="33180", categories=("hvac",), estimated_value=40_000),  # out of area → REJECT
]


def main() -> int:
    reg = default_registry()

    print(f"Owner: {PROFILE.name} · ZIP {PROFILE.service_zip} · {PROFILE.service_radius_miles:g}-mi service area\n")

    monitored = reg.within(PROFILE.service_zip, PROFILE.service_radius_miles)
    print(f"Monitored public entities within {PROFILE.service_radius_miles:g} mi of {PROFILE.service_zip}: {len(monitored)}")
    for mj in monitored:
        why = mj.detail
        print(f"  • {mj.jurisdiction.name:<38} [{mj.jurisdiction.kind.value:<15}] {mj.reason.value:<13} — {why}")

    print(f"\nObserved {len(SOLICITATIONS)} solicitations · qualifying against {PROFILE.name}:\n")
    buckets: dict[Decision, list] = {Decision.PURSUE: [], Decision.REVIEW: [], Decision.REJECT: []}
    for opp in SOLICITATIONS:
        q = qualify(opp, PROFILE, reg)
        buckets[q.decision].append((opp, q))
        marker = {"PURSUE": "✓", "REVIEW": "?", "REJECT": "✗"}[q.decision.value]
        print(f"  [{marker} {q.decision.value:<6}] {opp.issuing_entity} — {opp.title}")
        for r in q.reasons:
            print(f"            · {r}")

    p, r, x = len(buckets[Decision.PURSUE]), len(buckets[Decision.REVIEW]), len(buckets[Decision.REJECT])
    print(f"\nSummary: monitored {len(monitored)} entities · observed {len(SOLICITATIONS)} · "
          f"{p} PURSUE / {r} REVIEW / {x} REJECT")
    print(f"\n  \"A business at ZIP {PROFILE.service_zip} would be alerted to {p + r} public-sector opportunities "
          f"worth reviewing this cycle;\n   {p} passed qualification, {r} need a human look, and {x} were rejected with explicit reasons.\"")

    # Every PURSUE/REVIEW opportunity that becomes a Revenue Mission must satisfy the core invariant
    # before it counts as handled (owner + a next action / waiting condition / terminal disposition).
    actionable = buckets[Decision.PURSUE] + buckets[Decision.REVIEW]
    ok = all(has_valid_next_state(owner=PROFILE.name, next_action=f"owner review of {opp.opportunity_id}")
             for opp, _ in actionable)
    print(f"\n  invariant — every actionable opportunity has an owner + next action: {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
