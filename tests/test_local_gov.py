"""Local-Government Opportunity Discovery — geography + qualification, on real data."""
from __future__ import annotations

from context_runtime.integrations.local_gov import (
    BusinessProfile, Decision, InclusionReason, RevenueOpportunity, SolicitationType,
    default_registry, has_valid_next_state, haversine_miles, qualify,
)


def test_haversine_miles_matches_real_distances():
    reg = default_registry()
    z = reg.zip_area("33180").centroid
    sunny = next(j for j in reg.jurisdictions() if j.id == "fl-sunny-isles-beach").centroid
    keywest = next(j for j in reg.jurisdictions() if j.id == "fl-key-west").centroid
    # ~2 mi to Sunny Isles Beach; ~140 mi to Key West — real geography, generous tolerances.
    assert 1.0 < haversine_miles(z, sunny) < 4.0
    assert 120 < haversine_miles(z, keywest) < 160
    assert haversine_miles(z, z) == 0.0


def test_within_includes_local_and_excludes_far():
    reg = default_registry()
    ids = {m.jurisdiction.id: m for m in reg.within("33180", 50)}
    # local jurisdictions present
    for jid in ("fl-aventura", "fl-miami-dade-county", "fl-sunny-isles-beach", "fl-fort-lauderdale"):
        assert jid in ids, jid
    # far ones filtered out — proves the geography filters, not just includes everything
    assert "fl-key-west" not in ids
    assert "fl-orlando" not in ids


def test_containment_runs_through_the_geometry_engine():
    reg = default_registry()
    ids = {m.jurisdiction.id: m for m in reg.within("33180", 50)}
    # Aventura carries a real boundary polygon → containment via point_in_polygon, not distance.
    assert ids["fl-aventura"].reason is InclusionReason.CONTAINS
    assert "boundary" in ids["fl-aventura"].detail
    # the ZIP's county contains it even though the county area centroid is far away
    assert ids["fl-miami-dade-county"].reason is InclusionReason.CONTAINS


def test_within_is_deterministic_and_ordered():
    reg = default_registry()
    a = [m.jurisdiction.id for m in reg.within("33180", 50)]
    b = [m.jurisdiction.id for m in reg.within("33180", 50)]
    assert a == b                                   # deterministic
    # containers sort before proximity matches
    contains = [m for m in reg.within("33180", 50) if m.reason is InclusionReason.CONTAINS]
    assert a[: len(contains)] == [m.jurisdiction.id for m in contains]


_PROFILE = BusinessProfile(
    name="Coastal Mechanical", service_zip="33180", service_radius_miles=50,
    services=("hvac", "mechanical", "chillers"), naics=("238220",),
    licenses=("mechanical contractor",), certifications=("EPA 608",),
    min_contract_value=5_000, max_contract_value=2_000_000,
)


def _opp(**kw) -> RevenueOpportunity:
    base = dict(opportunity_id="X", source="s", issuing_entity="e", jurisdiction_id="fl-aventura",
                title="t", solicitation_type=SolicitationType.RFP, place_of_performance_zip="33180",
                categories=("hvac",))
    base.update(kw)
    return RevenueOpportunity(**base)


def test_qualify_pursue_on_strong_match():
    q = qualify(_opp(categories=("hvac", "mechanical"), response_due_at="2026-10-02"), _PROFILE, default_registry())
    assert q.decision is Decision.PURSUE
    assert q.service_match == 1.0


def test_qualify_reject_missing_license():
    q = qualify(_opp(categories=("electrical",), required_licenses=("electrical contractor",),
                     jurisdiction_id="fl-miami-dade-county"), _PROFILE, default_registry())
    assert q.decision is Decision.REJECT
    assert any("electrical contractor license not present" in r for r in q.reasons)


def test_qualify_reject_out_of_area():
    q = qualify(_opp(jurisdiction_id="fl-key-west"), _PROFILE, default_registry())
    assert q.decision is Decision.REJECT
    assert any("outside the 50-mi service area" in r for r in q.reasons)


def test_qualify_review_unknown_certification():
    q = qualify(_opp(categories=("hvac", "chillers"), required_certifications=("DBE",),
                     jurisdiction_id="fl-miami-dade-county"), _PROFILE, default_registry())
    assert q.decision is Decision.REVIEW
    assert any("DBE" in r for r in q.reasons)


def test_qualify_review_contract_too_large():
    q = qualify(_opp(estimated_value=6_500_000), _PROFILE, default_registry())
    assert q.decision is Decision.REVIEW
    assert any("above max contract size" in r for r in q.reasons)


def test_core_invariant():
    assert has_valid_next_state(owner="o", next_action="call back")
    assert has_valid_next_state(owner="o", waiting_condition="awaiting reply")
    assert has_valid_next_state(owner="o", terminal_disposition="WON")
    assert not has_valid_next_state(owner="o")               # owner but no action/waiting/terminal
    assert not has_valid_next_state(owner="", next_action="x")  # action but no owner
