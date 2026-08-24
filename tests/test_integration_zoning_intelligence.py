"""Zoning Intelligence tenant — the geospatial reference benchmark (offline, deterministic).

Asserts the plan's load-bearing properties: deterministic-first constraints, reconciliation that makes
false-permitted structurally impossible (the blocking SLO), the learned minimal-evidence policy per
use-difficulty bucket, that it beats both baselines, dependency-scoped incremental recomputation, and
population-governance drift detection.
"""
from __future__ import annotations

from context_runtime.geospatial.contracts import UseDisposition as D
from context_runtime.integrations.zoning_intelligence import (
    DEFAULT_ARMS, EvidencePlan, PopulationGovernor, ZoningIntelligenceTenant, affected_conclusions,
    build_providers, build_reference_world, gold_disposition, resolve_disposition, reward_zoning,
    zoning_bucket, _zoning_bandit,
)
from examples.zoning_intelligence import QUERIES, _ARM_BY_KEY, _eval


def _world():
    world, apn = build_reference_world()
    return world, apn, build_providers(world)


# ── vocabulary / arms / reward ──

def test_bucketing():
    assert zoning_bucket("RESIDENTIAL_SINGLE_FAMILY") == "residential"
    assert zoning_bucket("OFFICE") == "commercial" and zoning_bucket("RETAIL") == "commercial"
    assert zoning_bucket("WAREHOUSE") == zoning_bucket("DATA_CENTER") == "industrial"


def test_arm_key_and_cost_ordering():
    _, _, prov = _world()
    assert DEFAULT_ARMS[0].key == "regrid"
    assert DEFAULT_ARMS[-1].key == "regrid+attom+municipal_gis+ordinance"
    costs = [a.cost_units(prov) for a in DEFAULT_ARMS]
    assert costs == sorted(costs) and costs[0] < costs[-1]     # escalating bundles cost strictly more


def test_reward_penalizes_false_permit_and_rewards_cheaper():
    cheap, dear = DEFAULT_ARMS[1], DEFAULT_ARMS[-1]
    _, _, prov = _world()
    assert reward_zoning(correct=False, false_permitted=True, cost_units=1.0) == -1.0    # the SLO penalty
    assert reward_zoning(correct=False, false_permitted=False, cost_units=1.0) == 0.0    # a plain miss
    r_cheap = reward_zoning(correct=True, false_permitted=False, cost_units=cheap.cost_units(prov))
    r_dear = reward_zoning(correct=True, false_permitted=False, cost_units=dear.cost_units(prov))
    assert r_cheap > r_dear > 0.0                                                        # cheaper-correct wins


# ── deterministic-first + fail-safe resolver ──

def test_deterministic_constraint_precedes_interpretation():
    """A data center on an undersized lot (D-400) is PROHIBITED by the structured lot-area constraint even
    with the full evidence set — geometry before any ordinance/LLM interpretation (plan §4)."""
    world, apn, prov = _world()
    a = resolve_disposition(world, prov, apn["D-400"], "DATA_CENTER",
                            ("regrid", "attom", "municipal_gis", "ordinance"))
    assert a.disposition == D.PROHIBITED
    assert a.constraints and a.constraints[0].type.value == "MIN_LOT_AREA"


def test_single_provider_can_false_permit_but_reconciliation_cannot():
    """R-102's true base is C-2 but Regrid is stale (R-1). Regrid-only concludes a residence is PERMITTED
    where it is actually PROHIBITED — a false-permit. Adding the independent ATTOM source (reconciliation)
    turns it into a safe UNKNOWN. This is the blocking SLO in action (plan §16)."""
    world, apn, prov = _world()
    gold = gold_disposition(world, prov, apn["R-102"], "RESIDENTIAL_SINGLE_FAMILY")
    assert gold == D.PROHIBITED
    solo = resolve_disposition(world, prov, apn["R-102"], "RESIDENTIAL_SINGLE_FAMILY", ("regrid",))
    assert solo.disposition == D.PERMITTED       # single stale provider → false permit
    recon = resolve_disposition(world, prov, apn["R-102"], "RESIDENTIAL_SINGLE_FAMILY", ("regrid", "attom"))
    assert recon.disposition == D.UNKNOWN        # reconciliation refuses to guess


def test_thorough_bundle_has_zero_false_permits_everywhere():
    """The blocking SLO: across every parcel × target use, the full evidence set never concludes PERMITTED
    where the truth is PROHIBITED."""
    from context_runtime.integrations.zoning_intelligence import TARGET_USES
    world, apn, prov = _world()
    full = ("regrid", "attom", "municipal_gis", "ordinance")
    for pid in world.parcels:
        for use in TARGET_USES:
            gold = gold_disposition(world, prov, pid, use)
            d = resolve_disposition(world, prov, pid, use, full).disposition
            assert not (d == D.PERMITTED and gold == D.PROHIBITED)


def test_abstains_when_decisive_source_absent():
    """An overlay-sensitive commercial use with no official GIS cannot confirm 'no restrictive overlay',
    so it abstains to UNKNOWN rather than guessing PERMITTED (plan §9: don't force a binary)."""
    world, apn, prov = _world()
    a = resolve_disposition(world, prov, apn["O-200"], "OFFICE", ("regrid", "attom"))
    assert a.disposition == D.UNKNOWN


# ── tenant: choose / record / explain ──

def test_assess_records_pending_and_explains():
    world, apn, _ = _world()
    tenant = ZoningIntelligenceTenant(world=world)
    a = tenant.assess("q1", apn["O-200"], "OFFICE")
    assert a.disposition in set(D)
    exp = tenant.explain("q1")
    assert exp["bucket"] == "commercial"
    assert exp["selected"] in {arm.key for arm in DEFAULT_ARMS}
    assert len(exp["scores"]) == len(DEFAULT_ARMS)


# ── learning: converged policy per bucket + beats baselines ──

def test_learns_minimal_evidence_bundle_per_bucket():
    world, apn, prov = _world()
    tenant = ZoningIntelligenceTenant(world=world, bandit=_zoning_bandit(epsilon=0.1))
    for i in range(300):
        a_apn, use = QUERIES[i % len(QUERIES)]
        qid = f"q{i}"
        tenant.assess(qid, apn[a_apn], use)
        tenant.record(qid, gold_disposition(world, prov, apn[a_apn], use))
    pol = tenant.policy()
    # residential + commercial resolve WITHOUT the expensive ordinance; industrial needs it.
    assert "ordinance" not in pol["residential"]
    assert "ordinance" not in pol["commercial"]
    assert pol["industrial"] == DEFAULT_ARMS[-1].key


def test_learned_policy_beats_both_baselines_and_holds_the_slo():
    world, apn, prov = _world()
    tenant = ZoningIntelligenceTenant(world=world, bandit=_zoning_bandit(epsilon=0.1))
    for i in range(300):
        a_apn, use = QUERIES[i % len(QUERIES)]
        qid = f"q{i}"
        tenant.assess(qid, apn[a_apn], use)
        tenant.record(qid, gold_disposition(world, prov, apn[a_apn], use))
    learned = tenant.policy()
    lr, lc, lfp = _eval(world, prov, apn, lambda b: _ARM_BY_KEY[learned[b]])
    ar, ac, afp = _eval(world, prov, apn, lambda b: DEFAULT_ARMS[0])     # single provider
    br, bc, bfp = _eval(world, prov, apn, lambda b: DEFAULT_ARMS[-1])    # fixed thorough
    assert lr > ar                       # beats single-provider on correctness
    assert lc < bc                       # cheaper than always-thorough
    assert lr >= br - 1e-9               # at least matches thorough's reward (correctness)
    assert lfp == 0 and afp >= 1         # the SLO: learned trips 0, single-provider trips ≥1


# ── incremental change + population governance ──

def test_incremental_recomputation_is_dependency_scoped():
    world, apn, _ = _world()
    # An overlay covering only D-401's footprint must mark exactly D-401 stale — nothing else.
    ring = [(850, 1050), (1400, 1050), (1400, 1600), (850, 1600)]
    affected = affected_conclusions(world, changed_parcel_ids=set(), changed_overlay_rings=[ring])
    assert affected == {apn["D-401"]}
    assert len(affected) < len(world.parcels)


def test_population_governance_detects_systematic_drift():
    gov = PopulationGovernor(baseline_rate=0.0, threshold=0.25, window=20)
    for _ in range(20):
        gov.observe("ordinance", downgraded=False)
    assert gov.review_required("ordinance") is False           # healthy population, no review
    for i in range(20):
        gov.observe("ordinance", downgraded=(i % 2 == 0))       # ~50% silently downgraded
    assert gov.review_required("ordinance") is True            # cross-population shift → REQUIRE_REVIEW
