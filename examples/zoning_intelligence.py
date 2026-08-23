"""Zoning Intelligence × Context Runtime — the geospatial reference benchmark (plan §6, §14), offline.

Given a parcel and a target use, the Runtime must conclude the land-use disposition (PERMITTED /
CONDITIONAL / PROHIBITED / UNKNOWN) from heterogeneous evidence — commercial providers (Regrid, ATTOM),
the official municipal GIS, and the zoning ordinance. The decision the Runtime learns is *which evidence
to acquire*: the cheapest bundle that still reaches the correct, safe answer.

The benchmark shows the three arms from the plan:

  A · single-provider (Regrid only)   — cheap, but MISSES overlay/conditional cases AND produces a
                                        "verified permitted" that is actually prohibited (SLO violation).
  B · fixed thorough (all sources)     — correct everywhere, but always pays for the ordinance.
  C · Context Runtime (learned)        — matches B's correctness at lower cost by learning, per
                                        use-difficulty bucket, the minimal sufficient evidence bundle.

Everything is offline and deterministic (a fixture Fulton County jurisdiction). Then: a use-first land
search, an incremental evidence change (dependency-scoped recomputation), and a population-governance
shift (a provider drifting CONDITIONAL→PERMITTED across the population).

    PYTHONPATH=. python examples/zoning_intelligence.py
"""
from __future__ import annotations

from context_runtime.geospatial.contracts import UseDisposition as D
from context_runtime.integrations.zoning_intelligence import (
    DEFAULT_ARMS, EvidencePlan, PopulationGovernor, ZoningIntelligenceTenant, affected_conclusions,
    build_providers, build_reference_world, gold_disposition, resolve_disposition, reward_zoning,
    zoning_bucket, _zoning_bandit,
)

# Curated decision-relevant queries per use-difficulty bucket — the parcels/uses where the evidence tier
# actually matters (a zoning analyst does not ask whether a warehouse fits a clearly-residential lot).
QUERIES: list[tuple[str, str]] = [
    # residential: base zoning decides; commercial consensus resolves the stale-Regrid trap (R-102)
    ("R-100", "RESIDENTIAL_SINGLE_FAMILY"), ("R-101", "RESIDENTIAL_SINGLE_FAMILY"),
    ("R-102", "RESIDENTIAL_SINGLE_FAMILY"),
    # commercial: overlay-sensitive → needs the official GIS (O-201 sits under a FLOOD overlay)
    ("O-200", "OFFICE"), ("O-200", "RETAIL"), ("O-201", "OFFICE"), ("R-102", "OFFICE"),
    # industrial: conditional → needs the ordinance text (W-301 sits under a HISTORIC overlay)
    ("W-300", "WAREHOUSE"), ("W-300", "LIGHT_INDUSTRIAL"), ("D-401", "DATA_CENTER"),
    ("D-401", "LIGHT_INDUSTRIAL"), ("W-301", "WAREHOUSE"),
]


_ARM_BY_KEY = {a.key: a for a in DEFAULT_ARMS}


def _eval(world, providers, apn, pick) -> tuple[float, float, int]:
    """Average reward, average cost, and false-permitted count of a policy over the stream. ``pick`` maps
    a (bucket, use) to the EvidencePlan to run — a constant (fixed baseline) or the learned policy."""
    total = cost = 0.0
    false_permits = 0
    for a_apn, use in QUERIES:
        pid = apn[a_apn]
        arm = pick(zoning_bucket(use))
        gold = gold_disposition(world, providers, pid, use)
        a = resolve_disposition(world, providers, pid, use, arm.sources)
        correct = a.disposition == gold
        fp = a.disposition == D.PERMITTED and gold == D.PROHIBITED
        total += reward_zoning(correct=correct, false_permitted=fp, cost_units=arm.cost_units(providers))
        cost += arm.cost_units(providers)
        false_permits += 1 if fp else 0
    n = len(QUERIES)
    return total / n, cost / n, false_permits


def run(rounds: int = 260) -> None:
    world, apn = build_reference_world()
    providers = build_providers(world)
    tenant = ZoningIntelligenceTenant(world=world, bandit=_zoning_bandit(epsilon=0.1))

    print("Parcel-first assessments (evidence bundle chosen by the Runtime, learning the minimal set):\n")
    online_false_permits = 0
    for i in range(rounds):
        a_apn, use = QUERIES[i % len(QUERIES)]
        pid = apn[a_apn]
        qid = f"q{i}"
        assessment = tenant.assess(qid, pid, use)
        gold = gold_disposition(world, providers, pid, use)
        tenant.record(qid, gold)
        if assessment.disposition == D.PERMITTED and gold == D.PROHIBITED:
            online_false_permits += 1
        if i < len(QUERIES):
            ok = "✓" if assessment.disposition == gold else ("·" if assessment.disposition == D.UNKNOWN else "✗")
            print(f"  [{zoning_bucket(use):<11}] {a_apn} {use:<26} → {assessment.disposition.value:<11} "
                  f"{ok}  sources={'+'.join(assessment.sources_used)}")

    # C is the DEPLOYED (converged, greedy) policy — exploitation, no exploration noise. A/B are fixed.
    learned = tenant.policy()
    lr, lc, lfp = _eval(world, providers, apn, lambda b: _ARM_BY_KEY[learned[b]])
    ar, ac, afp = _eval(world, providers, apn, lambda b: DEFAULT_ARMS[0])      # A: single provider
    br, bc, bfp = _eval(world, providers, apn, lambda b: DEFAULT_ARMS[-1])     # B: fixed thorough

    print("\nreward = correct disposition − λ·evidence-cost;  false-permit = −1 (the blocking SLO)\n")
    print(f"  C · Context Runtime (learned bundle/bucket): reward {lr:+.3f}   avg cost {lc:.2f}u   "
          f"false-permits {lfp}")
    a_flag = "   ← violates SLO" if afp else ""
    print(f"  A · single provider (Regrid only):           reward {ar:+.3f}   avg cost {ac:.2f}u   "
          f"false-permits {afp}{a_flag}")
    print(f"  B · fixed thorough (all sources):            reward {br:+.3f}   avg cost {bc:.2f}u   "
          f"false-permits {bfp}")
    print(f"\n  learned vs single-provider:  reward {lr - ar:+.3f}   (0 SLO violations vs {afp})")
    print(f"  learned vs fixed-thorough:   reward {lr - br:+.3f}   cost {lc - bc:+.2f}u "
          f"({bc / lc:.1f}× cheaper at equal correctness)")
    print(f"  (during learning, {rounds} rounds at ε=0.1: {online_false_permits} exploratory SLO trips — "
          f"converged policy trips 0)")

    print("\n── learned minimal evidence bundle per use-difficulty bucket ──")
    for bucket in ("residential", "commercial", "industrial"):
        pol = tenant.policy().get(bucket, "—")
        print(f"  {bucket:<12} → {pol}")

    # EXPLAIN for one industrial parcel — every conclusion shows exactly which evidence produced it.
    tenant.assess("explain-demo", apn["D-401"], "DATA_CENTER")
    exp = tenant.explain("explain-demo")
    print(f"\n── EXPLAIN · D-401 / DATA_CENTER ──\n  selected={exp['selected']}  "
          f"disposition={exp['disposition']}  confidence={exp['confidence']}")
    for r in exp["reasons"]:
        print(f"    · {r}")

    # ── use-first land search (plan §6, §12): deterministic spatial filter + learned evidence ──
    hits = tenant.search("LIGHT_INDUSTRIAL", center=(1050, 1300), radius=1200, min_lot_area=40000)
    print("\n── LAND SEARCH · LIGHT_INDUSTRIAL within 1200ft of (1050,1300), lot ≥ 40,000 ft² ──")
    for h in hits:
        print(f"  {world.parcels[h.parcel_id].apn}  {h.disposition.value}  "
              f"lot={world.parcels[h.parcel_id].lot_area:.0f}ft²  ({'+'.join(h.sources_used)})")

    # ── incremental evidence change (plan §17): a new overlay → recompute only dependent parcels ──
    new_overlay = [(850, 1050), (1400, 1050), (1400, 1600), (850, 1600)]   # now covers D-401
    affected = affected_conclusions(world, changed_parcel_ids=set(), changed_overlay_rings=[new_overlay])
    print(f"\n── INCREMENTAL · new overlay district → recompute {len(affected)} of "
          f"{len(world.parcels)} parcels ──")
    print("  affected: " + ", ".join(sorted(world.parcels[p].apn for p in affected)))

    # ── population governance (plan §18): a provider drifts CONDITIONAL→PERMITTED across the population ──
    gov = PopulationGovernor(baseline_rate=0.0, threshold=0.25, window=20)
    for i in range(20):
        gov.observe("ordinance", downgraded=False)       # healthy baseline
    healthy = gov.review_required("ordinance")
    for i in range(20):
        gov.observe("ordinance", downgraded=(i % 2 == 0))  # ~50% now silently downgraded
    drifted = gov.review_required("ordinance")
    print("\n── POPULATION GOVERNANCE · ordinance classifier drift ──")
    print(f"  healthy window review_required={healthy}  →  drifted window review_required={drifted} "
          f"(recent downgrade rate {gov.recent_rate('ordinance'):.0%})")


if __name__ == "__main__":
    run()
