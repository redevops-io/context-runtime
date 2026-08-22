"""Revenue & Intelligence × Context Runtime — the GTM benchmark, offline.

The canonical mission: *find the accounts most likely to need us, enrich the right buyer at each, don't
overpay for data, and contact no one.* This runs it with no API keys over a labelled fixture and compares
four ways to choose providers:

    A  all providers        — call everything for every record (coverage ceiling, max cost)
    B  fixed pipeline       — one hand-engineered provider set for every record (misses some needs)
    C  cheapest-first       — always the free local source (cheap, usually insufficient)
    E  ReDevOps adaptive    — the runtime learns the cheapest provider bundle sufficient for each need

First publishable question (plan §17): can the runtime preserve enrichment quality while making materially
fewer / cheaper paid provider calls than a fixed pipeline? And it keeps outreach behind a NO_OUTREACH gate.

    python examples/revenue_intelligence.py
"""
from __future__ import annotations

from context_runtime.integrations.revenue_intelligence import (
    DEFAULT_BUNDLES, Fixture, ProviderBundle, RevenueIntelligenceTenant, _gtm_bandit,
    outcome_for, reward_enrich,
)

# ── labelled fixture: real GTM shape — each account has a decisive provider for its need, and two carry a
#    confident wrong-entity trap (a provider that returns the wrong company). Tier-2 disclosed labels. ──
FIXTURES: dict[str, Fixture] = {f.account_id: f for f in [
    Fixture("a01", "Northwind Health", "northwind.io", "company_identity", "pdl", "Northwind Health Inc"),
    Fixture("a02", "Contoso Robotics", "contoso.ai", "company_identity", "pdl", "Contoso Robotics",
            wrong=("apollo",)),                                   # apollo confidently returns a wrong org
    Fixture("a03", "Fabrikam Data", "fabrikam.com", "person_role", "apollo", "VP Data Platform"),
    Fixture("a04", "Tailspin Analytics", "tailspin.co", "person_role", "apollo", "Head of ML"),
    Fixture("a05", "Proseware AI", "proseware.ai", "firmographics", "apollo", "Series B, 180 staff"),
    Fixture("a06", "Wingtip Cloud", "wingtip.dev", "firmographics", "apollo", "Series A, 60 staff"),
    Fixture("a07", "Adventure Works", "adventure.works", "contact_verify", "hunter", "cto@adventure.works"),
    Fixture("a08", "Litware Systems", "litware.com", "contact_verify", "hunter", "vp.eng@litware.com",
            wrong=("apollo",)),                                   # apollo returns a stale/guessed address
    Fixture("a09", "Coho Vineyard", "coho.wine", "tech_signal", "builtwith", "Kubernetes + Snowflake"),
    Fixture("a10", "Fourth Coffee", "fourthcoffee.io", "tech_signal", "builtwith", "Databricks + Kafka"),
    Fixture("a11", "Graphic Design Inst", "gdi.edu", "company_identity", "pdl", "Graphic Design Institute"),
    Fixture("a12", "Humongous Insurance", "humongous.com", "person_role", "apollo", "Chief Data Officer"),
    Fixture("a13", "Lucerne Publishing", "lucerne.press", "tech_signal", "builtwith", "AWS + dbt"),
    Fixture("a14", "Margie's Travel", "margies.travel", "firmographics", "apollo", "Bootstrapped, 25 staff"),
    Fixture("a15", "Blue Yonder Air", "blueyonder.aero", "contact_verify", "hunter", "head.data@blueyonder.aero"),
]}

FIXED_PIPELINE = ProviderBundle(("crm", "apollo", "pdl"))    # arm B: competent, but no hunter/builtwith
CHEAPEST = ProviderBundle(("crm",))                          # arm C
ALL = DEFAULT_BUNDLES[-1]                                    # arm A


def _static_arm(bundle: ProviderBundle):
    """Quality (correct rate), avg external cost, and paid-call count for a fixed-bundle arm."""
    correct = cost = calls = 0
    for fx in FIXTURES.values():
        oc = outcome_for(bundle, fx)
        correct += int(oc == "correct")
        cost += bundle.cost
        calls += sum(1 for p in bundle.providers if p != "crm")   # crm/local is free, not a paid call
    n = len(FIXTURES)
    return {"quality": correct / n, "cost_per_acct": cost / n, "paid_calls": calls}


def run(rounds: int = 240) -> None:
    tenant = RevenueIntelligenceTenant(FIXTURES, bandit=_gtm_bandit(0.1),
                                       approver=lambda spec: False,  # no human here → send stays denied
                                       no_outreach=True)
    ids = list(FIXTURES)

    # Arm E: learn per-need which cheapest bundle is sufficient.
    for i in range(rounds):
        aid = ids[i % len(ids)]
        r = tenant.enrich(aid)
        tenant.record_outcome(aid, r.outcome)

    # steady-state pass over the learned policy (ε→0): use the current best bundle per need.
    e_correct = e_cost = e_calls = 0
    for aid, fx in FIXTURES.items():
        best_key = tenant.policy()[fx.need]
        bundle = next(b for b in DEFAULT_BUNDLES if b.key == best_key)
        oc = outcome_for(bundle, fx)
        e_correct += int(oc == "correct")
        e_cost += bundle.cost
        e_calls += sum(1 for p in bundle.providers if p != "crm")
    n = len(FIXTURES)
    arm_e = {"quality": e_correct / n, "cost_per_acct": e_cost / n, "paid_calls": e_calls}

    arms = {
        "A  all providers":   _static_arm(ALL),
        "B  fixed pipeline":  _static_arm(FIXED_PIPELINE),
        "C  cheapest-first":  _static_arm(CHEAPEST),
        "E  ReDevOps adaptive": arm_e,
    }

    print("Canonical mission: 15 candidate accounts · enrich the decisive need at each · contact no one\n")
    print(f"  {'arm':<22}{'quality':>9}{'$ / acct':>11}{'paid calls':>12}")
    for name, m in arms.items():
        print(f"  {name:<22}{m['quality']*100:>7.0f}% {m['cost_per_acct']:>10.3f} {m['paid_calls']:>11d}")

    b, e = arms["B  fixed pipeline"], arms["E  ReDevOps adaptive"]
    dq = (e["quality"] - b["quality"]) * 100
    dcost = (1 - e["cost_per_acct"] / b["cost_per_acct"]) * 100 if b["cost_per_acct"] else 0
    dcalls = b["paid_calls"] - e["paid_calls"]
    print(f"\n  ── adaptive (E) vs fixed pipeline (B) ──")
    print(f"     quality:    {dq:+.0f} pts   ({e['quality']*100:.0f}% vs {b['quality']*100:.0f}%)")
    print(f"     cost/acct:  {dcost:+.0f}%    (${e['cost_per_acct']:.3f} vs ${b['cost_per_acct']:.3f})")
    print(f"     paid calls: {dcalls:+d}     ({e['paid_calls']} vs {b['paid_calls']})")

    print("\n── learned provider policy per need ──")
    for need in sorted({f.need for f in FIXTURES.values()}):
        print(f"  {need:<18} → {tenant.policy()[need]}")

    # governance: prepare is allowed, send is denied under NO_OUTREACH
    prep = tenant.prepare_outreach("a03")
    sent = tenant.send_outreach("a03")
    print(f"\n── governance (NO_OUTREACH) ──")
    print(f"  outreach_prepare → ok={prep.ok}: {prep.text}")
    print(f"  outreach_send    → ok={sent.ok}: {sent.text}")
    print(f"  audit: {tenant.registry.audit[-1]}")


if __name__ == "__main__":
    run()
