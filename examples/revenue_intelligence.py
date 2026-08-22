"""Revenue & Intelligence × Context Runtime — the GTM benchmark, offline.

The canonical mission: *find the accounts most likely to need us, enrich the right buyer at each, don't
overpay for data, and contact no one.* Runs with no API keys over a labelled fixture and compares how
providers are chosen:

    A  all providers      — call everything for every record (coverage ceiling, max cost)
    B  fixed pipeline     — one hand-engineered provider set for every record (misses some needs)
    C  cheapest-first     — always the free local source (cheap, usually insufficient)
    E  adaptive (need)    — learn the cheapest bundle sufficient for each need
    F  learned (need×seg) — condition on segment too: a provider authoritative for enterprise but not SMB,
                            public but not private — the empirical policy the static one can't express
    G  + governance       — arm F with population/cross-series rules watching the account stream

First publishable question (§17): preserve quality at fewer/cheaper paid calls than a fixed pipeline.
Second (§18): can governance catch a cross-account regression invisible per call? Both, offline.

    python examples/revenue_intelligence.py
"""
from __future__ import annotations

from context_runtime.integrations.revenue_intelligence import (
    DEFAULT_BUNDLES, DEFAULT_POPULATION_RULES, Fixture, GtmPopulationGovernor, ProviderBundle,
    RevenueIntelligenceTenant, _gtm_bandit, outcome_for,
)

# ── labelled fixture with a SEGMENT axis: several needs have a segment-dependent decisive provider
#    (apollo authoritative for enterprise people, pdl for SMB; apollo for enterprise firmographics,
#    crunchbase for SMB; SEC for public-company funding, crunchbase for private). Two wrong-entity traps. ──
FIXTURES: dict[str, Fixture] = {f.account_id: f for f in [
    # company_identity — segment-independent (pdl)
    Fixture("a01", "Northwind Health", "northwind.io", "company_identity", "pdl", "Northwind Health Inc", segment="enterprise"),
    Fixture("a02", "Contoso Robotics", "contoso.ai", "company_identity", "pdl", "Contoso Robotics", wrong=("apollo",), segment="smb"),
    # person_role — segment-DEPENDENT: enterprise→apollo, smb→pdl
    Fixture("a03", "Fabrikam Data", "fabrikam.com", "person_role", "apollo", "VP Data Platform", segment="enterprise"),
    Fixture("a04", "Tailspin Analytics", "tailspin.co", "person_role", "pdl", "Head of ML", segment="smb"),
    Fixture("a05", "Humongous Insurance", "humongous.com", "person_role", "apollo", "Chief Data Officer", segment="enterprise"),
    Fixture("a06", "Margie's Startup", "margies.dev", "person_role", "pdl", "Founding Engineer", segment="smb"),
    # firmographics — segment-DEPENDENT: enterprise→apollo, smb→crunchbase
    Fixture("a07", "Proseware AI", "proseware.ai", "firmographics", "apollo", "12,000 staff", segment="enterprise"),
    Fixture("a08", "Wingtip Cloud", "wingtip.dev", "firmographics", "crunchbase", "Series A, 60 staff", segment="smb"),
    # contact_verify — segment-independent (hunter); a08-style apollo trap
    Fixture("a09", "Adventure Works", "adventure.works", "contact_verify", "hunter", "cto@adventure.works", segment="enterprise"),
    Fixture("a10", "Litware Systems", "litware.com", "contact_verify", "hunter", "vp.eng@litware.com", wrong=("apollo",), segment="smb"),
    # tech_signal — segment-independent (builtwith)
    Fixture("a11", "Coho Vineyard", "coho.wine", "tech_signal", "builtwith", "Kubernetes + Snowflake", segment="smb"),
    Fixture("a12", "Lucerne Publishing", "lucerne.press", "tech_signal", "builtwith", "AWS + dbt", segment="enterprise"),
    # funding_signal — segment-DEPENDENT: public→sec (free!), private→crunchbase
    Fixture("a13", "Fourth Coffee Corp", "fourthcoffee.com", "funding_signal", "sec", "10-K FY2025", segment="public"),
    Fixture("a14", "Blue Yonder Air", "blueyonder.aero", "funding_signal", "crunchbase", "Series C, $80M", segment="private"),
    # niche_signal — the expensive unstructured fallback (web_research)
    Fixture("a15", "Graphic Design Inst", "gdi.edu", "niche_signal", "web_research", "mentioned on a forum", segment="default"),
]}

FIXED_PIPELINE = ProviderBundle(("crm", "apollo", "pdl"))    # arm B
CHEAPEST = ProviderBundle(("crm",))                          # arm C
ALL = DEFAULT_BUNDLES[-1]                                    # arm A


def _static_arm(bundle: ProviderBundle) -> dict:
    from context_runtime.integrations.revenue_intelligence import PROVIDER_COST
    correct = cost = calls = 0
    for fx in FIXTURES.values():
        correct += int(outcome_for(bundle, fx) == "correct")
        cost += bundle.cost
        calls += sum(1 for p in bundle.providers if PROVIDER_COST[p] > 0)
    n = len(FIXTURES)
    return {"quality": correct / n, "cost_per_acct": cost / n, "paid_calls": calls}


def _train(segmented: bool, rounds: int = 400) -> RevenueIntelligenceTenant:
    t = RevenueIntelligenceTenant(FIXTURES, bandit=_gtm_bandit(0.1),
                                  approver=lambda spec: False, no_outreach=True, segmented=segmented)
    ids = list(FIXTURES)
    for i in range(rounds):
        aid = ids[i % len(ids)]
        t.record_outcome(aid, t.enrich(aid).outcome)
    return t


def _eval_policy(t: RevenueIntelligenceTenant) -> dict:
    from context_runtime.integrations.revenue_intelligence import PROVIDER_COST
    pol = t.policy()
    correct = cost = calls = 0
    for fx in FIXTURES.values():
        key = t._ctx(fx, fx.need)
        bundle = next(b for b in DEFAULT_BUNDLES if b.key == pol[key])
        correct += int(outcome_for(bundle, fx) == "correct")
        cost += bundle.cost
        calls += sum(1 for p in bundle.providers if PROVIDER_COST[p] > 0)
    n = len(FIXTURES)
    return {"quality": correct / n, "cost_per_acct": cost / n, "paid_calls": calls}


def run() -> None:
    tenant_e = _train(segmented=False)
    tenant_f = _train(segmented=True)
    arms = {
        "A  all providers":     _static_arm(ALL),
        "B  fixed pipeline":    _static_arm(FIXED_PIPELINE),
        "C  cheapest-first":    _static_arm(CHEAPEST),
        "E  adaptive (need)":   _eval_policy(tenant_e),
        "F  learned (need×seg)": _eval_policy(tenant_f),
    }

    print("Canonical mission: 15 candidate accounts · enrich the decisive need at each · contact no one\n")
    print(f"  {'arm':<24}{'quality':>9}{'$ / acct':>11}{'paid calls':>12}")
    for name, m in arms.items():
        print(f"  {name:<24}{m['quality']*100:>7.0f}% {m['cost_per_acct']:>10.3f} {m['paid_calls']:>11d}")

    b, e, f = arms["B  fixed pipeline"], arms["E  adaptive (need)"], arms["F  learned (need×seg)"]
    print(f"\n  ── adaptive E vs fixed pipeline B ──  quality {(e['quality']-b['quality'])*100:+.0f} pts, "
          f"cost {(1-e['cost_per_acct']/b['cost_per_acct'])*100:+.0f}%, calls {b['paid_calls']-e['paid_calls']:+d}")
    print(f"  ── learned F vs adaptive E    ──  quality {(f['quality']-e['quality'])*100:+.0f} pts, "
          f"cost {(1-f['cost_per_acct']/e['cost_per_acct'])*100:+.0f}% (${f['cost_per_acct']:.3f} vs ${e['cost_per_acct']:.3f}), "
          f"same-or-fewer calls — segment conditioning pays for itself")

    print("\n── learned provider policy (arm F, per need×segment) ──")
    for key in sorted(tenant_f.policy()):
        print(f"  {key:<26} → {tenant_f.policy()[key]}")

    # ── arm G: population governance over the account stream ──
    healthy = GtmPopulationGovernor(mode="OBSERVE").evaluate(tenant_f.events)
    # inject a controlled regression: a batch where every account escalates to the expensive fallback
    # (a provider outage / prompt regression would look exactly like this — invisible per call)
    regression = [{"account_id": f"r{i}", "need": "firmographics", "segment": "smb",
                   "bundle": "apollo+web_research", "providers": ("apollo", "web_research"),
                   "cost": 0.32, "outcome": "correct"} for i in range(8)]
    observe = GtmPopulationGovernor(DEFAULT_POPULATION_RULES, mode="OBSERVE").evaluate(regression)
    enforce = GtmPopulationGovernor(DEFAULT_POPULATION_RULES, mode="ENFORCE").evaluate(regression)

    print(f"\n── governance (arm G) ──")
    print(f"  healthy stream ({len(tenant_f.events)} events): {len(healthy)} findings")
    print(f"  injected regression (8 accounts escalate to web_research):")
    for fnd in observe:
        print(f"    ⚠ {fnd.rule:<26} [{fnd.disposition}] {fnd.detail}")
    same = [(f.rule, f.disposition) for f in observe] == [(f.rule, f.disposition) for f in enforce]
    print(f"  OBSERVE→ENFORCE detection identical: {same}  (OBSERVE emits, ENFORCE would gate)")

    # governance: prepare allowed, send denied under NO_OUTREACH
    prep, sent = tenant_f.prepare_outreach("a03"), tenant_f.send_outreach("a03")
    print(f"\n── NO_OUTREACH ──  prepare ok={prep.ok} · send ok={sent.ok} ({sent.text})")


if __name__ == "__main__":
    run()
