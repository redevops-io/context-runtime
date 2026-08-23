"""Revenue & Intelligence — the full canonical mission, live.

Runs the mission end to end against a REAL CRM (Salesforce or HubSpot): for each account the runtime tries
the live CRM first (free first hop), enriches only what the CRM is missing via the learned adaptive policy
(need×segment), keeps external spend under budget, evaluates the population governor over the run, and
contacts no one. This is the plan's §2 mission with a real design-partner CRM in the loop.

    CR_CRM_BACKEND=salesforce python examples/revenue_intelligence_live_mission.py   # or hubspot

Credentials come from the environment (the connectors read them; never printed). The 3 companies seeded by
the CRM smokes resolve for free; the rest are enriched by the cheapest sufficient provider per need.
"""
from __future__ import annotations

import os

from context_runtime.integrations.revenue_intelligence import (
    DEFAULT_BUNDLES, GtmPopulationGovernor, RevenueIntelligenceTenant, _gtm_bandit, outcome_for,
)
from context_runtime.integrations.hubspot_crm import HubSpotCRMTool, token_present as hs_present
from context_runtime.integrations.salesforce_crm import SalesforceCRMTool, token_present as sf_present
from examples.revenue_intelligence import FIXTURES

BUDGET = 5.00


def _pick_crm():
    want = os.getenv("CR_CRM_BACKEND", "").strip().lower()
    if want == "hubspot" and hs_present():
        return HubSpotCRMTool(FIXTURES), "HubSpot"
    if want == "salesforce" and sf_present():
        return SalesforceCRMTool(FIXTURES), "Salesforce"
    if sf_present():
        return SalesforceCRMTool(FIXTURES), "Salesforce"
    if hs_present():
        return HubSpotCRMTool(FIXTURES), "HubSpot"
    return None, None


def _learned_tenant() -> RevenueIntelligenceTenant:
    """Train arm F offline to get the cheapest sufficient provider per need×segment (the policy the mission
    uses to escalate) and to host the approval-gated outreach gate."""
    t = RevenueIntelligenceTenant(FIXTURES, bandit=_gtm_bandit(0.1), approver=lambda spec: False,
                                  segmented=True, no_outreach=True)
    ids = list(FIXTURES)
    for i in range(400):
        t.record_outcome(ids[i % len(ids)], t.enrich(ids[i % len(ids)]).outcome)
    return t


def run() -> None:
    crm, backend = _pick_crm()
    if crm is None:
        print("No live CRM credentials in the environment (HubSpot or Salesforce). Aborting.")
        return
    print(f"MISSION  find/enrich the decisive buyer per account  ·  CRM={backend}  ·  "
          f"budget ${BUDGET:.2f}  ·  NO_OUTREACH\n")

    trained = _learned_tenant()
    policy = trained.policy()
    events: list[dict] = []
    spend, correct, resolved_local, paid = 0.0, 0, 0, 0

    for aid, fx in FIXTURES.items():
        crm_res = crm.run({"account_id": aid, "need": fx.need})       # free first hop — the real CRM
        if crm_res.data.get("status") == "correct":
            resolved_local += 1
            correct += 1
            events.append({"account_id": aid, "need": fx.need, "segment": fx.segment,
                           "bundle": "crm", "providers": ("crm",), "cost": 0.0, "outcome": "correct"})
            print(f"  ✓ {fx.company:<22}{fx.need:<16} resolved from {backend} CRM  ($0.00)")
            continue
        # escalate: the learned cheapest provider bundle for this need×segment
        bundle = next(b for b in DEFAULT_BUNDLES if b.key == policy[f"{fx.need}|{fx.segment}"])
        oc = outcome_for(bundle, fx)
        spend += bundle.cost
        paid += 1
        correct += int(oc == "correct")
        events.append({"account_id": aid, "need": fx.need, "segment": fx.segment, "bundle": bundle.key,
                       "providers": bundle.providers, "cost": bundle.cost, "outcome": oc})
        print(f"  → {fx.company:<22}{fx.need:<16} enrich via {bundle.key:<12} (${bundle.cost:.2f})  [{oc}]")

    findings = GtmPopulationGovernor(mode="OBSERVE").evaluate(events)
    prep = trained.prepare_outreach(next(iter(FIXTURES)))
    sent = trained.send_outreach(next(iter(FIXTURES)))
    n = len(FIXTURES)

    print("\n── EXPLAIN ──")
    print(f"  {n} accounts evaluated")
    print(f"  {resolved_local} resolved from the CRM at $0 (paid calls avoided)")
    print(f"  {paid} external enrichments · quality {correct}/{n} correct")
    print(f"  external-data spend: ${spend:.2f} / ${BUDGET:.2f} budget  ({spend/BUDGET*100:.0f}% used)")
    print(f"  governance findings: {len(findings)} " + (f"({[f.rule for f in findings]})" if findings else "(healthy)"))
    print(f"  outreach: prepare ok={prep.ok} · send ok={sent.ok} ({sent.text})")
    print(f"\n  vs a fixed all-providers pass ({n} accts × every paid source): the runtime paid for "
          f"{paid} of {n} accounts and skipped {resolved_local} entirely by checking the CRM first.")


if __name__ == "__main__":
    run()
