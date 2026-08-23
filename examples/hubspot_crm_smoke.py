"""HubSpot CRM connector — live smoke (Phase 5). Run on the host that has the Service Key:

    HUBSPOT_SERVICE_KEY=... python examples/hubspot_crm_smoke.py     # (key from ~/.bashrc on evo-x2)

Seeds a few benchmark companies into your HubSpot account, resolves one, reads it back, and performs an
approval-gated property update — proving the connector's reads + governed write end to end. Idempotent:
re-running upserts by domain rather than duplicating. The token is read from the environment, never printed.
"""
from __future__ import annotations

from context_runtime.integrations.hubspot_crm import (
    HubSpotCRM, HubSpotCRMTool, HubSpotUpdateTool, seed_from_fixture, token_present,
)
from context_runtime.tools.base import ApprovalPolicy, ToolRegistry
from examples.revenue_intelligence import FIXTURES


def run() -> None:
    if not token_present():
        print("No HUBSPOT_SERVICE_KEY (or HUBSPOT_API_KEY) in the environment.")
        print("Create a HubSpot Service Key (Development → Keys → Service Keys) with scopes")
        print("crm.objects.{contacts,companies,deals}.read+write, then export HUBSPOT_SERVICE_KEY and re-run.")
        return

    client = HubSpotCRM()
    seed = {k: FIXTURES[k] for k in list(FIXTURES)[:3]}       # a few companies is enough for a smoke
    print(f"Seeding {len(seed)} companies into HubSpot (idempotent upsert by domain)…")
    ids = seed_from_fixture(seed, client)
    for aid, cid in ids.items():
        print(f"  {aid}  {seed[aid].company:<22} {seed[aid].domain:<20} → company {cid}")

    # resolve one via the tool the tenant uses (crm_enrich). HubSpot's search index is eventually
    # consistent, so a just-seeded record can take a few seconds to appear — retry briefly. (Real
    # accounts are already indexed; this lag only affects freshly-created ones.)
    import time
    aid = next(iter(seed))
    tool = HubSpotCRMTool(seed, client)
    for attempt in range(6):
        res = tool.run({"account_id": aid, "need": seed[aid].need})
        if res.data.get("status") == "correct":
            break
        time.sleep(3)
    print(f"\nresolve {aid} via crm_enrich → status={res.data.get('status')} "
          f"(after {attempt + 1} tries): {res.text}")

    # read back the raw record
    cid = ids[aid]
    company = client.get_company(cid)
    print(f"read company {cid}: {company.get('properties', {})}")

    # governed write: update a property, through the approval gate (approver says yes here).
    # industry is a HubSpot picklist — use a valid option.
    reg = ToolRegistry(ApprovalPolicy(mode="deny_side_effects", approver=lambda spec: True))
    reg.register(HubSpotUpdateTool(client))
    upd = reg.run("crm_update", {"company_id": cid, "properties": {"industry": "COMPUTER_SOFTWARE"}})
    print(f"\ngoverned crm_update → ok={upd.ok}: {upd.text}")
    print(f"audit: {reg.audit[-1]}")


if __name__ == "__main__":
    run()
