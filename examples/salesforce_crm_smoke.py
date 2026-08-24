"""Salesforce CRM connector — live smoke (Phase 5). Run on the host with the credentials:

    python examples/salesforce_crm_smoke.py     # reads SALESFORCE_CONSUMER_KEY/SECRET + _INSTANCE_URL

Fetches a client-credentials token, seeds a few benchmark accounts, resolves one, reads it back, and
performs an approval-gated field update — proving the connector's auth, reads + governed write end to end.
Idempotent (upsert by Website). Credentials are read from the environment, never printed.
"""
from __future__ import annotations

from context_runtime.integrations.salesforce_crm import (
    SalesforceCRM, SalesforceCRMTool, SalesforceUpdateTool, seed_from_fixture, token_present,
)
from context_runtime.tools.base import ApprovalPolicy, ToolRegistry
from examples.revenue_intelligence import FIXTURES


def run() -> None:
    if not token_present():
        print("Missing Salesforce credentials in the environment. Set:")
        print("  SALESFORCE_CONSUMER_KEY, SALESFORCE_CONSUMER_SECRET, SALESFORCE_INSTANCE_URL")
        print("and enable the Client Credentials Flow (with a Run-As user) on the External Client App.")
        return

    client = SalesforceCRM()
    print("Fetching client-credentials token…")
    client._fetch_token()                                    # surfaces auth errors early (never prints token)
    print("  token acquired; instance:", client._instance)

    seed = {k: FIXTURES[k] for k in list(FIXTURES)[:3]}
    print(f"\nSeeding {len(seed)} accounts (idempotent upsert by Website)…")
    ids = seed_from_fixture(seed, client)
    for aid, sid in ids.items():
        print(f"  {aid}  {seed[aid].company:<22} {seed[aid].domain:<20} → account {sid}")

    aid = next(iter(seed))
    tool = SalesforceCRMTool(seed, client)
    res = tool.run({"account_id": aid, "need": seed[aid].need})
    print(f"\nresolve {aid} via crm_enrich → status={res.data.get('status')}: {res.text}")

    acct = client.get_account(ids[aid])
    print(f"read account {ids[aid]}: name={acct.get('Name')} website={acct.get('Website')} "
          f"industry={acct.get('Industry')}")

    reg = ToolRegistry(ApprovalPolicy(mode="deny_side_effects", approver=lambda spec: True))
    reg.register(SalesforceUpdateTool(client))
    upd = reg.run("crm_update", {"account_id": ids[aid], "fields": {"Industry": "Technology"}})
    print(f"\ngoverned crm_update → ok={upd.ok}: {upd.text}")
    print(f"audit: {reg.audit[-1]}")


if __name__ == "__main__":
    run()
