"""HubSpot CRM connector — offline unit tests (no network, no key required).

Verifies the graceful no-token fallback, the request construction (method/path/body + Bearer auth via a
stubbed transport), and that the tenant transparently falls back to the offline fixture with no key. Live
end-to-end is covered by examples/hubspot_crm_smoke.py, which needs a Service Key.
"""
from __future__ import annotations

import pytest

from context_runtime.integrations import hubspot_crm as hs
from context_runtime.integrations.revenue_intelligence import Fixture, RevenueIntelligenceTenant


def test_no_token_reports_unavailable(monkeypatch):
    monkeypatch.delenv("HUBSPOT_SERVICE_KEY", raising=False)
    monkeypatch.delenv("HUBSPOT_API_KEY", raising=False)
    assert hs.token_present() is False
    fx = {"a1": Fixture("a1", "Acme", "acme.io", "company_identity", "crm", "Acme")}
    tool = hs.HubSpotCRMTool(fx)
    res = tool.run({"account_id": "a1", "need": "company_identity"})
    assert res.ok and res.data["status"] == "missing" and res.data["live"] is False


def test_request_uses_bearer_and_correct_endpoints(monkeypatch):
    calls = []

    def fake_request(method, path, body=None, *, timeout=20.0):
        calls.append((method, path, body))
        if path.endswith("/search"):
            return {"results": [{"id": "77", "properties": {"name": "Acme", "domain": "acme.io"}}]}
        if method == "POST":
            return {"id": "99"}
        if method == "PATCH":
            return {"id": path.rsplit("/", 1)[-1]}
        return {"id": "77", "properties": {"name": "Acme"}}

    monkeypatch.setattr(hs, "_request", fake_request)
    client = hs.HubSpotCRM()

    found = client.search_company(domain="acme.io")
    assert found["id"] == "77"
    assert calls[-1][0] == "POST" and calls[-1][1] == "/crm/v3/objects/companies/search"
    assert calls[-1][2]["filterGroups"][0]["filters"][0] == {
        "propertyName": "domain", "operator": "EQ", "value": "acme.io"}

    new_id = client.create_company({"name": "Beta", "domain": "beta.io"})
    assert new_id == "99" and calls[-1][:2] == ("POST", "/crm/v3/objects/companies")

    client.update_company("77", {"industry": "SOFTWARE"})
    assert calls[-1][0] == "PATCH" and calls[-1][1] == "/crm/v3/objects/companies/77"

    # upsert is idempotent: domain already exists → returns its id, no create
    n_before = len(calls)
    assert client.upsert_company({"name": "Acme", "domain": "acme.io"}) == "77"
    assert not any(c[0] == "POST" and c[1] == "/crm/v3/objects/companies" for c in calls[n_before:])


def test_bearer_header_attached(monkeypatch):
    """The token is sent as a Bearer header (checked via a stubbed urlopen — value never surfaced)."""
    monkeypatch.setenv("HUBSPOT_SERVICE_KEY", "pat-na2-test-000")
    seen = {}

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"results": []}'

    def fake_urlopen(req, timeout=20.0):
        seen["auth"] = req.get_header("Authorization")
        seen["url"] = req.full_url
        return _Resp()

    monkeypatch.setattr(hs.urllib.request, "urlopen", fake_urlopen)
    hs.HubSpotCRM().search_company(domain="x.io")
    assert seen["auth"] == "Bearer pat-na2-test-000"
    assert seen["url"].startswith("https://api.hubapi.com/crm/v3/objects/companies/search")


def test_tenant_falls_back_to_fixture_without_key(monkeypatch):
    monkeypatch.delenv("HUBSPOT_SERVICE_KEY", raising=False)
    monkeypatch.delenv("HUBSPOT_API_KEY", raising=False)
    fx = {"a1": Fixture("a1", "Acme", "acme.io", "company_identity", "pdl", "Acme Inc")}
    t = RevenueIntelligenceTenant(fx, approver=lambda spec: False)
    # with no key the crm provider is the offline ProviderTool: crm_enrich returns from the fixture,
    # never the network — so enrich runs fully offline.
    crm = t.registry.run("crm_enrich", {"account_id": "a1", "need": "company_identity"})
    assert crm.ok and "live" not in crm.data           # offline ProviderTool, not HubSpotCRMTool
    assert t.enrich("a1").outcome in ("correct", "wrong", "missing")


@pytest.mark.skipif(not hs.token_present(), reason="no HubSpot Service Key in env")
def test_live_search_smoke():
    # only runs where a real key is exported; a broad no-op query should return without raising
    hs.HubSpotCRM().search_company(domain="example-does-not-exist-zzz.io")
