"""Salesforce CRM connector — offline unit tests (no network, no creds required)."""
from __future__ import annotations

import pytest

from context_runtime.integrations import salesforce_crm as sf
from context_runtime.integrations.revenue_intelligence import Fixture


def _clear(monkeypatch):
    for k in ("SALESFORCE_CONSUMER_KEY", "SALESFORCE_CONSUMER_SECRET", "SALESFORCE_INSTANCE_URL",
              "SALESFORCE_CLIENT_ID", "SALESFORCE_CLIENT_SECRET"):
        monkeypatch.delenv(k, raising=False)


def test_no_creds_reports_unavailable(monkeypatch):
    _clear(monkeypatch)
    assert sf.token_present() is False
    fx = {"a1": Fixture("a1", "Acme", "acme.io", "company_identity", "crm", "Acme")}
    res = sf.SalesforceCRMTool(fx).run({"account_id": "a1", "need": "company_identity"})
    assert res.ok and res.data["status"] == "missing" and res.data["live"] is False


def test_client_credentials_token_request(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("SALESFORCE_CONSUMER_KEY", "ck")
    monkeypatch.setenv("SALESFORCE_CONSUMER_SECRET", "cs")
    monkeypatch.setenv("SALESFORCE_INSTANCE_URL", "https://x.my.salesforce.com")
    seen = {}

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            return b'{"access_token": "TOKEN123", "instance_url": "https://x.my.salesforce.com"}'

    def fake_urlopen(req, timeout=20):
        seen["url"] = req.full_url
        seen["body"] = req.data.decode()
        return _Resp()

    monkeypatch.setattr(sf.urllib.request, "urlopen", fake_urlopen)
    client = sf.SalesforceCRM()
    client._fetch_token()
    assert client._token == "TOKEN123"
    assert seen["url"] == "https://x.my.salesforce.com/services/oauth2/token"
    assert "grant_type=client_credentials" in seen["body"]
    assert "client_id=ck" in seen["body"] and "client_secret=cs" in seen["body"]


def test_soql_and_bearer_on_query(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("SALESFORCE_CONSUMER_KEY", "ck")
    monkeypatch.setenv("SALESFORCE_CONSUMER_SECRET", "cs")
    monkeypatch.setenv("SALESFORCE_INSTANCE_URL", "https://x.my.salesforce.com")
    seen = {}

    class _Resp:
        def __init__(self, payload): self._p = payload
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._p

    def fake_urlopen(req, timeout=20):
        if req.full_url.endswith("/token"):
            return _Resp(b'{"access_token": "T", "instance_url": "https://x.my.salesforce.com"}')
        seen["auth"] = req.get_header("Authorization")
        seen["url"] = req.full_url
        return _Resp(b'{"records": [{"Id": "001", "Name": "Acme", "Website": "acme.io"}]}')

    monkeypatch.setattr(sf.urllib.request, "urlopen", fake_urlopen)
    acct = sf.SalesforceCRM().search_account(domain="acme.io")
    assert acct["Id"] == "001"
    assert seen["auth"] == "Bearer T"
    assert "/services/data/" in seen["url"] and "query?q=" in seen["url"]
    assert "Website+LIKE" in seen["url"] or "Website%20LIKE" in seen["url"]


def test_token_refresh_on_401(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("SALESFORCE_CONSUMER_KEY", "ck")
    monkeypatch.setenv("SALESFORCE_CONSUMER_SECRET", "cs")
    monkeypatch.setenv("SALESFORCE_INSTANCE_URL", "https://x.my.salesforce.com")
    state = {"api_calls": 0, "tokens": 0}

    class _Resp:
        def __init__(self, payload): self._p = payload
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self._p

    def fake_urlopen(req, timeout=20):
        if req.full_url.endswith("/token"):
            state["tokens"] += 1
            return _Resp(b'{"access_token": "T", "instance_url": "https://x.my.salesforce.com"}')
        state["api_calls"] += 1
        if state["api_calls"] == 1:
            raise sf.urllib.error.HTTPError(req.full_url, 401, "expired", {}, None)
        return _Resp(b'{"records": []}')

    monkeypatch.setattr(sf.urllib.request, "urlopen", fake_urlopen)
    sf.SalesforceCRM().query("SELECT Id FROM Account")
    assert state["tokens"] == 2 and state["api_calls"] == 2   # refreshed once, retried once


@pytest.mark.skipif(not sf.token_present(), reason="no Salesforce credentials in env")
def test_live_query_smoke():
    sf.SalesforceCRM().search_account(domain="example-does-not-exist-zzz.io")
