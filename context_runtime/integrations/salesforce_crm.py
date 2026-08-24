"""Salesforce CRM connector (Phase 5) — a second live backend for the ``crm`` capability.

Reads Accounts over the Salesforce REST/SOQL API and, behind the approval gate, updates one. Authenticated
by the **OAuth 2.0 Client Credentials flow** (server-to-server, no user password): a Connected/External
Client App's consumer key + secret exchange for a short-lived bearer token, which is then sent on every API
call. Credentials come from ``SALESFORCE_CONSUMER_KEY`` / ``SALESFORCE_CONSUMER_SECRET`` (or the
``…_CLIENT_ID`` / ``…_CLIENT_SECRET`` aliases) and ``SALESFORCE_INSTANCE_URL`` — read from the environment,
never logged. With any missing the connector reports itself unavailable and the tenant falls back to the
offline fixture, so nothing here is required for the base benchmark.

Mirrors the HubSpot connector's shape (search/get/create/update + an approval-gated write + seed), so the
Revenue & Intelligence tenant treats the two design-partner CRMs interchangeably. Dependency-free (stdlib).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from ..tools.base import ToolResult, ToolSpec
from ..types import Hit

_API_VERSION = os.getenv("SALESFORCE_API_VERSION", "v60.0")
_ACCOUNT_FIELDS = ["Id", "Name", "Website", "Industry", "NumberOfEmployees", "BillingCity", "BillingCountry"]


class SalesforceError(RuntimeError):
    pass


def _creds() -> tuple[str, str, str]:
    cid = os.getenv("SALESFORCE_CONSUMER_KEY") or os.getenv("SALESFORCE_CLIENT_ID") or ""
    secret = os.getenv("SALESFORCE_CONSUMER_SECRET") or os.getenv("SALESFORCE_CLIENT_SECRET") or ""
    instance = (os.getenv("SALESFORCE_INSTANCE_URL") or "").rstrip("/")
    return cid, secret, instance


def token_present() -> bool:
    cid, secret, instance = _creds()
    return bool(cid and secret and instance)


class SalesforceCRM:
    """Thin client over the Salesforce Account API. Fetches a client-credentials token on first use and
    reuses it (re-fetching on a 401)."""

    def __init__(self):
        self._token: str | None = None
        self._instance = _creds()[2]

    # ── auth ──
    def _fetch_token(self) -> None:
        cid, secret, instance = _creds()
        if not (cid and secret and instance):
            raise SalesforceError("missing SALESFORCE_CONSUMER_KEY/SECRET or SALESFORCE_INSTANCE_URL")
        body = urllib.parse.urlencode({
            "grant_type": "client_credentials", "client_id": cid, "client_secret": secret}).encode()
        req = urllib.request.Request(f"{instance}/services/oauth2/token", data=body, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise SalesforceError(f"token request → {e.code}: {e.read().decode()[:300]}") from None
        except urllib.error.URLError as e:
            raise SalesforceError(f"token endpoint unreachable: {e.reason}") from None
        self._token = data["access_token"]
        self._instance = data.get("instance_url", self._instance).rstrip("/")

    def _request(self, method: str, path: str, body: dict | None = None, *, _retry: bool = True) -> dict:
        if self._token is None:
            self._fetch_token()
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(f"{self._instance}{path}", data=data, method=method)
        req.add_header("Authorization", f"Bearer {self._token}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read().decode()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            if e.code == 401 and _retry:                     # token expired → refresh once
                self._token = None
                return self._request(method, path, body, _retry=False)
            raise SalesforceError(f"Salesforce {method} {path} → {e.code}: {e.read().decode()[:300]}") from None
        except urllib.error.URLError as e:
            raise SalesforceError(f"Salesforce {method} {path} unreachable: {e.reason}") from None

    # ── accounts ──
    def query(self, soql: str) -> list[dict]:
        path = f"/services/data/{_API_VERSION}/query?q={urllib.parse.quote(soql)}"
        return self._request("GET", path).get("records", [])

    def search_account(self, *, domain: str | None = None, name: str | None = None) -> dict | None:
        fields = ", ".join(_ACCOUNT_FIELDS)
        if domain:
            where = f"Website LIKE '%{domain}%'"
        elif name:
            where = f"Name = '{name}'"
        else:
            return None
        rows = self.query(f"SELECT {fields} FROM Account WHERE {where} LIMIT 1")
        return rows[0] if rows else None

    def get_account(self, account_id: str) -> dict:
        return self._request("GET", f"/services/data/{_API_VERSION}/sobjects/Account/{account_id}")

    def create_account(self, fields: dict) -> str:
        res = self._request("POST", f"/services/data/{_API_VERSION}/sobjects/Account", fields)
        return res.get("id", "")

    def update_account(self, account_id: str, fields: dict) -> dict:
        # Salesforce PATCH returns 204 No Content on success
        self._request("PATCH", f"/services/data/{_API_VERSION}/sobjects/Account/{account_id}", fields)
        return {"id": account_id, "updated": True}

    def upsert_account(self, fields: dict) -> str:
        existing = self.search_account(domain=fields.get("Website"))
        if existing:
            return existing["Id"]
        return self.create_account(fields)


class SalesforceCRMTool:
    """The ``crm`` provider backed by Salesforce. Registers under ``crm_enrich`` so it swaps in for the
    offline fixture, resolving an account by domain and returning the CRM-known evidence."""

    name = "crm"

    def __init__(self, fixtures: dict | None = None, client: SalesforceCRM | None = None):
        self._fx = fixtures or {}
        self._client = client or SalesforceCRM()
        self.live = token_present()

    def spec(self) -> ToolSpec:
        return ToolSpec(name="crm_enrich", description="Resolve an account against Salesforce (read-only).",
                        parameters={"type": "object", "properties": {
                            "account_id": {"type": "string"}, "need": {"type": "string"}}})

    def run(self, args: dict) -> ToolResult:
        fx = self._fx.get(args.get("account_id", ""))
        domain = getattr(fx, "domain", None) or args.get("domain")
        if not (self.live and domain):
            return ToolResult(ok=True, hits=[], data={"status": "missing", "live": self.live}, text="no CRM record")
        try:
            acct = self._client.search_account(domain=domain)
        except SalesforceError:
            return ToolResult(ok=True, hits=[], data={"status": "missing", "error": "crm unreachable"}, text="crm error")
        if not acct:
            return ToolResult(ok=True, hits=[], data={"status": "missing"}, text="not in CRM")
        value = acct.get("Name") or domain
        hit = Hit(chunk_id=f"salesforce::{acct['Id']}", filename="salesforce", source="crm",
                  text=f"crm: {value} ({domain})", score=0.95)
        return ToolResult(ok=True, hits=[hit], data={"status": "correct", "account_id": acct["Id"],
                          "record": acct, "cost": 0.0}, text=hit.text)


class SalesforceUpdateTool:
    """Governed write-back: update an Account field. SIDE-EFFECTING + APPROVAL-REQUIRED."""

    def __init__(self, client: SalesforceCRM | None = None):
        self._client = client or SalesforceCRM()

    def spec(self) -> ToolSpec:
        return ToolSpec(name="crm_update", description="Update a Salesforce Account field.",
                        parameters={"type": "object", "properties": {
                            "account_id": {"type": "string"}, "fields": {"type": "object"}}},
                        side_effecting=True, approval_required=True)

    def run(self, args: dict) -> ToolResult:
        try:
            self._client.update_account(args["account_id"], args.get("fields", {}))
        except SalesforceError as e:
            return ToolResult(ok=False, data={"error": str(e)}, text=f"update failed: {e}")
        return ToolResult(ok=True, data={"updated": args["account_id"]}, text=f"updated account {args['account_id']}")


def seed_from_fixture(fixtures: dict, client: SalesforceCRM | None = None) -> dict[str, str]:
    """Idempotently create the benchmark companies as Salesforce Accounts. Returns {account_id: sf_id}."""
    client = client or SalesforceCRM()
    out: dict[str, str] = {}
    for aid, fx in fixtures.items():
        fields = {"Name": fx.company, "Website": fx.domain}
        if fx.need == "firmographics":
            n = "".join(ch for ch in fx.truth if ch.isdigit())
            if n:
                fields["NumberOfEmployees"] = int(n)
        out[aid] = client.upsert_account(fields)
    return out
