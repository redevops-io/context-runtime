"""HubSpot CRM connector (Phase 5) — a real backend for the ``crm`` capability of the Revenue &
Intelligence tenant.

Reads companies/contacts/deals over the HubSpot CRM v3 REST API and, behind an approval gate, updates a
record. Authenticated by a **Service Key** (HubSpot's system-to-system credential) supplied as
``HUBSPOT_SERVICE_KEY`` (or ``HUBSPOT_API_KEY``) and sent as ``Authorization: Bearer …`` — the token is
read from the environment and never logged. With no token the connector reports itself unavailable and the
tenant falls back to the offline fixture, so nothing here is required for the base benchmark.

This is the design-partner path (plan §I): a live mission can resolve an account against real CRM state
before paying for external enrichment, and stage a governed write back. Dependency-free (stdlib urllib).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from ..tools.base import ToolResult, ToolSpec
from ..types import Hit

_BASE = "https://api.hubapi.com"
_COMPANY_PROPS = ["name", "domain", "industry", "numberofemployees", "annualrevenue", "city", "country"]


class HubSpotError(RuntimeError):
    pass


def _token() -> str:
    return os.getenv("HUBSPOT_SERVICE_KEY") or os.getenv("HUBSPOT_API_KEY") or ""


def token_present() -> bool:
    return bool(_token())


def _request(method: str, path: str, body: dict | None = None, *, timeout: float = 20.0) -> dict:
    """One authenticated CRM v3 call. Raises HubSpotError on transport / non-2xx (the caller decides whether
    to fall back). The token is attached here and never returned or logged."""
    token = _token()
    if not token:
        raise HubSpotError("no HUBSPOT_SERVICE_KEY / HUBSPOT_API_KEY in environment")
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{_BASE}{path}", data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:300] if e.fp else ""
        raise HubSpotError(f"HubSpot {method} {path} → {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise HubSpotError(f"HubSpot {method} {path} unreachable: {e.reason}") from None


class HubSpotCRM:
    """Thin client over the CRM v3 company API (the subset the tenant needs)."""

    def search_company(self, *, domain: str | None = None, name: str | None = None) -> dict | None:
        if domain:
            flt = {"propertyName": "domain", "operator": "EQ", "value": domain}
        elif name:
            flt = {"propertyName": "name", "operator": "EQ", "value": name}
        else:
            return None
        body = {"filterGroups": [{"filters": [flt]}], "properties": _COMPANY_PROPS, "limit": 1}
        res = _request("POST", "/crm/v3/objects/companies/search", body)
        results = res.get("results", [])
        return results[0] if results else None

    def get_company(self, company_id: str) -> dict:
        props = ",".join(_COMPANY_PROPS)
        return _request("GET", f"/crm/v3/objects/companies/{company_id}?properties={props}")

    def create_company(self, properties: dict) -> str:
        res = _request("POST", "/crm/v3/objects/companies", {"properties": properties})
        return res.get("id", "")

    def update_company(self, company_id: str, properties: dict) -> dict:
        return _request("PATCH", f"/crm/v3/objects/companies/{company_id}", {"properties": properties})

    def upsert_company(self, properties: dict) -> str:
        """Idempotent seed: return the id of the company with this domain, creating it if absent."""
        existing = self.search_company(domain=properties.get("domain"))
        if existing:
            return existing["id"]
        return self.create_company(properties)


class HubSpotCRMTool:
    """The ``crm`` provider as a live ToolPlugin. Registered under the same name (``crm_enrich``) the tenant
    already calls, so it transparently replaces the offline fixture when a Service Key is present. Given an
    account it resolves the company in HubSpot by domain and returns the CRM-known evidence (identity /
    firmographics) — the "record resolved without paid enrichment" path. Falls back to a miss on any error."""

    name = "crm"

    def __init__(self, fixtures: dict | None = None, client: HubSpotCRM | None = None):
        self._fx = fixtures or {}
        self._client = client or HubSpotCRM()
        self.live = token_present()

    def spec(self) -> ToolSpec:
        return ToolSpec(name="crm_enrich", description="Resolve an account against HubSpot CRM (read-only).",
                        parameters={"type": "object", "properties": {
                            "account_id": {"type": "string"}, "need": {"type": "string"}}})

    def run(self, args: dict) -> ToolResult:
        fx = self._fx.get(args.get("account_id", ""))
        domain = getattr(fx, "domain", None) or args.get("domain")
        if not (self.live and domain):
            return ToolResult(ok=True, hits=[], data={"status": "missing", "live": self.live}, text="no CRM record")
        try:
            company = self._client.search_company(domain=domain)
        except HubSpotError:
            return ToolResult(ok=True, hits=[], data={"status": "missing", "error": "crm unreachable"}, text="crm error")
        if not company:
            return ToolResult(ok=True, hits=[], data={"status": "missing"}, text="not in CRM")
        props = company.get("properties", {})
        value = props.get("name") or domain
        hit = Hit(chunk_id=f"hubspot::{company['id']}", filename="hubspot", source="crm",
                  text=f"crm: {value} ({domain})", score=0.95)
        return ToolResult(ok=True, hits=[hit], data={"status": "correct", "company_id": company["id"],
                          "properties": props, "cost": 0.0}, text=hit.text)


class HubSpotUpdateTool:
    """Governed write-back: update a company property. SIDE-EFFECTING + APPROVAL-REQUIRED."""

    def __init__(self, client: HubSpotCRM | None = None):
        self._client = client or HubSpotCRM()

    def spec(self) -> ToolSpec:
        return ToolSpec(name="crm_update", description="Update a HubSpot company property.",
                        parameters={"type": "object", "properties": {
                            "company_id": {"type": "string"}, "properties": {"type": "object"}}},
                        side_effecting=True, approval_required=True)

    def run(self, args: dict) -> ToolResult:
        try:
            res = self._client.update_company(args["company_id"], args.get("properties", {}))
        except HubSpotError as e:
            return ToolResult(ok=False, data={"error": str(e)}, text=f"update failed: {e}")
        return ToolResult(ok=True, data={"updated": res.get("id")}, text=f"updated company {res.get('id')}")


def seed_from_fixture(fixtures: dict, client: HubSpotCRM | None = None) -> dict[str, str]:
    """Idempotently create the benchmark companies in HubSpot so a fresh test account has data to resolve
    against. Returns {account_id: company_id}."""
    client = client or HubSpotCRM()
    out: dict[str, str] = {}
    for aid, fx in fixtures.items():
        props = {"name": fx.company, "domain": fx.domain}
        if fx.need == "firmographics":
            props["numberofemployees"] = "".join(ch for ch in fx.truth if ch.isdigit()) or "0"
        out[aid] = client.upsert_company(props)
    return out
