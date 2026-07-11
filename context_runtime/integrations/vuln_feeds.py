"""vuln_feeds — vulnerability-feed ingestion for the Security & Compliance block.

Pulls normalized vulnerability records from public feeds (NVD 2.0, OSV.dev, and — when a
token is present — Snyk) and hands them to a :class:`VulnStore` (backed by Doris in the
deployed stack). This is the *feed* half of the block; SupplyChainScanner is the pre-deploy
scanner half and Edge Sentinel triages the merged findings.

All network I/O is factored through two private helpers (:func:`_get_json` /
:func:`_post_json`) built on stdlib ``urllib`` + ``json`` so tests can monkeypatch them and
the module carries no third-party deps. Every fetcher wraps its network call in try/except and
returns ``[]`` on any failure or missing key — a dead feed never breaks ingestion.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

_SEV_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "UNKNOWN": 4}


@dataclass(frozen=True)
class VulnRecord:
    cve_id: str
    source: str
    package: str
    ecosystem: str
    severity: str
    cvss: float
    fixed_version: str
    vulnerable_range: str
    published: str
    modified: str
    summary: str
    references: tuple = field(default_factory=tuple)
    aliases: tuple = field(default_factory=tuple)


# --------------------------------------------------------------------------- network I/O


def _get_json(url: str, headers: dict | None = None) -> dict:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(url: str, body: dict, headers: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8")
    hdrs = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


# --------------------------------------------------------------------------- helpers


def _band_from_score(score: float) -> str:
    """Derive a CVSS v3 severity band from a numeric score."""
    if score >= 9.0:
        return "CRITICAL"
    if score >= 7.0:
        return "HIGH"
    if score >= 4.0:
        return "MEDIUM"
    if score > 0.0:
        return "LOW"
    return "UNKNOWN"


def _as_tuple(seq) -> tuple:
    if not seq:
        return ()
    return tuple(seq)


# --------------------------------------------------------------------------- NVD


def fetch_nvd(query: str, limit: int = 20) -> list[VulnRecord]:
    """GET the NVD 2.0 keyword-search endpoint and normalize into VulnRecords."""
    url = (
        "https://services.nvd.nist.gov/rest/json/cves/2.0"
        f"?keywordSearch={urllib.parse.quote(query)}&resultsPerPage={int(limit)}"
    )
    headers: dict = {}
    api_key = os.environ.get("NVD_API_KEY")
    if api_key:
        headers["apiKey"] = api_key
    try:
        data = _get_json(url, headers=headers)
        out: list[VulnRecord] = []
        for item in data.get("vulnerabilities", [])[:limit]:
            cve = item.get("cve", {})
            cve_id = cve.get("id", "")
            if not cve_id:
                continue

            summary = ""
            for desc in cve.get("descriptions", []):
                if desc.get("lang") == "en":
                    summary = desc.get("value", "")
                    break

            severity, cvss = "UNKNOWN", 0.0
            metrics = cve.get("metrics", {})
            for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
                entries = metrics.get(key)
                if entries:
                    cdata = entries[0].get("cvssData", {})
                    cvss = float(cdata.get("baseScore", 0.0) or 0.0)
                    severity = (
                        cdata.get("baseSeverity")
                        or entries[0].get("baseSeverity")
                        or _band_from_score(cvss)
                    )
                    break

            refs = tuple(
                r.get("url", "") for r in cve.get("references", []) if r.get("url")
            )
            out.append(
                VulnRecord(
                    cve_id=cve_id,
                    source="nvd",
                    package="",
                    ecosystem="",
                    severity=str(severity).upper(),
                    cvss=cvss,
                    fixed_version="",
                    vulnerable_range="",
                    published=cve.get("published", ""),
                    modified=cve.get("lastModified", ""),
                    summary=summary,
                    references=refs,
                    aliases=(cve_id,),
                )
            )
        return out
    except Exception:
        return []


# --------------------------------------------------------------------------- OSV


def _osv_to_record(vuln: dict) -> VulnRecord | None:
    vuln_id = vuln.get("id", "")
    aliases = list(vuln.get("aliases", []) or [])
    if not vuln_id and not aliases:
        return None

    # Prefer a CVE identifier for cve_id, else fall back to the OSV id.
    cve_id = vuln_id
    for a in [vuln_id, *aliases]:
        if a.startswith("CVE-"):
            cve_id = a
            break

    all_aliases = tuple(dict.fromkeys([vuln_id, *aliases]))
    summary = vuln.get("summary") or vuln.get("details") or ""

    severity, cvss = "UNKNOWN", 0.0
    for sev in vuln.get("severity", []) or []:
        if sev.get("type") == "CVSS_V3":
            score = sev.get("score", "")
            try:
                cvss = float(score)
            except (TypeError, ValueError):
                # score is a CVSS vector string, not a number — leave 0.0 but band later.
                cvss = 0.0
            severity = _band_from_score(cvss)
            break

    refs = tuple(
        r.get("url", "") for r in vuln.get("references", []) or [] if r.get("url")
    )

    package, ecosystem, fixed_version, vulnerable_range = "", "", "", ""
    affected = vuln.get("affected", []) or []
    if affected:
        first = affected[0]
        pkg = first.get("package", {}) or {}
        package = pkg.get("name", "")
        ecosystem = pkg.get("ecosystem", "")
        introduced, fixed = "", ""
        for rng in first.get("ranges", []) or []:
            for ev in rng.get("events", []) or []:
                if "introduced" in ev and not introduced:
                    introduced = ev["introduced"]
                if "fixed" in ev:
                    fixed = ev["fixed"]
        fixed_version = fixed
        if introduced or fixed:
            lo = introduced or "0"
            vulnerable_range = f">={lo}" + (f", <{fixed}" if fixed else "")

    return VulnRecord(
        cve_id=cve_id,
        source="osv",
        package=package,
        ecosystem=ecosystem,
        severity=severity.upper(),
        cvss=cvss,
        fixed_version=fixed_version,
        vulnerable_range=vulnerable_range,
        published=vuln.get("published", ""),
        modified=vuln.get("modified", ""),
        summary=summary,
        references=refs,
        aliases=all_aliases,
    )


def fetch_osv(package: str, ecosystem: str, limit: int = 50) -> list[VulnRecord]:
    """POST the OSV.dev query endpoint for a package/ecosystem pair."""
    body = {"package": {"name": package, "ecosystem": ecosystem}}
    try:
        data = _post_json("https://api.osv.dev/v1/query", body)
        out: list[VulnRecord] = []
        for vuln in data.get("vulns", [])[:limit]:
            rec = _osv_to_record(vuln)
            if rec is not None:
                out.append(rec)
        return out
    except Exception:
        return []


def fetch_osv_by_id(vuln_id: str) -> list[VulnRecord]:
    """GET a single OSV advisory by id."""
    try:
        data = _get_json(f"https://api.osv.dev/v1/vulns/{vuln_id}")
        rec = _osv_to_record(data)
        return [rec] if rec is not None else []
    except Exception:
        return []


# --------------------------------------------------------------------------- Snyk


def fetch_snyk(org_id: str = "", limit: int = 50) -> list[VulnRecord]:
    """Best-effort GET of Snyk org issues; returns [] immediately when SNYK_TOKEN is unset."""
    token = os.environ.get("SNYK_TOKEN")
    if not token:
        return []
    try:
        org = org_id or os.environ.get("SNYK_ORG_ID", "")
        url = f"https://api.snyk.io/rest/orgs/{org}/issues?version=2024-01-01&limit={int(limit)}"
        headers = {"Authorization": f"token {token}"}
        data = _get_json(url, headers=headers)
        out: list[VulnRecord] = []
        for item in data.get("data", [])[:limit]:
            attrs = item.get("attributes", {}) or {}
            cvss = 0.0
            severity = str(attrs.get("effective_severity_level", "unknown")).upper()
            for sev in attrs.get("severities", []) or []:
                try:
                    cvss = float(sev.get("score", 0.0) or 0.0)
                except (TypeError, ValueError):
                    cvss = 0.0
                break
            problems = attrs.get("problems", []) or []
            cve_id = item.get("id", "")
            aliases = []
            for p in problems:
                pid = p.get("id", "")
                if pid:
                    aliases.append(pid)
                    if pid.startswith("CVE-"):
                        cve_id = pid
            out.append(
                VulnRecord(
                    cve_id=cve_id,
                    source="snyk",
                    package="",
                    ecosystem="",
                    severity=severity,
                    cvss=cvss,
                    fixed_version="",
                    vulnerable_range="",
                    published=attrs.get("created_at", ""),
                    modified=attrs.get("updated_at", ""),
                    summary=attrs.get("title", ""),
                    references=(),
                    aliases=tuple(dict.fromkeys(aliases)),
                )
            )
        return out
    except Exception:
        return []


# --------------------------------------------------------------------------- dedupe / ingest


def _keys(rec: VulnRecord) -> set:
    """The set of identifiers a record is known by (CVE id + aliases)."""
    ids = set(rec.aliases)
    if rec.cve_id:
        ids.add(rec.cve_id)
    return {i for i in ids if i}


def _richer(a: VulnRecord, b: VulnRecord) -> VulnRecord:
    """Prefer the record with more resolved data: a fixed_version, then higher CVSS."""
    if bool(a.fixed_version) != bool(b.fixed_version):
        return a if a.fixed_version else b
    if a.cvss != b.cvss:
        return a if a.cvss > b.cvss else b
    return a


def dedupe(records: list[VulnRecord]) -> list[VulnRecord]:
    """Collapse records sharing a CVE id or overlapping aliases, preferring richer data."""
    kept: list[VulnRecord] = []
    kept_keys: list[set] = []
    for rec in records:
        rkeys = _keys(rec)
        match = -1
        for i, ks in enumerate(kept_keys):
            if rkeys & ks:
                match = i
                break
        if match == -1:
            kept.append(rec)
            kept_keys.append(set(rkeys))
        else:
            kept[match] = _richer(kept[match], rec)
            kept_keys[match] |= rkeys
    return kept


@runtime_checkable
class VulnStore(Protocol):
    def upsert_vulns(self, records: list[VulnRecord]) -> int:
        ...

    def query_vulns(self, **filters) -> list[dict]:
        ...


def ingest(records: list[VulnRecord], store: VulnStore) -> int:
    """Dedupe then upsert into the store; returns the count reported by the store."""
    deduped = dedupe(records)
    return store.upsert_vulns(deduped)


def collect(
    *,
    nvd_query: str | None = None,
    osv_packages: list | None = None,
    snyk: bool = False,
) -> list[VulnRecord]:
    """Run the requested fetchers and return a single deduped, combined list."""
    records: list[VulnRecord] = []
    if nvd_query:
        records.extend(fetch_nvd(nvd_query))
    for pkg in osv_packages or []:
        name, ecosystem = pkg if isinstance(pkg, (tuple, list)) else (pkg, "")
        records.extend(fetch_osv(name, ecosystem))
    if snyk:
        records.extend(fetch_snyk())
    return dedupe(records)
