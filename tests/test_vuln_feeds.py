"""Hermetic tests for vuln_feeds — canned NVD/OSV payloads, no network I/O.

`_get_json` / `_post_json` are monkeypatched to return parsed canned payloads so the
fetchers exercise their normalization paths without touching the network.
"""
from __future__ import annotations

import json

from context_runtime.integrations import vuln_feeds
from context_runtime.integrations.vuln_feeds import (
    VulnRecord,
    dedupe,
    fetch_nvd,
    fetch_osv,
    fetch_snyk,
    ingest,
)

# --------------------------------------------------------------------------- canned data

_NVD_JSON = """
{"vulnerabilities":[
  {"cve":{
    "id":"CVE-2024-5555",
    "published":"2024-05-01T00:00:00.000",
    "lastModified":"2024-05-02T00:00:00.000",
    "descriptions":[
      {"lang":"es","value":"desbordamiento de bufer"},
      {"lang":"en","value":"A buffer overflow in acme-lib allows RCE."}
    ],
    "metrics":{"cvssMetricV31":[
      {"cvssData":{"baseScore":9.8,"baseSeverity":"critical"}}
    ]},
    "references":[
      {"url":"https://example.com/advisory/5555"},
      {"url":"https://nvd.nist.gov/vuln/detail/CVE-2024-5555"}
    ]
  }}
]}
"""

_OSV_JSON = """
{"vulns":[
  {
    "id":"GHSA-aaaa-bbbb-cccc",
    "aliases":["CVE-2024-5555"],
    "summary":"acme-lib buffer overflow",
    "published":"2024-05-01T00:00:00Z",
    "modified":"2024-05-03T00:00:00Z",
    "affected":[
      {"package":{"name":"acme-lib","ecosystem":"PyPI"},
       "ranges":[{"type":"ECOSYSTEM","events":[
         {"introduced":"0"},
         {"fixed":"2.31.0"}
       ]}]}
    ],
    "references":[{"url":"https://example.com/osv/5555"}]
  }
]}
"""

_NVD_PARSED = json.loads(_NVD_JSON)
_OSV_PARSED = json.loads(_OSV_JSON)


class _FakeStore:
    """In-memory VulnStore that records what it was asked to upsert."""

    def __init__(self):
        self.records: list[VulnRecord] = []

    def upsert_vulns(self, records: list[VulnRecord]) -> int:
        self.records = list(records)
        return len(self.records)

    def query_vulns(self, **filters) -> list[dict]:
        return []


# --------------------------------------------------------------------------- NVD


def test_nvd_normalization(monkeypatch):
    monkeypatch.setattr(vuln_feeds, "_get_json", lambda url, headers=None: _NVD_PARSED)
    recs = fetch_nvd("acme-lib")
    assert len(recs) == 1
    r = recs[0]
    assert r.cve_id == "CVE-2024-5555"
    assert r.source == "nvd"
    # severity uppercased from the canned lowercase "critical"
    assert r.severity == "CRITICAL"
    assert isinstance(r.cvss, float) and r.cvss == 9.8
    # english description picked, not the Spanish one
    assert r.summary == "A buffer overflow in acme-lib allows RCE."
    assert "https://example.com/advisory/5555" in r.references


# --------------------------------------------------------------------------- OSV


def test_osv_normalization(monkeypatch):
    monkeypatch.setattr(
        vuln_feeds, "_post_json", lambda url, body, headers=None: _OSV_PARSED
    )
    recs = fetch_osv("acme-lib", "PyPI")
    assert len(recs) == 1
    r = recs[0]
    assert r.package == "acme-lib"
    assert r.ecosystem == "PyPI"
    # fixed_version pulled from the ranges 'fixed' event
    assert r.fixed_version == "2.31.0"
    # cve_id resolved to the CVE alias, not the GHSA id
    assert r.cve_id == "CVE-2024-5555"
    assert r.source == "osv"


# --------------------------------------------------------------------------- dedupe


def test_dedupe_collapses_shared_cve(monkeypatch):
    monkeypatch.setattr(vuln_feeds, "_get_json", lambda url, headers=None: _NVD_PARSED)
    monkeypatch.setattr(
        vuln_feeds, "_post_json", lambda url, body, headers=None: _OSV_PARSED
    )
    combined = fetch_nvd("acme-lib") + fetch_osv("acme-lib", "PyPI")
    assert len(combined) == 2
    collapsed = dedupe(combined)
    assert len(collapsed) == 1
    r = collapsed[0]
    assert r.cve_id == "CVE-2024-5555"
    # richer record wins: OSV carried the fixed_version
    assert r.fixed_version == "2.31.0"


# --------------------------------------------------------------------------- Snyk


def test_snyk_returns_empty_without_token(monkeypatch):
    monkeypatch.delenv("SNYK_TOKEN", raising=False)
    assert fetch_snyk() == []


# --------------------------------------------------------------------------- ingest


def test_ingest_dedupes_and_upserts(monkeypatch):
    monkeypatch.setattr(vuln_feeds, "_get_json", lambda url, headers=None: _NVD_PARSED)
    monkeypatch.setattr(
        vuln_feeds, "_post_json", lambda url, body, headers=None: _OSV_PARSED
    )
    combined = fetch_nvd("acme-lib") + fetch_osv("acme-lib", "PyPI")
    store = _FakeStore()
    count = ingest(combined, store)
    assert count == 1
    assert len(store.records) == 1
    assert store.records[0].cve_id == "CVE-2024-5555"
