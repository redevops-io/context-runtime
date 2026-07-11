"""Hermetic tests for the supply-chain scanner — parse canned Trivy output; graceful degrade."""
from __future__ import annotations

from context_runtime.integrations.supply_chain import ScanResult, SupplyChainScanner

_TRIVY_JSON = """
{"Results":[
  {"Target":"requirements.txt","Vulnerabilities":[
    {"VulnerabilityID":"CVE-2024-0001","PkgName":"requests","InstalledVersion":"2.20.0","FixedVersion":"2.31.0","Severity":"HIGH","Title":"requests TLS bypass"},
    {"VulnerabilityID":"CVE-2023-9999","PkgName":"jinja2","InstalledVersion":"2.11.0","FixedVersion":"","Severity":"CRITICAL","Title":"jinja2 sandbox escape"},
    {"VulnerabilityID":"CVE-2022-1234","PkgName":"idna","InstalledVersion":"2.8","FixedVersion":"3.3","Severity":"LOW","Title":"idna dos"}],
   "Secrets":[{"RuleID":"aws-access-key"}],
   "Misconfigurations":[{"ID":"DS002"}]}
]}
"""


def test_parse_trivy_normalizes_and_sorts_by_severity():
    r = SupplyChainScanner._parse_trivy(_TRIVY_JSON, "requirements.txt")
    assert r.ok and r.scanner == "trivy"
    assert len(r.findings) == 3
    # CRITICAL first, then HIGH, then LOW
    assert [f.severity for f in r.findings] == ["CRITICAL", "HIGH", "LOW"]
    assert r.secrets == 1 and r.misconfigs == 1
    # the fixed/"patched pin" is carried through
    high = next(f for f in r.findings if f.id == "CVE-2024-0001")
    assert high.pkg == "requests" and high.fixed == "2.31.0"


def test_summary_counts():
    r = SupplyChainScanner._parse_trivy(_TRIVY_JSON, "requirements.txt")
    s = r.summary()
    assert s["total"] == 3
    assert s["by_severity"] == {"CRITICAL": 1, "HIGH": 1, "LOW": 1}
    assert s["fixable"] == 2  # jinja2 has no fix yet
    assert s["secrets"] == 1 and s["misconfigs"] == 1


def test_graceful_degrade_when_trivy_absent(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _name: None)
    r = SupplyChainScanner().scan_fs("/whatever")
    assert isinstance(r, ScanResult) and r.ok is False
    assert "not installed" in r.note


def test_bad_output_is_handled():
    r = SupplyChainScanner._parse_trivy("not json", "x")
    assert r.ok is False and "unparseable" in r.note
