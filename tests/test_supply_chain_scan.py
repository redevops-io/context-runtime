"""Hermetic tests for container scan + triage — NO docker, NO network."""
from __future__ import annotations

import pytest

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


def test_resolve_image_parses_docker_inspect(monkeypatch):
    sc = SupplyChainScanner()
    monkeypatch.setattr(sc, "_run", lambda cmd: (0, "sha256:abc123\n", ""))
    assert sc.resolve_image("foo") == "sha256:abc123"
    monkeypatch.setattr(sc, "_run", lambda cmd: (1, "", "boom"))
    assert sc.resolve_image("bar") == ""


def test_scan_container_happy_and_not_resolvable(monkeypatch):
    sc = SupplyChainScanner()
    monkeypatch.setattr("shutil.which", lambda n: True)
    monkeypatch.setattr(sc, "resolve_image", lambda c: "img:1")
    monkeypatch.setattr(sc, "scan_image", lambda img: SupplyChainScanner._parse_trivy(_TRIVY_JSON, img))
    r = sc.scan_container("c1")
    assert r.ok and "c1" in r.target and "img:1" in r.target

    monkeypatch.setattr(sc, "resolve_image", lambda c: "")
    r2 = sc.scan_container("c2")
    assert r2.ok is False and "could not resolve image" in r2.note


def test_list_scannable_containers_parses_and_filters(monkeypatch):
    sc = SupplyChainScanner()
    out = "c1\timg:1\nc2\timg:2\n"
    monkeypatch.setattr(sc, "_run", lambda cmd: (0, out, ""))
    rows = sc.list_scannable_containers()
    assert rows == [{"name": "c1", "image": "img:1"}, {"name": "c2", "image": "img:2"}]
    assert sc.list_scannable_containers("2") == [{"name": "c2", "image": "img:2"}]
    monkeypatch.setattr(sc, "_run", lambda cmd: (1, "", ""))
    assert sc.list_scannable_containers() == []


def test_triage_orders_fixable_and_caps_and_handles_bad_result():
    sc = SupplyChainScanner()
    bad = ScanResult(ok=False, target="x", scanner="trivy", note="fail")
    t = sc.triage(bad)
    assert t == {"summary": "fail", "fixes": [], "note": "fail"}

    findings = [
        type("F", (), {"id": "C1", "pkg": "p", "installed": "1", "fixed": "2", "severity": "CRITICAL"})(),
        type("F", (), {"id": "H1", "pkg": "q", "installed": "1", "fixed": "", "severity": "HIGH"})(),
        type("F", (), {"id": "L1", "pkg": "r", "installed": "1", "fixed": "3", "severity": "LOW"})(),
    ]
    good = ScanResult(ok=True, target="t", scanner="trivy", findings=findings)
    out = sc.triage(good, top=1)
    assert len(out["fixes"]) == 1
    assert out["fixes"][0]["action"] == "upgrade p 1 → 2"
    assert "critical" in out["summary"]

    nofix = ScanResult(ok=True, target="t", scanner="trivy", findings=[findings[1]])
    out2 = sc.triage(nofix)
    assert out2["note"] == "no fixable findings"