"""Hermetic tests for the runtime-DAST plane (deploy_scan MCP + RuntimeScanTenant).

No live scanners, no network: SCAN_LIVE is unset so DeploymentScanner returns simulated
findings. Covers scope (fail-closed), the async MCP job lifecycle, the approval gate, and reward.
"""
from __future__ import annotations

import os
import time

import pytest

from context_runtime.integrations.runtime_scan import (
    DEFAULT_BUNDLES, DeploymentScanner, OutOfScopeError, RuntimeScanTenant, ScanBundle,
    host_of, in_scope, mount_deploy_scan, passive_only_approver, reward_scan, scan_bucket,
)
from context_runtime.tools.base import ApprovalPolicy, ToolRegistry

ALLOW = ["192.168.40.105", "vibexgen.io", "*.vibexgen.io"]


# ── scope: fail-closed allowlist ────────────────────────────────────────────
def test_scope_fail_closed_when_empty():
    assert in_scope("vibexgen.io", []) is False
    assert in_scope("https://vibexgen.io/x", None if False else []) is False


def test_scope_exact_and_wildcard():
    assert in_scope("192.168.40.105", ALLOW) is True
    assert in_scope("https://vibexgen.io/login?q=1", ALLOW) is True
    assert in_scope("auth.vibexgen.io", ALLOW) is True          # *.vibexgen.io
    assert in_scope("https://app.vibexgen.io/api", ALLOW) is True
    assert in_scope("example.com", ALLOW) is False
    assert in_scope("notvibexgen.io", ALLOW) is False           # no false suffix match


def test_host_of():
    assert host_of("https://a.b.c/x?y=1") == "a.b.c"
    assert host_of("a.b.c:8080") == "a.b.c"
    assert host_of("A.B.C") == "a.b.c"


def test_scanner_guard_refuses_off_scope():
    s = DeploymentScanner(allowlist=ALLOW, live=False)
    with pytest.raises(OutOfScopeError):
        s.nmap_scan("example.com")
    # in-scope simulated scan works
    rep = s.nmap_scan("192.168.40.105")
    assert rep.ok and rep.simulated


def test_tenant_refuses_off_scope():
    t = RuntimeScanTenant(scanner=DeploymentScanner(allowlist=ALLOW, live=False))
    with pytest.raises(OutOfScopeError):
        t.scan("https://example.com/")


# ── simulated findings: decisive tool surfaces the class's exposure ─────────
def test_sim_decisive_tool_surfaces_exposure():
    s = DeploymentScanner(allowlist=ALLOW, live=False)
    # injection target: only the active scan surfaces the high-sev bug
    inj = "https://vibexgen.io/login"
    assert scan_bucket(inj) == "injection"
    passive = s.nuclei_scan(inj)
    assert not any(f.severity in ("high", "critical") for f in passive.findings)
    active = s.zap_active(inj)
    assert any(f.severity == "high" for f in active.findings)
    # tls target: nmap is decisive
    assert any(f.severity == "high" for f in s.nmap_scan("192.168.40.105").findings)


# ── reward: cheaper sufficient bundle scores higher; miss = 0 ───────────────
def test_reward_monotonic():
    cheap = ScanBundle(("nmap",))
    dear = ScanBundle(("nuclei", "zap_active"))
    assert reward_scan(True, cheap) > reward_scan(True, dear) > 0
    assert reward_scan(False, cheap) == 0.0


# ── deploy_scan MCP server: async lifecycle + scope + approval gate ─────────
def _mounted(approver=passive_only_approver):
    os.environ["SCAN_ALLOWLIST"] = ",".join(ALLOW)
    reg = ToolRegistry(ApprovalPolicy(mode="deny_side_effects", approver=approver))
    client, names = mount_deploy_scan(reg)
    return reg, client, names


def _drive(reg, target, profile):
    start = reg.run("scan.scan_start", {"target": target, "profile": profile})
    assert start.ok, start.error
    job = (start.data or {}).get("job_id")
    assert job
    for _ in range(50):
        st = reg.run("scan.scan_status", {"job_id": job})
        if (st.data or {}).get("state") in ("done", "error"):
            break
        time.sleep(0.1)
    return reg.run("scan.scan_results", {"job_id": job})


def test_mcp_async_lifecycle():
    reg, client, names = _mounted(approver=lambda a: True)   # allow active for the lifecycle test
    try:
        assert {"scan.scan_start", "scan.scan_status", "scan.scan_results"} <= set(names)
        res = _drive(reg, "https://vibexgen.io/login", "web-active")
        data = res.data or {}
        assert data.get("state") == "done"
        assert data["summary"]["total"] >= 1
        assert data["summary"]["max_severity"] in ("high", "critical")   # active caught injection
    finally:
        client.close()


def test_mcp_approval_gate_blocks_active():
    reg, client, _ = _mounted(approver=passive_only_approver)
    try:
        # passive auto-approved
        ok = reg.run("scan.scan_start", {"target": "https://vibexgen.io/", "profile": "web-baseline"})
        assert ok.ok
        # active blocked (no human approver granted it)
        blocked = reg.run("scan.scan_start", {"target": "https://vibexgen.io/login", "profile": "web-active"})
        assert not blocked.ok and "approval" in (blocked.error or "").lower()
    finally:
        client.close()


def test_mcp_off_scope_refused():
    reg, client, _ = _mounted(approver=lambda a: True)   # even with blanket approval…
    try:
        r = reg.run("scan.scan_start", {"target": "https://example.com/", "profile": "recon"})
        # …scope is enforced inside the tool: returns an error result, no job created
        assert (r.data or {}).get("error") and "scope" in r.data["error"].lower()
    finally:
        client.close()


def test_mcp_unknown_profile():
    reg, client, _ = _mounted(approver=lambda a: True)
    try:
        r = reg.run("scan.scan_start", {"target": "vibexgen.io", "profile": "nope"})
        assert (r.data or {}).get("error") and "profile" in r.data["error"].lower()
    finally:
        client.close()
