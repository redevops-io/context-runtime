"""runtime-scan × Context Runtime — the deployment-DAST plane for edge-sentinel.

The sibling of ``supply_chain`` (Trivy/Syft on images+lockfiles, i.e. *code & packages*).
This plane scans the **running deployment** for exploitable exposure with real DAST tools —
nmap (surface), nuclei (templated CVE/misconfig), ZAP (web/API DAST) — via the bundled
``deploy_scan`` MCP server. The engine + scope live in ``runtime_scan_engine``; this module
is the Context Runtime tenant: the decision is **which scan bundle to run** for a target, and
the reward is *catching the real exposure at the cheapest scan cost*.

Two invariants keep it safe to wire before you trust it:
  1. Scope is a fail-closed allowlist (``SCAN_ALLOWLIST``) — see ``in_scope``.
  2. Intrusive (active) scans are approval-gated; passive (recon/baseline) auto-approve.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..runtime.runtime import ContextRuntime
from ..tools.base import ToolRegistry
from ..types import Goal, Hit, Plan, Trace
from .bandit import EpsilonGreedyBandit
from .runtime_scan_engine import (  # re-exported for callers/tests
    DeploymentScanner, Finding, OutOfScopeError, ScanReport, host_of, in_scope,
    load_allowlist, scan_bucket,
)

__all__ = [
    "DeploymentScanner", "Finding", "OutOfScopeError", "ScanReport", "ScanBundle",
    "DEFAULT_BUNDLES", "COST_LAMBDA", "reward_scan", "RuntimeScanTenant", "ScanResult",
    "mount_deploy_scan", "passive_only_approver", "in_scope", "host_of", "load_allowlist",
    "scan_bucket",
]


# ──────────────────────────── the bandit arm ────────────────────────────


@dataclass(frozen=True)
class ScanBundle:
    """A bandit arm: which scanners to run. Fewer/passive = cheaper; active = pricier + gated."""

    tools: tuple[str, ...]

    @property
    def key(self) -> str:
        return "+".join(self.tools)

    @property
    def has_active(self) -> bool:
        return "zap_active" in self.tools


DEFAULT_BUNDLES: tuple[ScanBundle, ...] = (
    ScanBundle(("nmap",)),                           # recon only, cheapest
    ScanBundle(("nuclei",)),                         # templated web, passive
    ScanBundle(("zap_baseline",)),                   # passive web spider+scan
    ScanBundle(("nmap", "nuclei")),                  # standard
    ScanBundle(("nuclei", "zap_active")),            # deep web (INTRUSIVE, gated)
    ScanBundle(("nmap", "nuclei", "zap_baseline")),  # thorough passive
)
# active scanning is disproportionately expensive (time + intrusiveness)
_TOOL_COST = {"nmap": 1.0, "nuclei": 1.5, "zap_baseline": 2.0, "zap_active": 4.0}
COST_LAMBDA = 0.15
_MAX_COST = max(sum(_TOOL_COST[t] for t in b.tools) for b in DEFAULT_BUNDLES)


def _bundle_cost(b: ScanBundle) -> float:
    return sum(_TOOL_COST[t] for t in b.tools)


def reward_scan(caught: bool, bundle: ScanBundle) -> float:
    """Caught the real exposure at the cheapest sufficient bundle (the efficiency frontier)."""
    if not caught:
        return 0.0
    return round(1.0 - COST_LAMBDA * (_bundle_cost(bundle) / _MAX_COST), 4)


def _scan_bandit(epsilon: float = 0.15) -> EpsilonGreedyBandit:
    return EpsilonGreedyBandit(DEFAULT_BUNDLES, epsilon=epsilon)


def passive_only_approver(action: dict) -> bool:
    """Default ApprovalPolicy approver: auto-approve passive scans, require a human for active
    (intrusive) ones. Compose your own for the human step."""
    args = action.get("args") or {}
    prof = args.get("profile", "")
    if prof:
        from .runtime_scan_engine import ACTIVE_PROFILES
        return prof not in ACTIVE_PROFILES
    # bundle-shaped action (in-process path): allow unless it includes an active tool
    tools = args.get("tools") or ()
    return "zap_active" not in tools


# ──────────────────────────── the tenant ────────────────────────────


@dataclass
class ScanResult:
    target: str
    bucket: str
    bundle: ScanBundle
    report: ScanReport
    hits: tuple[Hit, ...]
    max_severity: str
    plan: Plan


class RuntimeScanTenant:
    """Context Runtime plans deployment DAST: classify the target, pick the cheapest scan bundle
    that still catches the exposure, run it (scope- and approval-gated), and learn from whether
    the exposure was really caught. Scans in-process via ``DeploymentScanner`` by default; pass a
    registry with the deploy_scan MCP server mounted (``mount_deploy_scan``) to route through MCP."""

    def __init__(self, runtime: ContextRuntime | None = None, registry: ToolRegistry | None = None,
                 bandit: EpsilonGreedyBandit | None = None, scanner: DeploymentScanner | None = None):
        self.runtime = runtime or ContextRuntime.default([])
        self.bandit = bandit or _scan_bandit()
        self.registry = registry
        self.scanner = scanner or DeploymentScanner()
        self._pending: dict[str, tuple[Plan, ScanBundle, str]] = {}

    def scan(self, target: str) -> ScanResult:
        if not in_scope(target, self.scanner.allowlist):
            raise OutOfScopeError(f"{host_of(target)!r} not in SCAN_ALLOWLIST — refusing (fail-closed)")
        bucket = scan_bucket(target)
        plan = self.runtime.plan(Goal(text=f"scan {target}"))
        bundle = self.bandit.select(bucket)
        findings: list[Finding] = []
        for tool in bundle.tools:
            rep = self.scanner.run_tool(tool, target)
            findings.extend(rep.findings)
        report = ScanReport(True, target, sorted({f.tool for f in findings}),
                            findings=findings, simulated=not self.scanner.live)
        hits = tuple(f.as_hit(i) for i, f in enumerate(findings))
        self._pending[self._key(target)] = (plan, bundle, bucket)
        return ScanResult(target, bucket, bundle, report, hits, report.summary()["max_severity"], plan)

    def record_outcome(self, target: str, exposure_confirmed: bool) -> float:
        key = self._key(target)
        if key not in self._pending:
            return 0.0
        plan, bundle, bucket = self._pending.pop(key)
        reward = reward_scan(exposure_confirmed, bundle)
        self.bandit.update(bucket, bundle, reward)
        trace = Trace(plan_id=plan.id, goal_text=f"scan {target}",
                      actual_tokens=int(_bundle_cost(bundle) * 200),
                      verification_passed=exposure_confirmed)
        self.runtime.estimator.observe(plan, trace)
        return reward

    def policy(self) -> dict[str, str]:
        return self.bandit.policy()

    @staticmethod
    def _key(t: str) -> str:
        import hashlib
        return hashlib.sha256(t.encode()).hexdigest()[:16]


def mount_deploy_scan(registry: ToolRegistry, *, python: str | None = None):
    """Mount the bundled deploy_scan MCP server (stdio) into a registry, so the scan tools ride
    the registry's ApprovalPolicy. Returns (client, tool_names)."""
    import sys
    from ..tools.mcp import MCPClient, mount_mcp
    client = MCPClient.stdio([python or sys.executable, "-m",
                              "context_runtime.tools.mcp_servers.deploy_scan"])
    names = mount_mcp(registry, client, prefix="scan", default_side_effecting=True)
    return client, names
