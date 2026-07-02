"""agentic-compliance × Context Runtime — evidence-selection tuning tenant.

Clone of ``agentic_billing``'s structure: the tenant chooses among discrete evidence
bundles (bandit arms) keyed by a finding bucket and learns the cheapest evidence set
that still reaches the correct remediation. ``examples/agentic_compliance.py`` drives a
72-round offline benchmark proving Context Runtime beats a fixed full-evidence bundle.

Licensed under AGPL-3.0 (see LICENSE).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable

from ..runtime.runtime import ContextRuntime
from ..tools.base import ToolRegistry, ToolResult, function_tool
from ..types import Goal, Trace
from .bandit import EpsilonGreedyBandit


# ──────────────────────────── evidence bundles (bandit arms) ────────────────────────────


@dataclass(frozen=True)
class ComplianceEvidenceBundle:
    """One concrete bundle of rule-family evidence to pull before staging a remediation."""

    include_access: bool
    include_crypto: bool
    include_audit: bool
    include_patch: bool
    name: str

    @property
    def key(self) -> str:
        return self.name

    def cost_units(self) -> float:
        cost = 1.0
        if self.include_access:
            cost += 0.7
        if self.include_crypto:
            cost += 0.8
        if self.include_audit:
            cost += 0.6
        if self.include_patch:
            cost += 0.9
        return cost


DEFAULT_COMPLIANCE: tuple[ComplianceEvidenceBundle, ...] = (
    ComplianceEvidenceBundle(True, True, True, True, "full_evidence"),
    ComplianceEvidenceBundle(True, True, False, False, "access_crypto"),
    ComplianceEvidenceBundle(False, False, True, True, "audit_patch"),
    ComplianceEvidenceBundle(True, False, True, False, "access_audit"),
    ComplianceEvidenceBundle(False, True, False, True, "crypto_patch"),
    ComplianceEvidenceBundle(True, False, False, True, "access_patch"),
)

DECISIVE_BY_BUCKET: dict[str, str] = {
    "access": "include_access",
    "crypto": "include_crypto",
    "logging": "include_audit",
    "patch": "include_patch",
    "general": "include_access",
}


# ──────────────────────────── buckets and rewards ────────────────────────────


def agentic_compliance_bucket(text: str) -> str:
    lowered = text.lower()
    if any(k in lowered for k in ("password", "access", "privilege", "sudo", "account", "login")):
        return "access"
    if any(k in lowered for k in ("cipher", "tls", "crypto", "encryption", "certificate", "ssh key")):
        return "crypto"
    if any(k in lowered for k in ("audit", "log", "logging", "rsyslog", "journald")):
        return "logging"
    if any(k in lowered for k in ("patch", "update", "cve", "version", "outdated", "upgrade")):
        return "patch"
    return "general"


def reward_from_remediation(value: float, bundle: ComplianceEvidenceBundle, cost: float | None = None) -> float:
    return value - (cost if cost is not None else bundle.cost_units())


# ──────────────────────────── tenant ────────────────────────────


def _compliance_bandit(*, epsilon: float = 0.15, arms: tuple[ComplianceEvidenceBundle, ...] = DEFAULT_COMPLIANCE,
                       bandit: EpsilonGreedyBandit | None = None) -> EpsilonGreedyBandit:
    return bandit or EpsilonGreedyBandit(arms, epsilon=epsilon)


def _simulate_pull(inputs: dict) -> str:
    return (f"Evidence bundle {inputs.get('bundle')} pulled: "
            f"access={inputs.get('access')} crypto={inputs.get('crypto')} "
            f"audit={inputs.get('audit')} patch={inputs.get('patch')}")


class AgenticComplianceTenant:
    def __init__(self, runtime: ContextRuntime | None = None,
                 arms: tuple[ComplianceEvidenceBundle, ...] = DEFAULT_COMPLIANCE,
                 bandit: EpsilonGreedyBandit | None = None, epsilon: float = 0.15,
                 pull_tool_factory: Callable[[dict], ToolResult] | None = None):
        self.runtime = runtime or ContextRuntime.default([])
        self.arms = arms
        self.bandit = _compliance_bandit(epsilon=epsilon, arms=arms, bandit=bandit)
        self.registry = ToolRegistry()
        pull_fn = pull_tool_factory or _simulate_pull
        self.registry.register(function_tool(
            name="pull_evidence",
            description="Pull the selected rule-family evidence bundle (simulated).",
            fn=pull_fn,
        ))
        self._pending: dict[str, tuple] = {}

    def choose(self, finding: str, bucket: str | None = None) -> ComplianceEvidenceBundle:
        plan = self.runtime.plan(Goal(text=finding))
        ctx_bucket = bucket or agentic_compliance_bucket(finding)
        bundle = self.bandit.select(ctx_bucket)
        _ = self.registry.run("pull_evidence", {
            "bundle": bundle.key,
            "access": bundle.include_access,
            "crypto": bundle.include_crypto,
            "audit": bundle.include_audit,
            "patch": bundle.include_patch,
        })
        self._pending[self._key(finding)] = (plan, bundle, ctx_bucket)
        return bundle

    def record_outcome(self, finding: str, value: float, cost: float | None = None) -> float:
        key = self._key(finding)
        entry = self._pending.pop(key, None)
        if entry is None:
            return 0.0
        plan, bundle, bucket = entry
        reward = reward_from_remediation(value, bundle, cost)
        self.bandit.update(bucket, bundle, reward)
        self.runtime.estimator.observe(plan, Trace(
            plan_id=plan.id,
            goal_text=finding,
            actual_tokens=12,
            actual_cost_usd=(cost if cost is not None else bundle.cost_units()) * 0.02,
            actual_latency_seconds=0.0,
            verification_passed=value >= (cost if cost is not None else bundle.cost_units()),
        ))
        return reward

    def policy(self) -> dict[str, str]:
        return self.bandit.policy()

    @staticmethod
    def _key(finding: str) -> str:
        return hashlib.sha256(finding.encode()).hexdigest()[:16]
