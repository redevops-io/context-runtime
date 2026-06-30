"""The business-module fleet — every agentic module as a ContextOS tenant.

The old agentic-os ran a fleet of hand-wired module controllers. ContextOS replaces
that with one pattern: each module is a **tenant** with a *goal* (answer its domain
question) and a *metric* (its own success signal), and ContextOS learns the cheapest
source bundle that meets the goal — exactly like the edge-sentinel SOC tenant, but
data-driven from a catalog so "migrate all" is a table, not 16 files.

  * ``ModuleSpec`` — declares a module: its OSS core, context sources, success metric,
    and approval-gated actions.
  * ``CATALOG`` — every existing agentic module + net-new tenants from the use-cases doc.
  * ``ModuleTenant`` — the generic tenant: classify the question → bandit picks a source
    bundle → run the source tools → assemble context → recommend an approval-gated
    action → learn from the reported outcome. Shared bandit + cost-model with every
    other tenant.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from itertools import combinations

from ..runtime.runtime import ContextRuntime
from ..tools.base import ApprovalPolicy, ToolRegistry, ToolResult, ToolSpec, function_tool
from ..types import Goal, Hit, Plan, Trace
from .bandit import EpsilonGreedyBandit

# ──────────────────────────── module declarations ────────────────────────────


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    core: str                       # the OSS core it wraps (Lago, Metabase, …)
    pain: str                       # the business question it answers
    sources: tuple[str, ...]        # available context sources (become source tools)
    metric: str                     # the reward signal (its own success measure)
    actions: tuple[str, ...] = ()   # side-effecting, approval-gated actions


# Every agentic-os-stack module (migrated) + net-new tenants from Use-cases.odt (new).
CATALOG: dict[str, ModuleSpec] = {
    # ── migrated from the agentic fleet ──
    "billing":        ModuleSpec("billing", "Lago", "payments & reconciliation",
                                 ("invoices", "ledger", "stripe", "dunning"), "reconciliation_match", ("refund", "dunning")),
    "support":        ModuleSpec("support", "Chatwoot", "customer support",
                                 ("tickets", "kb", "crm", "release_notes"), "resolution_rate", ("escalate",)),
    "control_tower":  ModuleSpec("control_tower", "Metabase", "business intelligence",
                                 ("warehouse", "salesforce", "stripe", "marketing"), "answer_correct", ()),
    "compliance":     ModuleSpec("compliance", "OpenSCAP", "compliance & data-privacy",
                                 ("scan_results", "policy", "evidence"), "control_pass_rate", ("remediate",)),
    "books":          ModuleSpec("books", "ERPNext", "bookkeeping & close",
                                 ("ledger", "bank", "invoices"), "close_correct", ("close",)),
    "crm":            ModuleSpec("crm", "ERPNext", "CRM & pipeline",
                                 ("contacts", "deals", "activity"), "lead_quality", ()),
    "market_radar":   ModuleSpec("market_radar", "changedetection.io", "competitive intel",
                                 ("competitor_pages", "news", "pricing"), "signal_precision", ()),
    "growth_engine":  ModuleSpec("growth_engine", "Umami", "marketing attribution",
                                 ("analytics", "campaigns", "attribution"), "attribution_accuracy", ()),
    "social":         ModuleSpec("social", "Postiz", "social-media growth",
                                 ("calendar", "engagement", "trends"), "engagement_lift", ("post",)),
    "lifecycle":      ModuleSpec("lifecycle", "Listmonk", "lifecycle messaging",
                                 ("subscribers", "segments", "campaigns"), "deliverability", ("send_campaign",)),
    "privacy":        ModuleSpec("privacy", "DSAR/GDPR", "data-subject requests",
                                 ("data_map", "requests", "consent"), "dsar_sla", ("fulfill_dsar",)),
    "edge_sentinel":  ModuleSpec("edge_sentinel", "CrowdSec", "network security (SOC)",
                                 ("crowdsec", "threat_intel", "edr"), "correct_verdict", ("block_ip",)),
    # ── net-new tenants from the use-cases doc ──
    "incident":       ModuleSpec("incident", "Kubernetes", "incident response",
                                 ("logs", "git", "metrics", "runbook"), "root_cause_found", ("rollback",)),
    "research":       ModuleSpec("research", "PubMed", "scientific research",
                                 ("pubmed", "citations", "reviews", "trials", "contradictions"), "answer_grounded", ()),
    "finance":        ModuleSpec("finance", "SEC EDGAR", "financial analysis",
                                 ("filings", "earnings_calls", "macro", "analyst", "news"), "thesis_support", ()),
    "personal":       ModuleSpec("personal", "Personal data", "personal AI",
                                 ("calendar", "email", "docs", "tasks", "memories"), "task_completed", ()),
}

COST_LAMBDA = 0.2   # efficiency penalty per source in a bundle (cheapest-sufficient frontier)


# ──────────────────────────── source bundles (the arms) ────────────────────────────


@dataclass(frozen=True)
class SourceBundle:
    sources: tuple[str, ...]

    @property
    def key(self) -> str:
        return "+".join(self.sources)


def _bundles(sources: tuple[str, ...], max_size: int = 2) -> tuple[SourceBundle, ...]:
    """Singletons + small combinations + the full set — spanning cheap→thorough."""
    arms: list[SourceBundle] = [SourceBundle((s,)) for s in sources]
    for n in range(2, min(max_size, len(sources)) + 1):
        arms += [SourceBundle(c) for c in combinations(sources, n)]
    arms.append(SourceBundle(tuple(sources)))
    # de-dup by key, keep order
    seen, out = set(), []
    for a in arms:
        if a.key not in seen:
            seen.add(a.key); out.append(a)
    return tuple(out)


def reward(success: bool, bundle: SourceBundle, n_sources: int) -> float:
    if not success:
        return 0.0
    return round(1.0 - COST_LAMBDA * (len(bundle.sources) / max(1, n_sources)), 4)


# ──────────────────────────── question kind (bandit context) ────────────────────────────

import re as _re

# word-boundary matched so "how" doesn't match "show", etc.
_ACTION_RE = _re.compile(r"\b(refund|close|send|block|escalate|remediate|rollback|post|"
                         r"fulfil\w*|issue|ban)\b", _re.I)
_ANALYSIS_RE = _re.compile(r"\b(why|how|analy\w*|trend\w*|explain|compare)\b", _re.I)


def question_kind(q: str) -> str:
    if _ACTION_RE.search(q):
        return "action"
    if _ANALYSIS_RE.search(q):
        return "analysis"
    return "lookup"


# ──────────────────────────── the generic tenant ────────────────────────────


@dataclass
class ModuleResult:
    module: str
    kind: str
    bundle: SourceBundle
    hits: tuple[Hit, ...]
    context: str
    recommended_action: str | None
    plan: Plan


def _source_tool(module: str, source: str):
    def run(args: dict) -> ToolResult:
        q = args.get("query", "")
        hit = Hit(chunk_id=f"{module}:{source}", filename=f"{module}/{source}", source=source,
                  text=f"[{module}/{source}] record relevant to: {q}", score=1.0)
        return ToolResult(ok=True, hits=[hit], text=f"{module}/{source}")
    return function_tool(f"{module}_{source}", run, description=f"{module} source: {source}")


def _action_tool(module: str, action: str):
    def run(args: dict) -> ToolResult:
        return ToolResult(ok=True, data={"action": action, "applied": False},
                          text=f"[dry-run] {module}.{action}({args})")
    return function_tool(f"{module}_{action}", run, description=f"{module} action: {action}",
                         side_effecting=True, approval_required=True)


class ModuleTenant:
    """One business module as a ContextOS tenant (goal + metric + learned policy)."""

    def __init__(self, spec: ModuleSpec, runtime: ContextRuntime | None = None,
                 bandit: EpsilonGreedyBandit | None = None, approver=None, epsilon: float = 0.12):
        self.spec = spec
        self.runtime = runtime or ContextRuntime.default([])
        self.arms = _bundles(spec.sources)
        self.bandit = bandit or EpsilonGreedyBandit(self.arms, epsilon=epsilon)
        self.registry = ToolRegistry(ApprovalPolicy(mode="deny_side_effects", approver=approver))
        for s in spec.sources:
            self.registry.register(_source_tool(spec.name, s))
        for a in spec.actions:
            self.registry.register(_action_tool(spec.name, a))
        self._pending: dict[str, tuple[Plan, SourceBundle, str]] = {}

    def handle(self, question: str) -> ModuleResult:
        kind = question_kind(question)
        plan = self.runtime.plan(Goal(text=question))
        bundle = self.bandit.select(f"{self.spec.name}:{kind}")
        hits: list[Hit] = []
        for src in bundle.sources:
            res = self.registry.run(f"{self.spec.name}_{src}", {"query": question})
            if res.ok:
                hits.extend(res.hits)
        context = "\n".join(f"[{i+1}] {h.text}" for i, h in enumerate(hits))
        action = (f"{self.spec.name}.{self.spec.actions[0]}"
                  if kind == "action" and self.spec.actions else None)
        self._pending[self._key(question)] = (plan, bundle, kind)
        return ModuleResult(self.spec.name, kind, bundle, tuple(hits), context, action, plan)

    def act(self, action: str, **args) -> ToolResult:
        return self.registry.run(f"{self.spec.name}_{action}", args)

    def record_outcome(self, question: str, success: bool) -> float:
        key = self._key(question)
        if key not in self._pending:
            return 0.0
        plan, bundle, kind = self._pending.pop(key)
        r = reward(success, bundle, len(self.spec.sources))
        self.bandit.update(f"{self.spec.name}:{kind}", bundle, r)
        self.runtime.estimator.observe(plan, Trace(
            plan_id=plan.id, goal_text=question, actual_tokens=len(bundle.sources) * 200,
            verification_passed=success))
        return r

    def policy(self) -> dict[str, str]:
        return self.bandit.policy()

    @staticmethod
    def _key(q: str) -> str:
        return hashlib.sha256(q.encode()).hexdigest()[:16]


def build_fleet(runtime: ContextRuntime | None = None, approver=None) -> dict[str, ModuleTenant]:
    """Instantiate the whole catalog as ContextOS tenants — the migrated fleet."""
    rt = runtime or ContextRuntime.default([])
    return {name: ModuleTenant(spec, runtime=rt, approver=approver) for name, spec in CATALOG.items()}
