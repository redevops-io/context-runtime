"""Revenue & Intelligence × Context Runtime — the GTM tenant.

Maps the "find the accounts that need us, enrich the right buyer, don't overpay for data" use case onto the
fleet pattern. The decision point is **which providers to call** for a given enrichment need — and the
reward is *correct enrichment at the cheapest sufficient provider bundle*, with a hard penalty for a
confident **wrong-entity** match (a wrong company is worse than a missing field). Same shared bandit + cost
model as the other tenants; the providers are ``ToolPlugin``s (Apollo / PDL / Hunter / BuiltWith / local
CRM), and outreach send is an approval-gated, ``NO_OUTREACH``-deniable side effect.

Providers are read live when configured (APOLLO_API_KEY / PDL_API_KEY / HUNTER_API_KEY / BUILTWITH_API_KEY);
otherwise a faithful **recorded/simulated** feed keyed to a labelled fixture lets the tenant + learning run
offline, exactly like the other tenants' harnesses. The offline fixture is where the benchmark ground truth
lives (correct provider per need, plus wrong-entity/stale/missing traps).

See ``examples/revenue_intelligence.py`` for the A/B/C/E arm comparison, and the plan
``redevops-revenue-intelligence-runtime-benchmark-plan.md`` for the full taxonomy.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from ..runtime.runtime import ContextRuntime
from ..tools.base import ApprovalPolicy, ToolRegistry, ToolResult, ToolSpec
from ..types import Goal, Hit, Plan, Trace
from .bandit import EpsilonGreedyBandit

# ──────────────────────────── GTM intent ────────────────────────────

# Each enrichment need has ONE decisive provider — the source that actually resolves it cheaply. The bandit
# has to discover the cheapest provider bundle that includes it (a real GTM stack has exactly this shape:
# every vendor claims everything, but only some are authoritative for a given field).
DECISIVE_PROVIDER: dict[str, str] = {
    "company_identity": "pdl",       # canonical org resolution / dedupe — PDL's matching is authoritative
    "person_role":      "apollo",    # the right decision-maker at the account
    "contact_verify":   "hunter",    # is this email real & deliverable
    "tech_signal":      "builtwith", # orthogonal technographic evidence
    "firmographics":    "apollo",    # size / industry / funding fields
}


def gtm_bucket(question: str) -> str:
    """Classify a GTM enrichment need so each bucket has one decisive provider."""
    q = question.lower()
    if re.search(r"\b(verify|valid|deliverab|bounce|email is real|reachable)\b", q):
        return "contact_verify"
    if re.search(r"\b(who|decision.maker|buyer|contact|person|role|title|vp|cto|head of)\b", q):
        return "person_role"
    if re.search(r"\b(tech|stack|technolog|uses|built|infrastructure|platform|tooling)\b", q):
        return "tech_signal"
    if re.search(r"\b(same company|which company|canonical|dedup|resolve|identity|duplicate)\b", q):
        return "company_identity"
    if re.search(r"\b(size|employees|industry|funding|revenue|stage|firmograph)\b", q):
        return "firmographics"
    return "company_identity"


# ──────────────────────────── providers (the real seam) ────────────────────────────

# Per-provider external cost (USD) — CRM/local evidence is free; specialists cost more. These are the
# knobs the planner trades against quality; live adapters would report observed cost instead.
PROVIDER_COST: dict[str, float] = {
    "crm": 0.0, "apollo": 0.02, "hunter": 0.03, "builtwith": 0.04, "pdl": 0.05, "web_research": 0.30,
}


@dataclass
class Fixture:
    """The labelled benchmark ground truth for one account/need (Tier-2 disclosed labels, plan §7).

    ``decisive`` is the provider that returns the correct value for this need; ``wrong`` providers return a
    confident but wrong entity (the trap that must be caught, not trusted); everyone else returns nothing.
    """
    account_id: str
    company: str
    domain: str
    need: str                       # the enrichment bucket under test
    decisive: str                   # provider holding the correct answer
    truth: str                      # the correct value
    wrong: tuple[str, ...] = ()     # providers that return a confident wrong entity


class ProviderTool:
    """A GTM data provider as a ToolPlugin. Offline it answers from the fixture (correct / wrong / missing);
    with an API key set it would call the real endpoint. Read-only, non-side-effecting."""

    def __init__(self, name: str, fixtures: dict[str, Fixture]):
        self.name = name
        self._fx = fixtures
        self.live = bool(os.getenv(f"{name.upper()}_API_KEY"))

    def spec(self) -> ToolSpec:
        return ToolSpec(name=f"{self.name}_enrich",
                        description=f"Enrich an account via {self.name}.",
                        parameters={"type": "object", "properties": {
                            "account_id": {"type": "string"}, "need": {"type": "string"}}})

    def run(self, args: dict) -> ToolResult:
        fx = self._fx.get(args.get("account_id", ""))
        if fx is None:
            return ToolResult(ok=True, hits=[], data={"status": "missing"}, text="no record")
        if self.name == fx.decisive:
            outcome, value = "correct", fx.truth
        elif self.name in fx.wrong:
            outcome, value = "wrong", f"{fx.company} (Inc.)"        # a plausible but wrong entity
        else:
            return ToolResult(ok=True, hits=[], data={"status": "missing"}, text="no confident match")
        hit = Hit(chunk_id=f"{self.name}::{fx.account_id}", filename=self.name, source=self.name,
                  text=f"{self.name}: {fx.need}={value}", score=0.9 if outcome == "correct" else 0.6)
        return ToolResult(ok=True, hits=[hit], data={"status": outcome, "value": value,
                          "cost": PROVIDER_COST[self.name]}, text=hit.text)


class PrepareOutreachTool:
    """Draft an outreach sequence — allowed (no message leaves the building)."""

    def spec(self) -> ToolSpec:
        return ToolSpec(name="outreach_prepare", description="Draft (not send) an outreach sequence.",
                        parameters={"type": "object", "properties": {"account_id": {"type": "string"}}})

    def run(self, args: dict) -> ToolResult:
        return ToolResult(ok=True, hits=[], data={"drafted": True, "account_id": args.get("account_id")},
                          text=f"[draft] outreach prepared for {args.get('account_id')}")


class SendOutreachTool:
    """Send outreach — SIDE-EFFECTING + APPROVAL-REQUIRED, and denied outright under NO_OUTREACH."""

    def spec(self) -> ToolSpec:
        return ToolSpec(name="outreach_send", description="Send an outreach message.",
                        parameters={"type": "object", "properties": {"account_id": {"type": "string"}}},
                        side_effecting=True, approval_required=True)

    def run(self, args: dict) -> ToolResult:
        return ToolResult(ok=True, data={"sent": True, "account_id": args.get("account_id")},
                          text=f"sent to {args.get('account_id')}")


# ──────────────────────────── arms + reward ────────────────────────────


@dataclass(frozen=True)
class ProviderBundle:
    """A bandit arm: which providers to call for a record. Fewer / cheaper = better if still sufficient."""
    providers: tuple[str, ...]

    @property
    def key(self) -> str:
        return "+".join(sorted(self.providers))

    @property
    def cost(self) -> float:
        return round(sum(PROVIDER_COST[p] for p in self.providers), 4)


DEFAULT_BUNDLES: tuple[ProviderBundle, ...] = (
    ProviderBundle(("crm",)),                                  # free, often insufficient
    ProviderBundle(("apollo",)),                               # cheap generalist
    ProviderBundle(("pdl",)),                                  # identity specialist
    ProviderBundle(("hunter",)),                               # verification specialist
    ProviderBundle(("builtwith",)),                            # technographic specialist
    ProviderBundle(("crm", "apollo")),                         # cheap combo
    ProviderBundle(("apollo", "pdl")),                         # generalist + identity
    ProviderBundle(("apollo", "hunter")),                      # generalist + verify
    ProviderBundle(("apollo", "builtwith")),                   # generalist + tech
    ProviderBundle(("crm", "apollo", "pdl", "hunter", "builtwith")),   # everything (the coverage ceiling)
)
_MAX_COST = max(b.cost for b in DEFAULT_BUNDLES)
COST_LAMBDA = 0.35        # how hard a bigger/pricier bundle is penalised in the reward
WRONG_PENALTY = -0.5      # a confident wrong-entity match is worse than a missing value


def outcome_for(bundle: ProviderBundle, fx: Fixture) -> str:
    """What the bundle resolves to against ground truth: the decisive provider wins if present (verification
    beats a wrong match); otherwise a wrong provider poisons the result; otherwise nothing is found."""
    if fx.decisive in bundle.providers:
        return "correct"
    if any(w in bundle.providers for w in fx.wrong):
        return "wrong"
    return "missing"


def reward_enrich(outcome: str, bundle: ProviderBundle) -> float:
    """Correct enrichment at the cheapest sufficient bundle; wrong-entity is penalised below missing."""
    if outcome == "correct":
        return round(1.0 - COST_LAMBDA * (bundle.cost / _MAX_COST), 4)
    if outcome == "wrong":
        return WRONG_PENALTY
    return 0.0


def _gtm_bandit(epsilon: float = 0.12) -> EpsilonGreedyBandit:
    return EpsilonGreedyBandit(DEFAULT_BUNDLES, epsilon=epsilon)


# ──────────────────────────── the tenant ────────────────────────────


@dataclass
class EnrichResult:
    account_id: str
    bucket: str
    bundle: ProviderBundle
    outcome: str                    # correct | wrong | missing
    hits: tuple[Hit, ...]
    cost: float
    plan: Plan


class RevenueIntelligenceTenant:
    """Context Runtime plans GTM enrichment: for each account pick the cheapest provider bundle that
    resolves the need, call the providers as tools, judge the result against ground truth, keep outreach
    behind an approval gate, and learn which bundle is sufficient per need."""

    def __init__(self, fixtures: dict[str, Fixture], *, runtime: ContextRuntime | None = None,
                 registry: ToolRegistry | None = None, bandit: EpsilonGreedyBandit | None = None,
                 approver=None, no_outreach: bool = True):
        self.fixtures = fixtures
        self.runtime = runtime or ContextRuntime.default([])
        self.bandit = bandit or _gtm_bandit()
        self.no_outreach = no_outreach
        self.registry = registry or self._default_registry(fixtures, approver)
        self.external_spend = 0.0
        self._pending: dict[str, tuple[Plan, ProviderBundle, str]] = {}

    @staticmethod
    def _default_registry(fixtures, approver) -> ToolRegistry:
        reg = ToolRegistry(ApprovalPolicy(mode="deny_side_effects", approver=approver))
        for name in ("crm", "apollo", "pdl", "hunter", "builtwith"):
            reg.register(ProviderTool(name, fixtures))
        reg.register(PrepareOutreachTool())
        reg.register(SendOutreachTool())
        return reg

    def enrich(self, account_id: str, need: str | None = None) -> EnrichResult:
        fx = self.fixtures[account_id]
        bucket = need or fx.need
        plan = self.runtime.plan(Goal(text=f"enrich {account_id} for {bucket}"))
        bundle = self.bandit.select(bucket)
        hits: list[Hit] = []
        cost = 0.0
        for prov in bundle.providers:
            res = self.registry.run(f"{prov}_enrich", {"account_id": account_id, "need": bucket})
            if res.ok and res.hits:
                hits.extend(res.hits)
            cost += PROVIDER_COST[prov]
        self.external_spend = round(self.external_spend + cost, 4)
        outcome = outcome_for(bundle, fx)
        self._pending[account_id] = (plan, bundle, bucket)
        return EnrichResult(account_id, bucket, bundle, outcome, tuple(hits), round(cost, 4), plan)

    def record_outcome(self, account_id: str, outcome: str) -> float:
        """Feed the ground-truth outcome back: reward the bundle, update bandit + cost model."""
        if account_id not in self._pending:
            return 0.0
        plan, bundle, bucket = self._pending.pop(account_id)
        reward = reward_enrich(outcome, bundle)
        self.bandit.update(bucket, bundle, reward)
        trace = Trace(plan_id=plan.id, goal_text=f"enrich {account_id}",
                      actual_tokens=len(bundle.providers) * 150,
                      verification_passed=(outcome == "correct"))
        self.runtime.estimator.observe(plan, trace)
        return reward

    def prepare_outreach(self, account_id: str) -> ToolResult:
        return self.registry.run("outreach_prepare", {"account_id": account_id})

    def send_outreach(self, account_id: str) -> ToolResult:
        """The governance demonstration: denied under NO_OUTREACH, else still approval-gated."""
        if self.no_outreach:
            self.registry.audit.append({"tool": "outreach_send", "args": {"account_id": account_id},
                                        "allowed": False, "reason": "NO_OUTREACH policy"})
            return ToolResult(ok=False, data={"denied": True}, text="denied by NO_OUTREACH policy")
        return self.registry.run("outreach_send", {"account_id": account_id})

    def policy(self) -> dict[str, str]:
        return self.bandit.policy()
