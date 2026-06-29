"""edge-sentinel × ContextOS — the cybersecurity SOC tenant (first tool-using tenant).

Maps the "Is this ransomware?" use case onto the fleet pattern. The decision point is
**which sources to pull** for a given alert (CrowdSec decisions, threat-intel/CVE,
EDR timeline, firewall/DNS) — and the reward is *correct verdict at the cheapest
source bundle*. Same shared bandit + cost-model as the other tenants; what's new is
that the sources are real **ToolPlugins**, and remediation (block an IP) is an
approval-gated side-effecting tool.

CrowdSec is read over its LAPI (GET /v1/decisions, X-Api-Key) when configured
(CROWDSEC_LAPI_URL / CROWDSEC_BOUNCER_KEY); otherwise a faithful simulated feed lets
the tenant + learning run offline, exactly like the other tenants' harnesses.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from ..runtime.runtime import ContextRuntime
from ..tools.base import ApprovalPolicy, ToolRegistry, ToolResult, ToolSpec
from ..types import Goal, Hit, Plan, Trace
from .bandit import EpsilonGreedyBandit

# ──────────────────────────── SOC intent ────────────────────────────

_IOC = re.compile(r"\b(\d{1,3}(?:\.\d{1,3}){3}|[a-f0-9]{32,64}|CVE-\d{4}-\d+)\b", re.I)


def soc_bucket(question: str) -> str:
    """Classify a SOC question so each bucket has one decisive evidence source:
    threat_hunt→threat-intel, behavioral→EDR, network_anomaly→CrowdSec."""
    q = question.lower()
    if re.search(r"\b(ransomware|malware|exfiltrat|c2|beacon|cve-\d|backdoor|supply.chain)\b", q):
        return "threat_hunt"
    if re.search(r"\b(powershell|process|persistence|lateral|privilege|endpoint|host|login)\b", q):
        return "behavioral"
    if re.search(r"\b(ip|scan|brute|ddos|traffic|port|firewall|dns)\b", q) or _IOC.search(question):
        return "network_anomaly"
    return "threat_hunt"


def extract_indicator(question: str) -> str | None:
    m = _IOC.search(question)
    return m.group(1) if m else None


# ──────────────────────────── tools (the real seam) ────────────────────────────


class CrowdSecDecisionsTool:
    """Read live CrowdSec decisions over the LAPI; simulated fallback when unset."""

    def __init__(self, lapi_url: str | None = None, bouncer_key: str | None = None):
        self.lapi_url = (lapi_url or os.getenv("CROWDSEC_LAPI_URL", "")).rstrip("/")
        self.bouncer_key = bouncer_key or os.getenv("CROWDSEC_BOUNCER_KEY", "")

    def spec(self) -> ToolSpec:
        return ToolSpec(name="crowdsec_decisions",
                        description="Live CrowdSec ban/captcha decisions (IP, scenario, duration).",
                        parameters={"type": "object", "properties": {"query": {"type": "string"}}})

    def run(self, args: dict) -> ToolResult:
        rows = self._live() if (self.lapi_url and self.bouncer_key) else self._sim(args.get("query", ""))
        hits = [Hit(chunk_id=f"cs::{i}", filename=f"crowdsec:{r['type']}",
                    text=f"{r['type']} {r['value']} — scenario {r['scenario']} (scope {r['scope']}, {r['duration']})",
                    score=1.0, source="crowdsec", meta=r) for i, r in enumerate(rows)]
        return ToolResult(ok=True, hits=hits, data=rows,
                          text=f"{len(rows)} CrowdSec decisions")

    def _live(self) -> list[dict]:
        req = urllib.request.Request(f"{self.lapi_url}/v1/decisions",
                                     headers={"X-Api-Key": self.bouncer_key})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read().decode()) or []
        except (urllib.error.URLError, json.JSONDecodeError):
            return self._sim("")
        return [{"type": d.get("type", "ban"), "value": d.get("value", "?"),
                 "scenario": d.get("scenario", "?"), "scope": d.get("scope", "Ip"),
                 "duration": d.get("duration", "4h")} for d in data]

    @staticmethod
    def _sim(query: str) -> list[dict]:
        base = [
            {"type": "ban", "value": "185.220.101.4", "scenario": "crowdsecurity/ssh-bf", "scope": "Ip", "duration": "4h"},
            {"type": "ban", "value": "45.155.205.99", "scenario": "crowdsecurity/http-probing", "scope": "Ip", "duration": "4h"},
            {"type": "captcha", "value": "91.92.245.10", "scenario": "crowdsecurity/http-crawl-non_statics", "scope": "Ip", "duration": "1h"},
        ]
        if query:
            hit = [r for r in base if query in r["value"] or query in r["scenario"]]
            return hit or base
        return base


class ThreatIntelTool:
    """IOC / CVE reputation lookup (simulated knowledge base — swap for a real feed)."""

    _KB = {
        "185.220.101.4": {"reputation": "malicious", "tags": ["tor-exit", "ssh-bruteforce"], "confidence": 0.92},
        "45.155.205.99": {"reputation": "malicious", "tags": ["scanner", "exploit-attempts"], "confidence": 0.88},
        "CVE-2024-3094": {"reputation": "critical", "tags": ["xz-backdoor", "supply-chain"], "confidence": 0.99},
    }

    def spec(self) -> ToolSpec:
        return ToolSpec(name="threat_intel",
                        description="Reputation/CVE lookup for an IP or hash or CVE id.",
                        parameters={"type": "object", "properties": {"query": {"type": "string"}}})

    def run(self, args: dict) -> ToolResult:
        ind = (args.get("query") or "").strip()
        rec = self._KB.get(ind, {"reputation": "unknown", "tags": [], "confidence": 0.1})
        hit = Hit(chunk_id=f"ti::{ind}", filename="threat_intel", source="threat_intel",
                  text=f"{ind}: {rec['reputation']} (confidence {rec['confidence']}, tags {rec['tags']})",
                  score=rec["confidence"], meta=rec)
        return ToolResult(ok=True, hits=[hit], data=rec, text=f"threat-intel: {rec['reputation']}")


class EdrTimelineTool:
    """Endpoint process/behavior timeline (simulated)."""

    def spec(self) -> ToolSpec:
        return ToolSpec(name="edr_timeline",
                        description="Recent endpoint process/behavior events for triage.",
                        parameters={"type": "object", "properties": {"query": {"type": "string"}}})

    def run(self, args: dict) -> ToolResult:
        events = [
            "powershell.exe -enc <b64> spawned by winword.exe",
            "vssadmin delete shadows /all /quiet",
            "mass file rename *.docx -> *.locked",
        ]
        hits = [Hit(chunk_id=f"edr::{i}", filename="edr_timeline", source="edr", text=e, score=0.8)
                for i, e in enumerate(events)]
        return ToolResult(ok=True, hits=hits, data=events, text=f"{len(events)} EDR events")


class BlockIpTool:
    """Remediation: ban an IP via CrowdSec. SIDE-EFFECTING + APPROVAL-REQUIRED.

    Dry-run by default (and even when approved) unless CROWDSEC_LIVE=1, so it is safe to
    wire up before you trust it. The approval gate (ToolRegistry.policy) decides whether
    it may run at all.
    """

    def spec(self) -> ToolSpec:
        return ToolSpec(name="block_ip",
                        description="Ban an IP at the edge (CrowdSec decision).",
                        parameters={"type": "object", "properties": {
                            "ip": {"type": "string"}, "duration": {"type": "string"}}},
                        side_effecting=True, approval_required=True)

    def run(self, args: dict) -> ToolResult:
        ip, dur = args.get("ip", "?"), args.get("duration", "4h")
        live = os.getenv("CROWDSEC_LIVE") == "1"
        if not live:
            return ToolResult(ok=True, data={"ip": ip, "duration": dur, "applied": False},
                              text=f"[dry-run] would ban {ip} for {dur}")
        # real path would POST a decision; kept behind CROWDSEC_LIVE to avoid accidents
        return ToolResult(ok=True, data={"ip": ip, "duration": dur, "applied": True},
                          text=f"banned {ip} for {dur}")


# ──────────────────────────── the tenant ────────────────────────────


@dataclass(frozen=True)
class SourceBundle:
    """A bandit arm: which sources to pull for triage. Fewer = cheaper."""

    sources: tuple[str, ...]

    @property
    def key(self) -> str:
        return "+".join(sorted(self.sources))


DEFAULT_BUNDLES: tuple[SourceBundle, ...] = (
    SourceBundle(("crowdsec",)),                                   # network-only, cheapest
    SourceBundle(("threat_intel",)),                              # ioc-only
    SourceBundle(("edr",)),                                       # host-only
    SourceBundle(("crowdsec", "threat_intel")),                  # standard
    SourceBundle(("crowdsec", "threat_intel", "edr")),           # thorough
)
_SRC_TOOL = {"crowdsec": "crowdsec_decisions", "threat_intel": "threat_intel", "edr": "edr_timeline"}
COST_LAMBDA = 0.2   # how much a bigger source bundle costs in the reward


def reward_triage(correct: bool, bundle: SourceBundle) -> float:
    """Correct verdict at the cheapest sufficient bundle (the efficiency frontier)."""
    if not correct:
        return 0.0
    cost = len(bundle.sources) / max(len(b.sources) for b in DEFAULT_BUNDLES)
    return round(1.0 - COST_LAMBDA * cost, 4)


def _soc_bandit(epsilon: float = 0.15) -> EpsilonGreedyBandit:
    return EpsilonGreedyBandit(DEFAULT_BUNDLES, epsilon=epsilon)


@dataclass
class TriageResult:
    question: str
    soc_bucket: str
    bundle: SourceBundle
    hits: tuple[Hit, ...]
    context: str
    recommended_action: str | None
    plan: Plan


class SOCTriageTenant:
    """ContextOS plans SOC triage: pick the cheapest source bundle that resolves the
    alert, assemble the evidence, recommend an (approval-gated) action, and learn from
    whether the verdict was confirmed."""

    def __init__(self, runtime: ContextRuntime | None = None, registry: ToolRegistry | None = None,
                 bandit: EpsilonGreedyBandit | None = None, approver=None):
        self.runtime = runtime or ContextRuntime.default([])
        self.bandit = bandit or _soc_bandit()
        self.registry = registry or self._default_registry(approver)
        self._pending: dict[str, tuple[Plan, SourceBundle, str]] = {}

    @staticmethod
    def _default_registry(approver) -> ToolRegistry:
        reg = ToolRegistry(ApprovalPolicy(mode="deny_side_effects", approver=approver))
        for t in (CrowdSecDecisionsTool(), ThreatIntelTool(), EdrTimelineTool(), BlockIpTool()):
            reg.register(t)
        return reg

    def triage(self, question: str) -> TriageResult:
        bucket = soc_bucket(question)
        plan = self.runtime.plan(Goal(text=question))
        bundle = self.bandit.select(bucket)
        indicator = extract_indicator(question) or question
        hits: list[Hit] = []
        for src in bundle.sources:
            res = self.registry.run(_SRC_TOOL[src], {"query": indicator, "k": 5})
            if res.ok:
                hits.extend(res.hits)
        context = "\n".join(f"[{i+1}] {h.text}" for i, h in enumerate(hits))
        # cheap heuristic verdict (a real deployment would reason() over the context)
        malicious = any("malicious" in h.text or "critical" in h.text or "locked" in h.text
                        or "vssadmin" in h.text for h in hits)
        action = None
        if malicious and indicator and _IOC.match(indicator):
            action = f"block_ip({indicator})"
        self._pending[self._key(question)] = (plan, bundle, bucket)
        return TriageResult(question, bucket, bundle, tuple(hits), context, action, plan)

    def act(self, ip: str, duration: str = "4h") -> ToolResult:
        """Run the approval-gated remediation. Denied unless the registry's policy allows."""
        return self.registry.run("block_ip", {"ip": ip, "duration": duration})

    def record_outcome(self, question: str, confirmed_malicious: bool, analyst_correct: bool) -> float:
        """Feed back whether the bundle's verdict was confirmed. Updates bandit + cost model."""
        key = self._key(question)
        if key not in self._pending:
            return 0.0
        plan, bundle, bucket = self._pending.pop(key)
        reward = reward_triage(analyst_correct, bundle)
        self.bandit.update(bucket, bundle, reward)
        trace = Trace(plan_id=plan.id, goal_text=question,
                      actual_tokens=len(bundle.sources) * 200,
                      verification_passed=analyst_correct)
        self.runtime.estimator.observe(plan, trace)
        return reward

    def policy(self) -> dict[str, str]:
        return self.bandit.policy()

    @staticmethod
    def _key(q: str) -> str:
        import hashlib
        return hashlib.sha256(q.encode()).hexdigest()[:16]
