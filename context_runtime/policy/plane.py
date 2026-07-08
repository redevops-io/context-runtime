"""The Policy Runtime plane — providers, decisions, and the mandatory PolicyDecision audit.

Thesis (docs/policy-runtime.md §0): policy is not post-processing; it defines the feasible execution
space before planning, and enforces across phases (planning · command · input · tool · retrieval · output).
This module carries the runtime-phase providers (input/output guardrails, tool approval) and the composed
``Policy`` that evaluates them and emits a ``PolicyDecision`` for EVERY decision so policy is visible in
EXPLAIN, not hidden in enforcement. The planning-phase check lives in the optimizer (PolicyEngine.feasible).
"""
from __future__ import annotations

import itertools
import re
import time
from collections import deque
from dataclasses import dataclass, field

from .store import RuleStore, scopes_for

ACTIONS = ("allow", "deny", "redact", "flag", "require_approval")


@dataclass(frozen=True)
class Decision:
    action: str = "allow"
    reason: str = ""
    rule_id: str = ""
    scope: str = ""
    provider: str = ""
    replacement: str | None = None      # for redact

    @property
    def ok(self) -> bool:
        return self.action == "allow"

    @property
    def blocked(self) -> bool:
        return self.action == "deny"


ALLOW = Decision("allow")

_ids = itertools.count(1)


@dataclass(frozen=True)
class PolicyDecision:
    """The audit/EXPLAIN event emitted for every policy decision (docs §4)."""
    policy_decision_id: str
    principal: str
    app: str
    scope: str
    rule_id: str
    decision: str
    reason: str
    phase: str                          # planning | command | input | tool | retrieval | output
    provider: str = ""
    at: float = 0.0

    def to_dict(self) -> dict:
        return {"policy_decision_id": self.policy_decision_id, "principal": self.principal, "app": self.app,
                "scope": self.scope, "rule_id": self.rule_id, "decision": self.decision, "reason": self.reason,
                "phase": self.phase, "provider": self.provider, "at": self.at}

    def summary(self) -> str:
        return f"{self.decision} · {self.phase}" + (f" · {self.reason}" if self.reason else "")


class DecisionSink:
    """Collects PolicyDecisions for audit/EXPLAIN, with an optional forward callback (e.g. a trace/OLAP)."""

    def __init__(self, forward=None, cap: int = 2000):
        self.events: deque = deque(maxlen=cap)
        self.forward = forward

    def emit(self, ev: PolicyDecision) -> PolicyDecision:
        self.events.append(ev)
        if self.forward is not None:
            try:
                self.forward(ev)
            except Exception:  # noqa: BLE001
                pass
        return ev

    def recent(self, n: int = 20, *, phase: str | None = None, decision: str | None = None) -> list[PolicyDecision]:
        out = [e for e in self.events
               if (phase is None or e.phase == phase) and (decision is None or e.decision == decision)]
        return out[-n:]


def _matches(rule, text: str) -> bool:
    pat = rule.match.get("pattern") or rule.text
    if rule.match.get("regex"):
        try:
            return re.search(pat, text or "", re.I) is not None
        except re.error:
            return False
    return pat.lower() in (text or "").lower()


class GuardrailProvider:
    """Input/output content safety over ``guardrail``-kind rules (global + app scope). Pattern-based v1;
    a rule with ``meta.semantic`` is a v2 LLM-judged extension behind the same seam."""

    def __init__(self, store: RuleStore):
        self.store = store

    def evaluate(self, principal, phase: str, text: str, *, app: str = "") -> Decision:
        for scope in scopes_for(principal, app):
            for r in self.store.list(scope=scope, kind="guardrail"):
                if r.match.get("phase", "both") in (phase, "both") and _matches(r, text):
                    return Decision(r.action, f"guardrail: {r.text}", r.id, r.scope, "guardrail",
                                    replacement=r.meta.get("replacement", "[redacted]"))
        return ALLOW


class ApprovalProvider:
    """Marks a tool call ``require_approval`` when an ``approval``-kind rule matches the tool (irreversible
    ops: send email, delete, charge, CRM/financial update, block IP)."""

    def __init__(self, store: RuleStore):
        self.store = store

    def evaluate(self, principal, tool: str, args: dict, *, app: str = "") -> Decision:
        for scope in scopes_for(principal, app):
            for r in self.store.list(scope=scope, kind="approval"):
                want = r.match.get("tool")
                if want in (None, "", tool) or (r.match.get("pattern") and _matches(r, tool)):
                    return Decision("require_approval", f"approval required: {r.text}", r.id, r.scope, "approval")
        return ALLOW


class Policy:
    """Composes the providers, evaluates the right one per phase, and emits a PolicyDecision for each."""

    def __init__(self, *, store: RuleStore | None = None, guardrail: GuardrailProvider | None = None,
                 approval: ApprovalProvider | None = None, sink: DecisionSink | None = None, app: str = ""):
        self.store = store
        self.guardrail = guardrail or (GuardrailProvider(store) if store else None)
        self.approval = approval or (ApprovalProvider(store) if store else None)
        self.sink = sink or DecisionSink()
        self.app = app

    def _emit(self, principal, phase: str, d: Decision) -> PolicyDecision:
        return self.sink.emit(PolicyDecision(
            policy_decision_id=f"pd-{next(_ids)}", principal=getattr(principal, "user", "") or "",
            app=self.app, scope=d.scope, rule_id=d.rule_id, decision=d.action, reason=d.reason,
            phase=phase, provider=d.provider, at=time.time()))

    def check(self, principal, phase: str, payload, *, app: str | None = None) -> Decision:
        """``payload`` is the text (input/output) or a ``(tool, args)`` tuple (tool phase)."""
        a = self.app if app is None else app
        d = ALLOW
        if phase in ("input", "output") and self.guardrail is not None:
            d = self.guardrail.evaluate(principal, phase, payload if isinstance(payload, str) else "", app=a)
        elif phase == "tool" and self.approval is not None:
            tool, args = (payload if isinstance(payload, (tuple, list)) and len(payload) == 2 else (str(payload), {}))
            d = self.approval.evaluate(principal, tool, args or {}, app=a)
        self._emit(principal, phase, d)      # every decision is audited (docs §4)
        return d

    def redact(self, text: str, decision: Decision) -> str:
        if decision.action == "redact" and self.store is not None:
            for scope in scopes_for(None, self.app):
                for r in self.store.list(scope=scope, kind="guardrail"):
                    if r.action == "redact" and _matches(r, text):
                        pat = r.match.get("pattern") or re.escape(r.text)
                        text = re.sub(pat, r.meta.get("replacement", "[redacted]"), text,
                                      flags=re.I if not r.match.get("regex") else re.I)
        return text


# ── fleet-wide install (like set_default_authorizer) ──
_default_policy = None


def set_default_policy(policy) -> None:
    global _default_policy
    _default_policy = policy


def current_policy():
    return _default_policy
