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
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .store import Rule, RuleStore, scopes_for

ACTIONS = ("allow", "deny", "redact", "flag", "require_approval")

# The ``ui_action`` phase: whether one UI step (a click, a fill, a navigation) may be performed. It is fail-closed (see
# UiActionProvider): no matching rule, an unknown side-effect class, a provider error and a timeout all deny, and only
# allow / deny / require_approval are meaningful, so a redact or flag rule is treated as deny.
UI_ACTION = "ui_action"
UI_ACTION_ACTIONS = ("allow", "deny", "require_approval")


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
    phase: str                          # planning | command | input | tool | retrieval | output | ui_action
    provider: str = ""
    at: float = 0.0
    ref: str = ""                       # ui_action: hash of the canonical decision record a receipt cites

    def to_dict(self) -> dict:
        d = {"policy_decision_id": self.policy_decision_id, "principal": self.principal, "app": self.app,
             "scope": self.scope, "rule_id": self.rule_id, "decision": self.decision, "reason": self.reason,
             "phase": self.phase, "provider": self.provider, "at": self.at}
        if self.ref:
            d["ref"] = self.ref
        return d

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


SIDE_EFFECTS = ("NONE", "READS_EXTERNAL", "WRITES_INTERNAL", "WRITES_EXTERNAL", "NOTIFIES_HUMAN", "SPENDS_MONEY")
EXTERNAL_SIDE_EFFECTS = ("WRITES_EXTERNAL", "NOTIFIES_HUMAN", "SPENDS_MONEY")
_KNOWN_IDEMPOTENCY = ("IDEMPOTENT", "AT_MOST_ONCE")
_UI_MATCH_KEYS = ("action_kind", "side_effect", "target_local")


@dataclass(frozen=True)
class UiActionRequest:
    """One UI step to be decided: plain data, so the plane needs no QA schema."""

    action_kind: str
    side_effect: str | None
    """The step's side-effect class. Absent or not one of ``SIDE_EFFECTS`` is an unknown class and is denied."""
    idempotency: str | None = None
    """``IDEMPOTENT`` / ``AT_MOST_ONCE`` / ``UNKNOWN``; ``None`` means not stated (not checked)."""
    allowed_side_effects: tuple = ()
    """The side-effect classes the reviewed spec lists as allowed."""
    target_local: bool = False
    """True only when the caller has established that the target is a local sink."""
    scope: tuple = ()


def _as_list(value) -> list:
    return list(value) if isinstance(value, (list, tuple, set, frozenset)) else [value]


class UiActionProvider:
    """Decides a ``ui_action`` request. Fail-closed by construction:

    * an unknown side-effect class (absent, ``UNKNOWN``, or not one of ``SIDE_EFFECTS``) is a deny;
    * a mutating step whose idempotency is ``UNKNOWN`` is a deny;
    * ``WRITES_EXTERNAL``, ``NOTIFIES_HUMAN`` and ``SPENDS_MONEY`` are a deny unless the reviewed spec lists the class in
      ``allowed_side_effects`` and the target is a local sink;
    * then the ``ui_action`` rules decide: the strictest matching action wins (deny, then require_approval, then
      allow); a ``redact`` / ``flag`` / unrecognised action is a deny; no matching rule is a deny;
    * a non-allow rule that cannot be evaluated (an unrecognised ``match`` key) is treated as matching, an allow rule
      that cannot be evaluated is treated as not matching.

    Rules come from ``ui_action``-kind rules of the store (global, app and app:user scope) and from static ``rules``.
    """

    name = "ui_action"

    def __init__(self, store: RuleStore | None = None, rules: Iterable[Rule] = ()):
        self.store = store
        self.rules = tuple(rules)

    def _rules(self, principal, app: str) -> list[Rule]:
        out = list(self.rules)
        if self.store is not None:
            for scope in scopes_for(principal, app):
                out.extend(self.store.list(scope=scope, kind=UI_ACTION))
        return sorted(out, key=lambda r: (r.id, r.scope))

    def _deny(self, reason: str, rule_id: str = "", scope: str = "") -> Decision:
        return Decision("deny", reason, rule_id, scope, self.name)

    @staticmethod
    def _effect(rule: Rule, req: UiActionRequest) -> str | None:
        """The action a rule contributes for ``req``, or None when it does not apply."""
        action = rule.action if rule.action in UI_ACTION_ACTIONS else "deny"   # redact / flag / unknown -> deny
        match = rule.match or {}
        evaluable = all(k in _UI_MATCH_KEYS for k in match)
        if not evaluable:
            return None if action == "allow" else action
        if "action_kind" in match and req.action_kind not in _as_list(match["action_kind"]):
            return None
        if "side_effect" in match and (req.side_effect or "") not in _as_list(match["side_effect"]):
            return None
        if "target_local" in match and bool(match["target_local"]) != bool(req.target_local):
            return None
        return action

    def evaluate(self, principal, request, *, app: str = "") -> Decision:
        if not isinstance(request, UiActionRequest) or not isinstance(request.action_kind, str) \
                or not request.action_kind:
            return self._deny("malformed ui_action request")
        effect = request.side_effect if isinstance(request.side_effect, str) else ""
        if effect not in SIDE_EFFECTS:
            return self._deny(f"unknown side-effect class {request.side_effect!r}")
        if request.idempotency is not None and request.idempotency not in _KNOWN_IDEMPOTENCY \
                and effect not in ("NONE", "READS_EXTERNAL"):
            return self._deny(f"unknown idempotency {request.idempotency!r} on a mutating step ({effect})")
        if effect in EXTERNAL_SIDE_EFFECTS:
            listed = {e for e in (request.allowed_side_effects or ()) if isinstance(e, str)}
            if effect not in listed:
                return self._deny(f"{effect} is not listed in the reviewed allowed side effects")
            if request.target_local is not True:
                return self._deny(f"{effect} is only permitted against a local sink")
        applied: list[tuple[str, Rule]] = []
        for r in self._rules(principal, app):
            act = self._effect(r, request)
            if act is not None:
                applied.append((act, r))
        if not applied:
            return self._deny("no matching ui_action rule (default deny)")
        strictness = {"deny": 0, "require_approval": 1, "allow": 2}
        act, rule = min(applied, key=lambda ar: (strictness[ar[0]], ar[1].id, ar[1].scope))
        return Decision(act, f"ui_action rule: {rule.text}", rule.id, rule.scope, self.name)


def _call_bounded(fn: Callable[[], object], timeout_s: float | None):
    """Run ``fn``; return ``("ok", value)``, ``("error", exc)`` or ``("timeout", None)``. A call that outlives the
    timeout is abandoned (its thread is a daemon), never waited for."""
    if timeout_s is None:
        try:
            return "ok", fn()
        except Exception as exc:  # noqa: BLE001
            return "error", exc
    box: dict = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except Exception as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        return "timeout", None
    if "error" in box:
        return "error", box["error"]
    return "ok", box.get("value")


class Policy:
    """Composes the providers, evaluates the right one per phase, and emits a PolicyDecision for each."""

    def __init__(self, *, store: RuleStore | None = None, guardrail: GuardrailProvider | None = None,
                 approval: ApprovalProvider | None = None, sink: DecisionSink | None = None, app: str = "",
                 ui_action: UiActionProvider | None = None, ui_action_timeout_s: float | None = 5.0):
        self.store = store
        self.guardrail = guardrail or (GuardrailProvider(store) if store else None)
        self.approval = approval or (ApprovalProvider(store) if store else None)
        self.ui_action = ui_action          # no default provider: without one every ui_action is denied
        self.ui_action_timeout_s = ui_action_timeout_s
        self.sink = sink or DecisionSink()
        self.app = app

    def _emit(self, principal, phase: str, d: Decision, *, app: str | None = None, ref: str = "") -> PolicyDecision:
        return self.sink.emit(PolicyDecision(
            policy_decision_id=f"pd-{next(_ids)}", principal=getattr(principal, "user", "") or "",
            app=self.app if app is None else app, scope=d.scope, rule_id=d.rule_id, decision=d.action,
            reason=d.reason, phase=phase, provider=d.provider, at=time.time(), ref=ref))

    def decide_ui_action(self, principal, request, *, app: str | None = None,
                         record: Callable[[Decision], str] | None = None) -> tuple[Decision, PolicyDecision]:
        """Decide one ``ui_action`` and emit its PolicyDecision, whatever the outcome (docs §4).

        Fail-closed: no provider, a provider error, a timeout, a result that is not a Decision with one of
        ``UI_ACTION_ACTIONS`` (a ``redact`` / ``flag`` result counts as ``deny``), or a ``record`` callback that
        raises, all produce ``deny``. ``record`` receives the final Decision and returns the hash of the canonical
        record the caller stores for it; that hash is put on the emitted event (``ref``).
        """
        a = self.app if app is None else app
        d = self._evaluate_ui_action(principal, request, a)
        ref = ""
        if record is not None:
            try:
                ref = str(record(d))
            except Exception as exc:  # noqa: BLE001
                d = Decision("deny", f"decision record failed: {type(exc).__name__}", "", "", "ui_action")
                ref = ""
        return d, self._emit(principal, UI_ACTION, d, app=a, ref=ref)

    def _evaluate_ui_action(self, principal, request, app: str) -> Decision:
        provider = self.ui_action
        if provider is None:
            return Decision("deny", "no ui_action provider configured (default deny)", "", "", "ui_action")
        outcome, value = _call_bounded(lambda: provider.evaluate(principal, request, app=app), self.ui_action_timeout_s)
        if outcome == "timeout":
            return Decision("deny", "ui_action provider timed out", "", "", "ui_action")
        if outcome == "error":
            return Decision("deny", f"ui_action provider error: {type(value).__name__}", "", "", "ui_action")
        if not isinstance(value, Decision):
            return Decision("deny", "ui_action provider returned no decision", "", "", "ui_action")
        if value.action in ("allow", "require_approval"):
            return Decision(value.action, value.reason, value.rule_id, value.scope, value.provider or "ui_action")
        reason = value.reason if value.action == "deny" else f"{value.action} is not meaningful for ui_action; {value.reason}"
        return Decision("deny", reason, value.rule_id, value.scope, value.provider or "ui_action")

    def check(self, principal, phase: str, payload, *, app: str | None = None) -> Decision:
        """``payload`` is the text (input/output) or a ``(tool, args)`` tuple (tool phase)."""
        a = self.app if app is None else app
        if phase == UI_ACTION:
            return self.decide_ui_action(principal, payload, app=a)[0]     # fail-closed, always audited
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
