"""The ``ui_action`` policy phase, bound to the canonical ``qa/v1`` ``QAPolicyDecision``.

The plane (``context_runtime.policy``) decides and audits; this module turns each decision into the canonical record
a receipt refers to. Every decision, allow or deny, produces one ``QAPolicyDecision`` whose ``content_hash`` is the
reference a receipt cites. An action without such a reference must be refused by whatever performs actions.

Fail-closed, as in the plane: no matching rule, an unknown side-effect class, a provider error or a timeout is a deny;
only ``allow``, ``deny`` and ``require_approval`` are meaningful; ``WRITES_EXTERNAL``, ``NOTIFIES_HUMAN`` and
``SPENDS_MONEY`` are denied unless the reviewed spec lists them and the target is a local sink. If the canonical record
itself cannot be built (for example an identifier that is not valid), the outcome is a deny with no reference.

``default_action_policy_ref()`` is the hash of the default QA action policy document. A plan carries it as
``action_policy_ref`` so that the policy a plan was built to run under is part of the plan's identity.
"""
from __future__ import annotations

import hashlib
import ipaddress
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Iterable

from ..policy.plane import (
    EXTERNAL_SIDE_EFFECTS, UI_ACTION, UI_ACTION_ACTIONS, Decision, DecisionSink, Policy,
    PolicyDecision, UiActionProvider, UiActionRequest,
)
from ..policy.store import Rule, rule_id
from ._contracts import SideEffect, canonical_json, content_hash, qa as contracts

DEFAULT_APP = "qa"
"""The application name on a decision when the policy has none (the canonical record requires one)."""
GUARDS_VERSION = "1"
"""Bump when the fixed guards of ``UiActionProvider`` change, so the default policy hash changes with them."""
POLICY_SCHEMA_ID = "context-runtime/qa-action-policy"

_MOST_SEVERE_SIDE_EFFECT = SideEffect.SPENDS_MONEY
_MOST_EGRESS_ACTION = contracts.ActionKind.NAVIGATE


# ── the default action policy ──────────────────────────────────────────────────────────────────────────────

def _rule(action: str, text: str, match: dict) -> Rule:
    return Rule(id=rule_id("global", UI_ACTION, text), scope="global", kind=UI_ACTION, text=text, action=action,
                match=match)


def default_action_rules() -> tuple[Rule, ...]:
    """Local-only effects are allowed; effects that leave the local sandbox need approval (and, before any rule is
    consulted, the reviewed spec must list them and the target must be local). Everything else is denied by default."""
    return (
        _rule("allow", "steps without external effect: none, external reads, internal writes",
              {"side_effect": ["NONE", "READS_EXTERNAL", "WRITES_INTERNAL"]}),
        _rule("require_approval", "steps with an external effect need approval",
              {"side_effect": list(EXTERNAL_SIDE_EFFECTS)}),
    )


def default_action_policy_document() -> dict:
    """The default policy as data. Its ``rcv1`` hash is ``default_action_policy_ref()``."""
    return {
        "schema_id": POLICY_SCHEMA_ID, "schema_version": "1", "guards_version": GUARDS_VERSION, "default": "deny",
        "meaningful_decisions": list(UI_ACTION_ACTIONS),
        "guards": [
            "unknown side-effect class: deny",
            "mutating step with unknown idempotency: deny",
            "external side effect not listed by the reviewed spec: deny",
            "external side effect against a target that is not a local sink: deny",
            "redact, flag or unrecognised rule action: deny",
            "strictest matching rule wins; no matching rule: deny",
            "provider error or timeout: deny",
        ],
        "rules": [{"id": r.id, "action": r.action, "text": r.text, "match": r.match}
                  for r in sorted(default_action_rules(), key=lambda r: r.id)],
    }


def default_action_policy_ref() -> str:
    return content_hash(default_action_policy_document())


def default_ui_action_policy(*, sink: DecisionSink | None = None, app: str = "",
                             ui_action_timeout_s: float | None = 5.0) -> Policy:
    """A ``Policy`` whose ``ui_action`` phase runs the default QA action policy."""
    return Policy(sink=sink, app=app, ui_action=UiActionProvider(rules=default_action_rules()),
                  ui_action_timeout_s=ui_action_timeout_s)


def is_local_target(host: str) -> bool:
    """True for ``localhost``, ``*.localhost`` and loopback addresses. No name resolution is done, so a name that is not
    literally local is not local."""
    if not isinstance(host, str) or not host:
        return False
    name = host.strip().strip("[]").lower()
    if name == "localhost" or name.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


# ── the decision ───────────────────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class UiActionDecision:
    action: str
    """``allow``, ``deny`` or ``require_approval``."""
    record: "contracts.QAPolicyDecision | None"
    """The canonical record, or ``None`` when it could not be built (the action is then a deny)."""
    event: PolicyDecision
    """The event the plane emitted (audit/EXPLAIN)."""
    reason: str

    @property
    def permitted(self) -> bool:
        """True only for an allow that has a canonical record to cite."""
        return self.action == "allow" and self.record is not None

    @property
    def decision_hash(self) -> str | None:
        """The reference a receipt cites (``ActionReceipt.policy_decision_ref``)."""
        return self.record.content_hash if self.record is not None else None


class UiActionPolicy:
    """Decides ``ui_action`` requests through a ``Policy`` and produces the canonical record for each."""

    def __init__(self, policy: Policy, *, clock: Callable[[], str] | None = None):
        self.policy = policy
        self._clock = clock or (lambda: datetime.now(timezone.utc).isoformat())
        self._seq: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()
        self.records: list = []
        """Every canonical record produced, in order (the caller may persist them)."""

    def _next_id(self, run_id: str, step_id: str, action_kind: str, side_effect: str) -> str:
        with self._lock:
            key = (str(run_id), str(step_id))
            n = self._seq.get(key, 0) + 1
            self._seq[key] = n
        digest = hashlib.sha256(canonical_json([run_id, step_id, action_kind, side_effect, n]).encode()).hexdigest()
        return f"pd-{digest[:24]}"

    def decide(self, *, principal, run_id: str, step_id: str, action_kind: str, side_effect: str | None,
               idempotency: str | None = None, allowed_side_effects: Iterable = (), target_local: bool = False,
               scope: Iterable[str] = (), app: str | None = None) -> UiActionDecision:
        req = UiActionRequest(
            action_kind=_text(action_kind), side_effect=_enum_text(side_effect), idempotency=_enum_text(idempotency),
            allowed_side_effects=tuple(_enum_text(e) for e in (allowed_side_effects or ())),
            target_local=target_local is True, scope=tuple(scope))
        app_name = (self.policy.app or DEFAULT_APP) if app is None else app
        built: dict = {}

        def record(d: Decision) -> str:
            rec = self._build(principal, run_id, step_id, req, d, app_name)
            built["record"] = rec
            return rec.content_hash

        d, event = self.policy.decide_ui_action(principal, req, app=app_name, record=record)
        rec = built.get("record")      # absent when the canonical record could not be built: the plane then denied
        if rec is not None:
            self.records.append(rec)
        return UiActionDecision(d.action, rec, event, d.reason)

    def _build(self, principal, run_id, step_id, req: UiActionRequest, d: Decision, app: str):
        markers = []
        try:
            kind = contracts.ActionKind(req.action_kind)
        except ValueError:
            kind = _MOST_EGRESS_ACTION
            markers.append("unrecognised-action-kind")
        try:
            effect = SideEffect(req.side_effect)
        except ValueError:
            effect = _MOST_SEVERE_SIDE_EFFECT
            markers.append("unrecognised-side-effect")
        scope = [*req.scope, *markers]
        if d.scope:
            scope.append(f"rule-scope:{d.scope}")
        return contracts.QAPolicyDecision(
            policy_decision_id=self._next_id(run_id, step_id, req.action_kind, str(req.side_effect)),
            run_id=run_id, step_id=step_id, principal=getattr(principal, "user", "") or "", app=app,
            decision=contracts.PolicyDecisionKind(d.action.upper()), provider=d.provider or "ui_action",
            action_kind=kind, side_effect=effect, scope=tuple(sorted(set(scope))), rule_id=d.rule_id or None,
            reason=d.reason[:4096] if d.reason else None, decided_at=self._clock())


def _text(value) -> str:
    return getattr(value, "value", value) if value is not None else ""


def _enum_text(value):
    value = getattr(value, "value", value)
    return value if isinstance(value, str) else None
