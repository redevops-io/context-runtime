"""The ``ui_action`` phase of the policy plane: fail-closed, and every decision audited.

Needs no QA schema package: the plane works on plain data. Existing phases are covered by test_policy_runtime.py and
are asserted unchanged here as well.
"""
from __future__ import annotations

import hashlib
import time
from types import SimpleNamespace

import pytest

from context_runtime.policy import (
    UI_ACTION, UI_ACTION_ACTIONS, Decision, DecisionSink, Policy, PolicyDecision, RuleStore, UiActionProvider,
    UiActionRequest,
)
from context_runtime.policy.store import Rule


def who(user="alex"):
    return SimpleNamespace(user=user, app="shop", roles=frozenset())


def rule(action, match=None, text="r", scope="global"):
    rid = "r-" + hashlib.sha256(repr((action, text, match, scope)).encode()).hexdigest()[:8]
    return Rule(id=rid, scope=scope, kind=UI_ACTION, text=text, action=action, match=match or {})


def policy(*rules, sink=None, **kw):
    return Policy(sink=sink or DecisionSink(), app="shop", ui_action=UiActionProvider(rules=rules), **kw)


def req(side_effect="NONE", kind="click", **kw):
    return UiActionRequest(action_kind=kind, side_effect=side_effect, **kw)


def decide(p, request, **kw):
    return p.decide_ui_action(who(), request, **kw)


ALLOW_ALL = rule("allow", {}, text="allow everything")


def test_phase_constants():
    assert UI_ACTION == "ui_action" and UI_ACTION_ACTIONS == ("allow", "deny", "require_approval")


# ── fail-closed defaults ───────────────────────────────────────────────────────────────────────────────────

def test_without_a_provider_every_request_is_denied_and_audited():
    sink = DecisionSink()
    p = Policy(sink=sink, app="shop")
    d, ev = decide(p, req("NONE"))
    assert d.action == "deny" and "no ui_action provider" in d.reason
    assert ev.phase == UI_ACTION and ev.decision == "deny" and sink.recent(phase=UI_ACTION) == [ev]


def test_no_matching_rule_is_a_deny():
    d, _ = decide(policy(), req("NONE"))
    assert d.action == "deny" and "default deny" in d.reason
    d, _ = decide(policy(rule("allow", {"side_effect": "READS_EXTERNAL"})), req("NONE"))
    assert d.action == "deny"


@pytest.mark.parametrize("bad", [None, "", "UNKNOWN", "bogus", "none", 7, ["NONE"]])
def test_an_unknown_side_effect_class_is_denied_even_when_a_rule_allows_everything(bad):
    d, _ = decide(policy(ALLOW_ALL), req(bad))
    assert d.action == "deny" and "unknown side-effect" in d.reason


def test_a_mutating_step_with_unknown_idempotency_is_denied_but_a_read_is_not():
    p = policy(ALLOW_ALL)
    assert decide(p, req("WRITES_INTERNAL", idempotency="UNKNOWN"))[0].action == "deny"
    assert decide(p, req("WRITES_INTERNAL", idempotency="nonsense"))[0].action == "deny"
    assert decide(p, req("WRITES_INTERNAL", idempotency="IDEMPOTENT"))[0].action == "allow"
    assert decide(p, req("WRITES_INTERNAL", idempotency="AT_MOST_ONCE"))[0].action == "allow"
    assert decide(p, req("WRITES_INTERNAL"))[0].action == "allow"                    # not stated: not checked
    assert decide(p, req("NONE", idempotency="UNKNOWN"))[0].action == "allow"
    assert decide(p, req("READS_EXTERNAL", idempotency="UNKNOWN"))[0].action == "allow"


@pytest.mark.parametrize("effect", ["WRITES_EXTERNAL", "NOTIFIES_HUMAN", "SPENDS_MONEY"])
def test_external_effects_need_the_reviewed_listing_and_a_local_sink(effect):
    p = policy(ALLOW_ALL)
    assert "not listed" in decide(p, req(effect))[0].reason
    assert decide(p, req(effect, allowed_side_effects=("WRITES_INTERNAL",), target_local=True))[0].action == "deny"
    assert "local sink" in decide(p, req(effect, allowed_side_effects=(effect,)))[0].reason
    assert decide(p, req(effect, allowed_side_effects=(effect,), target_local=False))[0].action == "deny"
    assert decide(p, req(effect, allowed_side_effects=(effect,), target_local=True))[0].action == "allow"
    assert decide(p, req(effect, allowed_side_effects=(effect,), target_local=1))[0].action == "deny"   # only True counts


def test_an_external_effect_listed_and_local_still_needs_a_matching_rule():
    p = policy()
    d, _ = decide(p, req("WRITES_EXTERNAL", allowed_side_effects=("WRITES_EXTERNAL",), target_local=True))
    assert d.action == "deny" and "default deny" in d.reason


def test_a_malformed_request_is_denied():
    p = policy(ALLOW_ALL)
    assert decide(p, "click")[0].action == "deny"
    assert decide(p, None)[0].action == "deny"
    assert decide(p, UiActionRequest(action_kind="", side_effect="NONE"))[0].action == "deny"


# ── rules ──────────────────────────────────────────────────────────────────────────────────────────────────

def test_allow_rule_matches_on_action_kind_side_effect_and_locality():
    p = policy(rule("allow", {"action_kind": ["click", "hover"], "side_effect": "NONE"}))
    assert decide(p, req("NONE", "click"))[0].action == "allow"
    assert decide(p, req("NONE", "fill"))[0].action == "deny"
    assert decide(p, req("READS_EXTERNAL", "click"))[0].action == "deny"
    q = policy(rule("allow", {"target_local": True}))
    assert decide(q, req("NONE", target_local=True))[0].action == "allow"
    assert decide(q, req("NONE", target_local=False))[0].action == "deny"


def test_the_strictest_matching_rule_wins():
    allow, ask, deny = rule("allow", text="a"), rule("require_approval", text="b"), rule("deny", text="c")
    assert decide(policy(allow, ask), req())[0].action == "require_approval"
    assert decide(policy(allow, ask, deny), req())[0].action == "deny"
    assert decide(policy(allow), req())[0].action == "allow"
    got = decide(policy(allow, deny), req())[0]
    assert got.rule_id == deny.id and "c" in got.reason


@pytest.mark.parametrize("action", ["redact", "flag", "block", "ALLOW", "", "maybe"])
def test_actions_that_are_not_meaningful_for_this_phase_are_treated_as_deny(action):
    d, _ = decide(policy(rule(action, text="odd"), ALLOW_ALL), req())
    assert d.action == "deny"


def test_a_rule_with_an_unrecognised_match_key_is_ignored_when_it_allows_and_applies_when_it_denies():
    typo_allow = rule("allow", {"side_effects": "NONE"}, text="typo allow")
    assert decide(policy(typo_allow), req("NONE"))[0].action == "deny"
    typo_deny = rule("deny", {"side_effects": "WRITES_INTERNAL"}, text="typo deny")
    assert decide(policy(ALLOW_ALL, typo_deny), req("NONE"))[0].action == "deny"


def test_rules_are_read_from_the_store_for_global_app_and_user_scopes(tmp_path):
    store = RuleStore(dir=str(tmp_path))
    store.add("global", UI_ACTION, "browse is fine", action="allow", match={"side_effect": "NONE"})
    p = Policy(store=store, app="shop", ui_action=UiActionProvider(store=store), sink=DecisionSink())
    assert decide(p, req("NONE"))[0].action == "allow"
    store.add("shop", UI_ACTION, "no fills here", action="deny", match={"action_kind": "fill"})
    assert decide(p, req("NONE", "fill"))[0].action == "deny"
    store.add("shop:alex", UI_ACTION, "alex allows everything", action="allow")
    assert decide(p, req("NONE", "fill"))[0].action == "deny"          # a user rule cannot lift a stricter one
    assert decide(p, req("NONE", "click"))[0].action == "allow"
    assert decide(p, req("bogus"))[0].action == "deny"


def test_a_guardrail_or_approval_rule_is_not_a_ui_action_rule(tmp_path):
    store = RuleStore(dir=str(tmp_path))
    store.add("global", "guardrail", "allow everything", action="allow")
    p = Policy(store=store, app="shop", ui_action=UiActionProvider(store=store), sink=DecisionSink())
    assert decide(p, req("NONE"))[0].action == "deny"


# ── provider failure ───────────────────────────────────────────────────────────────────────────────────────

class Raising:
    def evaluate(self, principal, request, *, app=""):
        raise RuntimeError("boom with details that stay out of the reason")


class Slow:
    def evaluate(self, principal, request, *, app=""):
        time.sleep(0.5)
        return Decision("allow", "late", "r-1", "global", "slow")


class Returns:
    def __init__(self, value):
        self.value = value

    def evaluate(self, principal, request, *, app=""):
        return self.value


def with_provider(provider, **kw):
    return Policy(sink=DecisionSink(), app="shop", ui_action=provider, **kw)


def test_a_provider_error_is_a_deny_that_does_not_leak_the_message():
    d, ev = decide(with_provider(Raising()), req())
    assert d.action == "deny" and "RuntimeError" in d.reason and "boom" not in d.reason
    assert ev.decision == "deny"


def test_a_provider_timeout_is_a_deny():
    d, ev = decide(with_provider(Slow(), ui_action_timeout_s=0.05), req())
    assert d.action == "deny" and "timed out" in d.reason and ev.decision == "deny"


def test_a_provider_that_returns_garbage_is_a_deny():
    for value in (None, "allow", 1, {"action": "allow"}, object()):
        assert decide(with_provider(Returns(value)), req())[0].action == "deny"


def test_a_provider_that_returns_redact_or_flag_is_a_deny_with_the_reason_noted():
    for action in ("redact", "flag", "weird"):
        d, _ = decide(with_provider(Returns(Decision(action, "x", "r-9", "global", "p"))), req())
        assert d.action == "deny" and "not meaningful" in d.reason and d.rule_id == "r-9"


def test_a_good_custom_provider_decision_passes_through():
    d, _ = decide(with_provider(Returns(Decision("require_approval", "ask", "r-2", "app", "mine"))), req())
    assert (d.action, d.rule_id, d.provider) == ("require_approval", "r-2", "mine")


# ── audit ──────────────────────────────────────────────────────────────────────────────────────────────────

def test_every_decision_emits_a_policy_decision_allow_or_deny():
    sink = DecisionSink()
    p = policy(rule("allow", {"side_effect": "NONE"}), rule("require_approval", {"side_effect": "WRITES_INTERNAL"}),
               sink=sink)
    outcomes = [decide(p, req(e))[0].action for e in ("NONE", "WRITES_INTERNAL", "READS_EXTERNAL", "bogus", None)]
    assert outcomes == ["allow", "require_approval", "deny", "deny", "deny"]
    events = sink.recent(50, phase=UI_ACTION)
    assert [e.decision for e in events] == outcomes
    assert all(isinstance(e, PolicyDecision) and e.phase == UI_ACTION and e.provider == "ui_action" for e in events)
    assert all(e.principal == "alex" and e.app == "shop" and e.policy_decision_id.startswith("pd-") for e in events)
    assert len({e.policy_decision_id for e in events}) == len(events)


def test_check_with_the_ui_action_phase_is_fail_closed_and_audited():
    sink = DecisionSink()
    p = policy(ALLOW_ALL, sink=sink)
    assert p.check(who(), UI_ACTION, req("NONE")).action == "allow"
    assert p.check(who(), UI_ACTION, ("click", {})).action == "deny"
    assert p.check(who(), UI_ACTION, None).action == "deny"
    assert len(sink.recent(10, phase=UI_ACTION)) == 3


def test_the_record_callback_hash_is_put_on_the_event():
    p = policy(ALLOW_ALL)
    d, ev = decide(p, req(), record=lambda decision: "rcv1:cited-by-a-receipt")
    assert d.action == "allow" and ev.ref == "rcv1:cited-by-a-receipt"
    assert ev.to_dict()["ref"] == "rcv1:cited-by-a-receipt"


def test_a_record_callback_that_fails_turns_the_decision_into_a_deny():
    def broken(decision):
        raise ValueError("cannot build the record")

    d, ev = decide(policy(ALLOW_ALL), req(), record=broken)
    assert d.action == "deny" and "decision record failed" in d.reason
    assert ev.decision == "deny" and ev.ref == ""


# ── existing behaviour is unchanged ────────────────────────────────────────────────────────────────────────

def test_policy_decision_dict_is_unchanged_when_there_is_no_reference():
    ev = PolicyDecision("pd-1", "u", "a", "s", "r", "allow", "why", "tool", "p", 1.0)
    assert ev.to_dict() == {"policy_decision_id": "pd-1", "principal": "u", "app": "a", "scope": "s", "rule_id": "r",
                            "decision": "allow", "reason": "why", "phase": "tool", "provider": "p", "at": 1.0}


def test_other_phases_still_allow_by_default_and_are_not_made_fail_closed():
    p = Policy(sink=DecisionSink(), app="shop")
    for phase in ("planning", "command", "input", "output", "tool", "retrieval"):
        assert p.check(who(), phase, "text").action == "allow"
    assert [e.phase for e in p.sink.recent(10)] == ["planning", "command", "input", "output", "tool", "retrieval"]
