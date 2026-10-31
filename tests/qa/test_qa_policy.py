"""The ui_action phase bound to the canonical QAPolicyDecision: every decision has a record, and its hash is what a
receipt cites."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from qa_fixtures import tnode
from runtime_contracts import qa as contracts
from runtime_contracts.canonical import content_hash

from context_runtime.policy import UI_ACTION, DecisionSink, Policy, UiActionProvider
from context_runtime.qa import (
    CostBasedQAPlanner, PlanBudget, PlanRequest, QAGraph, RunCaps, CostModel, Estimate, UiActionPolicy,
    default_action_policy_document, default_action_policy_ref, default_action_rules, default_ui_action_policy,
    is_local_target,
)

FIXED_CLOCK = lambda: "2026-01-01T00:00:00+00:00"  # noqa: E731


def make(**kw):
    sink = DecisionSink()
    return UiActionPolicy(default_ui_action_policy(sink=sink, app="shop"), clock=kw.pop("clock", FIXED_CLOCK)), sink


def decide(up, **kw):
    base = dict(principal=SimpleNamespace(user="alex"), run_id="run-1", step_id="step-1", action_kind="click",
                side_effect="NONE")
    base.update(kw)
    return up.decide(**base)


def test_an_allow_has_a_canonical_record_whose_hash_is_the_receipt_reference():
    up, sink = make()
    d = decide(up)
    assert d.action == "allow" and d.permitted
    rec = d.record
    assert type(rec) is contracts.QAPolicyDecision and rec.phase == "ui_action"
    assert rec.decision is contracts.PolicyDecisionKind.ALLOW
    assert (rec.run_id, rec.step_id, rec.principal, rec.app) == ("run-1", "step-1", "alex", "shop")
    assert rec.action_kind is contracts.ActionKind.CLICK and rec.rule_id and rec.provider == "ui_action"
    assert d.decision_hash == rec.content_hash == content_hash(rec.canonical_form())
    assert d.decision_hash.startswith("rcv1:") and len(d.decision_hash) == 69
    assert d.event.ref == d.decision_hash and d.event.phase == UI_ACTION
    assert contracts.verify_stored_hash(rec.to_json())
    assert contracts.QAPolicyDecision.from_json(rec.to_json()) == rec


def test_every_decision_allow_deny_or_require_approval_has_a_record_and_an_event():
    up, sink = make()
    cases = [dict(), dict(side_effect="bogus"), dict(side_effect=None),
             dict(side_effect="WRITES_EXTERNAL"),
             dict(side_effect="WRITES_EXTERNAL", allowed_side_effects=["WRITES_EXTERNAL"], target_local=True),
             dict(side_effect="WRITES_INTERNAL", idempotency="UNKNOWN")]
    outs = [decide(up, step_id=f"s-{i}", **c) for i, c in enumerate(cases)]
    assert [o.action for o in outs] == ["allow", "deny", "deny", "deny", "require_approval", "deny"]
    assert all(o.record is not None and o.decision_hash for o in outs)
    assert [o.record.decision.value for o in outs] == ["ALLOW", "DENY", "DENY", "DENY", "REQUIRE_APPROVAL", "DENY"]
    assert len({o.decision_hash for o in outs}) == len(outs)
    assert len(sink.recent(50, phase=UI_ACTION)) == len(cases) == len(up.records)
    assert all(e.ref == o.decision_hash for e, o in zip(sink.recent(50, phase=UI_ACTION), outs))
    assert not outs[1].permitted and not outs[4].permitted         # only an allow with a record is permitted


def test_the_hash_is_reproducible_and_does_not_depend_on_time_or_reason():
    a, _ = make(clock=lambda: "2026-01-01T00:00:00+00:00")
    b, _ = make(clock=lambda: "2031-09-09T09:09:09+00:00")
    assert decide(a).decision_hash == decide(b).decision_hash
    assert decide(a, step_id="other").decision_hash != decide(b).decision_hash
    assert decide(a, run_id="run-2").decision_hash != decide(b).decision_hash


def test_repeated_decisions_for_one_step_get_distinct_ids_and_hashes():
    up, _ = make()
    first, second = decide(up), decide(up)
    assert first.record.policy_decision_id != second.record.policy_decision_id
    assert first.decision_hash != second.decision_hash
    other, _ = make()
    assert decide(other).record.policy_decision_id == first.record.policy_decision_id    # deterministic per run/step


def test_an_unknown_side_effect_is_recorded_with_a_marker_and_denied():
    up, _ = make()
    d = decide(up, side_effect="TELEPORTS")
    assert d.action == "deny" and d.record is not None
    assert "unrecognised-side-effect" in d.record.scope
    assert d.record.side_effect.value == "SPENDS_MONEY"        # recorded as the most severe class, with the marker
    d2 = decide(up, side_effect=None, step_id="s-2")
    assert d2.action == "deny" and "unrecognised-side-effect" in d2.record.scope
    d3 = decide(up, action_kind="teleport", step_id="s-3")
    assert d3.record.action_kind is contracts.ActionKind.NAVIGATE and "unrecognised-action-kind" in d3.record.scope


def test_if_the_record_cannot_be_built_the_outcome_is_a_deny_with_no_reference_and_an_event():
    up, sink = make()
    d = decide(up, run_id="not a valid identifier")
    assert d.action == "deny" and not d.permitted and d.record is None and d.decision_hash is None
    assert "decision record failed" in d.reason and d.event.decision == "deny" and d.event.ref == ""
    assert len(sink.recent(10, phase=UI_ACTION)) == 1 and up.records == []


def test_a_policy_with_no_application_name_still_records_with_the_default_name():
    up = UiActionPolicy(default_ui_action_policy(), clock=FIXED_CLOCK)
    assert decide(up).record.app == "qa"


def test_scope_tags_are_part_of_the_record():
    up, _ = make()
    d = decide(up, scope=["host:localhost", "profile:ci"])
    assert set(d.record.scope) >= {"host:localhost", "profile:ci", "rule-scope:global"}


def test_default_policy_decisions_by_side_effect():
    up, _ = make()
    verdicts = {e: decide(up, side_effect=e, step_id=f"s-{e}").action
                for e in ("NONE", "READS_EXTERNAL", "WRITES_INTERNAL", "WRITES_EXTERNAL", "NOTIFIES_HUMAN",
                          "SPENDS_MONEY")}
    assert verdicts == {"NONE": "allow", "READS_EXTERNAL": "allow", "WRITES_INTERNAL": "allow",
                        "WRITES_EXTERNAL": "deny", "NOTIFIES_HUMAN": "deny", "SPENDS_MONEY": "deny"}
    ok = dict(allowed_side_effects=["WRITES_EXTERNAL"], target_local=True)
    assert decide(up, side_effect="WRITES_EXTERNAL", step_id="a", **ok).action == "require_approval"
    assert decide(up, side_effect="WRITES_EXTERNAL", step_id="b", allowed_side_effects=["WRITES_EXTERNAL"]).action == "deny"
    assert decide(up, side_effect="WRITES_EXTERNAL", step_id="c", target_local=True).action == "deny"


def test_enum_inputs_from_the_contracts_are_accepted():
    from runtime_contracts.models.capability import Idempotency, SideEffect

    up, _ = make()
    d = decide(up, action_kind=contracts.ActionKind.FILL, side_effect=SideEffect.WRITES_INTERNAL,
               idempotency=Idempotency.IDEMPOTENT)
    assert d.action == "allow" and d.record.action_kind is contracts.ActionKind.FILL
    assert decide(up, side_effect=SideEffect.WRITES_INTERNAL, idempotency=Idempotency.UNKNOWN,
                  step_id="s2").action == "deny"


def test_a_policy_without_a_provider_denies_and_still_produces_a_record():
    up = UiActionPolicy(Policy(sink=DecisionSink(), app="shop"), clock=FIXED_CLOCK)
    d = decide(up)
    assert d.action == "deny" and d.record is not None and d.record.rule_id is None


def test_a_provider_error_and_a_timeout_produce_denies_with_records():
    class Boom:
        def evaluate(self, *a, **k):
            raise RuntimeError("x")

    class Slow:
        def evaluate(self, *a, **k):
            import time
            time.sleep(0.5)

    for provider, kw in ((Boom(), {}), (Slow(), {"ui_action_timeout_s": 0.05})):
        up = UiActionPolicy(Policy(sink=DecisionSink(), app="shop", ui_action=provider, **kw), clock=FIXED_CLOCK)
        d = decide(up)
        assert d.action == "deny" and d.record is not None and d.record.decision is contracts.PolicyDecisionKind.DENY


def test_principal_without_a_user_name_is_recorded_as_empty():
    up, _ = make()
    assert decide(up, principal=None).record.principal == ""


def test_contracts_only_record_ui_action_decisions():
    up, _ = make()
    j = decide(up).record.to_json()
    j["phase"] = "tool"
    j.pop("content_hash")
    with pytest.raises(contracts.QAContractError):
        contracts.QAPolicyDecision.from_json(j)


# ── the default action policy and its reference ────────────────────────────────────────────────────────────

def test_the_policy_reference_is_the_rcv1_hash_of_the_policy_document_and_is_stable():
    doc = default_action_policy_document()
    assert default_action_policy_ref() == content_hash(doc) == default_action_policy_ref()
    assert doc["default"] == "deny" and doc["meaningful_decisions"] == ["allow", "deny", "require_approval"]
    assert {r["id"] for r in doc["rules"]} == {r.id for r in default_action_rules()}


def test_any_change_to_the_policy_changes_its_reference():
    doc = default_action_policy_document()
    base = content_hash(doc)
    doc2 = {**doc, "rules": doc["rules"][:-1]}
    assert content_hash(doc2) != base
    assert content_hash({**doc, "default": "allow"}) != base
    assert content_hash({**doc, "guards_version": "2"}) != base


def test_a_plan_can_carry_the_default_policy_reference_and_the_contracts_accept_it():
    g = QAGraph([tnode("a")])
    req = PlanRequest(g, ("a",), CostModel({}, default=Estimate(100)), PlanBudget(max_parallel=1),
                      RunCaps(deadline_ms=10000), action_policy_ref=default_action_policy_ref())
    plan = CostBasedQAPlanner().plan(req)
    assert plan.action_policy_ref == default_action_policy_ref()
    assert contracts.QAPlan.from_json(plan.to_json()) == plan
    other = CostBasedQAPlanner().plan(PlanRequest(g, ("a",), req.estimates, req.budget, req.caps))
    assert other.plan_hash != plan.plan_hash


@pytest.mark.parametrize("host,local", [("localhost", True), ("LOCALHOST", True), ("app.localhost", True),
                                        ("127.0.0.1", True), ("127.9.9.9", True), ("::1", True), ("[::1]", True),
                                        ("example.com", False), ("localhost.example.com", False), ("10.0.0.1", False),
                                        ("0.0.0.0", False), ("", False), (None, False), ("169.254.169.254", False),
                                        ("127.0.0.1.example.com", False)])
def test_is_local_target_does_no_resolution_and_is_conservative(host, local):
    assert is_local_target(host) is local


def test_the_default_policy_uses_only_the_three_meaningful_decisions():
    assert {r.action for r in default_action_rules()} <= {"allow", "require_approval"}
