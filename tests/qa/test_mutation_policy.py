"""Mutation tests for the ui_action phase: remove a fail-closed rule from a throw-away copy and the tests must fail."""
from __future__ import annotations

import pytest
from qa_subproc import MutantTree

SUITE = ("tests/test_policy_ui_action.py", "tests/qa/test_qa_policy.py")
PLANE = "context_runtime/policy/plane.py"


@pytest.fixture(scope="module")
def control():
    tree = MutantTree()
    try:
        out = tree.run_tests(*SUITE)
        assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-2000:]
        yield
    finally:
        tree.close()


def _run(edits):
    tree = MutantTree()
    try:
        for relpath, old, new in edits:
            tree.mutate(relpath, old, new)
        return tree.run_tests(*SUITE)
    finally:
        tree.close()


def test_removing_the_default_deny_fails_the_suite(control):
    out = _run([(PLANE, 'return self._deny("no matching ui_action rule (default deny)")',
                 'return Decision("allow", "no rule", "", "", self.name)')])
    assert out.returncode != 0, "the tests still pass when no matching rule means allow"


def test_removing_the_unknown_side_effect_deny_fails_the_suite(control):
    out = _run([(PLANE, "if effect not in SIDE_EFFECTS:", "if False:")])
    assert out.returncode != 0


def test_a_provider_error_that_allows_fails_the_suite(control):
    out = _run([(PLANE, 'return Decision("deny", f"ui_action provider error: {type(value).__name__}", "", "", "ui_action")',
                 'return Decision("allow", "error ignored", "", "", "ui_action")')])
    assert out.returncode != 0


def test_a_provider_timeout_that_allows_fails_the_suite(control):
    out = _run([(PLANE, 'return Decision("deny", "ui_action provider timed out", "", "", "ui_action")',
                 'return Decision("allow", "late", "", "", "ui_action")')])
    assert out.returncode != 0


def test_no_provider_that_allows_fails_the_suite(control):
    out = _run([(PLANE, 'return Decision("deny", "no ui_action provider configured (default deny)", "", "", "ui_action")',
                 'return Decision("allow", "no provider", "", "", "ui_action")')])
    assert out.returncode != 0


def test_treating_a_redact_rule_as_allow_fails_the_suite(control):
    out = _run([(PLANE, 'action = rule.action if rule.action in UI_ACTION_ACTIONS else "deny"',
                 'action = "allow" if rule.action in ("redact", "flag") else rule.action')])
    assert out.returncode != 0


def test_dropping_the_external_effect_gate_fails_the_suite(control):
    out = _run([(PLANE, "if effect in EXTERNAL_SIDE_EFFECTS:", "if False:")])
    assert out.returncode != 0


def test_not_emitting_the_event_for_a_deny_fails_the_suite(control):
    out = _run([(PLANE, "        return d, self._emit(principal, UI_ACTION, d, app=a, ref=ref)",
                 "        return d, (self._emit(principal, UI_ACTION, d, app=a, ref=ref) if d.action == 'allow' "
                 "else PolicyDecision('', '', '', '', '', d.action, '', UI_ACTION))")])
    assert out.returncode != 0
