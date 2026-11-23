"""A refused console call answers with its reason. Not with a 500.

``AgentConsole`` reads the failed result the registry hands back and has to tell two kinds of "no" apart: the
AUTHORIZATION gate refusing (no grant — pressing confirm changes nothing) and the APPROVAL gate holding a
side effect for confirmation. It asked the result for a ``denied`` flag that :mod:`agent_harness.tools` has
never set, so every authorization denial on a console tool raised ``AttributeError`` inside ``respond`` and the
app answered 500 — the one path where a refused user most needs a reason. The flag is honoured when a result
carries one; otherwise the registry's audit trail decides, because that is where the two gates sign their names.
"""
from __future__ import annotations

from types import SimpleNamespace

from agent_harness.tools import ApprovalPolicy, Authorization

from context_runtime.integrations.agent_console import AgentConsole
from context_runtime.tools.base import ToolResult, function_tool, local_trusted_authorizer


def _who(user="alice", app="growth-engine"):
    return SimpleNamespace(user=user, app=app, roles=frozenset())


def _deny_writes(principal, spec, args):
    """An authorizer in the engine's own words. It says YES out loud: returning nothing is silence, and
    silence is a denial (``missing_explicit_grant``), not permission — so the read needs a real decision."""
    if spec.name == "traffic_summary":
        return Authorization.allow(grant_ids=("local:read",), plane="permissions")
    return "no 'write' grant for this principal"


def _console(authorizer, **kw):
    return AgentConsole(
        "Growth Engine", "traffic and campaigns", app="growth-engine",
        tools=[function_tool("traffic_summary", lambda a: ToolResult(ok=True, text="12k visits"),
                             description="traffic summary", action="read"),
               function_tool("reallocate_budget", lambda a: ToolResult(ok=True, text="moved"),
                             description="move budget", side_effecting=True, action="write")],
        authorizer=authorizer, **kw)                 # no model: the routing falls back to the keyword path


def test_a_denied_action_reports_its_reason_instead_of_raising():
    answer = _console(_deny_writes).respond("reallocate the budget to linkedin", principal=_who())
    assert answer["intent"] == "action" and answer["approved"] is False
    assert "no 'write' grant" in answer["text"]
    assert "needs confirmation" not in answer["text"]          # refused, not held: confirming changes nothing


def test_an_allowed_read_reaches_the_tool():
    """The positive control: the same console, the same authorizer, a tool it does grant."""
    answer = _console(_deny_writes).respond("give me the traffic summary", principal=_who())
    assert answer["intent"] == "action" and answer["tool"] == "traffic_summary"
    assert answer["approved"] is True and "12k visits" in answer["text"]   # authorized: the tool ran


def test_a_held_side_effect_still_says_confirmation_not_refusal():
    """The other gate, the other sentence: authorized, but the approval policy has not been satisfied."""
    console = _console(local_trusted_authorizer())
    console.registry.policy = ApprovalPolicy("deny_side_effects", approver=lambda call: False)
    answer = console.respond("reallocate the budget to linkedin", principal=_who())
    assert answer["approved"] is False                                      # held, not run
    assert "[no_grant]" not in answer["text"] and "grant" not in answer["text"].lower()


def test_a_result_that_carries_the_flag_is_believed():
    """When the engine does provide the flag, the audit trail is not consulted — the result speaks for itself."""
    console = _console(_deny_writes)
    assert console._denied_by_authorization("traffic_summary",
                                            SimpleNamespace(error="denied: whatever", denied=False)) is False
    assert console._denied_by_authorization("traffic_summary",
                                            SimpleNamespace(error="denied: whatever", denied=True)) is True


def test_a_tool_that_crashed_is_not_reported_as_a_refusal():
    """No authorization entry behind the failure means it never reached the gate: not a denial."""
    console = _console(_deny_writes)
    assert console._denied_by_authorization("traffic_summary", SimpleNamespace(error="RuntimeError: boom")) is False
    assert console._denied_by_authorization("nope", SimpleNamespace(error="denied: not registered")) is False