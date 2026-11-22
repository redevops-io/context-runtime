"""A replaced implementation carries its own risk class, or the tenant refuses to start.

Every integration tenant here ships a local simulation of one tool — it builds a string and hands it back,
so the class it was reviewed at is ``read`` (``market_radar`` is the exception: a sweep fetches and stores,
so it is ``write``). Each tenant also exposes a ``*_tool_factory`` hook that replaces that simulation with
code this repository has never seen. Until now the class did not move with the code: pass a factory that
posts to a vendor and the tool stayed registered as ``read``, which means a principal holding only a read
grant could run it. That is the downgrade this file exists to keep closed.

The rule being pinned is the one ``resolve_action`` implements: the class describes the effect of the
function that actually runs. Nothing injected, the reviewed class stands. Something injected, whoever read
their own function says what it is — and construction refuses instead of inheriting.
"""
from __future__ import annotations

import importlib

import pytest

from agent_harness.tools import Authorization

from context_runtime.tools.base import RegistryConfigurationError, ToolResult

#: module, tenant class, factory hook, the class-declaring keyword, registered tool, reviewed class.
TENANTS = [
    ("control_tower", "ControlTowerTenant", "refresh_tool_factory", "refresh_action", "refresh_metabase", "read"),
    ("agentic_compliance", "AgenticComplianceTenant", "pull_tool_factory", "pull_action", "pull_evidence", "read"),
    ("agentic_books", "AgenticBooksTenant", "pull_tool_factory", "pull_action", "pull_reports", "read"),
    ("agentic_support", "AgenticSupportTenant", "retrieve_tool_factory", "retrieve_action", "retrieve_context",
     "read"),
    ("agentic_billing", "AgenticBillingTenant", "fetch_tool_factory", "fetch_action", "fetch_signals", "read"),
    ("growth_engine", "GrowthEngineTenant", "report_tool_factory", "report_action", "draft_attribution_report",
     "read"),
    ("market_radar", "MarketRadarTenant", "scrape_tool_factory", "scrape_action", "perform_sweep", "write"),
    ("outreach_engine", "OutreachEngineTenant", "teardown_tool_factory", "teardown_action", "draft_teardown",
     "read"),
    ("social_autopilot", "SocialAutopilotTenant", "caption_tool_factory", "caption_action", "draft_caption",
     "read"),
]
_IDS = [f"{mod}.{tool}" for mod, _, _, _, tool, _ in TENANTS]

_REASON = "offline audit of the injected-factory seam"


def _tenant(module: str, cls: str, **kwargs):
    return getattr(importlib.import_module(f"context_runtime.integrations.{module}"), cls)(
        local_trust=True, local_trust_reason=_REASON, **kwargs)


def _vendor_call(args: dict) -> ToolResult:
    """Stand-in for the thing an integrator actually swaps in: it leaves the process."""
    return ToolResult(ok=True, text="posted to a vendor")


def _read_only_plane(principal, spec, args):
    """A permissions plane that granted ``read`` and nothing else — the principal this test cares about."""
    if spec.action != "read":
        return f"no '{spec.action}' grant for this principal"
    return Authorization.allow(grant_ids=("role:reader",), plane="permissions")


@pytest.mark.parametrize("module,cls,factory,action_kw,tool,shipped", TENANTS, ids=_IDS)
def test_the_shipped_implementation_keeps_the_class_it_was_reviewed_at(module, cls, factory, action_kw, tool,
                                                                       shipped):
    tenant = _tenant(module, cls)
    assert tenant.registry.get(tool).spec().action == shipped


@pytest.mark.parametrize("module,cls,factory,action_kw,tool,shipped", TENANTS, ids=_IDS)
def test_a_replacement_without_a_class_of_its_own_refuses_to_start(module, cls, factory, action_kw, tool,
                                                                   shipped):
    """No silent inheritance: the integrator that swapped the function is the one who has to grade it."""
    with pytest.raises(RegistryConfigurationError) as exc:
        _tenant(module, cls, **{factory: _vendor_call})
    message = str(exc.value)
    assert action_kw in message                      # it names the parameter that answers the question
    assert tool in message                           # and the tool the answer is about


@pytest.mark.parametrize("module,cls,factory,action_kw,tool,shipped", TENANTS, ids=_IDS)
def test_a_replacement_declares_its_own_class(module, cls, factory, action_kw, tool, shipped):
    tenant = _tenant(module, cls, **{factory: _vendor_call, action_kw: "write"})
    spec = tenant.registry.get(tool).spec()
    assert spec.action == "write"


@pytest.mark.parametrize("module,cls,factory,action_kw,tool,shipped", TENANTS, ids=_IDS)
def test_a_class_outside_the_vocabulary_refuses_to_start(module, cls, factory, action_kw, tool, shipped):
    """A typo is not a risk class — and neither is a word invented for one tool."""
    with pytest.raises(RegistryConfigurationError):
        _tenant(module, cls, **{factory: _vendor_call, action_kw: "generate"})


def test_a_replaced_caption_no_longer_runs_on_a_read_grant():
    """The end of the story: the class is what a grant is matched against, so this is the denial that matters.

    ``draft_caption`` simulated is a read and runs for a read-only principal. The same tool with a real
    generator behind it, declared as such, does not — while an integrator who genuinely replaced it with
    another pure function can still declare ``read`` and keep working.
    """
    shipped = _tenant("social_autopilot", "SocialAutopilotTenant")
    assert shipped.registry.run("draft_caption", {"goal": "g", "channel": "linkedin"}).ok
    # ^ that tenant ran on the declared local authority; the two below name the plane, the way a
    # deployment does, because the grant is the thing under test.

    live = _tenant("social_autopilot", "SocialAutopilotTenant", caption_tool_factory=_vendor_call,
                   caption_action="write", authorizer=_read_only_plane)
    denied = live.registry.run("draft_caption", {"goal": "g", "channel": "linkedin"})
    assert not denied.ok and "no 'write' grant" in (denied.error or "")
    assert live.registry.audit[-1]["allowed"] is False

    # and the shipped read-only simulation still runs for that same read-only principal
    assert _tenant("social_autopilot", "SocialAutopilotTenant",
                   authorizer=_read_only_plane).registry.run(
        "draft_caption", {"goal": "g", "channel": "linkedin"}).ok

    pure = _tenant("social_autopilot", "SocialAutopilotTenant", caption_tool_factory=_vendor_call,
                   caption_action="read", authorizer=_read_only_plane)
    assert pure.registry.run("draft_caption", {"goal": "g", "channel": "linkedin"}).ok