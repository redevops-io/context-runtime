"""An injected implementation earns its class: a floor by default, a contract to get under it.

Every integration tenant ships a local simulation of one tool — it builds a string and hands it back, so the
class it was reviewed at is ``read`` (``market_radar`` is the exception: a sweep fetches and stores, so it is
``write``). Each tenant also exposes a ``*_tool_factory`` hook that replaces that simulation with code this
repository has never seen. Two failures were possible there, and they are the same failure — the code asking
to run grading its own risk:

* **inherited** — pass a factory and the tool stayed registered at the shipped simulation's class, so a
  function that posts to a vendor ran on a principal holding only a read grant;
* **self-declared** — let the caller simply say ``read`` and nothing has been gained, only moved.

So an injected function sits at a floor of ``write``, and the only way below it is :func:`pure_read`, whose
argument has to be a sentence about why the function is inert — checked against the function's own source,
using the sink vocabulary the platform's FR-04 sweep uses. A visible ``post`` beats any declaration, and so
does a visible ``rmtree`` one class lower. The shipped simulations keep the ``read`` they were audited at:
this file exists to prove that ``draft_caption`` really is one.
"""
from __future__ import annotations

import importlib

import pytest

from agent_harness.tools import Authorization

from context_runtime.tools import pure_read
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
_INERT_REASON = "formats its arguments into a string; no I/O — reviewed 2026-11-22"


def _tenant(module: str, cls: str, **kwargs):
    return getattr(importlib.import_module(f"context_runtime.integrations.{module}"), cls)(
        local_trust=True, local_trust_reason=_REASON, **kwargs)


# ─────────── three injected implementations, one per kind of claim ───────────


def _inert(args: dict) -> ToolResult:
    """Builds a string out of its arguments. Nothing here leaves the process."""
    return ToolResult(ok=True, text=f"[{args.get('channel', 'channel')}] drafted locally")


class _VendorClient:
    def post(self, path: str, payload: dict) -> dict:
        return {"id": "remote-draft"}


def _vendor_post(args: dict) -> ToolResult:
    """What an integrator actually swaps in: the arguments go off to a remote service."""
    return ToolResult(ok=True, text=str(_VendorClient().post("/captions", args)))


class _Disk:
    def rmtree(self, path: str) -> None:
        return None


def _wipes(args: dict) -> ToolResult:
    """A tool named for tidiness. Nothing in the name says delete; the call does."""
    _Disk().rmtree(args.get("path", "/tmp/nothing"))
    return ToolResult(ok=True, text="removed")


def _read_only_plane(principal, spec, args):
    """A permissions plane that granted ``read`` and nothing else — the principal this file is about."""
    if spec.action != "read":
        return f"no '{spec.action}' grant for this principal"
    return Authorization.allow(grant_ids=("role:reader",), plane="permissions")


# ───────────────────────── the shipped implementations ─────────────────────────


@pytest.mark.parametrize("module,cls,factory,action_kw,tool,shipped", TENANTS, ids=_IDS)
def test_the_shipped_implementation_keeps_the_class_it_was_reviewed_at(module, cls, factory, action_kw, tool,
                                                                       shipped):
    """The pure built-ins are not made to defend themselves: ``draft_caption`` is a read, and stays one."""
    assert _tenant(module, cls).registry.get(tool).spec().action == shipped


def test_the_shipped_caption_is_actually_inert_not_just_labelled_inert():
    """A class is only worth having if the function behind it does what the class says."""
    social = importlib.import_module("context_runtime.integrations.social_autopilot")
    result = social._draft_caption({"goal": "g", "channel": "linkedin", "tone": "bold", "format": "post"})
    assert result.ok and result.data["channel"] == "linkedin"


# ─────────────────────────── the floor: no guessing ───────────────────────────


@pytest.mark.parametrize("module,cls,factory,action_kw,tool,shipped", TENANTS, ids=_IDS)
def test_a_replacement_without_a_class_of_its_own_refuses_to_start(module, cls, factory, action_kw, tool,
                                                                   shipped):
    with pytest.raises(RegistryConfigurationError) as exc:
        _tenant(module, cls, **{factory: _vendor_post})
    message = str(exc.value)
    assert action_kw in message and tool in message        # names the keyword and the tool it is about
    assert "pure_read" in message                          # and how to claim below the floor


@pytest.mark.parametrize("module,cls,factory,action_kw,tool,shipped", TENANTS, ids=_IDS)
def test_a_replacement_declared_at_or_above_the_floor_is_registered_as_declared(module, cls, factory, action_kw,
                                                                                tool, shipped):
    tenant = _tenant(module, cls, **{factory: _vendor_post, action_kw: "write"})
    assert tenant.registry.get(tool).spec().action == "write"


@pytest.mark.parametrize("module,cls,factory,action_kw,tool,shipped", TENANTS, ids=_IDS)
def test_a_class_outside_the_vocabulary_refuses_to_start(module, cls, factory, action_kw, tool, shipped):
    """A typo is not a risk class — and neither is a word invented for one tool."""
    with pytest.raises(RegistryConfigurationError):
        _tenant(module, cls, **{factory: _vendor_post, action_kw: "generate"})


# ───────────── intentional misclassification: the part that matters ─────────────


@pytest.mark.parametrize("module,cls,factory,action_kw,tool,shipped", TENANTS, ids=_IDS)
def test_an_injected_function_cannot_claim_read_by_saying_read(module, cls, factory, action_kw, tool, shipped):
    """The word is the downgrade. Below the floor has to be a deliberate claim, in its own type."""
    with pytest.raises(RegistryConfigurationError) as exc:
        _tenant(module, cls, **{factory: _inert, action_kw: "read"})
    assert "below the floor" in str(exc.value) and "pure_read" in str(exc.value)


@pytest.mark.parametrize("module,cls,factory,action_kw,tool,shipped", TENANTS, ids=_IDS)
def test_a_function_reaching_a_sink_cannot_be_registered_as_a_read(module, cls, factory, action_kw, tool,
                                                                   shipped):
    """A contract the source contradicts is refused, and the refusal names the sink."""
    with pytest.raises(RegistryConfigurationError) as exc:
        _tenant(module, cls, **{factory: _vendor_post, action_kw: pure_read(_INERT_REASON)})
    message = str(exc.value)
    assert "post" in message and "not a read" in message


def test_a_deleting_function_cannot_hide_one_class_below_delete():
    """The same under-declaration one rung lower: a ``rmtree`` registered as ``write`` still loses data."""
    with pytest.raises(RegistryConfigurationError) as exc:
        _tenant("social_autopilot", "SocialAutopilotTenant", caption_tool_factory=_wipes,
                caption_action="write")
    assert "rmtree" in str(exc.value) and "delete" in str(exc.value)

    strict = _tenant("social_autopilot", "SocialAutopilotTenant", caption_tool_factory=_wipes,
                     caption_action="delete")
    assert strict.registry.get("draft_caption").spec().action == "delete"


def test_a_function_with_no_readable_source_cannot_claim_read():
    """No source, no evidence, no waiver: an unreadable callable is assumed consequential."""
    with pytest.raises(RegistryConfigurationError) as exc:
        _tenant("social_autopilot", "SocialAutopilotTenant", caption_tool_factory=len,
                caption_action=pure_read("a built-in that reads nothing at all"))
    assert "source" in str(exc.value)


def test_a_verb_counts_as_a_sink_when_the_object_it_is_called_on_says_so():
    """``os.remove`` deletes a file; ``rows.remove(x)`` is a list. One of these is not a write.

    The ambiguous verbs — remove, rename, move — are sinks when the receiver names a filesystem and ordinary
    code otherwise. A rule that flags every ``.remove(`` is a rule that trains integrators to declare
    ``write`` without reading anything, which is the thing being defended against.
    """
    import importlib

    actions = importlib.import_module("context_runtime.tools.actions")

    def removes_a_file(args):
        import os
        os.remove(args.get("path", "/tmp/none"))
        return ToolResult(ok=True, text="gone")

    def removes_a_row(args):
        rows = list(args.get("rows", []))
        if rows:
            rows.remove(rows[0])
        return ToolResult(ok=True, text=str(rows))

    assert any("os.remove" in sink for sink in actions.write_sinks(removes_a_file))
    assert actions.write_sinks(removes_a_row) == ()

    with pytest.raises(RegistryConfigurationError):
        _tenant("social_autopilot", "SocialAutopilotTenant", caption_tool_factory=removes_a_file,
                caption_action=pure_read(_INERT_REASON))
    inert = _tenant("social_autopilot", "SocialAutopilotTenant", caption_tool_factory=removes_a_row,
                    caption_action=pure_read("pops the first row off a local list; no I/O — reviewed 2026-11-22"))
    assert inert.registry.get("draft_caption").spec().action == "read"


def test_the_capability_contract_refuses_a_reason_that_says_nothing():
    """The reason is the reviewable part. A flag wearing a sentence's clothes is what it replaced."""
    for empty in ("", "   ", "true", "yes", "1", "pure", "safe", "n/a"):
        with pytest.raises(RegistryConfigurationError):
            pure_read(empty)
    assert pure_read(_INERT_REASON).reason == _INERT_REASON


def test_a_declared_pure_read_contract_survives_construction():
    """The object that crosses the seam is the contract itself, not a string someone can typo."""
    from context_runtime.tools import PureRead

    assert PureRead is not None and pure_read(_INERT_REASON).reason == _INERT_REASON


# ─────────── and the decision the class is actually matched against ───────────


def test_an_inert_injected_function_contracted_down_to_read_runs_for_a_read_only_principal():
    tenant = _tenant("social_autopilot", "SocialAutopilotTenant", caption_tool_factory=_inert,
                     caption_action=pure_read(_INERT_REASON), authorizer=_read_only_plane)
    assert tenant.registry.get("draft_caption").spec().action == "read"
    assert tenant.registry.run("draft_caption", {"goal": "g", "channel": "linkedin"}).ok


def test_a_replaced_caption_no_longer_runs_on_a_read_grant():
    """The end of the story: the class is what a grant is matched against, so this is the denial that matters."""
    shipped = _tenant("social_autopilot", "SocialAutopilotTenant", authorizer=_read_only_plane)
    assert shipped.registry.run("draft_caption", {"goal": "g", "channel": "linkedin"}).ok

    live = _tenant("social_autopilot", "SocialAutopilotTenant", caption_tool_factory=_vendor_post,
                   caption_action="write", authorizer=_read_only_plane)
    denied = live.registry.run("draft_caption", {"goal": "g", "channel": "linkedin"})
    assert not denied.ok and "no 'write' grant" in (denied.error or "")
    assert live.registry.audit[-1]["allowed"] is False


def test_an_unclassified_replacement_matches_no_grant_and_is_denied_everywhere():
    """What a registration that goes around the seam still cannot do: an empty class matches no grant."""
    from context_runtime.tools.base import ToolRegistry, function_tool

    registry = ToolRegistry(authorizer=_read_only_plane)
    registry.register(function_tool("unclassified_thing", lambda a: ToolResult(ok=True, text="ran"), action=""))
    assert not registry.run("unclassified_thing", {}).ok