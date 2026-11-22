"""The five refusals, asserted as refusals.

Every test in this file asks for something the design is supposed to say no to, and fails if anything runs.
They are grouped by the shape of the attack rather than by the module that implements the check, because the
point of the seam is that no single layer is the gate: a wrapper that lies about what it calls still meets
the plane, a caller who skips the wrapper meets the registry, and a service whose plane disappears meets its
own start-up rule.

Two decisions from the owner of the stack set the posture these tests hold:

* **delegation is refused by default** — permitted only when it is explicitly authorized, non-escalating and
  carries provenance a reviewer can check. There is no implicit floor: no class is assigned to foreign code
  because nobody felt like choosing one.
* **a privileged service with no authorization plane does not run** — readiness and start-up fail. A separate
  diagnostic instance may boot precisely because privileged execution is not available inside it.
"""
from __future__ import annotations

import pytest

from agent_harness.tools import Authorization, Decision, ToolRegistry, function_tool

from context_runtime.integrations.agent_console import AgentConsole
from context_runtime.integrations.modules import CATALOG, ModuleTenant
from context_runtime.tools import (
    Delegated,
    PrivilegedExecutionUnavailable,
    delegated,
    execution_mode,
    pure_read,
    require_plane,
    resolve_action,
    resolve_authorizer,
)
from context_runtime.tools.authz import EXECUTION_MODE_ENV
from context_runtime.tools.base import (RegistryConfigurationError, ToolClassificationError, ToolResult,
                                        set_default_authorizer)

_AUTH = dict(authority="grant agentic-os:fleet/sec-41 block_ip (ticket SEC-41)",
             via="sentinel_client.Client.block_ip, an HTTPS PUT to the Sentinel API",
             review="read the client on 2026-11-22: it writes to one vendor API and holds nothing locally")
_INERT_CLAIM = "formats its arguments into a string; no I/O — reviewed 2026-11-22"


class _Vendor:
    """A client object built outside the injected callable — the effect is over there, not in the source."""

    def dispatch(self, payload: dict) -> dict:
        return {"ok": True}


def _wrapper(args: dict) -> ToolResult:
    """A plugin wrapper with nothing to see: no sink in its own body, everything delegated away."""
    return ToolResult(ok=True, text=str(_Vendor().dispatch(args)))


def _reader(principal, spec, args):
    """A plane that granted ``read`` and nothing else."""
    if spec.action != "read":
        return f"no '{spec.action}' grant for this principal"
    return Authorization.allow(grant_ids=("role:reader",), plane="permissions")


# ── 1. a plugin wrapper attempting privilege escalation ──────────────────────────────────────


def test_a_wrapper_cannot_register_itself_as_a_read():
    """The escalation, in every form it arrives in.

    A wrapper's own body is clean by construction — the work happens in whatever it calls — so the two
    things that could grant it a ``read`` are a claim about the source and a word in a keyword argument.
    Neither is accepted: the claim is refused because it cannot be checked, and the word is refused because
    a class with no authority behind it is a self-grant with better manners.
    """
    with pytest.raises(RegistryConfigurationError, match="_Vendor"):
        resolve_action(shipped="read", injected=_wrapper, declared=pure_read(_INERT_CLAIM),
                       tool="draft_caption")
    for bare in ("read", "write", "delete"):
        with pytest.raises(RegistryConfigurationError, match="no authority"):
            resolve_action(shipped="read", injected=_wrapper, declared=bare, tool="draft_caption")


def test_an_authorized_wrapper_still_cannot_run_on_a_readers_authority():
    """Registering at ``write`` is not what stops the call — the class is only what the call is checked
    against. This is the assertion that classification and authorization are two layers, and that passing
    cleanly through the first does not buy a way past the second.
    """
    tenant = ModuleTenant(CATALOG["edge_sentinel"], authorizer=_reader)
    denied = tenant.registry.run("edge_sentinel_block_ip", {"ip": "1.2.3.4"})
    assert denied.ok is False and "no 'write' grant" in (denied.error or "")
    last = tenant.registry.audit[-1]
    assert last["allowed"] is False


def test_reregistering_a_tool_at_a_lower_class_is_refused():
    """Hot reload, plugin overwrite, second builder on the same registry: none may lower a class."""
    registry = ToolRegistry(authorizer=_reader)
    registry.register(function_tool("wipe", lambda a: ToolResult(ok=True), action="delete"))
    with pytest.raises(ToolClassificationError, match="downgrading the previous classification"):
        registry.register(function_tool("wipe", lambda a: ToolResult(ok=True), action="read"))
    with pytest.raises(ToolClassificationError, match="re-registered without a risk class"):
        registry.register(function_tool("wipe", lambda a: ToolResult(ok=True)))


def test_a_spec_that_admits_side_effects_cannot_claim_read():
    registry = ToolRegistry(authorizer=_reader)
    with pytest.raises(ToolClassificationError, match="classified 'read'"):
        registry.register(function_tool("mutates", lambda a: ToolResult(ok=True), side_effecting=True,
                                        action="read"))


# ── 2. delegation without provenance ─────────────────────────────────────────────────────────


def test_delegation_without_provenance_never_reaches_the_registry():
    """Nothing is registered on a guess: every unattributed form is refused before a tool exists."""
    with pytest.raises(RegistryConfigurationError, match="who authorized it"):
        resolve_action(shipped="read", injected=_wrapper, tool="draft_caption")
    for thin in ({**_AUTH, "authority": ""}, {**_AUTH, "via": "client"}, {**_AUTH, "review": "checked"}):
        with pytest.raises(RegistryConfigurationError):
            delegated(action="write", **thin)
    with pytest.raises(RegistryConfigurationError, match="prose"):
        delegated(action="read", **_AUTH)


def test_a_hand_built_authorization_record_is_checked_like_a_built_one():
    """Constructing the dataclass is not a way around the builder — the sentences are the authorization."""
    with pytest.raises(RegistryConfigurationError, match="authority"):
        resolve_action(shipped="read", injected=_wrapper,
                       declared=Delegated(action="write", authority="", via="", review=""),
                       tool="draft_caption")


def test_an_authorization_below_the_strongest_class_is_refused_wherever_it_is_made():
    """``delegated()`` refuses a ``read`` delegation, and the resolver refuses one that arrived anyway."""
    with pytest.raises(RegistryConfigurationError, match="prose"):
        delegated(action="read", **_AUTH)
    with pytest.raises(RegistryConfigurationError, match="never authorized below"):
        resolve_action(shipped="write", injected=_wrapper,
                       declared=Delegated(action="read", **_AUTH), tool="draft_caption")


# ── 3. a missing or incompatible authorization plane ─────────────────────────────────────────


def test_a_privileged_service_refuses_to_start_without_a_plane():
    """Three layers, one answer.

    ``require_plane`` is what a service calls; ``ModuleTenant`` is what a fleet builds; the harness
    ``validate(strict=True)`` is what a console runs. None of them is allowed to hand back something that
    serves and denies, because that state looks healthy.
    """
    with pytest.raises(RegistryConfigurationError, match="refusing to start"):
        require_plane(context="the fleet")
    with pytest.raises(RegistryConfigurationError, match="no authorizer is configured"):
        ModuleTenant(CATALOG["edge_sentinel"])
    with pytest.raises(RegistryConfigurationError, match="deny_later. is refused"):
        resolve_authorizer(on_missing="deny_later", context="a registry that can run tools")
    with pytest.raises(RegistryConfigurationError, match="not one of"):
        resolve_authorizer(on_missing="later_please", context="a caller who guessed the keyword")


def test_an_incompatible_plane_is_a_denial_not_a_pass_through():
    """The old contract, met by the new engine: silence, an exception, and nonsense all deny.

    ``None`` used to mean *allowed* on this seam — which is how a plane that answers in its own vocabulary
    ended up authorizing everything it never answered about. It is in the vocabulary of denials now.
    """
    def silent(principal, spec, args):
        return None                                        # the pre-authz/v1 shape for "allowed"

    def exploding(principal, spec, args):
        raise ConnectionError("the policy plane is unreachable")

    def nonsense(principal, spec, args):
        return {"decision": "maybe"}

    for plane, reason in ((silent, "missing_explicit_grant"), (exploding, "no_resource_plane"),
                          (nonsense, "malformed_request")):
        registry = ToolRegistry(authorizer=plane)
        registry.register(function_tool("any_tool", lambda a: ToolResult(ok=True, text="ran"), action="read"))
        result = registry.run("any_tool", {})
        assert result.ok is False, reason
        assert registry.audit[-1]["allowed"] is False
        assert registry.audit[-1]["policy_reason"] == reason


def test_a_plane_that_allows_by_returning_true_is_audited_as_legacy():
    """Still honoured — the coercion exists so old callers fail visibly, not silently."""
    registry = ToolRegistry(authorizer=lambda p, s, a: True)
    registry.register(function_tool("any_tool", lambda a: ToolResult(ok=True, text="ran"), action="read"))
    assert registry.run("any_tool", {}).ok
    assert registry.audit[-1]["legacy"] is True
    assert [a["tool"] for a in registry.unattributed_allows()] == ["any_tool"]


# ── 4. calling the engine directly, past the wrappers ────────────────────────────────────────


def test_the_wrapper_is_not_the_gate_direct_execution_meets_the_same_plane():
    """Skipping the console, the fleet and the tenant still does not skip the registry.

    The wrappers add policy that the registry does not know about; they do not add the gate. A caller with
    the registry object in hand gets the same answer as a caller who came in through the UI, which is what
    makes "we route everything through the console" a statement about tidiness rather than security.
    """
    tenant = ModuleTenant(CATALOG["edge_sentinel"], authorizer=_reader)
    direct = tenant.registry.run("edge_sentinel_block_ip", {"ip": "1.2.3.4"})
    through_act = tenant.act("block_ip", ip="1.2.3.4")
    assert direct.ok is False and through_act.ok is False
    assert direct.error == through_act.error               # same reason, same depth of door
    assert [a["allowed"] for a in tenant.registry.audit] == [False, False]


def test_a_diagnostic_surface_cannot_be_asked_to_execute():
    """An authorizer that would say yes to everything is installed, and the call is refused anyway.

    In a diagnostic instance execution is not a policy question with a negative answer — the path is not
    built. Nothing a later grant, a wrapper or a caller can do reaches a tool from here.
    """
    set_default_authorizer(lambda p, s, a: True)
    try:
        tenant = ModuleTenant(CATALOG["edge_sentinel"], privileged=False)
        with pytest.raises(PrivilegedExecutionUnavailable):
            tenant.act("block_ip", ip="1.2.3.4")

        console = AgentConsole("Ops", "A console that reads.", tools=[
            function_tool("read_something", lambda a: ToolResult(ok=True, text="ok"), action="read")],
            privileged=False)
        assert console.registry.list() == []               # the tools were never registered
        with pytest.raises(PrivilegedExecutionUnavailable):
            console._answer_tool("read_something", {})
        with pytest.raises(PrivilegedExecutionUnavailable):
            console.mount_mcp(object())
    finally:
        set_default_authorizer(None)


# ── 5. execution after the service has already started ───────────────────────────────────────


def test_removing_the_plane_after_start_up_leaves_nothing_authorized():
    """A grant that walked out of the room does not stay in the room.

    The registry reads the plane at call time, so the lifetime of an authorization is the lifetime of the
    decision, not of the object it was asked through. The reverse half — a plane installed after the tenant
    was built governs it — is what lets an application import a console before its gate exists.
    """
    tenant = ModuleTenant(CATALOG["edge_sentinel"], privileged=False, authorizer_on_missing="deny_later")
    registry = tenant.registry
    registry.register(function_tool("later_tool", lambda a: ToolResult(ok=True, text="ran"), action="write"))

    def deny_everything(principal, spec, args):
        return Authorization(Decision.DENY, reason="revoked", detail="the grant was withdrawn")

    set_default_authorizer(deny_everything)
    try:
        assert registry.run("later_tool", {}).ok is False
        assert registry.audit[-1]["policy_reason"] == "revoked"
    finally:
        set_default_authorizer(None)

    denied = registry.run("later_tool", {})                # nothing cached from when a plane was there
    assert denied.ok is False and "no authorizer is installed" in (denied.error or "")


def test_a_tool_registered_before_the_plane_arrives_is_governed_by_it_afterwards():
    """The ordering an embedding application depends on: build first, gate second, govern from then on."""
    registry = ToolRegistry()                                # authorizer read at call time
    registry.register(function_tool("later_tool", lambda a: ToolResult(ok=True, text="ran"), action="write"))
    assert registry.run("later_tool", {}).ok is False        # not yet: nothing has answered for it

    set_default_authorizer(_reader)
    try:
        assert registry.run("later_tool", {}).ok is False    # a plane now says no: write asked, read held
        # The plane answered in the pre-authz/v1 shape (a string means deny), which the engine honours and
        # marks. What matters here is that the plane was consulted at all, after the registry was built.
        assert registry.audit[-1]["policy_reason"] == "missing_explicit_grant"
        assert registry.audit[-1]["legacy"] is True
    finally:
        set_default_authorizer(None)


def test_the_posture_of_a_process_is_not_decided_by_an_unset_or_typoed_variable(monkeypatch):
    """The mode is read where it is used, and an unset variable lands on the strict side.

    No mode means "this process runs tools", which is the assumption that has to be paid for at start-up;
    a typo must not fall through to the permissive branch, because the permissive branch is the one that
    ships.
    """
    monkeypatch.delenv(EXECUTION_MODE_ENV, raising=False)
    assert execution_mode() == ("execute", "")
    monkeypatch.setenv(EXECUTION_MODE_ENV, "diagnostc")
    with pytest.raises(RegistryConfigurationError, match="is not a mode"):
        execution_mode()
    monkeypatch.setenv(EXECUTION_MODE_ENV, "read-only")
    with pytest.raises(RegistryConfigurationError, match="CR_DIAGNOSTIC_REASON"):
        execution_mode()