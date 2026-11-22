"""The tool-access authorization seam, pinned the way agent-harness C1/C2 define it.

The contract this file exists to defend is the two directions the old seam got wrong:

* **no authorizer** used to mean "pass everything through"; it now means "nothing granted this", and the
  registration says so at start-up rather than in front of a user;
* **an authorizer returning ``None``** used to mean "allow"; a decision is now required, and the absence of
  one is a denial (the old ``Optional[str]`` shapes still work, but arrive tagged ``legacy=True`` so a
  reviewer can find the call sites still on the pre-``authz/v1`` contract).

Each test also reads the risk class the call is asked about, because the resource/action a tenant is checked
against is exactly what the per-tenant classification in ``integrations/`` decides.
"""
from __future__ import annotations

import pytest

from context_runtime.tools.base import (
    Authorization, PolicyReason, RegistryConfigurationError, ToolClassificationError, ToolRegistry, ToolResult,
    function_tool, set_default_authorizer,
)


def _reg(authorizer=None, *, action="read") -> ToolRegistry:
    r = ToolRegistry(authorizer=authorizer)
    r.register(function_tool("read_vulns", lambda a: ToolResult(ok=True, text="rows"), action=action))
    return r


# -- no authorizer installed ────────────────────────────────────────────────────────────────────────────

def test_missing_authorizer_denies_rather_than_passing_through():
    reg = _reg()
    res = reg.run("read_vulns", {})
    assert not res.ok                                              # was: ok, the pass-through C1 removed
    assert "missing_explicit_grant" in (res.error or "")
    denial = reg.audit[-1]
    assert denial["allowed"] is False and denial["policy_reason"] == PolicyReason.MISSING_EXPLICIT_GRANT


def test_missing_authorizer_is_a_start_up_problem_not_a_runtime_surprise():
    reg = _reg()
    report = reg.validate()                                        # a boot check can read this...
    assert not report.ok and report.authorizer == "none"
    assert any("authorizer" in p for p in report.problems)
    with pytest.raises(RegistryConfigurationError):                # ...or a constructor can just fail
        reg.validate(strict=True)


def test_local_trusted_authorizer_runs_and_says_who_authorized_it(local_run):
    reg = _reg()                                                   # inherits the declared local authority
    assert reg.run("read_vulns", {}).ok
    allow = reg.audit[-1]                                          # an unattributed allow stays visible
    assert allow["allowed"] is True and allow["plane"] == local_run
    assert allow["policy_reason"] == PolicyReason.WILDCARD_GRANT
    assert reg.unattributed_allows()                               # a wildcard is reported as one


# -- the decision an authorizer gives ───────────────────────────────────────────────────────────────────

def test_authorizer_denies_by_principal():
    def authz(principal, spec, args):
        roles = (principal or {}).get("roles", set())
        if spec.name == "read_vulns" and "security" not in roles:
            return "not permitted"                                 # the legacy shape still denies
        return Authorization.allow(grant_ids=("role:security",), plane="permissions")

    reg = _reg(authz)
    assert reg.run("read_vulns", {}, principal={"roles": {"security"}}).ok
    # An ordinary named grant is not audit noise, so it is not in the trail — but it is attributable:
    decision = reg.authorize(reg.get("read_vulns").spec(), {}, principal={"roles": {"security"}})
    assert decision.grant_ids == ("role:security",) and decision.plane == "permissions"
    denied = reg.run("read_vulns", {}, principal={"roles": {"guest"}})
    assert not denied.ok and "not permitted" in denied.error
    assert reg.audit[-1]["allowed"] is False                       # and every denial is audited


def test_an_authorizer_that_returns_nothing_denies():
    """The bug this change is about: an authorizer with a missing branch used to allow everything."""
    reg = _reg(lambda principal, spec, args: None)
    res = reg.run("read_vulns", {})
    assert not res.ok and "no decision" in (res.error or "")
    assert reg.audit[-1]["legacy"] is True                         # the old shape stays visible, never silent


def test_a_raising_authorizer_is_a_denial_not_a_crash():
    def down(principal, spec, args):
        raise ConnectionError("policy plane unreachable")

    res = _reg(down).run("read_vulns", {})
    assert not res.ok and "policy plane unreachable" in (res.error or "")


def test_default_authorizer_governs_all_registries_then_denies_again():
    reg = _reg()
    set_default_authorizer(lambda p, spec, args: "blocked fleet-wide")
    try:
        assert not reg.run("read_vulns", {}).ok                    # a registry with no authorizer inherits the plane
    finally:
        set_default_authorizer(None)
    # Clearing the plane is not a return to pass-through: with nothing installed, nothing is granted.
    assert not reg.run("read_vulns", {}).ok


def test_a_registrys_own_authorizer_wins_over_the_installed_plane(local_run):
    """The per-registry door stays the first one: a tenant that names its authority is not overridden by a
    process-wide local-trust default."""
    reg = _reg(lambda p, spec, args: "denied by this tenant's own plane")
    assert not reg.run("read_vulns", {}).ok
    assert "this tenant's own plane" in (reg.run("read_vulns", {}).error or "")


# -- the risk class a call is asked about ───────────────────────────────────────────────────────────────

def test_the_capability_triple_comes_from_the_registry_that_owns_the_tool():
    assert _reg().required_capability("read_vulns") == {
        "resource_kind": "tool", "resource_name": "read_vulns", "requested_action": "read"}


def test_an_unclassified_tool_asks_for_nothing_and_therefore_matches_no_grant():
    reg = _reg(action="")
    assert reg.required_capability("read_vulns")["requested_action"] == ""
    assert reg.unclassified() == ["read_vulns"]
    assert any("no risk class" in w for w in reg.validate().warnings)   # reported, not swallowed


def test_a_read_class_on_a_side_effecting_tool_never_reaches_the_call_path():
    with pytest.raises(ToolClassificationError):
        ToolRegistry().register(function_tool("nuke", lambda a: ToolResult(ok=True),
                                              side_effecting=True, action="read"))


def test_re_registering_a_tool_cannot_lower_its_risk_class():
    reg = _reg(action="write")
    with pytest.raises(ToolClassificationError):
        reg.register(function_tool("read_vulns", lambda a: ToolResult(ok=True), action="read"))