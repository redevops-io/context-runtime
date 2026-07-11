"""The tool-access authorization seam — an optional authorizer gates every tool call by principal,
before the side-effect approval gate. The enterprise permissions plane plugs in here (open-core)."""
from __future__ import annotations

from context_runtime.tools.base import ToolRegistry, ToolResult, function_tool, set_default_authorizer


def _reg(authorizer=None) -> ToolRegistry:
    r = ToolRegistry(authorizer=authorizer)
    r.register(function_tool("read_vulns", lambda a: ToolResult(ok=True, text="rows")))
    return r


def test_no_authorizer_is_unchanged_behavior():
    assert _reg().run("read_vulns", {}).ok


def test_authorizer_denies_by_principal():
    # deny 'read_vulns' unless the principal carries the 'security' role
    def authz(principal, spec, args):
        roles = (principal or {}).get("roles", set())
        return None if spec.name != "read_vulns" or "security" in roles else "not permitted"

    reg = _reg(authz)
    assert reg.run("read_vulns", {}, principal={"roles": {"security"}}).ok
    denied = reg.run("read_vulns", {}, principal={"roles": {"guest"}})
    assert not denied.ok and "not permitted" in denied.error
    assert reg.audit[-1]["allowed"] is False       # the denial is audited


def test_default_authorizer_governs_all_registries_then_clears():
    reg = _reg()
    set_default_authorizer(lambda p, spec, args: "blocked fleet-wide")
    try:
        assert not reg.run("read_vulns", {}).ok    # a registry with no authorizer inherits the default
    finally:
        set_default_authorizer(None)
    assert reg.run("read_vulns", {}).ok
