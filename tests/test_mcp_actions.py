"""MCP tools are mounted with a class the operator declared, never one the server claimed.

An MCP server is code pulled from somewhere else, and ``readOnlyHint`` is its own announcement about itself.
The registry's rule is that the caller declares what it is about to run: the hint may relax the *approval*
gate — an announced read-only tool need not be confirmed — and it may never relax the authorization class,
which is what decides whether the call happens at all.

The doubles here return a typed :class:`~context_runtime.tools.base.Authorization`, and that is deliberate:
under ``authz/v1`` an authorizer that returns ``None`` is a *denial*, so the instinctive ``lambda *a: None``
is not an allow-all double, it is a double that denies everything while looking like one that allows it.
"""
from __future__ import annotations

import pytest

from context_runtime.integrations.agent_console import AgentConsole
from context_runtime.tools.authz import RegistryConfigurationError
from context_runtime.tools.base import ApprovalPolicy, Authorization, PolicyReason, ToolRegistry
from context_runtime.tools.mcp import mount_mcp


class FakeMCPClient:
    """The surface mount_mcp touches: an initialized client that can list and call tools."""

    def __init__(self, tools):
        self._initialized = True
        self._tools = tools
        self.calls = []

    def initialize(self):  # pragma: no cover - mount_mcp only calls it when uninitialized
        self._initialized = True

    def list_tools(self):
        return list(self._tools)

    def call_tool(self, name, arguments=None):
        self.calls.append((name, arguments))
        return {"content": [{"type": "text", "text": f"{name} ran"}], "isError": False}


def _tool(name, **annotations):
    desc = {"name": name, "description": f"the {name} tool",
            "inputSchema": {"type": "object", "properties": {}}}
    if annotations:
        desc["annotations"] = annotations
    return desc


def _grant(principal, spec, args):
    """An authorizer that says yes and names the grant standing behind the yes."""
    return Authorization.allow(reason=PolicyReason.EXPLICIT_GRANT, grant_ids=("grant-test",),
                               plane="test-plane")


def _spy():
    """An attributing allow that also records what it was asked about."""
    seen = []

    def spy(principal, spec, args):
        seen.append((spec.name, spec.action, spec.side_effecting))
        return _grant(principal, spec, args)
    return spy, seen


def _runnable(*names, authorizer):
    """A registry whose approval gate has been told these tools may run — so a refusal in these tests can
    only have come from the authorization layer, which is the layer under test. The approval list is named
    per tool rather than bypassed: bypass would hide a tool that never reached the gate at all."""
    return ToolRegistry(policy=ApprovalPolicy("allowlist", allow=list(names)), authorizer=authorizer)


def test_mounted_mcp_tool_defaults_to_write():
    """No class declared means the strictest class every capability must clear — not ``read`` because the
    server said it was one, and not a free pass from having no class at all."""
    spy, seen = _spy()
    reg = _runnable("convert_video", authorizer=spy)
    mount_mcp(reg, FakeMCPClient([_tool("convert_video")]))
    assert reg.run("convert_video", {}).ok
    assert seen == [("convert_video", "write", True)]


def test_read_only_hint_relaxes_approval_but_never_the_class():
    """The hint is an announcement from the code being run: it may stop the call needing confirmation,
    and it stops there."""
    spy, seen = _spy()
    reg = ToolRegistry(authorizer=spy)
    mount_mcp(reg, FakeMCPClient([_tool("search", readOnlyHint=True)]))
    assert reg.run("search", {}).ok
    assert seen == [("search", "write", False)]  # not confirmed, still authorized as a write


def test_declared_action_is_the_one_asked_about():
    reg = ToolRegistry(authorizer=_grant)
    mount_mcp(reg, FakeMCPClient([_tool("convert_video"), _tool("list_jobs")]),
              actions={"convert_video": "read", "list_jobs": "write"})
    assert [reg.get(n).spec().action for n in ("convert_video", "list_jobs")] == ["read", "write"]
    # a tool the operator classed read cannot also claim to mutate: the spec is made consistent with the
    # class, rather than the class being quietly raised to match a flag
    assert reg.get("convert_video").spec().side_effecting is False


def test_default_action_is_honoured_for_tools_left_out_of_the_map():
    """Mounting with ``default_action="delete"`` is how an operator says "I have not read this server,
    assume the worst"; the grant the plane needs then has to be the one that matches — so the question put
    to the plane has to carry ``delete``, not the tool's own opinion of itself."""
    spy, seen = _spy()
    reg = _runnable("wipe", authorizer=spy)
    mount_mcp(reg, FakeMCPClient([_tool("wipe")]), default_action="delete")
    assert reg.run("wipe", {}).ok
    assert seen == [("wipe", "delete", True)]
    assert reg.get("wipe").spec().action == "delete"


def test_both_keyings_are_honoured_and_the_stricter_one_wins():
    """A map may be keyed by the server's own name or by the name the mount registers it under. Spelling
    cannot be allowed to decide how much damage is authorised, so disagreement resolves to the stricter."""
    reg = ToolRegistry(authorizer=_grant)
    mount_mcp(reg, FakeMCPClient([_tool("convert_video")]), prefix="ffmpeg",
              actions={"convert_video": "read", "ffmpeg.convert_video": "write"})
    assert reg.get("ffmpeg.convert_video").spec().action == "write"

    reg2 = ToolRegistry(authorizer=_grant)
    mount_mcp(reg2, FakeMCPClient([_tool("convert_video")]), prefix="ffmpeg",
              actions={"convert_video": "read"})
    assert reg2.get("ffmpeg.convert_video").spec().action == "read"


def test_a_class_outside_the_vocabulary_is_refused_at_mount_time():
    """A typo here ("readonly", "modify") would otherwise become a class nothing in a policy can match."""
    client = FakeMCPClient([_tool("convert_video")])
    with pytest.raises(RegistryConfigurationError, match="outside the vocabulary"):
        mount_mcp(ToolRegistry(authorizer=_grant), client, actions={"convert_video": "readonly"})
    with pytest.raises(RegistryConfigurationError, match="default_action"):
        mount_mcp(ToolRegistry(authorizer=_grant), client, default_action="dangerous")


def test_an_unmounted_server_cannot_be_asked_for():
    """Nothing is registered until the server is mounted, so a tool the process has never seen cannot run —
    the mount is the only door through which third-party code becomes callable."""
    spy, seen = _spy()
    reg = ToolRegistry(authorizer=spy)
    assert reg.list() == []
    assert reg.run("runtime_registered", {}).ok is False
    assert seen == []           # the authorizer was not even consulted about a tool that is not there


def test_console_mount_appends_to_the_approval_allowlist(local_run):
    """How an app's chat entry point mounts a server: declared classes, plus the tools chat may run without
    a confirmation step. The two axes stay separate, and the console inherits the installed plane."""
    console = AgentConsole("Billing", "Billing is about invoices.", tools=[])
    console.mount_mcp(FakeMCPClient([_tool("convert_video", readOnlyHint=True)]),
                      actions={"convert_video": "read"}, allow=["convert_video"])
    assert console.registry.policy.allow == {"convert_video"}

    result = console.registry.run("convert_video", {})
    assert result.ok, result.error
    allowed = [a for a in console.registry.audit if a.get("allowed")]
    assert allowed and allowed[-1]["plane"] == local_run        # who said yes, not just that it ran
    # Local trust is a wildcard grant, so it shows up in unattributed_allows() on purpose: whoever reads
    # that list is meant to see that no named grant stood behind these calls.
    assert [a["tool"] for a in console.registry.unattributed_allows()] == ["convert_video"]