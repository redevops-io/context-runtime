"""MCP tools are mounted with a class the operator declared, never one the server claimed.

An MCP server is code pulled from somewhere else, and ``readOnlyHint`` is its own announcement about itself.
The registry's rule is that the caller declares what it is about to run: the hint affects neither the
authorization class nor the approval gate. (A prior version let the hint relax the approval gate for a
non-``read`` action — ``side_effecting`` dropped to ``False`` whenever a server claimed ``readOnlyHint: true``,
which made :class:`ApprovalPolicy` skip the approval step entirely for a tool the operator had explicitly
classified ``write``/``delete``. That was Finding M1 of the authorization-stack review: a malicious or
careless server could talk its own consequential tool out of needing approval. Fixed — see below.)

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


def test_a_tool_nobody_classed_is_not_mounted_at_all():
    """Refused by default, and refused structurally: the tool is not in the registry, so there is no call
    for a plane to deny and no grant that could accidentally match it. The signature used to carry
    ``default_action="write"``, which handed out a risk class to a mount that had not read the server —
    a floor in the same shape as the one the injected-factory seam just lost."""
    spy, seen = _spy()
    reg = _runnable("convert_video", authorizer=spy)
    with pytest.warns(UserWarning, match="no risk class"):
        names = mount_mcp(reg, FakeMCPClient([_tool("convert_video")]))
    assert names == [] and reg.list() == []
    assert reg.run("convert_video", {}).ok is False
    assert seen == []            # nothing was registered, so nothing was ever asked about


def test_read_only_hint_has_no_effect_on_approval_or_class():
    """The hint is an announcement from the code being run, not from anyone this registry trusts to grade
    its own risk: it changes neither the authorization class nor whether the approval gate engages. A tool
    mounted as ``write`` still needs approval even when the server calls itself read-only — the exploit this
    test used to validate as correct behavior (M1): under the default ``deny_side_effects`` policy with no
    approver, the call is denied, not silently waved through."""
    spy, seen = _spy()
    reg = ToolRegistry(authorizer=spy)  # default policy: deny_side_effects, no approver
    mount_mcp(reg, FakeMCPClient([_tool("search", readOnlyHint=True)]), default_action="write")
    result = reg.run("search", {})
    assert result.ok is False
    assert "requires approval" in result.error
    assert seen == [("search", "write", True)]  # side_effecting is true despite the hint; nothing executed


def test_read_only_hint_cannot_downgrade_delete_either():
    """The same bug, one class stricter: a server cannot talk a ``delete`` tool out of needing approval by
    claiming ``readOnlyHint: true``."""
    spy, seen = _spy()
    reg = ToolRegistry(authorizer=spy)
    mount_mcp(reg, FakeMCPClient([_tool("wipe", readOnlyHint=True)]), default_action="delete")
    result = reg.run("wipe", {})
    assert result.ok is False
    assert "requires approval" in result.error
    assert seen == [("wipe", "delete", True)]


@pytest.mark.parametrize("annotations", [
    {},                                   # hint absent entirely
    {"readOnlyHint": False},              # hint explicitly false
    {"readOnlyHint": True},               # hint explicitly true (the exploit shape)
    {"readOnlyHint": "yes"},              # malformed: a truthy non-bool
    {"readOnlyHint": None},               # malformed: present but null
], ids=["absent", "false", "true", "truthy-string", "null"])
def test_no_mcp_metadata_shape_can_downgrade_approval(annotations):
    """Every combination of missing, false, true, or malformed ``readOnlyHint`` must leave a ``write`` tool
    requiring approval — none of them is a code path that was ever meant to grant a pass."""
    spy, seen = _spy()
    reg = ToolRegistry(authorizer=spy)
    mount_mcp(reg, FakeMCPClient([_tool("search", **annotations)]), default_action="write")
    result = reg.run("search", {})
    assert result.ok is False
    assert "requires approval" in result.error


def test_an_explicitly_approved_write_still_executes_correctly():
    """The fix must not break the legitimate path: a write tool with a real approver that says yes still
    runs, hint or no hint."""
    client = FakeMCPClient([_tool("search", readOnlyHint=True)])
    reg = ToolRegistry(authorizer=_grant,
                        policy=ApprovalPolicy("deny_side_effects", approver=lambda call: True))
    mount_mcp(reg, client, default_action="write")
    result = reg.run("search", {})
    assert result.ok, result.error
    assert client.calls == [("search", {})]


def test_delegated_classification_already_refuses_a_remote_readonlyhint_as_a_source():
    """A second place this class of bug could have been reintroduced: a ``delegated()``/``pure_read()``
    wrapper around an MCP client claiming its own class from the server's hint instead of from whoever read
    the code. This module's own classification seam (``context_runtime.tools.actions.resolve_action``)
    already refuses to accept a remote ``readOnlyHint`` as a classification source at all — it checks the
    wrapper's own source for write sinks and opaque callees, never anything the wrapped MCP client reports
    about itself. Confirmed here so a future change to either module cannot silently reopen this path
    without a test noticing."""
    from context_runtime.tools import actions

    def wraps_a_consequential_mcp_call(args: dict):
        # the fake client's `readOnlyHint=True` on this tool must have no bearing on this wrapper's own
        # classification — its own source (an opaque call to a client object) is what gets checked.
        client = FakeMCPClient([_tool("delete_everything", readOnlyHint=True)])
        return client.call_tool("delete_everything", args)

    with pytest.raises(actions.RegistryConfigurationError, match="pure_read"):
        actions.resolve_action(
            shipped="read",
            injected=wraps_a_consequential_mcp_call,
            declared=actions.pure_read("this just calls the mcp client, which is a readOnlyHint tool"),
            tool="delete_everything", context="test",
        )


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