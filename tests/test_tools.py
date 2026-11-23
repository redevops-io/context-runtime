"""The ToolPlugin seam: specs, the approval gate, the audit log, ToolRetriever.

Two things every test here states out loud, because both are now part of the contract: a tool declares its
risk class with ``action=``, and a registry that is expected to *run* a tool runs under ``local_run`` — with no
authorizer installed the registry denies the call (agent-harness C1), which is the opposite of what it did
before that change.
"""
from __future__ import annotations

from context_runtime.plugins import base
from context_runtime.tools import (
    ApprovalPolicy, ToolRegistry, ToolResult, ToolRetriever, ToolSpec,
)
from context_runtime.tools.base import function_tool
from context_runtime.types import Hit


def _read_tool():
    return function_tool("echo_docs", lambda a: ToolResult(ok=True, hits=[
        Hit(chunk_id="x", filename="f", text=f"doc for {a.get('query')}", score=1.0)]),
        description="returns docs", action="read")


def _side_effect_tool():
    return function_tool("nuke", lambda a: ToolResult(ok=True, text="boom"),
                         side_effecting=True, approval_required=True, action="write")


def test_read_tool_runs_and_satisfies_protocol(local_run):
    reg = ToolRegistry()
    t = reg.register(_read_tool())
    assert isinstance(t, base.ToolPlugin)
    res = reg.run("echo_docs", {"query": "k8s"})
    assert res.ok and res.hits and "k8s" in res.hits[0].text
    assert reg.classification() == {"echo_docs": "read"}      # the risk class travels with the registration


def test_side_effecting_tool_denied_by_default(local_run):
    # local_run so the *approval* gate is what denies: without an authorizer the call never reaches it.
    reg = ToolRegistry()
    reg.register(_side_effect_tool())
    res = reg.run("nuke", {})
    assert not res.ok and "denied" in res.error
    assert reg.audit[-1]["allowed"] is False         # the gate is audited
    assert "approval" in reg.audit[-1]["reason"]     # ...and it says which gate denied it


def test_side_effecting_tool_allowed_with_approver(local_run):
    reg = ToolRegistry(ApprovalPolicy(mode="deny_side_effects", approver=lambda a: True))
    reg.register(_side_effect_tool())
    res = reg.run("nuke", {})
    assert res.ok and res.text == "boom"
    assert reg.audit[-1]["allowed"] is True


def test_allowlist_mode(local_run):
    reg = ToolRegistry(ApprovalPolicy(mode="allowlist", allow=["nuke"]))
    reg.register(_side_effect_tool())
    assert reg.run("nuke", {}).ok


def test_openai_specs():
    reg = ToolRegistry()
    reg.register(_read_tool())
    specs = reg.specs()
    assert specs[0]["type"] == "function" and specs[0]["function"]["name"] == "echo_docs"


def test_tool_retriever_exposes_tool_as_retrieval_source(local_run):
    reg = ToolRegistry()
    reg.register(_read_tool())
    retr = ToolRetriever(reg, source_tool={"api": "echo_docs"}, default_tool="echo_docs")
    assert isinstance(retr, base.RetrieverPlugin)
    hits = retr.search("incident", k=3, method="api")
    assert hits and "incident" in hits[0].text


def test_unknown_tool_is_a_failed_result_not_a_crash():
    reg = ToolRegistry()
    res = reg.run("missing", {})
    assert not res.ok and "unknown tool" in res.error


def test_tool_that_raises_becomes_failed_result(local_run):
    def boom(args):
        raise ValueError("kaboom")
    reg = ToolRegistry()
    reg.register(function_tool("boomer", boom, description="raises", action="write"))
    res = reg.run("boomer", {})
    assert res.ok is False and "kaboom" in (res.error or "")   # a crash is a failed result, not a runtime crash


def test_registry_unknown_tool_is_failed_result():
    res = ToolRegistry().run("does-not-exist", {})
    assert res.ok is False
