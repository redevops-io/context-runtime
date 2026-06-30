"""The ToolPlugin seam: specs, the approval gate, the audit log, ToolRetriever."""
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
        description="returns docs")


def _side_effect_tool():
    return function_tool("nuke", lambda a: ToolResult(ok=True, text="boom"),
                         side_effecting=True, approval_required=True)


def test_read_tool_runs_and_satisfies_protocol():
    reg = ToolRegistry()
    t = reg.register(_read_tool())
    assert isinstance(t, base.ToolPlugin)
    res = reg.run("echo_docs", {"query": "k8s"})
    assert res.ok and res.hits and "k8s" in res.hits[0].text


def test_side_effecting_tool_denied_by_default():
    reg = ToolRegistry()
    reg.register(_side_effect_tool())
    res = reg.run("nuke", {})
    assert not res.ok and "denied" in res.error
    assert reg.audit[-1]["allowed"] is False         # the gate is audited


def test_side_effecting_tool_allowed_with_approver():
    reg = ToolRegistry(ApprovalPolicy(mode="deny_side_effects", approver=lambda a: True))
    reg.register(_side_effect_tool())
    res = reg.run("nuke", {})
    assert res.ok and res.text == "boom"
    assert reg.audit[-1]["allowed"] is True


def test_allowlist_mode():
    reg = ToolRegistry(ApprovalPolicy(mode="allowlist", allow=["nuke"]))
    reg.register(_side_effect_tool())
    assert reg.run("nuke", {}).ok


def test_openai_specs():
    reg = ToolRegistry()
    reg.register(_read_tool())
    specs = reg.specs()
    assert specs[0]["type"] == "function" and specs[0]["function"]["name"] == "echo_docs"


def test_tool_retriever_exposes_tool_as_retrieval_source():
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
