"""Tool contracts, registry and approval gate.

These definitions live in ``agent_harness.tools`` (github.com/redevops-io/agent-harness) and are re-exported here
unchanged so existing imports keep working. They used to be a copy; the copy had no tests tying it to the original
and could drift. ``agent-harness`` is standard-library only, so depending on it adds no transitive dependencies.
"""
from __future__ import annotations

from agent_harness.tools import (  # noqa: F401  (re-exports)
    ApprovalPolicy,
    Approver,
    Authorizer,
    ToolError,
    ToolPlugin,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    current_principal,
    function_tool,
    set_default_authorizer,
)

__all__ = [
    "ApprovalPolicy", "Approver", "Authorizer", "ToolError", "ToolPlugin", "ToolRegistry", "ToolResult",
    "ToolSpec", "current_principal", "function_tool", "set_default_authorizer",
]
