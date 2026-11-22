"""Tool contracts, registry and approval gate.

These definitions live in ``agent_harness.tools`` (github.com/redevops-io/agent-harness) and are re-exported here
unchanged so existing imports keep working. They used to be a copy; the copy had no tests tying it to the original
and could drift. ``agent-harness`` is standard-library only, so depending on it adds no transitive dependencies.

Two of these names are load-bearing outside this repository. ``set_default_authorizer`` is the install point an
enterprise permissions plane uses once at start-up (``agentic_os.app_gate.install_gate`` reaches the engine through
this module), and the risk-class vocabulary (``ACTIONS``, ``ToolSpec.action``) is what a registration declares its
risk in. Where Context Runtime decides *who* authorizes one of its registries, as opposed to which engine answers,
is :mod:`context_runtime.tools.authz`.
"""
from __future__ import annotations

from agent_harness.tools import (  # noqa: F401  (re-exports)
    ACTIONS,
    ApprovalPolicy,
    Approver,
    Authorization,
    Authorizer,
    Decision,
    PolicyReason,
    RegistryConfigurationError,
    ToolClassificationError,
    ToolError,
    ToolPlugin,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    ValidationReport,
    current_principal,
    default_authorizer_installed,
    function_tool,
    local_trusted_authorizer,
    set_default_authorizer,
)

__all__ = [
    "ACTIONS", "ApprovalPolicy", "Approver", "Authorization", "Authorizer", "Decision", "PolicyReason",
    "RegistryConfigurationError", "ToolClassificationError", "ToolError", "ToolPlugin", "ToolRegistry",
    "ToolResult", "ToolSpec", "ValidationReport", "current_principal", "default_authorizer_installed",
    "function_tool", "local_trusted_authorizer", "set_default_authorizer",
]
