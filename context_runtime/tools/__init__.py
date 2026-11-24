"""Tools — the seam by which plans reach external systems (SPEC §4, new in this layer).

Context Runtime plans *what context a model sees*; many of those sources live behind an API
(a SIEM, a BI warehouse, a firewall's decision engine). A ``ToolPlugin`` is how a plan
calls one. Read-only tools become retrieval sources (wrap with ``ToolRetriever``);
side-effecting tools (block an IP, file a ticket) are approval-gated.

The contract, registry and approval gate are defined in ``agent_harness.tools`` and re-exported by ``.base``.
"""
from .base import (
    ApprovalPolicy,
    ToolError,
    ToolPlugin,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from .authz import (
    DIAGNOSTIC,
    DIAGNOSTIC_REASON_ENV,
    EXECUTE,
    EXECUTION_MODE_ENV,
    LOCAL_AUTHORITY_ENV,
    PrivilegedExecutionUnavailable,
    execution_mode,
    local_authority,
    local_authority_from_env,
    require_plane,
    resolve_authorizer,
)
from .actions import (
    INERT_MODULES,
    INERT_ROOTS,
    MIN_DELEGATED_CLASS,
    Delegated,
    PureRead,
    delegated,
    opaque_callees,
    pure_read,
    resolve_action,
)
from .retriever import ToolRetriever

__all__ = [
    "ToolSpec", "ToolResult", "ToolPlugin", "ToolRegistry", "ApprovalPolicy",
    "ToolError", "ToolRetriever",
    "resolve_authorizer", "local_authority", "local_authority_from_env", "LOCAL_AUTHORITY_ENV",
    "require_plane", "execution_mode", "PrivilegedExecutionUnavailable",
    "EXECUTION_MODE_ENV", "DIAGNOSTIC_REASON_ENV", "EXECUTE", "DIAGNOSTIC",
    "resolve_action",
    "PureRead", "pure_read",
    "delegated", "Delegated", "opaque_callees", "INERT_ROOTS", "INERT_MODULES", "MIN_DELEGATED_CLASS",
]
