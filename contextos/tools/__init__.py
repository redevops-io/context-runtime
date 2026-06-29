"""Tools — the seam by which plans reach external systems (SPEC §4, new in this layer).

ContextOS plans *what context a model sees*; many of those sources live behind an API
(a SIEM, a BI warehouse, a firewall's decision engine). A ``ToolPlugin`` is how a plan
calls one. Read-only tools become retrieval sources (wrap with ``ToolRetriever``);
side-effecting tools (block an IP, file a ticket) are approval-gated.

Ported from agent-harness ``tools.py`` (registry) + ``approval.py`` (gate).
"""
from .base import (
    ApprovalPolicy,
    ToolError,
    ToolPlugin,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from .retriever import ToolRetriever

__all__ = [
    "ToolSpec", "ToolResult", "ToolPlugin", "ToolRegistry", "ApprovalPolicy",
    "ToolError", "ToolRetriever",
]
