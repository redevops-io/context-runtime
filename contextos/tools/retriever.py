"""ToolRetriever — expose read-only tools as a RetrieverPlugin (SPEC §4.5).

This is the bridge that turns "call an API" into "a retrieval source the planner can
pick." A SIEM tool, a BI query tool, a firewall-decisions tool — each becomes a method
ContextOS can route to. Side-effecting tools are NOT exposed here (they go through the
registry's approval gate explicitly).
"""
from __future__ import annotations

from .base import ToolRegistry
from ..types import Hit, PluginInfo, Retrieval


class ToolRetriever:
    """Dispatch ``search`` to a registered read-only tool.

    ``source_tool`` maps a retrieval method/source name → tool name. The tool is run
    with ``{"query": ..., "k": ...}`` and must return ``ToolResult.hits``.
    """

    def __init__(self, registry: ToolRegistry, source_tool: dict[str, str], default_tool: str | None = None):
        self.registry = registry
        self.source_tool = source_tool
        self.default_tool = default_tool

    def search(self, query: str, k: int, method: Retrieval = "api") -> list[Hit]:
        tool_name = self.source_tool.get(method) or self.default_tool
        if not tool_name:
            return []
        res = self.registry.run(tool_name, {"query": query, "k": k})
        return res.hits[:k] if res.ok else []

    def info(self) -> PluginInfo:
        return PluginInfo(name="tool_retriever", kind="retriever",
                          capabilities=frozenset(self.source_tool))
