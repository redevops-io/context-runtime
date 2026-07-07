"""The bundled web_search MCP server + AgentConsole.mount_mcp — the end-to-end proof that an
app can mount a real MCP tool through the agent-harness registry.

The mount/registration test is hermetic (tools/list is static, no network). The actual search
test hits keyless public APIs and skips if the network is unavailable.
"""
from __future__ import annotations

import sys

import pytest

from context_runtime.integrations.agent_console import AgentConsole, tool
from context_runtime.tools.mcp import MCPClient


def _client() -> MCPClient:
    return MCPClient.stdio([sys.executable, "-m", "context_runtime.tools.mcp_servers.web_search"])


def test_web_search_mounts_as_readonly_tool():
    console = AgentConsole("X", "A watch monitors a URL.", tools=[tool("noop", "noop", lambda a: {"text": "ok"})])
    client = _client()
    try:
        names = console.mount_mcp(client)
        assert "web_search" in names
        assert "web_search" in console._tools                       # visible to classify + dispatch
        spec = console.registry.get("web_search").spec()
        assert spec.side_effecting is False                          # readOnlyHint → ungated → runs from chat
        assert "query" in spec.parameters.get("properties", {})
    finally:
        client.close()


def test_web_search_returns_results():
    from context_runtime.tools.mcp_servers.web_search import web_search

    out = web_search("python programming language", 4)
    if "no web results" in out.lower():
        pytest.skip("network unavailable / providers returned nothing")
    assert "results" in out.lower()
    assert "http" in out.lower()
