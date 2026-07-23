"""Context Runtime as an MCP tool — the context-serving sidecar for AgentCore / Strands agents (§2.4).

An AgentCore **Gateway** *is* MCP, and Strands agents consume MCP tools — so exposing "give me an
optimal context bundle" as an MCP tool lets an agent running on the AWS stack call Context Runtime as
a first-class tool, riding AWS's own tool plane instead of a bespoke integration. The tool returns the
exact bundle `/librechat/retrieve` does (routed knowledge representation, cost-optimal method, hits +
assembled context, an EXPLAIN-able plan id), drawn from whatever arms the deployment wired — Bedrock
KB, OpenSearch, local hybrid.

fastmcp is an optional dep (like the agentic-os MCP server); it's imported lazily in ``build_mcp`` so
this module imports without it. Run:  ``python -m context_runtime.control_plane.mcp_server``
"""
from __future__ import annotations

import os


def retrieve_context(request: str, model: str | None = None, *, tenant_resolver=None) -> dict:
    """Core tool logic (fastmcp-free, so it's unit-testable). ``tenant_resolver(model) -> tenant``
    defaults to the control plane's; injectable for tests."""
    if tenant_resolver is None:
        from .app import _tenant_for as tenant_resolver  # noqa: N813
    from .app import retrieve_bundle
    return retrieve_bundle(tenant_resolver(model), request)


def build_mcp():
    """Construct the FastMCP server exposing the retrieval tool (lazy fastmcp import)."""
    from fastmcp import FastMCP

    mcp = FastMCP("context-runtime")

    @mcp.tool()
    def retrieve_context_tool(request: str, model: str | None = None) -> dict:
        """Return a cost-optimal context bundle for `request`: the chosen knowledge representation +
        retrieval method, the ranked hits, the assembled context string, and a plan id you can
        EXPLAIN. `model` selects the corpus/tenant (optional)."""
        return retrieve_context(request, model)

    return mcp


def main() -> None:  # pragma: no cover — process entrypoint
    build_mcp().run(transport="streamable-http", host="0.0.0.0",
                    port=int(os.getenv("CR_MCP_PORT", "8081")))


if __name__ == "__main__":  # pragma: no cover
    main()
