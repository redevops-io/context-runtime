"""Hermetic MCP adapter tests: real stdio transport + ApprovalPolicy gating."""
from __future__ import annotations

import json
import sys
import tempfile
from context_runtime.tools import ApprovalPolicy, ToolRegistry, ToolResult
from context_runtime.tools.mcp import MCPClient, mount_mcp


FAKE_SERVER = r'''import sys, json

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        rid = req.get("id")
        meth = req.get("method")
        params = req.get("params") or {}
        if meth == "initialize":
            res = {
                "jsonrpc": "2.0",
                "id": rid,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake-mcp", "version": "0"},
                },
            }
            print(json.dumps(res, separators=(",", ":")), flush=True)
            continue
        if rid is None:
            # notification (e.g. initialized), no response
            continue
        if meth == "tools/list":
            tools = [
                {
                    "name": "web_search",
                    "description": "search the web",
                    "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
                {
                    "name": "echo",
                    "description": "echo text",
                    "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}},
                },
            ]
            reply = {"jsonrpc": "2.0", "id": rid, "result": {"tools": tools}}
            print(json.dumps(reply, separators=(",", ":")), flush=True)
            continue
        if meth == "tools/call":
            nm = (params or {}).get("name", "")
            args = (params or {}).get("arguments") or {}
            if nm == "echo":
                txt = args.get("text", "")
                if txt == "ERR":
                    res = {
                        "isError": True,
                        "content": [{"type": "text", "text": "failed"}],
                        "error": "tool error",
                    }
                else:
                    res = {"content": [{"type": "text", "text": "echoed:" + txt}]}
                print(json.dumps({"jsonrpc": "2.0", "id": rid, "result": res}, separators=(",", ":")), flush=True)
                continue
            # web_search or default
            q = args.get("query", "")
            res = {"content": [{"type": "text", "text": "web:" + q}]}
            print(json.dumps({"jsonrpc": "2.0", "id": rid, "result": res}, separators=(",", ":")), flush=True)
            continue
        if rid is not None:
            err = {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "method not found"}}
            print(json.dumps(err, separators=(",", ":")), flush=True)

if __name__ == "__main__":
    main()
'''


def _run_with_client_and_registry(policy: ApprovalPolicy | None = None):
    reg = ToolRegistry(policy=policy)
    with tempfile.TemporaryDirectory() as td:
        fake_path = f"{td}/fake_mcp_server.py"
        with open(fake_path, "w") as f:
            f.write(FAKE_SERVER)
        client = MCPClient.stdio([sys.executable, fake_path])
        try:
            names = mount_mcp(reg, client)
            return reg, names, client
        except Exception:
            client.close()
            raise


def test_mcp_mount_registers_tools_via_real_stdio():
    reg, names, client = _run_with_client_and_registry(policy=ApprovalPolicy(mode="bypass"))
    try:
        assert set(names) == {"web_search", "echo"}
        assert set(reg.list()) == {"web_search", "echo"}
    finally:
        client.close()


def test_mcp_echo_runs_under_bypass_and_returns_text():
    reg, names, client = _run_with_client_and_registry(policy=ApprovalPolicy(mode="bypass"))
    try:
        assert "echo" in reg.list()
        res = reg.run("echo", {"text": "hi"})
        assert isinstance(res, ToolResult)
        assert res.ok is True
        assert "hi" in res.text
        assert "echoed" in res.text
    finally:
        client.close()


def test_mcp_tools_are_gated_by_approval_policy():
    # default policy (deny_side_effects) blocks side-effecting MCP tool (default_side=True)
    reg, names, client = _run_with_client_and_registry()
    try:
        res = reg.run("echo", {"text": "hi"})
        assert res.ok is False
        assert "denied" in (res.error or "")
    finally:
        client.close()

    # with allowlist it passes
    reg2 = ToolRegistry(ApprovalPolicy(mode="allowlist", allow=["echo", "web_search"]))
    with tempfile.TemporaryDirectory() as td:
        fake_path = f"{td}/fake_mcp_server.py"
        with open(fake_path, "w") as f:
            f.write(FAKE_SERVER)
        client2 = MCPClient.stdio([sys.executable, fake_path])
        try:
            mount_mcp(reg2, client2)
            res2 = reg2.run("echo", {"text": "allow"})
            assert res2.ok is True
            assert "allow" in res2.text
        finally:
            client2.close()


def test_mcp_iserror_maps_to_toolresult_failure():
    reg, names, client = _run_with_client_and_registry(policy=ApprovalPolicy(mode="bypass"))
    try:
        res = reg.run("echo", {"text": "ERR"})
        assert res.ok is False
        assert res.error is not None and "error" in res.error.lower()
        assert "failed" in (res.text or "")
    finally:
        client.close()
