"""MCP adapter: mount external MCP tool servers into the agent-harness registry gated by ApprovalPolicy.

Minimal JSON-RPC 2.0 client for the Model Context Protocol. Supports stdio (subprocess)
and HTTP (httpx POST) transports. Tools are wrapped as ToolPlugins so side-effects are
routed through the registry's ApprovalPolicy. Do not list tools at import time.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Any

import httpx

from .base import ToolPlugin, ToolRegistry, ToolResult, ToolSpec


class MCPClient:
    """Minimal JSON-RPC 2.0 MCP client.

    Create via classmethods:
        MCPClient.stdio(command, env=None)
        MCPClient.http(base_url, headers=None)
    """

    def __init__(self) -> None:
        self._req_id: int = 0
        self._stdio_proc: subprocess.Popen[str] | None = None
        self._http_base: str | None = None
        self._http_headers: dict[str, str] = {}
        self._initialized: bool = False
        self._closed: bool = False

    @classmethod
    def stdio(cls, command: list[str], env: dict[str, str] | None = None) -> MCPClient:
        self = cls()
        env_dict = os.environ.copy()
        if env:
            env_dict.update({k: str(v) for k, v in env.items()})
        self._stdio_proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env_dict,
        )
        return self

    @classmethod
    def http(cls, base_url: str, headers: dict[str, str] | None = None) -> MCPClient:
        self = cls()
        self._http_base = base_url
        self._http_headers = dict(headers or {})
        return self

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _exchange(self, req: dict[str, Any]) -> dict[str, Any]:
        """Send req and return the JSON-RPC response dict (may contain 'error')."""
        if self._closed:
            return {"jsonrpc": "2.0", "id": req.get("id"), "error": {"code": -32000, "message": "client closed"}}
        if self._stdio_proc is not None:
            try:
                line = json.dumps(req, separators=(",", ":")) + "\n"
                assert self._stdio_proc.stdin is not None
                self._stdio_proc.stdin.write(line)
                self._stdio_proc.stdin.flush()
                while True:
                    outline = self._stdio_proc.stdout.readline()
                    if not outline:
                        try:
                            err = self._stdio_proc.stderr.read(256) or ""
                        except Exception:
                            err = ""
                        return {"jsonrpc": "2.0", "id": req.get("id"), "error": {"code": -32000, "message": f"stdio closed: {err}"}}
                    try:
                        resp = json.loads(outline.strip())
                    except json.JSONDecodeError:
                        continue
                    rid = resp.get("id")
                    if rid == req.get("id") or "id" not in resp:
                        # accept matching or (rare) no-id but we treat as notif only on dedicated path
                        if "id" in resp:
                            return resp
                        continue  # notification while waiting, keep reading for our id
                    # other id? ignore
            except Exception as e:
                return {"jsonrpc": "2.0", "id": req.get("id"), "error": {"code": -32603, "message": f"stdio transport: {e}"}}
        elif self._http_base is not None:
            try:
                headers = {"Content-Type": "application/json", **self._http_headers}
                r = httpx.post(self._http_base, json=req, headers=headers, timeout=30.0)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                return {"jsonrpc": "2.0", "id": req.get("id"), "error": {"code": -32603, "message": f"http transport: {e}"}}
        return {"jsonrpc": "2.0", "id": req.get("id"), "error": {"code": -32603, "message": "no transport configured"}}

    def _send_notif(self, notif: dict[str, Any]) -> None:
        if self._closed:
            return
        if self._stdio_proc is not None:
            try:
                line = json.dumps(notif, separators=(",", ":")) + "\n"
                assert self._stdio_proc.stdin is not None
                self._stdio_proc.stdin.write(line)
                self._stdio_proc.stdin.flush()
            except Exception:
                pass
        elif self._http_base is not None:
            try:
                headers = {"Content-Type": "application/json", **self._http_headers}
                httpx.post(self._http_base, json=notif, headers=headers, timeout=10.0)
            except Exception:
                pass

    def _rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        rid = self._next_id()
        req = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}
        resp = self._exchange(req)
        if "error" in resp and resp.get("error"):
            err = resp["error"]
            code = err.get("code", -1)
            msg = err.get("message", str(err))
            data = err.get("data")
            raise RuntimeError(f"MCP {method} error {code}: {msg}{f' {data}' if data else ''}")
        return resp.get("result") or {}

    def initialize(self) -> None:
        if self._initialized:
            return
        params: dict[str, Any] = {
            "protocolVersion": "2024-11-05",
            "clientInfo": {"name": "context-runtime", "version": "0.1"},
        }
        self._rpc("initialize", params)
        notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        self._send_notif(notif)
        self._initialized = True

    def list_tools(self) -> list[dict[str, Any]]:
        if not self._initialized:
            self.initialize()
        result = self._rpc("tools/list")
        if isinstance(result, dict):
            return result.get("tools", [])
        return result if isinstance(result, list) else []

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._rpc("tools/call", {"name": name, "arguments": arguments or {}})

    def close(self) -> None:
        self._closed = True
        if self._stdio_proc is not None:
            try:
                assert self._stdio_proc.stdin is not None
                self._stdio_proc.stdin.close()
                self._stdio_proc.terminate()
                self._stdio_proc.wait(timeout=0.5)
            except Exception:
                pass
            self._stdio_proc = None
        self._http_base = None


@dataclass
class MCPToolPlugin:
    """ToolPlugin wrapper for a single MCP tool descriptor + shared live MCPClient."""

    _desc: dict[str, Any]
    _client: MCPClient
    _name: str
    _default_side: bool = True

    def __init__(self, tool_desc: dict[str, Any], client: MCPClient, *, name: str | None = None, default_side_effecting: bool = True) -> None:
        # support dataclass init + override for manual
        object.__setattr__(self, "_desc", tool_desc)
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_name", name or tool_desc.get("name", "unnamed"))
        object.__setattr__(self, "_default_side", default_side_effecting)

    def spec(self) -> ToolSpec:
        mcp_name = self._desc.get("name") or self._name
        desc = self._desc.get("description") or ""
        schema = self._desc.get("inputSchema") or self._desc.get("input_schema") or {"type": "object", "properties": {}}
        ann = self._desc.get("annotations") or {}
        read_only = bool(ann.get("readOnlyHint")) if "readOnlyHint" in ann else False
        side = (not read_only) if "readOnlyHint" in ann else bool(self._default_side)
        return ToolSpec(
            name=self._name,
            description=desc,
            parameters=schema,
            side_effecting=side,
        )

    def run(self, args: dict) -> ToolResult:
        try:
            res = self._client.call_tool(self._desc.get("name") or self._name, args)
        except Exception as e:
            return ToolResult(ok=False, error=str(e), text=f"[error] {self._name}: {e}")
        is_err = bool(res.get("isError", False))
        content_list = res.get("content") or []
        texts: list[str] = []
        for block in content_list:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text") or ""
                if t:
                    texts.append(t)
            elif block:
                texts.append(str(block))
        text = "\n".join(texts)
        if "structuredContent" in res and res.get("structuredContent") is not None:
            data = res["structuredContent"]
        else:
            data = res.get("content", res)
        err = None
        if is_err:
            err = res.get("error") or (text or "mcp tool returned isError")
        return ToolResult(ok=not is_err, data=data, text=text, error=err)


def mount_mcp(
    registry: ToolRegistry,
    client: MCPClient,
    *,
    prefix: str | None = None,
    default_side_effecting: bool = True,
) -> list[str]:
    """Initialize client if needed, list tools, register each wrapped MCPToolPlugin.

    Registered name = f'{prefix}.{name}' if prefix else name.
    Returns the list of registered tool names.
    """
    try:
        if not client._initialized:
            client.initialize()
    except Exception:
        pass  # list_tools may still work or caller will see []
    registered: list[str] = []
    try:
        tools = client.list_tools()
    except Exception:
        return registered
    for tdesc in tools:
        if not isinstance(tdesc, dict) or not tdesc.get("name"):
            continue
        raw = tdesc["name"]
        reg_name = f"{prefix}.{raw}" if prefix else raw
        plugin = MCPToolPlugin(tdesc, client, name=reg_name, default_side_effecting=default_side_effecting)
        registry.register(plugin)
        registered.append(reg_name)
    return registered
