"""MCP adapter: mount external MCP tool servers into the agent-harness registry gated by ApprovalPolicy.

Minimal JSON-RPC 2.0 client for the Model Context Protocol. Supports stdio (subprocess)
and HTTP (httpx POST) transports. Tools are wrapped as ToolPlugins so side-effects are
routed through the registry's ApprovalPolicy. Do not list tools at import time.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Any

import httpx

from .base import ToolPlugin, ToolRegistry, ToolResult, ToolSpec

_log = logging.getLogger(__name__)

#: Parent environment variables a stdio MCP server receives by default: only what interpreters and common tools need to
#: start. Credentials (API keys, tokens, passwords, cloud and database settings) are NOT in this list; pass what a server
#: needs through the ``env`` mapping, or opt out with ``inherit_env=True``. Names are compared case-insensitively.
SAFE_ENV_VARS = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LANGUAGE", "TZ", "TERM", "TMPDIR", "TEMP", "TMP",
    "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "PATHEXT",
    "PYTHONPATH", "VIRTUAL_ENV", "PYTHONUTF8", "PYTHONIOENCODING",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "NO_PROXY",
})
SAFE_ENV_PREFIXES = ("LC_",)
#: Proxy settings are passed on only when the value carries no embedded credentials (no ``user:password@``).
_PROXY_ENV_VARS = frozenset({"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"})
#: Process-wide escape hatch that restores the old behaviour (the child inherits the whole parent environment).
INHERIT_ENV_VAR = "CONTEXT_RUNTIME_MCP_INHERIT_ENV"


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def stdio_env(
    parent: dict[str, str] | None = None,
    explicit: dict[str, str] | None = None,
    inherit_env: bool = False,
) -> dict[str, str]:
    """The environment a stdio MCP server is started with.

    Default: the allowlisted variables of ``parent`` (``os.environ`` when omitted) plus the ``explicit`` mapping, which wins.
    With ``inherit_env`` the whole parent environment is passed on, as before.
    """
    parent = os.environ if parent is None else parent
    if inherit_env:
        out = dict(parent)
    else:
        out = {}
        for key, value in parent.items():
            name = key.upper()
            if name in SAFE_ENV_VARS or name.startswith(SAFE_ENV_PREFIXES) or (name in _PROXY_ENV_VARS and "@" not in value):
                out[key] = value
    if explicit:
        out.update({k: str(v) for k, v in explicit.items()})
    return out


class MCPClient:
    """Minimal JSON-RPC 2.0 MCP client.

    Create via classmethods:
        MCPClient.stdio(command, env=None, inherit_env=None)
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
    def stdio(cls, command: list[str], env: dict[str, str] | None = None, *, inherit_env: bool | None = None) -> MCPClient:
        """Start a stdio MCP server.

        The child gets an allowlisted environment (``SAFE_ENV_VARS``) plus ``env``; secrets in the parent environment do not
        reach it. ``inherit_env=True`` passes the whole parent environment on (the previous behaviour); when ``inherit_env``
        is None the process-wide variable ``CONTEXT_RUNTIME_MCP_INHERIT_ENV`` decides (default: off).
        """
        self = cls()
        if inherit_env is None:
            inherit_env = _truthy(os.environ.get(INHERIT_ENV_VAR))
        if inherit_env:
            _log.warning("MCP stdio server %r starts with the full parent environment (inherit_env)", os.path.basename(command[0]))
        env_dict = stdio_env(explicit=env, inherit_env=bool(inherit_env))
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
