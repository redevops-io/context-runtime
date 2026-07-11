"""Tool contracts, registry, and the approval gate.

A ``ToolPlugin`` exposes an OpenAI-style spec and a ``run(args) -> ToolResult``. The
``ToolRegistry`` is what a plan (or a tenant) calls; it enforces that side-effecting
tools pass an ``ApprovalPolicy`` before they execute. Default policy DENIES
side-effects unless explicitly allowed — never an open grant (agent-harness lineage).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from ..types import Hit


class ToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    side_effecting: bool = False          # mutates the outside world (block_ip, file_ticket)
    approval_required: bool = False       # gate even read tools if they touch sensitive data

    def openai(self) -> dict:
        """OpenAI-style function spec for tool-calling models."""
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}


@dataclass
class ToolResult:
    ok: bool = True
    data: Any = None                      # raw structured result
    hits: list[Hit] = field(default_factory=list)   # if the tool is a retrieval source
    text: str = ""                        # human/agent-readable summary
    error: str | None = None


@runtime_checkable
class ToolPlugin(Protocol):
    def spec(self) -> ToolSpec: ...
    def run(self, args: dict) -> ToolResult: ...


# An approver: given an action dict, return True to allow a side-effecting/gated tool.
Approver = Callable[[dict], bool]


class ApprovalPolicy:
    """Decide whether a side-effecting/gated tool call may run.

    Modes:
      * ``deny_side_effects`` (default) — read tools allowed; side-effects need an
        explicit approver returning True.
      * ``allowlist`` — only tools whose name is in ``allow`` may side-effect.
      * ``bypass`` — everything allowed (use only in trusted automation).
    """

    def __init__(self, mode: str = "deny_side_effects", allow: list[str] | None = None,
                 approver: Approver | None = None):
        self.mode = mode
        self.allow = set(allow or [])
        self.approver = approver

    def decide(self, spec: ToolSpec, args: dict) -> tuple[bool, str]:
        gated = spec.side_effecting or spec.approval_required
        if not gated:
            return True, "read-only"
        if self.mode == "bypass":
            return True, "bypass"
        if self.mode == "allowlist":
            return (spec.name in self.allow, "allowlisted" if spec.name in self.allow else "not in allowlist")
        # deny_side_effects: require an explicit approver
        if self.approver is not None and self.approver({"tool": spec.name, "args": args}):
            return True, "approved"
        return False, "side-effecting tool requires approval"


# ── data-access authorization seam (enterprise open-core) ─────────────────────────────────────────
# An optional authorizer runs on every tool call BEFORE the tool executes: given the caller's principal
# (opaque to the engine), the ToolSpec, and the args, it returns None to allow or a short deny reason.
# The engine imports no enterprise code — it calls this callable, exactly like the PolicyProvider seam.
# ``set_default_authorizer`` installs one fleet-wide, so every AgentConsole app is gated by a single call.
Authorizer = "Callable[[object, ToolSpec, dict], str | None]"

_default_authorizer = None


def set_default_authorizer(fn) -> None:
    """Install a process-wide authorizer used by every ToolRegistry that wasn't given its own."""
    global _default_authorizer
    _default_authorizer = fn


import contextvars as _contextvars

_principal_ctx = _contextvars.ContextVar("cr_current_principal", default=None)


def current_principal():
    """The principal for the in-flight tool call (set by ``ToolRegistry.run``), or None. Lets a tool
    scope its behavior to the caller — e.g. per-user config — without changing the ToolPlugin interface."""
    return _principal_ctx.get()


class ToolRegistry:
    """Register tools; run them through the approval gate; describe them to a model."""

    def __init__(self, policy: ApprovalPolicy | None = None, authorizer=None):
        self._tools: dict[str, ToolPlugin] = {}
        self.policy = policy or ApprovalPolicy()
        self.authorizer = authorizer       # data-access gate; falls back to the process default
        self.audit: list[dict] = []        # append-only record of every gated decision

    def register(self, tool: ToolPlugin) -> ToolPlugin:
        self._tools[tool.spec().name] = tool
        return tool

    def get(self, name: str) -> ToolPlugin:
        if name not in self._tools:
            raise ToolError(f"unknown tool: {name}")
        return self._tools[name]

    def list(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict]:
        """OpenAI-style tool specs for a tool-calling model."""
        return [t.spec().openai() for t in self._tools.values()]

    def run(self, name: str, args: dict | None = None, principal=None) -> ToolResult:
        try:
            tool = self.get(name)
        except ToolError as e:
            return ToolResult(ok=False, error=str(e), text=f"[error] {e}")
        spec = tool.spec()
        args = args or {}
        # expose the caller to the tool for the duration of the call (per-user scoping), reset after.
        _token = _principal_ctx.set(principal)
        try:
            # data-access authorization (enterprise): gate the tool by who is asking, before side-effect
            # approval. None principal + no authorizer ⇒ unchanged behavior.
            authorizer = self.authorizer or _default_authorizer
            if authorizer is not None:
                deny = authorizer(principal, spec, args)
                if deny:
                    self.audit.append({"tool": name, "args": args, "allowed": False, "reason": deny})
                    return ToolResult(ok=False, error=f"denied: {deny}", text=f"[blocked] {name}: {deny}")
            ok, reason = self.policy.decide(spec, args)
            if spec.side_effecting or spec.approval_required:
                self.audit.append({"tool": name, "args": args, "allowed": ok, "reason": reason})
            if not ok:
                return ToolResult(ok=False, error=f"denied: {reason}", text=f"[blocked] {name}: {reason}")
            return tool.run(args)
        except Exception as e:  # a tool crash is a failed result, not a runtime crash
            return ToolResult(ok=False, error=f"{type(e).__name__}: {e}", text=f"[error] {name}: {e}")
        finally:
            _principal_ctx.reset(_token)


def function_tool(name: str, fn: Callable[[dict], ToolResult], description: str = "",
                  parameters: dict | None = None, side_effecting: bool = False,
                  approval_required: bool = False) -> ToolPlugin:
    """Adapt a plain ``fn(args)->ToolResult`` into a ToolPlugin (agent-harness register())."""

    _spec = ToolSpec(name=name, description=description, parameters=parameters or {"type": "object", "properties": {}},
                     side_effecting=side_effecting, approval_required=approval_required)

    class _FnTool:
        def spec(self) -> ToolSpec:
            return _spec

        def run(self, args: dict) -> ToolResult:
            return fn(args)

    return _FnTool()
