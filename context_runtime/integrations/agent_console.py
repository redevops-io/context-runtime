"""AgentConsole — a reusable conversational agent that runs entirely on the Context
Runtime stack.

Every agentic-os app wraps a complex OSS core (ERPNext, Lago, OpenSCAP, CrowdSec, …); the
dashboards show state but leave users to learn the tool and hunt for the right action.
``AgentConsole`` is the one chat surface that fixes both: ask *"how do I…?"* or *"explain
this"* and it answers grounded in the app's guide; ask it to *do* something and it runs a
tool — always showing its work.

It is deliberately built on our own three planes, not a raw provider SDK:

* **context-runtime** — generation goes through a Context Runtime ``ModelPlugin``
  (``OpenAICompatibleModel`` when a key is present, else the offline ``StubModel``), so the
  same cost-tiered ``ModelRequest → ModelResult`` contract and accounting apply.
* **redevops-rag** — the *"how do I / explain"* path is grounded by retrieval over the app's
  primer, honouring redevops-rag's ``hybrid_search`` contract. The dependency-free
  ``PrimerIndex`` here is the offline path; the ``[rag]`` extra swaps in the real reranked
  index without changing callers.
* **agent-harness** — actions are dispatched through the harness ``ToolRegistry`` +
  ``ApprovalPolicy``, so a side-effecting tool is gated and every decision is audited.

One console per app: give it a ``tenant`` name, a ``primer`` (the product knowledge), and a
list of ``tools`` (each bound to that app's core API). ``respond()`` returns a transparent
payload the chat panel renders; ``panel_html()`` returns the panel itself.
"""
from __future__ import annotations

import json
import math
import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..adapters.model_openai import OpenAICompatibleModel
from ..adapters.model_stub import StubModel
from ..tools.authz import (
    DIAGNOSTIC,
    PrivilegedExecutionUnavailable,
    RegistryConfigurationError,
    execution_mode,
    resolve_authorizer,
)
from ..tools.base import (ACTIONS, ApprovalPolicy, ToolRegistry, ToolResult, ToolSpec,
                          ValidationReport, default_authorizer_installed)
from ..types import ModelRequest

_log = logging.getLogger(__name__)

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


# ──────────────────────────── grounding (redevops-rag contract) ────────────────────────────


@dataclass
class Chunk:
    idx: int
    text: str
    score: float = 0.0


class PrimerIndex:
    """Dependency-free ranked retrieval over the app primer — the offline redevops-rag path.

    Splits the primer into blank-line-separated passages and ranks them by TF·IDF overlap
    with the query. Same ``search(query, k) -> ranked passages`` shape as redevops-rag's
    ``hybrid_search``; the ``[rag]`` extra replaces this with the reranked vector path.
    """

    def __init__(self, primer: str):
        passages = [p.strip() for p in re.split(r"\n\s*\n", primer.strip()) if p.strip()]
        self.passages = passages
        self._toks = [_tokens(p) for p in passages]
        df: Counter = Counter()
        for toks in self._toks:
            for t in set(toks):
                df[t] += 1
        n = max(1, len(passages))
        self._idf = {t: math.log(1 + n / c) for t, c in df.items()}

    def search(self, query: str, k: int = 4) -> list[Chunk]:
        q = set(_tokens(query))
        if not q or not self.passages:
            return []
        scored: list[Chunk] = []
        for i, toks in enumerate(self._toks):
            tf = Counter(toks)
            score = sum(tf[t] * self._idf.get(t, 0.0) for t in q)
            if score > 0:
                scored.append(Chunk(idx=i, text=self.passages[i], score=round(score, 3)))
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:k]


# ──────────────────────────── tools (agent-harness) ────────────────────────────


ToolFn = Callable[[dict], Any]


class _Tool:
    """Adapt a plain callable into an agent-harness ``ToolPlugin``."""

    def __init__(self, spec: ToolSpec, fn: ToolFn):
        self._spec = spec
        self._fn = fn

    def spec(self) -> ToolSpec:
        return self._spec

    def run(self, args: dict) -> ToolResult:
        try:
            out = self._fn(args or {})
        except Exception as e:  # noqa: BLE001 — never let a tool crash the console
            return ToolResult(ok=False, error=str(e), text=f"[tool error] {e}")
        if isinstance(out, ToolResult):
            return out
        if isinstance(out, dict) and ("text" in out or "data" in out):
            return ToolResult(ok=True, data=out.get("data"), text=str(out.get("text", "")))
        return ToolResult(ok=True, data=out, text="" if out is None else str(out))


def tool(
    name: str,
    description: str,
    fn: ToolFn,
    *,
    parameters: dict | None = None,
    side_effecting: bool = False,
    action: str = "write",
) -> _Tool:
    """Declare an app capability. ``fn`` takes an args dict and returns a dict/str/ToolResult.

    ``action`` is the risk class the console asks for when this tool runs — declared by whoever wrote
    ``fn``, because the console cannot see what it does. It defaults to ``write``, the strictest class
    every capability must clear; a capability that erases data says so with ``action="delete"``, and a
    genuinely read-only one with ``action="read"``. ``side_effecting`` remains the *approval* axis —
    the two are separate checks and neither substitutes for the other.
    """
    if action not in ACTIONS:
        raise RegistryConfigurationError(
            f"tool {name!r} declares action={action!r}, which is not one of the vocabulary {ACTIONS}")
    return _Tool(
        ToolSpec(
            name=name,
            description=description,
            parameters=parameters or {"type": "object", "properties": {}},
            side_effecting=side_effecting,
            action=action,
        ),
        fn,
    )


# ──────────────────────────── the console ────────────────────────────


_CLASSIFY_SYS = (
    "You route a user's message for the assistant of a business app. Reply with ONLY a JSON "
    'object: {"mode":"tool"|"help","tool":<tool name or null>,"args":{...},"reason":<short>}. '
    'Use "tool" when the message asks to DO or SHOW something a listed tool covers; use "help" '
    "for how-to / explain / what-is questions. Never invent a tool name."
)

# Interrogative/how-to leads that the deterministic fallback routes to help (not a tool). Kept as
# explicit phrases so action queries like "what changed this week?" / "how's my traffic?" still route.
_HELP_MARKERS = (
    "how do i", "how do you", "how can i", "how to ", "how does ", "how would i",
    "what is ", "what are ", "what does ", "what's a ", "explain", "tell me about", "why ",
)


def _policy_items(decisions) -> list[dict]:
    """Compact policy list for the panel (§10.1): only non-allow decisions become chips; allows fold."""
    out = []
    for d, phase in decisions:
        if d.action == "allow":
            continue
        out.append({"decision": d.action, "phase": phase, "reason": d.reason, "rule_id": d.rule_id,
                    "scope": d.scope, "provider": d.provider, "summary": f"{d.action} · {phase}"})
    return out


class AgentConsole:
    def __init__(
        self,
        tenant: str,
        primer: str,
        tools: list[_Tool] | tuple[_Tool, ...] = (),
        *,
        suggestions: list[str] | tuple[str, ...] = (),
        subtitle: str = "",
        model: Any = None,
        allow_side_effects: list[str] | None = None,
        authorizer=None,
        local_trust: bool = False,
        local_trust_reason: str = "",
        commands=None,
        policy=None,
        app: str = "",
        privileged: Optional[bool] = None,
    ):
        self.tenant = tenant
        self.subtitle = subtitle or f"Ask about {tenant} — how things work, or get something done."
        self.primer = primer.strip()
        self.index = PrimerIndex(self.primer)
        self.suggestions = list(suggestions)
        self.model = model if model is not None else (OpenAICompatibleModel.from_env() or StubModel())
        # The console is a request-facing surface: an enterprise deployment hands it the request's principal
        # as ``authorizer``, otherwise it inherits the process-wide plane installed with
        # set_default_authorizer, so one install() governs every console (tools/authz.py).
        #
        # ``on_missing="defer_to_plane"`` is the one place this module resolves differently from a tenant, and
        # the distinction is the whole point of it: a console is a COMPONENT, not a service. An app builds one
        # at module import and its gate installs the plane moments later, so refusing to build here refuses to
        # import an app that was about to be configured correctly. What is NOT negotiable is what happens
        # next — the application that owns this console must gate on ``validate()`` before it serves
        # (``agentic_os.app_gate.validate_console``: no plane means no start when authorization was asked for,
        # and a readiness failure in the health record when it was not). Building without a plane is allowed;
        # SERVING without one is the failure mode, and it belongs to the service, not to this constructor.
        self._authorizer = resolve_authorizer(authorizer, local_trust=local_trust,
                                              local_trust_reason=local_trust_reason,
                                              context=f"agent console for tenant {tenant!r}",
                                              on_missing="defer_to_plane")
        if self._authorizer is None and not default_authorizer_installed():
            _log.warning("agent console %r has no authorization plane: every tool call will be denied with "
                         "missing_explicit_grant. Pass authorizer=..., install one with "
                         "context_runtime.tools.set_default_authorizer(...) before building the console, or "
                         "declare local execution with local_trust=True and a written reason.", tenant)
        self.registry = ToolRegistry(
            ApprovalPolicy(mode="allowlist", allow=list(allow_side_effects or [])),
            authorizer=self._authorizer)
        # Policy Runtime: /commands are dispatched here (dual-path); `policy` enforces input/output/tool
        # phases and emits PolicyDecisions. Both optional and fall back to the process default → one
        # install governs the fleet. app = the policy scope slug (e.g. "market-radar").
        self.commands = commands
        self._policy = policy
        self.app = app
        self._tools: dict[str, _Tool] = {}
        # A diagnostic instance does not get tools, which is what makes its read-only promise structural
        # rather than a denial the operator has to trust: nothing is registered, so nothing can be offered to
        # the model, dispatched, or re-approved into running. The alternative — register everything and let
        # the plane say no on every call — is the state this repository just stopped calling healthy.
        # ``privileged=False`` is the same declaration made in code rather than in the environment, for the
        # surfaces an application builds as read-only regardless of how the process is configured.
        mode_privileged = execution_mode(context=f"agent console for tenant {tenant!r}")[0] != DIAGNOSTIC
        self.privileged = mode_privileged if privileged is None else privileged
        if not self.privileged:
            _log.warning("agent console %r is running in the diagnostic mode: %d tool(s) the app passed are "
                         "not registered, so this instance can plan and answer but cannot act.", tenant,
                         len(list(tools)))
        else:
            for t in tools:
                self.registry.register(t)
                self._tools[t.spec().name] = t

    def mount_mcp(self, client, *, prefix: str | None = None, allow: list[str] | None = None,
                  actions: dict[str, str] | None = None,
                  default_action: Optional[str] = None) -> list[str]:
        """Mount an external MCP tool server's tools into this console.

        Registers each tool in the agent-harness registry AND the classify/dispatch catalog, so
        the assistant can pick and run them like any native tool.

        Risk classes come from ``actions`` — ``{"ffmpeg.convert_video": "read", ...}`` — decided by
        whoever mounts the server, after reading what that server does. A tool outside that map gets
        ``default_action``, and with no ``default_action`` it is not mounted at all: an MCP server is
        code this console did not write, so a class it would fall back to is a grant nobody chose.
        A server's own ``readOnlyHint`` still moves the *approval* gate (an announced read-only tool
        need not be confirmed) but never the authorization class, because the thing asking to be run is
        not allowed to grade its own risk.

        ``allow`` is the approval allowlist and is unchanged. Returns the mounted tool names.
        """
        if not self.privileged:
            raise PrivilegedExecutionUnavailable(
                "mount_mcp() is refused on a diagnostic console: mounting a tool server is how a read-only "
                "instance acquires an execution path, which is the one thing its mode promise rules out.")
        from ..tools.mcp import mount_mcp as _mount_mcp

        names = _mount_mcp(self.registry, client, prefix=prefix, actions=actions,
                           default_action=default_action)
        for n in names:
            self._tools[n] = self.registry.get(n)  # classify() sees it, respond() dispatches it
        if allow:
            self.registry.policy.allow.update(allow)
        return names

    def validate(self, *, strict: bool = True) -> ValidationReport:
        """Start-up check: every tool must carry a class the plane can ask about, and there must be a plane.

        An app calls this at boot (``console = AgentConsole(...); console.validate()``) and refuses to serve
        when it answers anything. With no plane the console still builds — an app's gate installs one a moment
        later, and refusing to import an app is a worse failure than the console denying — but a console that
        *serves* without one answers every tool call with ``missing_explicit_grant``, which is a support ticket
        rather than a diagnostic. This turns that into a start-up failure. The report's ``problems`` are the
        reasons not to serve, its ``warnings`` the reachability facts a reviewer should see (a gated tool the
        approval policy can never approve); ``strict=True``, the default, raises
        :class:`~context_runtime.tools.base.RegistryConfigurationError` on a problem instead of returning it,
        which is what a constructor wants.
        """
        return self.registry.validate(strict=strict)

    # ---- helpers -------------------------------------------------------------

    @property
    def _live(self) -> bool:
        return not isinstance(self.model, StubModel)

    def _tool_catalog(self) -> str:
        return "\n".join(f"- {n}: {t.spec().description}" for n, t in self._tools.items()) or "(no tools)"

    def _keyword_route(self, message: str) -> dict:
        """Deterministic fallback when the model can't return JSON (offline / parse fail)."""
        # How-to / explain questions are help, not actions — mirror the LLM classify rule so a
        # single incidental word ("...part of a page?" vs a "monitor a page" tool) can't misroute
        # an interrogative into a side-effecting tool call.
        low = (message or "").strip().lower()
        if any(low.startswith(mk) or (" " + mk) in low for mk in _HELP_MARKERS):
            return {"mode": "help", "tool": None, "args": {}, "reason": "how-to → help"}
        toks = set(_tokens(message))
        best, best_score = None, 0
        for name, t in self._tools.items():
            vocab = set(_tokens(name.replace("_", " ") + " " + t.spec().description))
            score = len(toks & vocab)
            if score > best_score:
                best, best_score = name, score
        if best and best_score >= 1:
            return {"mode": "tool", "tool": best, "args": {}, "reason": "keyword match"}
        return {"mode": "help", "tool": None, "args": {}, "reason": "default to help"}

    def classify(self, message: str) -> dict:
        if not self._tools or not self._live:
            return self._keyword_route(message)
        prompt = f"Tools:\n{self._tool_catalog()}\n\nMessage: {message}\n\nJSON:"
        try:
            res = self.model.complete(
                ModelRequest(messages=({"role": "user", "content": prompt},), system=_CLASSIFY_SYS, max_tokens=200)
            )
            m = re.search(r"\{.*\}", res.text, re.S)
            data = json.loads(m.group(0)) if m else {}
            mode = data.get("mode")
            tool_name = data.get("tool")
            if mode == "tool" and tool_name in self._tools:
                return {"mode": "tool", "tool": tool_name, "args": data.get("args") or {}, "reason": data.get("reason", "")}
            if mode == "help":
                return {"mode": "help", "tool": None, "args": {}, "reason": data.get("reason", "")}
        except Exception:  # noqa: BLE001 — any failure drops to the deterministic router
            pass
        return self._keyword_route(message)

    def _answer_help(self, message: str) -> dict:
        hits = self.index.search(message, k=4)
        evidence = [{"n": i + 1, "text": h.text, "score": h.score} for i, h in enumerate(hits)]
        if not hits:
            body = "I don't have that in the guide yet. Try the dashboard, or ask about a specific action."
            return {"intent": "help", "text": body, "evidence": [], "model": self.model.info().name}
        context = "\n\n".join(f"[{e['n']}] {e['text']}" for e in evidence)
        if not self._live:
            # extractive offline answer: lead with the top passage, cite it
            body = f"{hits[0].text}\n\n(Grounded in the {self.tenant} guide [1].)"
            return {"intent": "help", "text": body, "evidence": evidence, "model": self.model.info().name}
        system = (
            f"You are the assistant for {self.tenant}. Answer the user's question using ONLY the numbered "
            "guide passages, citing them inline like [1]. For a 'what is' / explain question, define the term "
            "AND explain how it fits this system from the passage context — keep the relevant specifics, don't "
            "collapse it to a bare one-line definition. For a how-to question, give the concrete steps. Keep it "
            "tight but complete; if the passages don't cover it, say so plainly."
        )
        prompt = f"Guide:\n{context}\n\nQuestion: {message}"
        try:
            res = self.model.complete(
                ModelRequest(messages=({"role": "user", "content": prompt},), system=system, max_tokens=600)
            )
            return {"intent": "help", "text": res.text or hits[0].text, "evidence": evidence,
                    "model": res.model, "cost_usd": res.est_cost_usd}
        except Exception:  # noqa: BLE001 — model outage → fall back to the top grounded passage
            return {"intent": "help", "text": f"{hits[0].text}\n\n(Grounded in the {self.tenant} guide [1].)",
                    "evidence": evidence, "model": self.model.info().name}

    def _effective_policy(self):
        from ..policy import current_policy
        return self._policy or current_policy()

    def _denied_by_authorization(self, name: str, result) -> bool:
        """Whether a failed result is the AUTHORIZATION gate refusing, not the APPROVAL gate holding the call.

        The two answers read differently to a user — "you are not allowed to do that" versus "this needs a
        confirmation first" — and conflating them tells someone with no grant that pressing confirm would help.

        :mod:`agent_harness.tools` has never carried a ``denied`` flag on ``ToolResult``, so asking for one
        raised ``AttributeError`` and a denied console call answered 500 instead of its reason. The flag is used
        when a result has one; otherwise the registry's own audit trail is read: an authorization outcome is the
        entry that carries the plane's attribution (``policy_reason``/``plane``), while an approval hold writes
        only the four keys it has always written. No entry at all is not a denial — an unregistered tool, a tool
        that crashed, and a policy block each failed somewhere else, and say so.
        """
        explicit = getattr(result, "denied", None)
        if explicit is not None:
            return bool(explicit)
        if not (result.error or "").startswith("denied:"):
            return False
        for entry in reversed(self.registry.audit):
            if entry.get("tool") != name:
                continue
            return entry.get("allowed") is False and "policy_reason" in entry
        return False

    def _answer_tool(self, name: str, args: dict, principal=None) -> dict:
        # Nothing is registered in a diagnostic console, so this is unreachable by accident of construction —
        # and an answer this important is not left to an accident. Explicit refusal, not a lookup miss.
        if not self.privileged:
            raise PrivilegedExecutionUnavailable(
                f"{name} was refused: this agent console is a diagnostic instance, so privileged execution is "
                f"not available in it — no tool is registered. Run the execution instance instead.")
        # tool-phase policy: an approval rule pauses an irreversible action; a deny rule blocks it.
        policy = self._effective_policy()
        if policy is not None:
            td = policy.check(principal, "tool", (name, args), app=getattr(principal, "app", "") or self.app)
            if td.action in ("require_approval", "deny"):
                verb = "needs approval before it runs" if td.action == "require_approval" else "is blocked"
                return {"intent": "action", "tool": name, "text": f"This action {verb}: {td.reason}",
                        "evidence": [{"tool": name, "gate": td.action, "ok": False}], "data": None, "approved": False,
                        "policy": [{"decision": td.action, "phase": "tool", "reason": td.reason, "rule_id": td.rule_id,
                                    "scope": td.scope, "provider": td.provider, "summary": f"{name} → {td.action}"}]}
        result = self.registry.run(name, args, principal=principal)
        gate = self.registry.audit[-1] if self.registry.audit else {"decision": "read-only"}
        evidence = [{"tool": name, "args": args, "gate": gate.get("reason") or gate.get("decision", ""), "ok": result.ok}]
        summary = result.text or (json.dumps(result.data)[:800] if result.data is not None else "")
        if not result.ok:
            # A denial carries its own reason and must show it: "this needs confirmation" is what a held
            # side effect says, and it is the wrong thing to tell someone whose action was denied.
            if self._denied_by_authorization(name, result):
                body = summary or result.error or "That action is not authorized."
            else:
                body = summary or "That action needs confirmation before it can run."
            return {"intent": "action", "tool": name, "text": body, "evidence": evidence, "data": result.data, "approved": False}
        if self._live and summary:
            system = (
                f"You are the assistant for {self.tenant}. The user asked something and a tool returned the data "
                "below. Answer them directly and concisely from that data. Do not invent numbers."
            )
            prompt = f"Tool `{name}` returned:\n{summary}\n\nAnswer the user."
            try:
                res = self.model.complete(
                    ModelRequest(messages=({"role": "user", "content": prompt},), system=system, max_tokens=500)
                )
                body = res.text or summary
                model_name = res.model
            except Exception:  # noqa: BLE001 — model outage → return the raw tool summary
                body = summary
                model_name = self.model.info().name
        else:
            body = summary or "Done."
            model_name = self.model.info().name
        return {"intent": "action", "tool": name, "text": body, "evidence": evidence, "data": result.data, "approved": True, "model": model_name}

    def respond(self, message: str, principal=None) -> dict:
        message = (message or "").strip()
        # 1) command path — deterministic, no LLM, permission-gated (dual-path dispatch)
        if self.commands is not None and self.commands.is_command(message):
            return self.commands.dispatch(message, principal)
        if not message:
            return {"intent": "help", "text": "Ask me anything about " + self.tenant + ".", "evidence": []}
        policy = self._effective_policy()
        app = getattr(principal, "app", "") or self.app
        decisions: list[tuple] = []
        # 2) input policy (guardrails)
        if policy is not None:
            di = policy.check(principal, "input", message, app=app)
            decisions.append((di, "input"))
            if di.action == "deny":
                return {"intent": "blocked", "text": f"Blocked by policy: {di.reason}", "evidence": [],
                        "policy": _policy_items(decisions)}
            if di.action == "redact":
                message = policy.redact(message, di)
        # 3) route (help / tool). tool-phase policy is enforced inside _answer_tool.
        route = self.classify(message)
        if route["mode"] == "tool" and route["tool"] in self._tools:
            out = self._answer_tool(route["tool"], route.get("args") or {}, principal=principal)
        else:
            out = self._answer_help(message)
        out["route"] = route.get("reason", "")
        # 4) output policy (guardrails)
        if policy is not None:
            do = policy.check(principal, "output", out.get("text", ""), app=app)
            decisions.append((do, "output"))
            if do.action == "deny":
                return {"intent": "blocked", "text": f"Blocked by policy: {do.reason}", "evidence": [],
                        "policy": _policy_items(decisions)}
            if do.action == "redact":
                out["text"] = policy.redact(out["text"], do)
        # 5) attach compact policy summary (progressive disclosure — §10.1). Merge any tool-phase items.
        #    Only when a policy is active, so consoles without policy are byte-for-byte unchanged.
        if policy is not None:
            items = _policy_items(decisions) + [p for p in (out.get("policy") or []) if p.get("decision") != "allow"]
            out["policy"] = items
            out["policy_checks"] = len(decisions) + (1 if out.get("tool") else 0)
        return out

    # ---- the panel -----------------------------------------------------------

    def panel_html(self, mount: str = "agent", endpoint: str = "api/agent") -> str:
        chips = "".join(
            f'<button class="ac-chip" onclick="acAsk(this.textContent)">{_esc(s)}</button>' for s in self.suggestions
        )
        # ``mount`` becomes element ids and ``endpoint`` a script string: neither is escaped by the template, so a host
        # page that passes a request value through must not be able to add markup or script with it.
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", mount or ""):
            raise ValueError("mount must be an element id: a letter, then letters, digits, '_' or '-' (at most 64)")
        if not re.fullmatch(r"[A-Za-z0-9_./:?&=%~+-]{1,512}", endpoint or "") or endpoint.lower().startswith(("javascript:", "data:")):
            raise ValueError("endpoint must be a plain URL or path")
        return _PANEL_TMPL.format(
            mount=mount, endpoint=endpoint, tenant=_esc(self.tenant), subtitle=_esc(self.subtitle), chips=chips
        )


def _esc(s: str) -> str:
    return (
        str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


_PANEL_TMPL = """
<div id="{mount}" class="ac-panel">
  <div class="ac-head"><span class="ac-dot"></span><div><div class="ac-title">{tenant} · Assistant</div>
    <div class="ac-sub">{subtitle}</div></div></div>
  <div id="{mount}-log" class="ac-log"></div>
  <div class="ac-chips">{chips}</div>
  <form class="ac-form" onsubmit="acSend(event)">
    <input id="{mount}-in" class="ac-in" autocomplete="off" placeholder="Ask, or tell me what to do…"/>
    <button class="ac-send" type="submit">Send</button>
  </form>
</div>
<style>
  .ac-panel{{background:#141416;border:1px solid #2b2b30;border-radius:14px;padding:14px;color:#e7e5ea;
    font:14px/1.5 Roboto,system-ui,sans-serif;display:flex;flex-direction:column;gap:10px;max-width:720px}}
  .ac-head{{display:flex;gap:10px;align-items:center}}
  .ac-dot{{width:9px;height:9px;border-radius:50%;background:#4fd1c5;box-shadow:0 0 10px #4fd1c5}}
  .ac-title{{font-weight:600}} .ac-sub{{color:#9b99a1;font-size:12px}}
  .ac-log{{display:flex;flex-direction:column;gap:10px;max-height:52vh;overflow:auto;padding:2px}}
  .ac-msg{{padding:9px 12px;border-radius:12px;max-width:88%}}
  .ac-you{{align-self:flex-end;background:#2b3a49}} .ac-bot{{align-self:flex-start;background:#1d1d21;border:1px solid #2b2b30}}
  .ac-work{{margin-top:7px;font-size:12px;color:#8f8d95}}
  .ac-work summary{{cursor:pointer;color:#4fd1c5}}
  .ac-ev{{border-top:1px solid #2b2b30;padding:5px 0;white-space:pre-wrap}}
  .ac-chips{{display:flex;flex-wrap:wrap;gap:6px}}
  .ac-chip{{background:#1d1d21;border:1px solid #2b2b30;color:#c7c5ca;border-radius:20px;padding:5px 11px;
    font-size:12px;cursor:pointer}} .ac-chip:hover{{border-color:#4fd1c5}}
  .ac-form{{display:flex;gap:8px}}
  .ac-in{{flex:1;background:#0e0e10;border:1px solid #2b2b30;border-radius:10px;padding:10px;color:#e7e5ea}}
  .ac-send{{background:#4fd1c5;color:#08312d;border:0;border-radius:10px;padding:0 16px;font-weight:600;cursor:pointer}}
  .ac-tag{{display:inline-block;font-size:10px;text-transform:uppercase;letter-spacing:.5px;color:#08312d;
    background:#4fd1c5;border-radius:4px;padding:1px 6px;margin-right:6px}}
  .ac-tag.help{{background:#c7b3ff}} .ac-tag.deny{{background:#ff9b9b}}
</style>
<script>
(function(){{
  var M="{mount}",EP="{endpoint}";
  window.acAsk=function(t){{document.getElementById(M+"-in").value=t;acSend();}};
  window.acSend=function(e){{
    if(e&&e.preventDefault)e.preventDefault();
    var inp=document.getElementById(M+"-in"),q=(inp.value||"").trim();if(!q)return;inp.value="";
    add("you",esc(q));var wait=add("bot","…");
    fetch(EP,{{method:"POST",headers:{{"Content-Type":"application/json"}},body:JSON.stringify({{message:q}})}})
      .then(function(r){{return r.json();}}).then(function(d){{wait.remove();render(d);}})
      .catch(function(){{wait.remove();add("bot","(the assistant is offline right now)");}});
  }};
  function add(who,html){{var log=document.getElementById(M+"-log");var m=document.createElement("div");
    m.className="ac-msg ac-"+(who==="you"?"you":"bot");m.innerHTML=html;log.appendChild(m);
    log.scrollTop=log.scrollHeight;return m;}}
  function render(d){{
    var tag=d.intent==="action"?(d.approved===false?'<span class="ac-tag deny">needs ok</span>':'<span class="ac-tag">action'+(d.tool?" · "+esc(d.tool):"")+'</span>'):'<span class="ac-tag help">guide</span>';
    var html=tag+"<div>"+esc(d.text||"").replace(/\\n/g,"<br>")+"</div>";
    var ev=(d.evidence||[]);
    if(ev.length){{var rows=ev.map(function(e){{
      if(e.text!=null)return '<div class="ac-ev">['+esc(e.n)+'] '+esc(e.text)+'</div>';
      return '<div class="ac-ev">tool <b>'+esc(e.tool||"")+'</b> · gate: '+esc(e.gate||"")+' · '+(e.ok?"ok":"blocked")+'</div>';
    }}).join("");
      html+='<details class="ac-work"><summary>show work'+(d.model?" · "+esc(d.model):"")+'</summary>'+rows+'</details>';}}
    add("bot",html);
  }}
  function esc(s){{return String(s==null?"":s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}}
}})();
</script>
"""
