"""Slash-command framework (docs/policy-runtime.md §7). Dual-path dispatch at the input boundary:
input starting with ``/`` → a deterministic, no-LLM, permission-gated handler; otherwise → the agent loop.
Commands read/write the RuleStore (long-term memory) and reply with the resulting state. The permission
check (``can``) is injected — the enterprise ``command_gate`` satisfies it from the PermissionsPlane.
"""
from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Command:
    name: str
    run: Callable[[dict, object], dict]      # (args, principal) -> {"text", "data"?, "ok"?}
    description: str
    usage: str = ""
    requires: str = ""                        # capability/role; "" = any (unrestricted)
    aliases: tuple[str, ...] = ()


def parse_args(argstr: str) -> dict:
    """`--key value` / `--flag` → args[key]; the rest → args['text'] and comma-split args['items']."""
    args: dict = {"_raw": argstr}
    try:
        toks = shlex.split(argstr)
    except ValueError:
        toks = argstr.split()
    rest: list[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            key = t[2:]
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                args[key] = toks[i + 1]
                i += 2
            else:
                args[key] = True
                i += 1
        else:
            rest.append(t)
            i += 1
    text = " ".join(rest).strip()
    args["text"] = text
    args["items"] = [s.strip() for s in text.split(",") if s.strip()]
    return args


class CommandRegistry:
    def __init__(self, *, can: Callable[[object, str], bool] | None = None):
        self._cmds: dict[str, Command] = {}
        self._alias: dict[str, str] = {}
        self.can = can or (lambda principal, requires: not requires)
        self.audit: list[dict] = []

    def register(self, cmd: Command) -> Command:
        self._cmds[cmd.name] = cmd
        for a in cmd.aliases:
            self._alias[a] = cmd.name
        return cmd

    def register_all(self, cmds) -> None:
        for c in cmds:
            self.register(c)

    def is_command(self, text: str) -> bool:
        return (text or "").lstrip().startswith("/")

    def parse(self, text: str) -> tuple[str, str]:
        s = (text or "").lstrip()
        if s.startswith("/"):
            s = s[1:]
        head, _, rest = s.partition(" ")
        return head.strip().lower(), rest.strip()

    def _resolve(self, name: str) -> Command | None:
        return self._cmds.get(name) or self._cmds.get(self._alias.get(name, ""))

    def dispatch(self, text: str, principal=None) -> dict:
        name, argstr = self.parse(text)
        if name in ("help", ""):
            return self._help(principal)
        cmd = self._resolve(name)
        if cmd is None:
            return {"text": f"Unknown command /{name}. Try /help.", "ok": False}
        if cmd.requires and not self.can(principal, cmd.requires):
            self.audit.append({"cmd": name, "user": getattr(principal, "user", None), "allowed": False,
                               "requires": cmd.requires})
            return {"text": f"You don't have permission to run /{name} (requires: {cmd.requires}).", "ok": False,
                    "policy": [{"decision": "deny", "phase": "command", "reason": f"requires {cmd.requires}",
                                "summary": f"/{name} → denied"}]}
        args = parse_args(argstr)
        self.audit.append({"cmd": name, "user": getattr(principal, "user", None), "allowed": True, "args": args})
        out = cmd.run(args, principal)
        out.setdefault("ok", True)
        return out

    def _help(self, principal) -> dict:
        visible = [c for c in self._cmds.values() if not c.requires or self.can(principal, c.requires)]
        visible.sort(key=lambda c: c.name)
        lines = [f"/{c.name}{(' ' + c.usage) if c.usage else ''} — {c.description}" for c in visible]
        return {"text": "Commands you can run:\n" + "\n".join(lines) if lines else "No commands available.",
                "data": {"commands": [c.name for c in visible]}, "ok": True}

    def names(self) -> list[str]:
        return list(self._cmds)
