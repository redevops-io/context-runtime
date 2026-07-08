"""Ready-made command sets over a RuleStore (docs/policy-runtime.md §8). Generic + reusable: the
permission `requires` strings are declared here; the injected `can` (enterprise `command_gate`) enforces
them. `policy_commands` manages a rule tier (global by default); `rule_commands` is the app-rule variant.
"""
from __future__ import annotations

from .commands import Command
from .store import RuleStore


def _fmt(rules) -> str:
    return "\n".join(f"{r.id} [{r.kind}·{r.action}] {r.text}" for r in rules) or "(none)"


def policy_commands(store: RuleStore, *, scope: str = "global", requires: str = "policy-admin",
                    default_kind: str = "guardrail", prefix: str = "policy",
                    read_aliases: tuple = (), write_aliases: tuple = ()) -> list[Command]:
    """`/show<prefix>` (read, open) + `/add<prefix>` `/remove<prefix>` `/modify<prefix>` (write, gated)."""

    def _show(args, p):
        rules = store.list(scope=scope, kind=args.get("kind"))
        return {"text": "Policy rules:\n" + _fmt(rules), "data": {"rules": [r.to_dict() for r in rules]}}

    def _add(args, p):
        text = args.get("text", "").strip()
        if not text:
            return {"text": f"Usage: /add{prefix} <text> [--kind guardrail|approval] "
                    "[--action deny|redact|require_approval] [--phase input|output] [--tool <name>] [--regex]",
                    "ok": False}
        match = {}
        for k in ("phase", "tool"):
            if args.get(k):
                match[k] = args[k]
        if args.get("regex"):
            match["regex"] = True
        r = store.add(scope, args.get("kind", default_kind), text, action=args.get("action", "deny"),
                      match=match, created_by=getattr(p, "user", "") or "")
        return {"text": f"Added {r.id} ({r.kind}·{r.action}): {r.text}", "data": {"id": r.id}}

    def _remove(args, p):
        rid = args.get("text", "").strip()
        ok = store.remove(rid, scope=scope)
        return {"text": f"Removed {rid}." if ok else f"No rule {rid} in this scope.", "ok": ok}

    def _modify(args, p):
        parts = args.get("text", "").split(" ", 1)
        if len(parts) < 2:
            return {"text": f"Usage: /modify{prefix} <id> <new text>", "ok": False}
        r = store.modify(parts[0].strip(), scope=scope, text=parts[1].strip())
        return {"text": f"Updated {parts[0]}." if r else f"No rule {parts[0]}.", "ok": bool(r)}

    return [
        Command(f"show{prefix}", _show, f"show the {scope} policy rules", requires="", aliases=read_aliases),
        Command(f"add{prefix}", _add, f"add a {scope} policy rule", usage="<text> [--kind …] [--action …]",
                requires=requires, aliases=write_aliases),
        Command(f"remove{prefix}", _remove, f"remove a {scope} policy rule by id", usage="<id>", requires=requires),
        Command(f"modify{prefix}", _modify, f"modify a {scope} policy rule", usage="<id> <new text>", requires=requires),
    ]


def global_policy_commands(store: RuleStore) -> list[Command]:
    """The universal Global-Policy set every app registers (privileged to write; read is open)."""
    return policy_commands(store, scope="global", requires="policy-admin",
                           read_aliases=("showguardrails",), write_aliases=("addguardrail",))
