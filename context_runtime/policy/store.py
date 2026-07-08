"""Rules = the long-term memory of the Policy Runtime (see docs/policy-runtime.md §5).

A ``Rule`` is a persisted, scoped, hard fact — distinct from chat history. Three tiers by scope:
``global`` (universal policy), ``<app>`` (app rules), ``<app>:<user>`` (user rules). Backed by JSONL files,
atomic writes, so a change is visible to the next request. This is the source of truth for v1; it can be
projected into the OLAP / snapshot layers later.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Rule:
    id: str                                     # stable content id (r-…) — the handle for modify/remove
    scope: str                                  # "global" | "<app>" | "<app>:<user>"
    kind: str                                   # "guardrail" | "approval" | "budget" | "target" | "rule" | …
    text: str                                   # the rule content (phrase/regex/instruction/limit)
    action: str = "deny"                        # allow | deny | redact | flag | require_approval
    match: dict = field(default_factory=dict)   # {"regex":bool,"phase":..,"tool":..,"pattern":..}
    meta: dict = field(default_factory=dict)
    created_by: str = ""
    at: float = 0.0

    @property
    def tier(self) -> str:
        return "global" if self.scope == "global" else ("user" if ":" in self.scope else "app")

    def to_dict(self) -> dict:
        return {"id": self.id, "scope": self.scope, "kind": self.kind, "text": self.text,
                "action": self.action, "match": self.match, "meta": self.meta,
                "created_by": self.created_by, "at": self.at}

    @classmethod
    def from_dict(cls, d: dict) -> "Rule":
        return cls(id=d["id"], scope=d["scope"], kind=d.get("kind", "rule"), text=d.get("text", ""),
                   action=d.get("action", "deny"), match=d.get("match", {}), meta=d.get("meta", {}),
                   created_by=d.get("created_by", ""), at=d.get("at", 0.0))


def rule_id(scope: str, kind: str, text: str) -> str:
    return "r-" + hashlib.sha1(f"{scope}|{kind}|{text}".encode("utf-8")).hexdigest()[:6]


class RuleStore:
    """Per-scope JSONL rule files under ``dir``. ``global.jsonl`` / ``<app>/app.jsonl`` /
    ``<app>/users/<user>.jsonl``."""

    def __init__(self, dir: str | None = None):
        self.dir = dir or os.getenv("POLICY_DIR", "/data/context-runtime/policy")

    # ── scope → file ──
    def _path(self, scope: str) -> Path:
        base = Path(self.dir)
        if scope == "global":
            return base / "global.jsonl"
        if ":" in scope:
            app, user = scope.split(":", 1)
            return base / _safe(app) / "users" / f"{_safe(user)}.jsonl"
        return base / _safe(scope) / "app.jsonl"

    def _read(self, scope: str) -> list[Rule]:
        p = self._path(scope)
        if not p.exists():
            return []
        out: list[Rule] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(Rule.from_dict(json.loads(line)))
                except Exception:  # noqa: BLE001
                    pass
        return out

    def _write(self, scope: str, rules: list[Rule]) -> None:
        p = self._path(scope)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(p) + ".tmp"
        Path(tmp).write_text("".join(json.dumps(r.to_dict()) + "\n" for r in rules), encoding="utf-8")
        os.replace(tmp, p)

    # ── CRUD ──
    def add(self, scope: str, kind: str, text: str, *, action: str = "deny", match: dict | None = None,
            meta: dict | None = None, created_by: str = "") -> Rule:
        rules = self._read(scope)
        rid = rule_id(scope, kind, text)
        rules = [r for r in rules if r.id != rid]        # idempotent by content
        rule = Rule(id=rid, scope=scope, kind=kind, text=text, action=action, match=match or {},
                    meta=meta or {}, created_by=created_by, at=time.time())
        rules.append(rule)
        self._write(scope, rules)
        return rule

    def list(self, *, scope: str | None = None, kind: str | None = None) -> list[Rule]:
        rules = self._read(scope) if scope is not None else self._all()
        return [r for r in rules if kind is None or r.kind == kind]

    def get(self, rule_id: str, *, scope: str | None = None) -> Rule | None:
        return next((r for r in self.list(scope=scope) if r.id == rule_id), None)

    def remove(self, rule_id: str, *, scope: str) -> bool:
        rules = self._read(scope)
        kept = [r for r in rules if r.id != rule_id]
        if len(kept) == len(rules):
            return False
        self._write(scope, kept)
        return True

    def modify(self, rule_id: str, *, scope: str, text: str | None = None, action: str | None = None,
               match: dict | None = None) -> Rule | None:
        rules = self._read(scope)
        for i, r in enumerate(rules):
            if r.id == rule_id:
                if text is not None:
                    r.text = text
                if action is not None:
                    r.action = action
                if match is not None:
                    r.match = match
                rules[i] = r
                self._write(scope, rules)
                return r
        return None

    def _all(self) -> list[Rule]:
        base = Path(self.dir)
        out: list[Rule] = []
        if not base.exists():
            return out
        for f in base.rglob("*.jsonl"):
            if f.name.endswith(".tmp"):
                continue
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        out.append(Rule.from_dict(json.loads(line)))
                    except Exception:  # noqa: BLE001
                        pass
        return out


def _safe(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9_-]+", "_", (s or "default").lower())


def scopes_for(principal, app: str) -> list[str]:
    """The rule scopes that apply to a request: global always, then the app, then the app:user tier."""
    out = ["global"]
    if app:
        out.append(app)
        user = getattr(principal, "user", "") or ""
        if user:
            out.append(f"{app}:{user}")
    return out
