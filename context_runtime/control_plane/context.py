"""Shared business context + an append-only approvals / audit log.

The context is what every agent in the fleet knows about the business (profile, customers,
policies). The approvals log is the human-in-the-loop record: any action a module marks as
`approval_required` is recorded here as PENDING and only executed once approved.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterator

from . import safety


@dataclass
class Approval:
    id: str
    module: str
    action: str
    summary: str
    status: str = "pending"        # pending | approved | rejected
    payload: dict | None = None
    # Hermes 0.17 safety scan: dangerous-pattern findings shown as a ⚠ badge.
    findings: list[str] = field(default_factory=list)


class Context:
    """Durable, file-backed shared state. JSON on disk — no database required to start."""

    def __init__(self, home: str | Path = ".agentic-os", notifier=None):
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)
        self._profile_path = self.home / "business.json"
        self._approvals_path = self.home / "approvals.jsonl"
        # Optional Hermes 0.17 chat notifier — pings on approval request/resolve.
        self.notifier = notifier

    # --- business profile (shared knowledge) ---------------------------------
    def profile(self) -> dict[str, Any]:
        if self._profile_path.exists():
            return json.loads(self._profile_path.read_text(encoding="utf-8"))
        return {}

    def set_profile(self, **fields: Any) -> dict[str, Any]:
        p = self.profile()
        p.update(fields)
        self._profile_path.write_text(json.dumps(p, indent=2), encoding="utf-8")
        return p

    # --- approvals / audit log -----------------------------------------------
    def request_approval(self, module: str, action: str, summary: str, payload: dict | None = None) -> Approval:
        findings = safety.scan_action(action, (payload or {}).get("prompt", ""), payload)
        ap = Approval(id=self._next_id(), module=module, action=action, summary=summary,
                      payload=payload, findings=findings)
        self._append(ap)
        if self.notifier is not None:
            try:
                self.notifier.approval_requested(ap)
            except Exception:  # noqa: BLE001 — notifications are best-effort
                pass
        return ap

    def resolve(self, approval_id: str, approved: bool) -> Approval | None:
        rows = list(self._read())
        target = None
        for ap in rows:
            if ap.id == approval_id and ap.status == "pending":
                ap.status = "approved" if approved else "rejected"
                target = ap
        if target is not None:
            self._rewrite(rows)
            if self.notifier is not None:
                try:
                    self.notifier.approval_resolved(target)
                except Exception:  # noqa: BLE001
                    pass
        return target

    def pending(self) -> list[Approval]:
        return [ap for ap in self._read() if ap.status == "pending"]

    # --- storage helpers -----------------------------------------------------
    def _next_id(self) -> str:
        return f"ap-{sum(1 for _ in self._read()) + 1:05d}"

    def _append(self, ap: Approval) -> None:
        with self._approvals_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(ap)) + "\n")

    def _rewrite(self, rows: list[Approval]) -> None:
        # Write to a temp file and atomically rename over the log so a concurrent
        # reader never sees a half-written file (no partial/truncated rewrite).
        tmp_path = self._approvals_path.with_suffix(self._approvals_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as fh:
            for ap in rows:
                fh.write(json.dumps(asdict(ap)) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, self._approvals_path)

    def _read(self) -> Iterator[Approval]:
        if not self._approvals_path.exists():
            return iter(())
        out = []
        for line in self._approvals_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(Approval(**json.loads(line)))
        return iter(out)
