"""Action safety scan — a port of Hermes 0.17's ``sidekick.skills.scan_skill``.

Before an approval-gated action reaches a human, scan its prompt/payload for
dangerous shell patterns and attach human-readable findings to the Approval, so
the dashboard can show a ``⚠ safety`` badge and the approver sees the risk
*before* clicking approve.
"""
from __future__ import annotations

import re

# (compiled pattern, human-readable finding) — same spirit as sidekick.skills.
_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\brm\s+-rf?\b"), "recursive force-delete (rm -rf)"),
    (re.compile(r"(curl|wget)\s+[^|]*\|\s*(sudo\s+)?(ba)?sh"), "pipe remote download into shell"),
    (re.compile(r"\bsudo\b"), "privilege escalation (sudo)"),
    (re.compile(r":\(\)\s*\{\s*:\|:&\s*\}\s*;"), "fork bomb"),
    (re.compile(r"base64\s+-d[^|]*\|\s*(ba)?sh"), "decode-and-execute"),
    (re.compile(r"\b(nc|netcat|ncat)\b[^\n]*-e\b"), "reverse shell (netcat -e)"),
    (re.compile(r"\bdd\b[^\n]*of=/dev/(sd|nvme|vd)"), "raw block-device write"),
    (re.compile(r"\bgit\s+push\b[^\n]*(--force|-f)\b"), "force-push (git push -f)"),
    (re.compile(r">\s*/dev/(sd|nvme|vd)"), "raw block-device write"),
    (re.compile(r"\bchmod\s+-R?\s*777\b"), "world-writable permissions (chmod 777)"),
    (re.compile(r"\biptables\s+-F\b"), "flush all firewall rules (iptables -F)"),
    (re.compile(r"\bdrop\s+(table|database)\b", re.I), "destructive SQL (DROP TABLE/DATABASE)"),
]


def scan_text(text: str) -> list[str]:
    """Return a list of dangerous-pattern findings (empty = clean)."""
    if not text:
        return []
    found: list[str] = []
    for pat, label in _PATTERNS:
        if pat.search(text) and label not in found:
            found.append(label)
    return found


def scan_action(action: str, prompt: str = "", payload: dict | None = None) -> list[str]:
    """Scan everything a module action carries (action verb, prompt, payload values)."""
    blob = " ".join([action or "", prompt or ""])
    if payload:
        for v in payload.values():
            if isinstance(v, str):
                blob += " " + v
    return scan_text(blob)
