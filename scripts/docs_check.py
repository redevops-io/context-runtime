#!/usr/bin/env python3
"""Documentation checks (stdlib only). Exit non-zero on any finding.

1. Relative links in *.md resolve to an existing file or directory (anchors and URLs are not fetched).
2. Fenced code blocks are balanced; ```bash/sh/shell blocks pass `bash -n` (syntax only, never executed).
3. No bare `pip install <ReDevOps package>` that implies a PyPI release (use git+ URL, -e ., or a path).
4. docs/evidence/*.yaml cards carry the required fields and a permitted status.

Run: python scripts/docs_check.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "build", "dist", "apps", "go", "uv.lock"}
LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE = re.compile(r"^(\s*)(```+|~~~+)\s*([A-Za-z0-9_+-]*)")
UNPUBLISHED = (
    "agentic-os", "redevops-mission", "runtime-contracts", "context-runtime", "discovery-runtime",
    "redevops-rag", "redevops-connectors", "knowledge-frontier", "sidekick", "agent-harness",
)
EVIDENCE_FIELDS = (
    "claim_id", "claim", "status", "code_commit", "dataset", "protocol", "model", "metrics",
    "baselines", "reproduce", "artifacts", "limitations", "review",
)
STATUSES = (
    "Illustrative/synthetic", "Internally reproduced", "Externally reproduced", "Field validated",
    "internally_reproduced?: UNVERIFIED",
)
BLOCKED_STATUSES = ("Externally reproduced", "Field validated")  # nothing here may carry these (none exist yet)

errors: list[str] = []


def md_files():
    for p in sorted(ROOT.rglob("*.md")):
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts[:-1]):
            continue
        yield p


def check_markdown(p: Path) -> None:
    text = p.read_text(encoding="utf-8", errors="replace")
    in_fence, fence_tok, lang, buf, start = False, "", "", [], 0
    for n, line in enumerate(text.splitlines(), 1):
        m = FENCE.match(line)
        if m and not in_fence:
            in_fence, fence_tok, lang, buf, start = True, m.group(2)[:3], m.group(3).lower(), [], n
            continue
        if in_fence:
            if line.strip().startswith(fence_tok):
                in_fence = False
                if lang in ("bash", "sh", "shell"):
                    check_shell("\n".join(buf), p, start)
            else:
                buf.append(line)
            continue
        for target in LINK.findall(line):
            if re.match(r"^(https?:|mailto:|#|tel:)", target):
                continue
            path = target.split("#", 1)[0]
            if path and not (p.parent / path).exists():
                errors.append(f"{p.relative_to(ROOT)}:{n}: broken relative link -> {target}")
    if in_fence:
        errors.append(f"{p.relative_to(ROOT)}:{start}: unterminated code fence")


def check_shell(code: str, p: Path, start: int) -> None:
    rel = p.relative_to(ROOT)
    r = subprocess.run(["bash", "-n"], input=code, capture_output=True, text=True)
    if r.returncode:
        errors.append(f"{rel}:{start}: bash -n failed: {r.stderr.strip().splitlines()[-1] if r.stderr else '?'}")
    for line in code.splitlines():
        m = re.match(r"^\s*pip3?\s+install\s+(.*)$", line)
        if not m:
            continue
        args = m.group(1)
        if "git+" in args or re.search(r"(^|\s)-e\s", args) or re.search(r"(^|\s)\.(\[|\s|$)", args + " ") \
                or "/" in args.split("#")[0].replace("'", "").replace('"', "").split("[")[0]:
            continue
        for name in UNPUBLISHED:
            if re.search(rf"[\s'\"]{re.escape(name)}([\[=<>~'\" ]|$)", " " + args):
                errors.append(f"{rel}:{start}: bare `pip install {name}` implies a PyPI release; use git+URL or -e .")
                break


def check_evidence() -> None:
    d = ROOT / "docs" / "evidence"
    if not d.is_dir():
        return
    for f in sorted(d.glob("*.yaml")):
        text = f.read_text(encoding="utf-8")
        for key in EVIDENCE_FIELDS:
            if not re.search(rf"^{key}:", text, re.M):
                errors.append(f"{f.relative_to(ROOT)}: missing field `{key}`")
        m = re.search(r"^status:\s*(.+?)\s*$", text, re.M)
        if m:
            status = m.group(1).strip("\"'")
            if not any(status == s or status.startswith(s) for s in STATUSES):
                errors.append(f"{f.relative_to(ROOT)}: status `{status}` is not a permitted class")
            if status in BLOCKED_STATUSES:
                errors.append(f"{f.relative_to(ROOT)}: status `{status}` requires external evidence not present")
        cid = re.search(r"^claim_id:\s*(\S+)", text, re.M)
        if cid and cid.group(1).strip("\"'") != f.stem:
            errors.append(f"{f.relative_to(ROOT)}: claim_id does not match filename")


def main() -> int:
    for p in md_files():
        check_markdown(p)
    check_evidence()
    for e in errors:
        print(e)
    print(f"docs_check: {len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
