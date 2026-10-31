#!/usr/bin/env python3
"""Gate for the mandatory QA test job (stdlib only).

Runs ``tests/qa`` and fails, never skips, when any of these holds:

* ``runtime-contracts`` cannot be imported (the message carries the install hint);
* the pin the job fetched differs from the pin declared for the ``qa`` extra in ``pyproject.toml``;
* pytest exits non-zero (this includes "no tests collected", exit status 5);
* zero tests were collected, or a required test is absent from the report;
* any test was skipped or marked expected-failure. The allow-list of skips is ``ALLOWED_SKIPS`` and it is empty.

Run:  python scripts/qa_ci_gate.py [--tests-path tests/qa] [--pin <commit-or-tag>]
Exit status 0 means every condition passed.
"""
from __future__ import annotations

import argparse
import importlib
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Tests that may be skipped. Empty by design: a skipped QA test is a failure.
ALLOWED_SKIPS: frozenset[str] = frozenset()
#: Test names that must be present and passed, so that renaming or deleting them cannot weaken the gate.
REQUIRED_TESTS: tuple[str, ...] = ("test_qa_contracts_installed",)

INSTALL_HINT = ("runtime-contracts is not importable. Install the qa extra: pip install -e \".[qa]\" "
                "(or install the repository pinned in pyproject.toml).")

_QA_PIN = re.compile(r'^\s*qa\s*=\s*\[\s*"runtime-contracts @ git\+https://github\.com/redevops-io/'
                     r'runtime-contracts\.git@([^"\s]+)"', re.M)


def declared_pin(pyproject: Path) -> str | None:
    m = _QA_PIN.search(pyproject.read_text(encoding="utf-8"))
    return m.group(1) if m else None


def evaluate_junit(xml_text: str, *, allowed_skips: frozenset[str] = ALLOWED_SKIPS,
                   required: tuple[str, ...] = REQUIRED_TESTS) -> list[str]:
    """Problems found in a pytest junit report. An empty list means the report is acceptable."""
    problems: list[str] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        return [f"the junit report is not valid XML: {exc}"]
    cases = list(root.iter("testcase"))
    if not cases:
        return ["zero QA tests were collected"]
    passed: set[str] = set()
    for case in cases:
        ident = f"{case.get('classname', '')}::{case.get('name', '')}"
        name = case.get("name", "")
        failed = case.find("failure") is not None or case.find("error") is not None
        skipped = case.find("skipped") is not None
        if failed:
            problems.append(f"failed: {ident}")
        elif skipped:
            if ident not in allowed_skips and name not in allowed_skips:
                problems.append(f"skipped (not allowed): {ident}")
        else:
            passed.add(name.split("[", 1)[0])
    for name in required:
        if name not in passed:
            problems.append(f"required test did not pass or is missing: {name}")
    return problems


def run_gate(tests_path: str, *, pin: str | None, pyproject: Path = ROOT / "pyproject.toml") -> list[str]:
    problems: list[str] = []
    if pin:
        declared = declared_pin(pyproject)
        if declared != pin:
            problems.append(f"the fetched runtime-contracts pin ({pin}) differs from the pin declared for the qa "
                            f"extra in {pyproject.name} ({declared})")
    try:
        importlib.import_module("runtime_contracts.qa")
    except ImportError as exc:
        return [*problems, f"{INSTALL_HINT} ({exc})"]
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "qa-junit.xml"
        env = {**os.environ, "CR_QA_REQUIRED": "1"}
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", tests_path, "-p", "no:cacheprovider", "-q", "-rs",
             f"--junitxml={report}"], cwd=ROOT, env=env)
        if proc.returncode != 0:
            problems.append(f"pytest exited with status {proc.returncode}"
                            + (" (no tests collected)" if proc.returncode == 5 else ""))
        if report.exists():
            problems.extend(evaluate_junit(report.read_text(encoding="utf-8")))
        else:
            problems.append("pytest wrote no junit report")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tests-path", default="tests/qa")
    ap.add_argument("--pin", default=os.environ.get("RUNTIME_CONTRACTS_PIN"),
                    help="the commit or tag the job fetched (default: $RUNTIME_CONTRACTS_PIN)")
    args = ap.parse_args(argv)
    problems = run_gate(args.tests_path, pin=args.pin)
    for p in problems:
        print(f"qa-gate: FAIL: {p}", file=sys.stderr)
    print(f"qa-gate: {'FAILED' if problems else 'ok'} ({len(problems)} problem(s))")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
