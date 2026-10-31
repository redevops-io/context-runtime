"""The data-only boundary: Context Runtime consumes the runtime-contracts schemas and emits a canonical QAPlan. It never
imports agent_harness.qa, and the QA planning package imports nothing that executes a plan (no browser driver, no
runner, no harness)."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
from qa_subproc import MutantTree, REPO, run_py

PACKAGE = REPO / "context_runtime"
QA_PACKAGE = PACKAGE / "qa"
FORBIDDEN_EVERYWHERE = ("agent_harness.qa",)
#: Top-level modules the QA planning package may import besides the standard library.
ALLOWED_THIRD_PARTY = {"runtime_contracts", "context_runtime"}
EXECUTION_NAMES = ("playwright", "selenium", "pyppeteer", "agent_harness", "redevops_qa", "qa_runner", "qa_bridge")


def _names(tree: ast.AST):
    """Every module an import statement or a dynamic import by literal name refers to."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom):
            base = ("." * node.level) + (node.module or "")
            yield base
            if not node.level and node.module:
                for a in node.names:
                    yield f"{node.module}.{a.name}"
        elif isinstance(node, ast.Call):
            fn = node.func
            called = fn.id if isinstance(fn, ast.Name) else fn.attr if isinstance(fn, ast.Attribute) else ""
            if called in ("import_module", "__import__") and node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str):
                yield node.args[0].value
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("agent_harness.qa"):
            yield node.value


def violations(source: str, forbidden=FORBIDDEN_EVERYWHERE) -> list[str]:
    return sorted({n for n in _names(ast.parse(source)) if any(n == f or n.startswith(f + ".") for f in forbidden)})


def python_files(root: Path):
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


# ── the scanner itself ─────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("source", [
    "import agent_harness.qa",
    "import agent_harness.qa as q",
    "from agent_harness.qa import verifier",
    "from agent_harness.qa.verifier import classify",
    "from agent_harness import qa",
    "import importlib\nimportlib.import_module('agent_harness.qa.runner')",
    "__import__('agent_harness.qa')",
    "NAME = 'agent_harness.qa.verifier'",
])
def test_the_scanner_detects_every_way_of_reaching_agent_harness_qa(source):
    assert violations(source), source


@pytest.mark.parametrize("source", [
    "import agent_harness", "from agent_harness.tools import ToolSpec", "from agent_harness import tools",
    "import agent_harness_extras", "from .qa import closure", "from runtime_contracts import qa",
])
def test_the_scanner_does_not_flag_the_allowed_harness_tool_contract_or_lookalikes(source):
    assert violations(source) == [], source


# ── this repository ────────────────────────────────────────────────────────────────────────────────────────

def test_no_module_of_context_runtime_imports_agent_harness_qa():
    bad = {str(p.relative_to(REPO)): v for p in python_files(PACKAGE) if (v := violations(p.read_text()))}
    assert bad == {}, bad


def test_the_qa_package_imports_only_the_standard_library_the_contracts_and_this_package():
    assert python_files(QA_PACKAGE), "the QA package was not found"
    stdlib = set(sys.stdlib_module_names)
    bad = {}
    for p in python_files(QA_PACKAGE):
        for name in _names(ast.parse(p.read_text())):
            if name.startswith("."):
                continue
            top = name.split(".")[0]
            if top not in stdlib and top not in ALLOWED_THIRD_PARTY and top != "agent_harness.qa":
                bad.setdefault(str(p.relative_to(REPO)), set()).add(name)
    assert bad == {}, bad


def test_the_qa_package_imports_nothing_that_drives_a_browser_runs_a_plan_or_belongs_to_the_harness():
    bad = {str(p.relative_to(REPO)): v for p in python_files(QA_PACKAGE)
           if (v := violations(p.read_text(), EXECUTION_NAMES))}
    assert bad == {}, bad


def test_importing_the_qa_package_loads_no_agent_harness_qa_and_no_browser_driver():
    out = run_py("""
import sys
import context_runtime
before = set(sys.modules)
import context_runtime.qa
import context_runtime.qa.policy, context_runtime.qa.explain, context_runtime.qa.planner, context_runtime.qa.closure
new = sorted(set(sys.modules) - before)
bad = [m for m in new if m.split('.')[0] in ('playwright', 'selenium', 'pyppeteer', 'agent_harness', 'redevops_qa')]
print('BAD=' + ','.join(bad))
print('QA_LOADED=' + str(any(m.startswith('agent_harness.qa') for m in sys.modules)))
""")
    assert "BAD=\n" in out and "QA_LOADED=False" in out, out


def test_the_data_only_statement_is_documented():
    text = (REPO / "docs" / "qa_planning.md").read_text()
    assert "never imports `agent_harness.qa`" in text


# ── mutation: an import of agent_harness.qa is caught ──────────────────────────────────────────────────────

@pytest.mark.parametrize("relpath,old", [
    ("context_runtime/qa/closure.py", "from __future__ import annotations\n"),
    ("context_runtime/policy/plane.py", "from __future__ import annotations\n"),
])
def test_adding_an_agent_harness_qa_import_fails_the_boundary_tests(relpath, old):
    tree = MutantTree()
    try:
        tree.mutate(relpath, old, old + "import agent_harness.qa  # noqa\n")
        out = tree.run_tests("tests/qa/test_qa_import_boundary.py::test_no_module_of_context_runtime_imports_agent_harness_qa")
        assert out.returncode != 0
    finally:
        tree.close()
