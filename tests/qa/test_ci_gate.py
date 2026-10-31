"""The mandatory QA job: it fails, and never skips, when its dependency is missing, when nothing is collected, or when
any QA test is skipped. The workflow cannot run in this suite; its definition and the gate script it calls are checked
here, and the gate is exercised end to end in child processes."""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml
from qa_subproc import REPO, child_env

SPEC = importlib.util.spec_from_file_location("qa_ci_gate", REPO / "scripts" / "qa_ci_gate.py")
gate = importlib.util.module_from_spec(SPEC)
sys.modules["qa_ci_gate"] = gate
SPEC.loader.exec_module(gate)

WORKFLOWS = REPO / ".github" / "workflows"


def junit(*cases: str) -> str:
    return "<testsuites><testsuite>" + "".join(cases) + "</testsuite></testsuites>"


def ok(name="test_qa_contracts_installed"):
    return f'<testcase classname="tests.qa.test_x" name="{name}"/>'


def test_the_skip_allow_list_is_empty():
    assert gate.ALLOWED_SKIPS == frozenset()


def test_report_with_all_tests_passing_is_acceptable():
    assert gate.evaluate_junit(junit(ok(), ok("test_other"))) == []


def test_zero_collected_tests_fail():
    assert gate.evaluate_junit(junit()) == ["zero QA tests were collected"]
    assert gate.evaluate_junit("<testsuites/>") == ["zero QA tests were collected"]


def test_a_skipped_test_fails():
    skipped = '<testcase classname="tests.qa.test_x" name="test_b"><skipped message="needs a thing"/></testcase>'
    problems = gate.evaluate_junit(junit(ok(), skipped))
    assert any(p.startswith("skipped (not allowed)") for p in problems)


def test_an_expected_failure_counts_as_a_skip():
    xfail = ('<testcase classname="tests.qa.test_x" name="test_b"><skipped type="pytest.xfail" message="x"/>'
             "</testcase>")
    assert gate.evaluate_junit(junit(ok(), xfail))


def test_a_failure_or_error_fails():
    bad = '<testcase classname="tests.qa.test_x" name="test_b"><failure message="boom"/></testcase>'
    err = '<testcase classname="tests.qa.test_x" name="test_c"><error message="boom"/></testcase>'
    assert any(p.startswith("failed") for p in gate.evaluate_junit(junit(ok(), bad)))
    assert any(p.startswith("failed") for p in gate.evaluate_junit(junit(ok(), err)))


def test_the_dedicated_install_test_must_be_present():
    problems = gate.evaluate_junit(junit(ok("test_other")))
    assert problems == ["required test did not pass or is missing: test_qa_contracts_installed"]


def test_an_allow_listed_skip_would_be_accepted_proving_the_list_is_what_decides():
    skipped = '<testcase classname="tests.qa.test_x" name="test_b"><skipped/></testcase>'
    assert gate.evaluate_junit(junit(ok(), skipped), allowed_skips=frozenset({"test_b"})) == []


def test_malformed_report_fails():
    assert gate.evaluate_junit("not xml")


def _workflow():
    return yaml.safe_load((WORKFLOWS / "tests.yml").read_text(encoding="utf-8"))


def test_workflow_defines_a_read_only_qa_job_that_cannot_be_skipped():
    job = _workflow()["jobs"]["qa"]
    assert job["permissions"] == {"contents": "read"}
    text = (WORKFLOWS / "tests.yml").read_text(encoding="utf-8")
    qa_text = text[text.index("\n  qa:"):]
    assert "if:" not in qa_text, "a conditional step or job could skip the check"
    assert "continue-on-error" not in qa_text
    assert "secrets." not in qa_text and "github.token" not in qa_text.lower() and "GITHUB_TOKEN" not in qa_text
    assert "persist-credentials: false" in qa_text
    runs = [s.get("run", "") for s in job["steps"]]
    assert any("scripts/qa_ci_gate.py" in r for r in runs)
    assert any("rev-parse HEAD" in r and "is not the pin" in r for r in runs), "HEAD must be compared to the pin"
    assert any("set -euo pipefail" in r for r in runs)


def test_workflow_pin_equals_the_pin_declared_for_the_qa_extra():
    text = (WORKFLOWS / "tests.yml").read_text(encoding="utf-8")
    pin = re.search(r"RUNTIME_CONTRACTS_PIN:\s*(\S+)", text).group(1)
    assert gate.declared_pin(REPO / "pyproject.toml") == pin
    assert re.fullmatch(r"[0-9a-f]{40}", pin)


def test_existing_unit_test_job_is_unchanged_in_substance():
    job = _workflow()["jobs"]["pytest"]
    runs = [s.get("run", "") for s in job["steps"] if "run" in s]
    assert runs[-1] == "python -m pytest tests -q -rs"
    assert 'python -m pip install -e ".[dev,test,control-plane,duckdb]" numpy pillow' in runs


def test_no_workflow_can_write_publish_or_tag():
    for wf in WORKFLOWS.glob("*.yml"):
        text = wf.read_text(encoding="utf-8")
        assert "contents: write" not in text and "packages: write" not in text, wf.name
        assert "tags:" not in text.split("jobs:", 1)[0], f"{wf.name} has a tag trigger"


def run_gate(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "scripts" / "qa_ci_gate.py"), *args], capture_output=True,
                          text=True, cwd=str(REPO), env=env or child_env(), timeout=300)


def _tests_dir(tmp_path: Path, body: str) -> str:
    d = tmp_path / "qa_probe"
    d.mkdir()
    (d / "test_probe.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return str(d)


PASSING = """
    def test_qa_contracts_installed():
        import runtime_contracts.qa
    def test_other():
        assert True
"""


def test_gate_passes_when_everything_holds(tmp_path):
    out = run_gate("--tests-path", _tests_dir(tmp_path, PASSING))
    assert out.returncode == 0, out.stdout + out.stderr


def test_gate_fails_on_a_skipped_test(tmp_path):
    body = PASSING + """
    import pytest
    @pytest.mark.skip(reason="not today")
    def test_skipped():
        pass
"""
    out = run_gate("--tests-path", _tests_dir(tmp_path, body))
    assert out.returncode == 1 and "skipped (not allowed)" in out.stderr


def test_gate_fails_when_zero_tests_are_collected(tmp_path):
    out = run_gate("--tests-path", _tests_dir(tmp_path, "x = 1\n"))
    assert out.returncode == 1 and "zero QA tests were collected" in out.stderr


def test_gate_fails_on_a_failing_test(tmp_path):
    out = run_gate("--tests-path", _tests_dir(tmp_path, PASSING + "\ndef test_bad():\n    assert False\n"))
    assert out.returncode == 1 and "failed" in out.stderr


def test_gate_fails_on_a_pin_that_differs_from_the_declared_one(tmp_path):
    out = run_gate("--tests-path", _tests_dir(tmp_path, PASSING), "--pin", "0" * 40)
    assert out.returncode == 1 and "differs from the pin declared" in out.stderr


def _hidden_contracts_env(tmp_path: Path) -> dict:
    stub = tmp_path / "hide"
    (stub / "runtime_contracts").mkdir(parents=True)
    (stub / "runtime_contracts" / "__init__.py").write_text(
        "raise ImportError('runtime_contracts is not installed (simulated)')\n", encoding="utf-8")
    env = child_env()
    env["PYTHONPATH"] = os.pathsep.join([str(stub), env["PYTHONPATH"]])
    return env


def test_gate_fails_when_runtime_contracts_is_missing(tmp_path):
    out = run_gate(env=_hidden_contracts_env(tmp_path))
    assert out.returncode == 1
    assert "runtime-contracts is not importable" in out.stderr and "[qa]" in out.stderr


def test_the_qa_tests_fail_to_collect_without_runtime_contracts(tmp_path):
    env = _hidden_contracts_env(tmp_path)
    env["CR_QA_REQUIRED"] = "1"
    out = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q",
                          str(REPO / "tests" / "qa" / "test_contracts_installed.py")], capture_output=True, text=True,
                         cwd=str(REPO), env=env, timeout=300)
    assert out.returncode != 0 and "skipped" not in out.stdout.lower().split("passed")[-1]
    assert "runtime_contracts is not installed" in out.stdout + out.stderr


def test_the_general_suite_omits_tests_qa_only_when_contracts_is_absent_and_not_requested(tmp_path):
    env = _hidden_contracts_env(tmp_path)
    env.pop("CR_QA_REQUIRED", None)
    cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "--co", "-q", "--continue-on-collection-errors"]
    plain = subprocess.run([*cmd, "tests"], capture_output=True, text=True, cwd=str(REPO), env=env, timeout=300)
    assert "tests/qa/" not in plain.stdout and "tests/qa was not collected" in plain.stdout
    named = subprocess.run([*cmd, "tests/qa"], capture_output=True, text=True, cwd=str(REPO), env=env, timeout=300)
    assert named.returncode != 0 and "tests/qa was not collected" not in named.stdout
    env["CR_QA_REQUIRED"] = "1"
    required = subprocess.run([*cmd, "tests"], capture_output=True, text=True, cwd=str(REPO), env=env, timeout=300)
    assert required.returncode != 0 and "tests/qa was not collected" not in required.stdout
    assert "runtime_contracts is not installed" in required.stdout + required.stderr
