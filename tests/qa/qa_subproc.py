"""Run Python in a child process with a controlled hash seed and locale, and mutate a throw-away copy of the package."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def child_env(**extra: str) -> dict[str, str]:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": tempfile.gettempdir(),
           "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    env.update(extra)
    return env


def run_py(code: str, *, seed: str = "0", locale: str = "C", timeout: int = 120) -> str:
    env = child_env(PYTHONHASHSEED=seed, LC_ALL=locale, LANG=locale)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=timeout,
                         cwd=str(REPO))
    if out.returncode != 0:
        raise AssertionError(f"child failed ({out.returncode}):\n{out.stderr[-2000:]}")
    return out.stdout


class MutantTree:
    """A throw-away copy of ``context_runtime`` and the QA tests, with source edits applied.

    ``mutate`` replaces an exact snippet and insists it occurs exactly once, so a refactor that moves or renames the
    guard fails loudly instead of leaving a mutant that changes nothing.
    """

    def __init__(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="qa-mutant-"))
        shutil.copytree(REPO / "context_runtime", self.root / "context_runtime",
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(REPO / "tests" / "qa", self.root / "tests" / "qa", ignore=shutil.ignore_patterns("__pycache__"))
        for extra in ("conftest.py", "test_policy_ui_action.py"):
            if (REPO / "tests" / extra).exists():
                shutil.copy(REPO / "tests" / extra, self.root / "tests" / extra)
        if (REPO / "pyproject.toml").exists():
            shutil.copy(REPO / "pyproject.toml", self.root / "pyproject.toml")

    def mutate(self, relpath: str, old: str, new: str) -> None:
        path = self.root / relpath
        text = path.read_text(encoding="utf-8")
        assert text.count(old) == 1, f"{relpath}: expected exactly one occurrence of {old!r}, found {text.count(old)}"
        path.write_text(text.replace(old, new), encoding="utf-8")

    def run_tests(self, *targets: str, timeout: int = 300) -> subprocess.CompletedProcess:
        paths = [t for t in targets]
        env = child_env(PYTHONPATH=os.pathsep.join([str(self.root), *[p for p in sys.path if p and p != str(REPO)]]),
                        CR_QA_REQUIRED="1")
        return subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-x", *paths],
                              capture_output=True, text=True, env=env, timeout=timeout, cwd=str(self.root))

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)
