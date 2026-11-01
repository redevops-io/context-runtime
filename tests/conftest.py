"""Shared test setup: keep every test run self-contained.

* Import-time state: ``control_plane.app`` builds its fleet when it is imported and resolves its home
  from ``CONTEXT_RUNTIME_HOME`` at that moment, so the variable has to be set while pytest is still
  collecting, before any test module is imported. A session-wide temporary directory is used unless the
  caller already chose one.
* Per-test state: an autouse fixture points the home variables at the test's own ``tmp_path`` so no test
  reads another test's files or writes into the working directory.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_HOME_VARS = ("CONTEXT_RUNTIME_HOME", "AGENTIC_OS_HOME")
_SESSION_HOME = None
if not any(os.environ.get(v) for v in _HOME_VARS):
    _SESSION_HOME = tempfile.TemporaryDirectory(prefix="cr-test-home-")
    os.environ["CONTEXT_RUNTIME_HOME"] = _SESSION_HOME.name

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRAY = (".context-runtime", ".agentic-os")


@pytest.fixture(autouse=True)
def _per_test_home(tmp_path, monkeypatch):
    """A fresh state directory for every test (override inside a test with monkeypatch if needed)."""
    home = tmp_path / "cr-home"
    monkeypatch.setenv("CONTEXT_RUNTIME_HOME", str(home))
    monkeypatch.delenv("AGENTIC_OS_HOME", raising=False)
    yield home


@pytest.fixture(scope="session", autouse=True)
def _no_stray_state_in_cwd():
    """Fail the run if a test left a state directory in the working directory or the repository root."""
    places = sorted({Path.cwd(), _REPO_ROOT})
    before = {p / s for p in places for s in _STRAY if (p / s).exists()}
    yield
    created = [str(p / s) for p in places for s in _STRAY if (p / s).exists() and (p / s) not in before]
    assert not created, f"tests wrote state into the working tree: {created}"
