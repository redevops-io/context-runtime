"""Collection rules for the whole suite.

``tests/qa`` needs the ``qa`` extra (``runtime-contracts``). Where that package is installed it is collected like
any other directory. Where it is not (the general unit-test job installs no git extras) the directory is left out
of that run and the run says so in its summary. This never applies to a run that asks for the QA tests, whether by
naming ``tests/qa`` on the command line or by setting ``CR_QA_REQUIRED=1`` (the dedicated ``qa`` job does the
latter): there a missing package is a collection error, not an omission, and the job also refuses any skipped test.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

_QA_DIR = Path(__file__).resolve().parent / "qa"
_omitted = False


def _contracts_installed() -> bool:
    try:
        importlib.import_module("runtime_contracts")
    except ImportError:
        return False
    return True


def _qa_requested(config) -> bool:
    if os.environ.get("CR_QA_REQUIRED"):
        return True
    for arg in getattr(config, "args", ()) or ():
        try:
            path = Path(str(arg).split("::", 1)[0]).resolve()
        except OSError:
            continue
        if path == _QA_DIR or _QA_DIR in path.parents:
            return True
    return False


def pytest_ignore_collect(collection_path, config):
    global _omitted
    if Path(collection_path).resolve() == _QA_DIR and not _contracts_installed() and not _qa_requested(config):
        _omitted = True
        return True
    return None


def pytest_terminal_summary(terminalreporter):
    if _omitted:
        terminalreporter.write_line(
            "tests/qa was not collected: runtime-contracts is not installed in this environment. "
            "The dedicated `qa` job installs it and runs those tests.", yellow=True)
