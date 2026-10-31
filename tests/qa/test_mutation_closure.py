"""Mutation tests: take a guard out of a throw-away copy of the package and the closure tests must fail.

A test that still passes with its guard removed proves nothing about that guard. The control run (nothing removed)
must pass, so a failure of a mutant is down to the removal and not to the environment.
"""
from __future__ import annotations

import pytest
from qa_subproc import MutantTree

SUITE = ("tests/qa/test_closure.py",)


@pytest.fixture(scope="module")
def control():
    tree = MutantTree()
    try:
        out = tree.run_tests(*SUITE)
        assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-2000:]
        yield
    finally:
        tree.close()


def _mutant_fails(relpath: str, old: str, new: str):
    tree = MutantTree()
    try:
        tree.mutate(relpath, old, new)
        return tree.run_tests(*SUITE)
    finally:
        tree.close()


def test_removing_cycle_detection_fails_the_suite(control):
    out = _mutant_fails("context_runtime/qa/closure.py", "cycles = _find_cycles(declared)", "cycles = []")
    assert out.returncode != 0, "the closure tests still pass with cycle detection removed"


def test_removing_the_unknown_reference_check_fails_the_suite(control):
    out = _mutant_fails("context_runtime/qa/closure.py", "if target is None:", "if False:")
    assert out.returncode != 0


def test_removing_the_pin_check_fails_the_suite(control):
    out = _mutant_fails("context_runtime/qa/closure.py", "if ref.pin is not None and target.pin != ref.pin:",
                        "if False:")
    assert out.returncode != 0


def test_removing_conflict_detection_fails_the_suite(control):
    out = _mutant_fails("context_runtime/qa/closure.py",
                        "if c.resource in held and (held[c.resource] or c.exclusive):", "if False:")
    assert out.returncode != 0
