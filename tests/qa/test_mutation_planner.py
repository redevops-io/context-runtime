"""Mutation tests for the planner: remove a guarantee from a throw-away copy and the planner tests must fail.

The control run (nothing removed) must pass, so a mutant's failure is down to the removal.
"""
from __future__ import annotations

import pytest
from qa_subproc import MutantTree

SUITE = ("tests/qa/test_qa_planner.py",)
PLANNER = "context_runtime/qa/planner.py"


@pytest.fixture(scope="module")
def control():
    tree = MutantTree()
    try:
        out = tree.run_tests(*SUITE)
        assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-2000:]
        yield
    finally:
        tree.close()


def _run(edits):
    tree = MutantTree()
    try:
        for relpath, old, new in edits:
            tree.mutate(relpath, old, new)
        return tree.run_tests(*SUITE)
    finally:
        tree.close()


def test_removing_the_conflict_key_check_fails_the_suite(control):
    out = _run([(PLANNER, "return [j for j in placed if closure.conflicts(test_id, j)]", "return []")])
    assert out.returncode != 0, "the planner tests still pass with the conflict-key check removed"


def test_removing_conflict_detection_from_the_closure_fails_the_suite(control):
    out = _run([("context_runtime/qa/closure.py",
                 "if c.resource in held and (held[c.resource] or c.exclusive):", "if False:")])
    assert out.returncode != 0


def test_removing_the_full_coverage_validator_fails_the_suite(control):
    out = _run([(PLANNER, "def _require_full_coverage(plan, closure: Closure, problems: list[Problem]) -> None:",
                 "def _require_full_coverage(plan, closure: Closure, problems: list[Problem]) -> None:\n    return")])
    assert out.returncode != 0, "nothing notices when the coverage check is a no-op"


def test_a_planner_that_drops_a_test_and_has_no_coverage_check_fails_the_suite(control):
    out = _run([
        (PLANNER, "def _require_full_coverage(plan, closure: Closure, problems: list[Problem]) -> None:",
         "def _require_full_coverage(plan, closure: Closure, problems: list[Problem]) -> None:\n    return"),
        (PLANNER, "    for n, t in enumerate(run.order):\n        ready, conflict, starts, bound = run.detail[t]",
         "    for n, t in enumerate(run.order[:-1]):\n        ready, conflict, starts, bound = run.detail[t]"),
    ])
    assert out.returncode != 0


def test_accepting_a_subset_request_fails_the_suite(control):
    out = _run([(PLANNER, 'if value != "FULL":', "if False:")])
    assert out.returncode != 0


def test_trimming_instead_of_refusing_an_infeasible_wall_time_fails_the_suite(control):
    out = _run([(PLANNER, "chosen = min(feasible, key=lambda n: (runs[n].makespan, runs[n].lanes_used, n))",
                 "chosen = min(runs, key=lambda n: (runs[n].makespan, runs[n].lanes_used, n))"),
                (PLANNER, "if not feasible:", "if False:")])
    assert out.returncode != 0


def test_removing_the_pool_capacity_bound_fails_the_suite(control):
    out = _run([(PLANNER, "p_max = max(1, min(limits.values()))", "p_max = max(1, budget_parallel(request))"),
                (PLANNER, "def _limiting(", "def budget_parallel(request):\n    return request.budget.max_parallel\n\n\ndef _limiting(")])
    assert out.returncode != 0
