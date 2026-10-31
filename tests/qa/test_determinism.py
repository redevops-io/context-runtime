"""Closure output is identical across hash seeds, locales and processes."""
from __future__ import annotations

import pytest
from qa_subproc import run_py

SCRIPT = """
import sys
sys.path.insert(0, {tests!r})
from qa_fixtures import random_dag
from context_runtime.qa import resolve
g = random_dag(11, 60)
c = resolve(g, ["t059", "t040", "t013"])
print(c.digest())
print(",".join(n.key for n in c.nodes))
print(";".join(f"{{t}}:{{','.join(c.exclusive_keys(t))}}" for t in c.test_ids))
"""


@pytest.mark.parametrize("seed", ["0", "1", "4242", "random"])
@pytest.mark.parametrize("locale", ["C", "C.UTF-8", "tr_TR.UTF-8"])
def test_closure_is_identical_across_hash_seeds_and_locales(seed, locale):
    from pathlib import Path

    code = SCRIPT.format(tests=str(Path(__file__).parent))
    reference = run_py(code, seed="0", locale="C")
    assert reference.count("\n") == 3
    assert run_py(code, seed=seed, locale=locale) == reference


PLAN_SCRIPT = """
import sys
sys.path.insert(0, {tests!r})
from qa_fixtures import random_dag
from context_runtime.qa import (CostBasedQAPlanner, CostModel, Estimate, PlanBudget, PlanRequest, RunCaps)
g = random_dag(21, 70, p_edge=0.08)
names = [f"t{{i:03d}}" for i in range(70)]
est = CostModel({{n: Estimate(50 + (i * 37) % 400, i % 7, (i * 911) % 10001) for i, n in enumerate(names)}})
req = PlanRequest(g, tuple(names), est, PlanBudget(max_parallel=6), RunCaps(deadline_ms=900000, max_recovery_attempts=2, recovery_ceiling=2))
r = CostBasedQAPlanner().plan_detailed(req)
print(r.plan.plan_hash)
print(r.request_digest)
print(r.chosen_lanes, r.makespan_ms)
print(";".join(f"{{s.test_id}}@{{s.lane}}#{{s.order}}:{{s.start_ms}}" for s in r.schedule))
"""


@pytest.mark.parametrize("seed", ["0", "7", "31337", "random"])
@pytest.mark.parametrize("locale", ["C", "C.UTF-8", "tr_TR.UTF-8"])
def test_plan_and_plan_hash_are_identical_across_hash_seeds_and_locales(seed, locale):
    from pathlib import Path

    code = PLAN_SCRIPT.format(tests=str(Path(__file__).parent))
    reference = run_py(code, seed="0", locale="C")
    assert reference.splitlines()[0].startswith("rcv1:")
    assert run_py(code, seed=seed, locale=locale) == reference
