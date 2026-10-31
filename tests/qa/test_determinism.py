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


EXPLAIN_SCRIPT = """
import sys
sys.path.insert(0, {tests!r})
from qa_fixtures import random_dag
from context_runtime.qa import (CostBasedQAPlanner, CostModel, Estimate, PlanBudget, PlanRequest, RunCaps,
                                UiActionPolicy, default_action_policy_ref, default_ui_action_policy,
                                plan_with_explanation)
g = random_dag(33, 50, p_edge=0.1)
names = [f"t{{i:03d}}" for i in range(50)]
est = CostModel({{n: Estimate(40 + (i * 53) % 300, i % 5, (i * 777) % 10001) for i, n in enumerate(names)}})
req = PlanRequest(g, tuple(names), est, PlanBudget(max_parallel=5), RunCaps(deadline_ms=900000, max_recovery_attempts=1, recovery_ceiling=1),
                  action_policy_ref=default_action_policy_ref())
ex = plan_with_explanation(CostBasedQAPlanner(), req)
print(ex.plan.plan_hash)
print(ex.explanation.hash)
print(default_action_policy_ref())
up = UiActionPolicy(default_ui_action_policy(app="shop"), clock=lambda: "2026-01-01T00:00:00+00:00")
for i, effect in enumerate(["NONE", "WRITES_INTERNAL", "WRITES_EXTERNAL", "bogus"]):
    print(up.decide(principal=None, run_id="run-1", step_id=f"s-{{i}}", action_kind="click", side_effect=effect).decision_hash)
"""


@pytest.mark.parametrize("seed", ["0", "3", "99999", "random"])
@pytest.mark.parametrize("locale", ["C", "C.UTF-8", "tr_TR.UTF-8"])
def test_explanation_policy_reference_and_decision_hashes_are_identical_across_seeds_and_locales(seed, locale):
    from pathlib import Path

    code = EXPLAIN_SCRIPT.format(tests=str(Path(__file__).parent))
    reference = run_py(code, seed="0", locale="C")
    assert reference.count("rcv1:") == 7
    assert run_py(code, seed=seed, locale=locale) == reference
