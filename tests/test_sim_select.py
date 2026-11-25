"""Selecting from a Self-Improvement Memory never loses evidence. Records here are synthetic test data, not results."""
from __future__ import annotations

import copy
import random

import pytest

from context_runtime.learning import sim_select as ss

MEASURED = ("round", "sample", "files", "predicted", "fixed", "missed", "regressed", "passed", "total", "accepted",
            "backfilled", "evaluated", "candidate_harness", "provenance")


def attempt(rng, round_, tasks, *, regress=False, files=("prompts/system.md",)):
    predicted = rng.sample(tasks, k=min(3, len(tasks)))
    fixed = predicted[: rng.randrange(len(predicted) + 1)]
    return {"round": round_, "sample": 0, "what": f"edit in round {round_}: " + "describe the change in detail " * 12,
            "files": list(files), "rationale": "why this should help " * 15, "predicted": predicted, "fixed": fixed,
            "missed": [t for t in predicted if t not in fixed], "regressed": [rng.choice(tasks)] if regress else [],
            "passed": len(fixed), "total": len(predicted), "accepted": rng.random() < 0.4, "backfilled": False,
            "evaluated": True, "candidate_harness": f"h{round_}",
            "provenance": {"domain": "d", "benchmark": "b", "policy": "model-x", "harness_version": "h0", "optimizer": "o",
                           "evaluator": "e", "run": "r"}}


def record(seed=0, problems=12):
    rng = random.Random(seed)
    tasks = [f"task-{i}" for i in range(30)]
    out = []
    for n in range(1, problems + 1):
        out.append({"id": f"P{n}", "name": f"root cause number {n}: the parser drops field {n}", "opened_round": n,
                    "attempts": [attempt(rng, n + k, tasks, regress=rng.random() < 0.3, files=(f"tools/t{n}.py",))
                                 for k in range(1 + rng.randrange(3))]})
    return {"schema": ss.SCHEMA, "run": "r", "next_id": problems + 2, "carried_from": {"arm": "a", "domain": "d", "into": "r"},
            "aliases": {f"P{problems + 1}": "P2"}, "problems": out}


def need(rec, query):
    """The smallest budget that holds the mandatory evidence and the index."""
    with pytest.raises(ss.BudgetTooSmall) as e:
        ss.select(rec, query, budget_tokens=1)
    return e.value.needed


def test_the_complete_record_is_returned_unchanged_when_it_fits():
    rec = record()
    view, report = ss.select(rec, ss.Query.of(["task-1"]), budget_tokens=10**7)
    assert view == rec and report["complete"] and report["tokens"] == report["tokens_complete"] and "selection" not in view


@pytest.mark.parametrize("seed", range(25))
def test_no_view_loses_a_problem_an_attempt_or_anything_measured(seed):
    rec = record(seed)
    before = copy.deepcopy(rec)
    rng = random.Random(seed)
    query = ss.Query.of(rng.sample([f"task-{i}" for i in range(30)], 4), [f"tools/t{rng.randrange(1, 13)}.py"], "parser drops field")
    low, whole = need(rec, query), ss.estimate_tokens(ss.render(rec))
    for budget in (low, low + (whole - low) // 3, low + 2 * (whole - low) // 3, whole - 1):
        view, report = ss.select(rec, query, budget_tokens=budget)
        meta = view["selection"]
        assert rec == before, "the record was modified"
        assert ss.estimate_tokens(ss.render(view)) == report["tokens"] <= budget
        shown, omitted = {p["id"] for p in view["problems"]}, {o["id"] for o in meta["omitted"]}
        assert shown | omitted == {p["id"] for p in rec["problems"]} and not shown & omitted
        assert view["aliases"] == rec["aliases"] and view["next_id"] == rec["next_id"] and view["carried_from"] == rec["carried_from"]
        source = {p["id"]: p for p in rec["problems"]}
        for p in view["problems"]:
            assert p["name"] == source[p["id"]]["name"] and len(p["attempts"]) == len(source[p["id"]]["attempts"])
            for got, want in zip(p["attempts"], source[p["id"]]["attempts"]):
                assert {k: got[k] for k in MEASURED} == {k: want[k] for k in MEASURED}
                if p["id"] not in meta["compacted"]:
                    assert got == want
        for o in meta["omitted"]:
            attempts = source[o["id"]]["attempts"]
            assert o["name"] == source[o["id"]]["name"] and o["attempts"] == len(attempts)
            assert o["failed_predictions"] == sum(len(a["missed"]) for a in attempts)
            assert o["regressed_tasks"] == sorted({t for a in attempts for t in a["regressed"]})
        assert set(meta["mandatory"]) <= shown and meta["view_not_record"] and not meta["complete"]


def test_regression_evidence_that_bears_on_the_decision_is_always_shown():
    rec = record(3, problems=15)
    quiet = rec["problems"][9]                                       # otherwise irrelevant, and old
    quiet["name"] = "an unrelated cause"
    for a in quiet["attempts"]:
        a.update(predicted=["task-90"], fixed=[], missed=["task-90"], regressed=[], files=["other.py"], what="x", rationale="y")
    quiet["attempts"][0]["regressed"] = ["task-77"]
    query = ss.Query.of(["task-77"])
    view, report = ss.select(rec, query, budget_tokens=need(rec, query))
    assert quiet["id"] in report["mandatory"] and quiet["id"] in report["selected"]
    quiet["attempts"][0]["files"] = ["tools/under_review.py"]        # same through the file under consideration
    query = ss.Query.of(["task-1"], ["tools/under_review.py"])
    assert quiet["id"] in ss.select(rec, query, budget_tokens=need(rec, query))[1]["mandatory"]


def test_a_budget_that_cannot_hold_the_mandatory_evidence_is_an_error_not_a_smaller_view():
    rec = record(1)
    broken = next(t for p in rec["problems"] for a in p["attempts"] for t in a["regressed"])
    query = ss.Query.of([broken])
    low = need(rec, query)
    with pytest.raises(ss.BudgetTooSmall) as e:
        ss.select(rec, query, budget_tokens=low - 1)
    assert e.value.mandatory and e.value.needed == low
    ss.select(rec, query, budget_tokens=low)


def test_a_failed_prediction_about_a_task_that_still_fails_ranks_first():
    rec = record(5, problems=10)
    for p in rec["problems"]:
        for a in p["attempts"]:
            a.update(regressed=[], predicted=["task-1"], fixed=["task-1"], missed=[])
    target = rec["problems"][6]
    target["attempts"][0].update(predicted=["task-42"], fixed=[], missed=["task-42"])
    query = ss.Query.of(["task-42"])
    # exactly enough to show that one problem compacted, and nothing more
    exact = ss.estimate_tokens(ss.render(ss._assemble(rec, {target["id"]: "compact"}, ss.relationships(rec["problems"]), [])))
    view, report = ss.select(rec, query, budget_tokens=exact)
    assert report["selected"] == [target["id"]] and report["compacted"] == [target["id"]] and report["relevant_omitted"] == [], report
    view, report = ss.select(rec, query, budget_tokens=need(rec, query))      # too small for it: the report says what was lost
    assert report["relevant_omitted"] == [target["id"]]
    assert next(o for o in view["selection"]["omitted"] if o["id"] == target["id"])["failed_predictions"] >= 1


def test_relationships_are_stated_even_toward_an_omitted_problem():
    rec = record(2, problems=8)
    a, b = rec["problems"][0], rec["problems"][1]
    for p in rec["problems"]:
        for x in p["attempts"]:
            x.update(regressed=[], predicted=[f"only-{p['id']}"], fixed=[], missed=[f"only-{p['id']}"])
    a["attempts"][0]["regressed"] = ["shared"]
    b["attempts"][0].update(predicted=["shared"], fixed=["shared"], missed=[])
    query = ss.Query.of(["shared"])
    low = need(rec, query)
    view, report = ss.select(rec, query, budget_tokens=low)
    assert a["id"] in report["selected"]
    assert {"kind": "broke_task_of", "from": a["id"], "to": b["id"], "tasks": ["shared"]} in view["selection"]["relationships"]


def test_selection_is_deterministic_and_refuses_another_schema():
    rec = record(7)
    query = ss.Query.of(["task-3"], text="parser")
    budget = need(rec, query) + 800
    assert ss.select(rec, query, budget_tokens=budget) == ss.select(copy.deepcopy(rec), query, budget_tokens=budget)
    with pytest.raises(ss.SelectionError):
        ss.select({**rec, "schema": "something-else/1"}, query, budget_tokens=budget)
    with pytest.raises(ss.SelectionError):
        ss.select(rec, query, budget_tokens=0)


def test_a_view_loads_as_agent_harness_memory_when_that_version_is_installed():
    sim = pytest.importorskip("agent_harness.sim.memory", reason="the installed agent-harness predates the SIM record")
    rec = record(4)
    query = ss.Query.of(["task-5"])
    view, _ = ss.select(rec, query, budget_tokens=need(rec, query) + 1500)
    loaded = sim.SelfImprovementMemory.from_json(view)
    assert [p.id for p in loaded.problems] == view["selection"]["selected"]
