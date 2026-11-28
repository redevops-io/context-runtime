"""An analytical question answered from document retrieval is marked as a fallback, never silently (mirrors the Go engine's plan.Extra["representation_fallback"])."""
from __future__ import annotations

import pytest

from context_runtime.planner import representations
from context_runtime.types import Constraints, Goal
from test_exactness_feasibility import ANALYTICAL_METHODS, FakeAnalytical, QUESTIONS, runtime

ANALYTICAL = [q for q, a in QUESTIONS if a]
OTHER = [q for q, a in QUESTIONS if not a]


@pytest.mark.parametrize("q", ANALYTICAL)
def test_with_no_exact_provider_an_analytical_question_is_marked_a_fallback(q):
    rt, _ = runtime()
    plan = rt.plan(Goal(text=q))
    note = plan.extra.get("representation_fallback")
    assert note and note["detected"] == "analytical" and note["exact"] is False and note["used"] != "analytical"


@pytest.mark.parametrize("q", ANALYTICAL)
def test_with_an_exact_provider_it_is_not_a_fallback_and_goes_to_sql(q):
    rt, _ = runtime(FakeAnalytical())
    plan = rt.plan(Goal(text=q))
    assert "representation_fallback" not in plan.extra and plan.chosen.steps[0].params["method"] in ANALYTICAL_METHODS


@pytest.mark.parametrize("q", OTHER)
def test_a_question_that_is_not_analytical_is_never_marked(q):
    rt, _ = runtime()
    assert "representation_fallback" not in rt.plan(Goal(text=q)).extra


@pytest.mark.parametrize("q", ANALYTICAL)
def test_a_required_exactness_is_refused_not_marked(q):
    from context_runtime.plugins.base import NoEligiblePlan
    rt, _ = runtime()
    with pytest.raises(NoEligiblePlan):
        rt.plan(Goal(text=q, constraints=Constraints(require_exact=True)))
