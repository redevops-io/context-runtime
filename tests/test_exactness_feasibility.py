"""Execution-correctness: exactness/completeness are HARD requirements checked before cost; no silent top-k fallback; typed limitations.

Cross-domain on purpose (finance, procurement, inventory, temporal): nothing here is domain- or customer-specific. The fake analytical
provider records the exact query text it receives, so the tests also pin that text-to-SQL gets the user's words (not the plan-cache key).
"""
from __future__ import annotations

import pytest

from context_runtime.adapters.model_stub import StubModel
from context_runtime.adapters.store_inmemory import InMemoryStore
from context_runtime.adapters.store_router import HopRouterRetriever
from context_runtime.plancache.cache import build_key
from context_runtime.plugins.base import NoEligiblePlan
from context_runtime.planner import representations
from context_runtime.runtime.runtime import ContextRuntime
from context_runtime.types import Constraints, Goal, Hit, PluginInfo

_TIERS = ("local", "cheap", "premium")


class FakeAnalytical:
    natural_query_methods = frozenset({"sql", "mongo", "elastic", "logs", "api"})

    def __init__(self, rows=1, at_cap=False):
        self.queries, self.rows, self.at_cap = [], rows, at_cap

    def search(self, query, k, method="sql"):
        self.queries.append(query)
        return [Hit(chunk_id=f"sql:{i}", filename="fake:analytical", text=f"n={i}", score=1.0, source="analytical:fake",
                    meta={"row": {"n": i}, "sql": "SELECT 1", "at_row_cap": self.at_cap}) for i in range(self.rows)]

    def info(self):
        return PluginInfo(name="fake-analytical", kind="retriever", capabilities=frozenset({"sql", "api", "exact"}))


class CountingModel(StubModel):
    calls = 0

    def complete(self, req):
        CountingModel.calls += 1
        return super().complete(req)


DOCS = [{"chunk_id": f"d{i}", "filename": "f", "text": f"supplier invoice stock price order {i}", "created_at": None} for i in range(5)]


def runtime(analytical=None, **kw):
    models = {t: CountingModel(tier=t) for t in _TIERS}
    doc = InMemoryStore(DOCS)
    retr = HopRouterRetriever(single_hop=doc, graph=None, analytical=analytical, on_unbound="abstain") if analytical is not None else doc
    return ContextRuntime(models=models, retriever=retr, **kw), analytical


# finance / procurement / inventory / temporal; the second column is whether the shipped hint rules call it analytical
QUESTIONS = [
    ("What is the total amount invoiced to customer ACME-17 in fiscal year 2024, broken down by quarter?", True),
    ("How many invoices are more than 60 days overdue, per customer?", True),
    ("Sum the refunds issued between 2024-01-01 and 2024-03-31.", True),
    ("Which supplier submitted the lowest quote for RFQ-5521 and what is the spread to the highest quote?", False),
    ("How many purchase orders did each buyer raise last quarter?", True),
    ("What is the average unit price paid for item SKU-883 across all purchase orders?", True),
    ("How many units of SKU-883 are on hand in warehouse W2 as of 2024-06-30?", True),
    ("List the top 10 items by quantity shipped last month.", True),
    ("What was the stock level of SKU-883 at the end of each month in 2023?", False),
    ("What was the unit price of SKU-883 on 2024-02-14, before the March price change?", False),
    ("As of 2024-06-30, how many open purchase orders existed and what was their total value?", True),
    ("What changed in the supplier list between 2023-12-31 and 2024-12-31?", False),
]
ANALYTICAL_METHODS = set(representations.methods_for("analytical"))


@pytest.mark.parametrize("q,_hint", QUESTIONS)
def test_require_exact_never_selects_a_document_method(q, _hint):
    rt, _ = runtime(FakeAnalytical())
    plan = rt.plan(Goal(text=q, sources=(), constraints=Constraints(require_exact=True)))
    method = plan.chosen.steps[0].params["method"]
    assert method in ANALYTICAL_METHODS and "exact" in rt._method_capabilities(method)
    assert all(s.type != "rerank" for s in plan.chosen.steps)


@pytest.mark.parametrize("q,_hint", QUESTIONS)
def test_require_exact_without_an_exact_provider_is_a_typed_limitation_not_a_topk_answer(q, _hint):
    rt, _ = runtime(analytical=None)                  # document store only
    with pytest.raises(NoEligiblePlan):
        rt.plan(Goal(text=q, sources=(), constraints=Constraints(require_exact=True)))
    CountingModel.calls = 0
    res = rt.run(Goal(text=q, sources=(), constraints=Constraints(require_exact=True)))
    assert res.answer == "" and res.limitation is not None and res.limitation.kind == "no_eligible_plan"
    assert CountingModel.calls == 0                     # no model call, no prose guess


@pytest.mark.parametrize("q,hinted", QUESTIONS)
def test_analytical_representation_is_not_shadowed_by_the_temporal_or_multihop_bucket(q, hinted):
    """D-1: for a question the intent analyser puts in the analytical representation, the estimator must prefer the analytical method
    over the document fallback whatever the bucket ("as of ..." -> temporal, "per ..." -> multi_hop)."""
    rt, _ = runtime(FakeAnalytical())
    g = Goal(text=q, sources=(), constraints=Constraints())
    intent = rt.intent.analyze(g)
    plan, _, _ = rt._make_plan(g)
    if getattr(intent, "representation", None) == "analytical":
        assert plan.chosen.steps[0].params["method"] in ANALYTICAL_METHODS


def test_default_constraints_are_unchanged_and_not_exact():
    c = Constraints()
    assert (c.require_exact, c.require_complete) == (False, False)
    g = Goal(text="how many orders per month", sources=(), constraints=c)
    rt, _ = runtime(FakeAnalytical())
    assert rt.plan(g).chosen.steps[0].params["method"] in ANALYTICAL_METHODS | {"hybrid", "vector", "bm25"}


def test_the_plan_cache_never_replays_a_plan_across_the_exactness_requirement():
    q = "How many invoices are overdue per customer?"
    rt, _ = runtime(analytical=None)
    rt.plan(Goal(text=q, sources=(), constraints=Constraints()))                     # caches a document plan
    with pytest.raises(NoEligiblePlan):
        rt.plan(Goal(text=q, sources=(), constraints=Constraints(require_exact=True)))
    g1 = Goal(text=q, sources=(), constraints=Constraints())
    g2 = Goal(text=q, sources=(), constraints=Constraints(require_exact=True))
    intent = rt.intent.analyze(g1)
    assert build_key(intent, g1) != build_key(intent, g2)


def test_empty_exact_result_is_a_typed_limitation_with_no_model_call():
    rt, fake = runtime(FakeAnalytical(rows=0))
    CountingModel.calls = 0
    res = rt.run(Goal(text="How many invoices are overdue per customer?", sources=(), constraints=Constraints(require_exact=True)))
    assert res.limitation and res.limitation.kind == "empty_exact_result" and res.answer == "" and CountingModel.calls == 0


def test_a_result_cut_at_the_row_cap_is_incomplete_evidence_when_completeness_is_required():
    q = "List the top 10 items by quantity shipped last month."
    rt, _ = runtime(FakeAnalytical(rows=3, at_cap=True))
    res = rt.run(Goal(text=q, sources=(), constraints=Constraints(require_complete=True)))
    assert res.limitation and res.limitation.kind == "incomplete_evidence"
    rt2, _ = runtime(FakeAnalytical(rows=3, at_cap=True))
    ok = rt2.run(Goal(text=q, sources=(), constraints=Constraints(require_exact=True)))        # exact but completeness not required
    assert ok.limitation is None and ok.answer


def test_text_to_sql_receives_the_users_words_and_document_retrieval_still_receives_the_normalised_key():
    q = "As of 2024-06-30, how many open purchase orders existed and what was their total value?"
    rt, fake = runtime(FakeAnalytical())
    rt.run(Goal(text=q, sources=(), constraints=Constraints(require_exact=True)))
    assert fake.queries == [q]                                                              # word order and the date intact (D-3)
    rt_doc, _ = runtime(analytical=None)
    seen = []
    orig = rt_doc.retriever.search
    rt_doc.retriever.search = lambda query, k, method="hybrid": (seen.append(query), orig(query, k, method))[1]
    rt_doc.run(Goal(text="What does the supplier invoice say about stock price?", sources=(), constraints=Constraints()))
    assert seen and seen[0] == rt_doc.intent.analyze(Goal(text="What does the supplier invoice say about stock price?", sources=(),
                                                           constraints=Constraints())).normalized       # unchanged legacy behaviour


def test_a_plain_retriever_declaring_exact_is_eligible_and_one_that_does_not_is_not():
    class Plain(FakeAnalytical):
        pass
    rt = ContextRuntime(models={t: CountingModel(tier=t) for t in _TIERS}, retriever=Plain())
    assert rt.plan(Goal(text="total sales per region", sources=(), constraints=Constraints(require_exact=True))).chosen.steps[0].params["method"] in ANALYTICAL_METHODS
    rt2 = ContextRuntime(models={t: CountingModel(tier=t) for t in _TIERS}, retriever=InMemoryStore(DOCS))
    with pytest.raises(NoEligiblePlan):
        rt2.plan(Goal(text="total sales per region", sources=(), constraints=Constraints(require_exact=True)))
