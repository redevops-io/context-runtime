"""Typed operations behind the retriever seam, and the opt-in deterministic fast path. Inventory/finance only; the runtime names no domain."""
import re

import pytest

from context_runtime.adapters.model_stub import StubModel
from context_runtime.adapters.store_inmemory import InMemoryStore
from context_runtime.adapters.store_router import HopRouterRetriever
from context_runtime.adapters.store_typed import TypedOperation, TypedOperationRetriever
from context_runtime.runtime.config import Config
from context_runtime.runtime.runtime import ContextRuntime
from context_runtime.types import Constraints, Goal, Hit, PluginInfo
from context_runtime.verification.narration import Fact

STOCK = {"SKU-8841": {"north": 120, "south": 80}, "SKU-1002": {"north": 5}}
CALLS = []


def stock_op():
    def match(q):
        m = re.search(r"\b(SKU-\d{4})\b", q)
        return (m.group(1),) if m else None

    def run(sku):
        CALLS.append(sku)
        return {"sku": sku, "by_warehouse": STOCK.get(sku, {}), "total": sum(STOCK.get(sku, {}).values())}

    def render(r):
        return f"{r['sku']}: total {r['total']} units; " + ", ".join(f"{w}: {n}" for w, n in r["by_warehouse"].items())

    def facts(r):
        return [Fact("identifier", r["sku"]), Fact("number", str(r["total"]), "units", "total")] + [Fact("number", str(n), None, w) for w, n in r["by_warehouse"].items()]

    return TypedOperation("stock_by_warehouse", match, run, render, facts, provides=frozenset({"quantity"}))


VOCAB = {"quantity": r"\b(how many|units|stock|quantity)\b", "money": r"\b(price|cost|dollar|amount|value)\b"}


class Counting(StubModel):
    n = 0

    def complete(self, req):
        Counting.n += 1
        return super().complete(req)


class FallbackSQL:
    natural_query_methods = frozenset({"sql"})

    def __init__(self):
        self.queries = []

    def search(self, q, k, method="sql"):
        self.queries.append(q)
        return [Hit(chunk_id="sql:0", filename="w", text="n=1", score=1.0, source="analytical:fake", meta={"sql": "SELECT 1", "at_row_cap": False})]

    def info(self):
        return PluginInfo(name="fake-sql", kind="retriever", capabilities=frozenset({"sql", "exact"}))


def runtime(fast=True, fallback=None, **kw):
    prov = TypedOperationRetriever([stock_op()], fallback=fallback, vocabulary=VOCAB)
    Counting.n = 0
    CALLS.clear()
    models = {t: Counting(tier=t) for t in ("local", "cheap", "premium")}
    docs = InMemoryStore([{"chunk_id": "d", "filename": "f", "text": "stock", "created_at": None}])
    return ContextRuntime(models=models, retriever=HopRouterRetriever(single_hop=docs, graph=None, analytical=prov, on_unbound="abstain"),
                          config=Config(fast_path=fast), **kw), prov


EXACT = Constraints(require_exact=True)


def test_provider_declares_exact_and_typed_and_returns_provenance_bearing_hits():
    prov = TypedOperationRetriever([stock_op()], vocabulary=VOCAB)
    assert {"exact", "typed"} <= set(prov.info().capabilities)
    h = prov.search("How many units of SKU-8841 are in stock?", 5)[0]
    assert h.meta["op"] == "stock_by_warehouse" and h.meta["result"]["total"] == 200 and h.meta["at_row_cap"] is False


def test_unambiguous_covered_request_is_answered_without_a_model_call_or_a_plan():
    rt, _ = runtime()
    res = rt.run(Goal(text="How many units of SKU-8841 are in stock per warehouse?", sources=(), constraints=EXACT))
    assert res.path == "fast" and res.plan is None and Counting.n == 0
    assert "total 200 units" in res.answer and res.trace.spans[0].attrs["model_calls"] == 0 and CALLS == ["SKU-8841"]


def test_a_measure_the_operations_do_not_provide_is_not_covered_and_takes_the_full_path():
    rt, prov = runtime()
    q = "What is the total cost value of SKU-8841 in stock?"
    assert prov.resolve(q).covered is False and prov.resolve(q).uncovered == ("money",)
    res = rt.run(Goal(text=q, sources=(), constraints=EXACT))
    assert res.path == "full" and Counting.n >= 1                                    # LLM fallback for the ambiguous request


def test_not_taken_without_the_flag_without_require_exact_and_when_a_policy_is_configured():
    q = "How many units of SKU-8841 are in stock per warehouse?"
    assert runtime(fast=False)[0].run(Goal(text=q, sources=(), constraints=EXACT)).path == "full"
    assert runtime()[0].run(Goal(text=q, sources=(), constraints=Constraints())).path == "full"

    class Allow:
        def feasible(self, candidate, goal, score):
            return None
    from context_runtime.optimizer.knapsack import KnapsackOptimizer
    from context_runtime.costmodel.estimators import HeuristicEstimator
    rt, _ = runtime(optimizer=KnapsackOptimizer(HeuristicEstimator(), policy=Allow()))
    assert rt.run(Goal(text=q, sources=(), constraints=EXACT)).path == "full"      # the policy is evaluated per plan: no shortcut around it


def test_no_matching_operation_falls_through_to_the_fallback_provider_with_the_users_question():
    fb = FallbackSQL()
    rt, _ = runtime(fallback=fb)
    q = "How many invoices are more than 60 days overdue, per customer?"
    res = rt.run(Goal(text=q, sources=(), constraints=EXACT))
    assert res.path == "full" and fb.queries == [q]


def test_a_narration_that_changes_a_typed_figure_is_replaced_by_the_structured_result():
    class Wrong(StubModel):
        def complete(self, req):
            r = super().complete(req)
            from dataclasses import replace
            return replace(r, text="SKU-8841 has 210 units in total (north: 120, south: 80).")   # 200 -> 210

    rt, _ = runtime(fast=False)
    rt.models = {t: Wrong(tier=t) for t in ("local", "cheap", "premium")}
    res = rt.run(Goal(text="How many units of SKU-8841 are in stock per warehouse?", sources=(), constraints=EXACT))
    assert res.narration.status == "replaced" and "210" in res.narration.unsupported_numbers and "total 200 units" in res.answer
    assert any(s.name == "narration" for s in res.trace.spans)
