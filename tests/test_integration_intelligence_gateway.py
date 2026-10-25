"""Intelligence Gateway integration: the planner offers gateway retrieval only when the gateway clears it, execution never spends or reaches past the plan, and
rights, entitlement, spend and privacy refusals all end in "no gateway evidence", never a wider route. A fake gateway stands in for the HTTP API."""
from __future__ import annotations

import json

import pytest

from context_runtime.adapters.store_inmemory import InMemoryStore
from context_runtime.integrations.intelligence_gateway import (
    GatewayCandidateGenerator, GatewayClient, GatewayNeed, GatewayRetriever, NeedBook, gateway_source, needs_from_goal)
from context_runtime.planner.candidates_registry import RegistryCandidateGenerator
from context_runtime.runtime.runtime import ContextRuntime
from context_runtime.types import Constraints, Goal


class FakeGateway:
    """Models the parts of the real gateway the integration depends on: entitlement, quote, rights (451), caps and a price that can move."""

    def __init__(self, entitled=("company_profile",), price=0.05, rights_blocked=(), unavailable=(), cap=None):
        self.entitled, self.price, self.rights_blocked, self.unavailable, self.cap = list(entitled), price, set(rights_blocked), set(unavailable), cap
        self.calls: list[tuple[str, str, dict]] = []
        self.spent = 0.0

    def __call__(self, method, url, headers, body):
        path = url.split("//", 1)[1].split("/", 1)[1]
        req = json.loads(body) if body else {}
        self.calls.append((method, "/" + path, {"auth": headers.get("Authorization"), **req}))
        if headers.get("Authorization") != "Bearer tok":
            return 401, {"detail": "bad credential"}
        cap = req.get("capability")
        if path == "v1/capabilities":
            return 200, {"served": self.entitled, "entitled": self.entitled}
        if cap in self.rights_blocked:
            return 451, {"detail": "unavailable for rights reasons"}
        if path == "v1/quote":
            if cap in self.unavailable:
                return 200, {"available": False, "providers": [], "total": 0.0}
            return 200, {"available": True, "providers": ["p"], "provider_cost": self.price, "total": round(self.price * 1.15, 6)}
        if path == "v1/resolve":
            if self.cap is not None and self.spent + self.price > self.cap:
                return 402, {"detail": "monthly spend cap would be exceeded"}
            if float(req.get("max_cost") or 0.0) < self.price - 1e-9:            # the real gateway: a ceiling below the price (0 = spend nothing) buys nothing
                return 200, {"answer": "", "confidence": 0.0, "charge_total": 0.0, "provider_receipts": [], "decision_need_id": "dn-0", "fingerprint": "fp0"}
            self.spent += self.price
            return 200, {"answer": {"name": "Acme Ltd", "lei": "X1"}, "confidence": 0.9, "charge_total": round(self.price * 1.15, 6), "as_of": "2026-10-25T00:00:00Z",
                         "known_at": "2026-10-25T00:00:00Z", "fingerprint": "fp1", "decision_need_id": "dn-1",
                         "provider_receipts": [{"provider": "p", "cost": self.price, "ok": True}]}
        return 404, {}


def _rig(gw: FakeGateway, **kw):
    client = GatewayClient("https://gw.test", lambda: "tok", transport=gw)
    book = NeedBook()
    planner = GatewayCandidateGenerator(RegistryCandidateGenerator(), client, book, **kw)
    retr = GatewayRetriever(InMemoryStore([{"chunk_id": "d1", "filename": "local.md", "text": "acme local notes"}]), client, book)
    rt = ContextRuntime.default(candidates=planner)
    rt.retriever = retr
    return rt, planner, retr, gw


def _goal(text="what do we know about acme", cap="company_profile", **cons):
    return Goal(text, sources=(gateway_source(cap, subjects=["acme"], fields=["name"], purpose="onboarding check"),), constraints=Constraints(**cons))


def _methods(planner, goal):
    from context_runtime.planner.intent import RuleIntentAnalyzer
    intent = RuleIntentAnalyzer().analyze(goal)
    return {s.params["method"] for c in planner.generate(intent, goal) for s in c.steps if s.type == "retrieve"}


# ───────── the goal declares a need; nothing else changes ─────────

def test_a_goal_without_a_gateway_source_plans_and_calls_exactly_as_before():
    rt, planner, retr, gw = _rig(FakeGateway())
    goal = Goal("what do we know about acme")
    assert all(not m.startswith("gateway:") for m in _methods(planner, goal))
    assert gw.calls == []


def test_the_need_round_trips_through_the_source_and_a_malformed_one_is_no_need():
    g = _goal()
    n = needs_from_goal(g)
    assert n == [GatewayNeed("company_profile", ("acme",), ("name",), "onboarding check")]
    bad = Goal("x", sources=(gateway_source("company_profile"),))
    from dataclasses import replace
    assert needs_from_goal(Goal("x", sources=(replace(bad.sources[0], uri="gateway://need?max_cost=abc&capability=c"),))) == []


# ───────── planning: every check must clear ─────────

def test_a_cleared_need_offers_gateway_plans_and_prunes_the_rest_in_required_mode():
    rt, planner, retr, gw = _rig(FakeGateway())
    ms = _methods(planner, _goal())
    assert ms and all(m.startswith("gateway:") for m in ms)
    assert planner.decisions[0].offered and planner.decisions[0].quote_total == 0.0575


def test_augment_mode_keeps_the_base_plans_too():
    rt, planner, retr, gw = _rig(FakeGateway(), mode="augment")
    ms = _methods(planner, _goal())
    assert any(m.startswith("gateway:") for m in ms) and any(not m.startswith("gateway:") for m in ms)


@pytest.mark.parametrize("gw,cons,why", [
    (FakeGateway(entitled=()), {}, "not served or not entitled"),
    (FakeGateway(rights_blocked={"company_profile"}), {}, "rights-blocked"),
    (FakeGateway(unavailable={"company_profile"}), {}, "no provider available"),
    (FakeGateway(price=0.50), {"max_cost_usd": 0.10}, "over the budget"),
    (FakeGateway(), {"sensitivity": "restricted"}, "restricted goals do not use an external service"),
    (FakeGateway(), {"sensitivity": "internal"}, "internal goals do not use an external service"),
])
def test_any_refusal_removes_the_gateway_plan_and_the_goal_falls_back_to_local_plans(gw, cons, why):
    rt, planner, retr, gw = _rig(gw)
    ms = _methods(planner, _goal(**cons))
    assert ms and not any(m.startswith("gateway:") for m in ms)           # the base plans remain: the goal still runs locally
    assert why in planner.decisions[0].reason and not planner.decisions[0].offered


def test_restricted_goals_never_even_ask_the_gateway():
    rt, planner, retr, gw = _rig(FakeGateway())
    _methods(planner, _goal(sensitivity="restricted"))
    assert gw.calls == []


def test_internal_goals_can_opt_in():
    rt, planner, retr, gw = _rig(FakeGateway(), allow_internal=True)
    assert any(m.startswith("gateway:") for m in _methods(planner, _goal(sensitivity="internal")))


def test_a_gateway_that_is_down_or_has_no_credential_offers_nothing():
    down = GatewayClient("https://gw.test", lambda: "tok", transport=lambda *a: (0, {}))
    planner = GatewayCandidateGenerator(RegistryCandidateGenerator(), down, NeedBook())
    assert not any(m.startswith("gateway:") for m in _methods(planner, _goal()))

    def no_token():
        raise RuntimeError("no identity")
    nocred = GatewayClient("https://gw.test", no_token, transport=FakeGateway())
    planner = GatewayCandidateGenerator(RegistryCandidateGenerator(), nocred, NeedBook())
    assert not any(m.startswith("gateway:") for m in _methods(planner, _goal()))


def test_a_per_need_ceiling_is_honoured_even_with_no_goal_budget():
    rt, planner, retr, gw = _rig(FakeGateway(price=0.50))
    g = Goal("x", sources=(gateway_source("company_profile", subjects=["a"], max_cost=0.10),))
    assert not any(m.startswith("gateway:") for m in _methods(planner, g))


# ───────── privacy: what is sent ─────────

def test_the_goal_text_never_leaves_only_the_declared_need_does():
    gw = FakeGateway()
    rt, planner, retr, _ = _rig(gw)
    secret = "CONFIDENTIAL: our client acme owes us 4,000,000 and we plan to sue"
    rt.run(_goal(text=secret))
    sent = json.dumps(gw.calls)
    assert "CONFIDENTIAL" not in sent and "4,000,000" not in sent and "sue" not in sent
    body = [c[2] for c in gw.calls if c[1] == "/v1/resolve"][0]
    assert body["capability"] == "company_profile" and body["subject_refs"] == ["acme"] and body["question"] == ""


def test_no_provider_list_or_jurisdiction_is_ever_sent_so_the_caller_cannot_widen_rights():
    gw = FakeGateway()
    rt, *_ = _rig(gw)
    rt.run(_goal())
    for _m, _p, body in gw.calls:
        assert not ({"permitted_providers", "jurisdiction", "license_scope", "providers"} & set(body))


# ───────── execution ─────────

def test_end_to_end_the_gateway_answer_becomes_evidence_with_provenance_and_the_spend_is_recorded():
    rt, planner, retr, gw = _rig(FakeGateway())
    res = rt.run(_goal())
    assert gw.spent == 0.05 and retr.spent_usd == 0.0575
    assert retr.receipts[0]["decision_need_id"] == "dn-1"
    ctx = rt.build_context(rt.plan(_goal()), _goal())
    h = ctx.hits[0]
    assert h.source == "intelligence-gateway" and h.content_hash == "fp1" and h.observed_at == "2026-10-25T00:00:00Z" and "Acme Ltd" in h.text
    assert res is not None


def test_a_price_that_rose_after_planning_is_refused_not_paid():
    gw = FakeGateway(price=0.05)
    rt, planner, retr, _ = _rig(gw)
    plan = rt.plan(_goal())
    gw.price = 0.90                                                       # moved between planning and execution
    ctx = rt.build_context(plan, _goal())
    assert ctx.hits == () and gw.spent == 0.0
    assert "price rose" in retr.refusals[0]["reason"]


def test_rights_revoked_after_planning_is_honoured_at_execution():
    gw = FakeGateway()
    rt, planner, retr, _ = _rig(gw)
    plan = rt.plan(_goal())
    gw.rights_blocked.add("company_profile")
    ctx = rt.build_context(plan, _goal())
    assert ctx.hits == () and gw.spent == 0.0 and retr.refusals[0]["reason"] == "rights-blocked"


def test_a_spend_cap_refusal_gives_no_evidence_and_no_retry():
    gw = FakeGateway(cap=0.0)
    rt, planner, retr, _ = _rig(gw)
    ctx = rt.build_context(rt.plan(_goal()), _goal())
    assert ctx.hits == () and sum(1 for c in gw.calls if c[1] == "/v1/resolve") == 1 and retr.refusals[0]["status"] == 402


def test_an_unplanned_need_cannot_reach_the_gateway():
    gw = FakeGateway()
    rt, planner, retr, _ = _rig(gw)
    assert retr.search("anything", 5, method="gateway:deadbeefdeadbeef") == []
    assert [c for c in gw.calls if c[1] in ("/v1/quote", "/v1/resolve")] == []
    assert retr.refusals[0]["reason"] == "unknown need"


def test_other_retrieval_methods_pass_through_untouched():
    rt, planner, retr, gw = _rig(FakeGateway())
    hits = retr.search("acme local notes", 5, method="hybrid")
    assert hits and hits[0].filename == "local.md" and gw.calls == []


def test_an_unreadable_gateway_response_gives_no_evidence():
    gw = FakeGateway()
    rt, planner, retr, _ = _rig(gw)
    plan = rt.plan(_goal())
    retr.client._send = lambda *a: (200, ["not", "a", "dict"])
    assert rt.build_context(plan, _goal()).hits == ()


def test_the_client_carries_a_fresh_token_each_call_and_stores_none():
    seen = []

    def tokens():
        seen.append(1)
        return "tok"
    c = GatewayClient("https://gw.test", tokens, transport=FakeGateway())
    c.capabilities(); c.capabilities()
    assert len(seen) == 2 and not any("tok" == v for v in vars(c).values())


def test_the_gateway_is_given_the_quoted_price_as_its_ceiling_never_the_default_of_zero():
    """Found by the contract test against the real gateway: max_cost 0 means "spend nothing", so a default-0 resolve returned an empty answer."""
    gw = FakeGateway()
    rt, planner, retr, _ = _rig(gw)
    rt.run(_goal())
    body = [c[2] for c in gw.calls if c[1] == "/v1/resolve"][0]
    assert body["max_cost"] == 0.05
    assert rt.build_context(rt.plan(_goal()), _goal()).hits           # and with it the evidence arrives


def test_a_charged_call_that_returns_no_answer_is_recorded_not_hidden():
    gw = FakeGateway()
    rt, planner, retr, _ = _rig(gw)
    plan = rt.plan(_goal())
    orig = gw.__call__

    def empty_answer(method, url, headers, body):
        st, payload = orig(method, url, headers, body)
        if url.endswith("/v1/resolve") and st == 200:
            payload = dict(payload, answer="")
        return st, payload
    retr.client._send = empty_answer
    assert rt.build_context(plan, _goal()).hits == ()
    r = retr.refusals[-1]
    assert r["reason"] == "empty answer" and r["charged_usd"] == 0.0575 and retr.spent_usd == 0.0575
