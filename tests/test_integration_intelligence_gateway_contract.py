"""Contract test against the REAL gateway app (in process), not the fake: skipped unless the gateway package and its test kit are importable.

    PYTHONPATH=.:/path/to/intelligence-gateway:/path/to/intelligence-gateway/tests python -m pytest tests/test_integration_intelligence_gateway_contract.py
"""
from __future__ import annotations

import pytest

pytest.importorskip("intelligence_gateway")
rightskit = pytest.importorskip("rightskit")
pytest.importorskip("fastapi.testclient")

from dataclasses import dataclass                                        # noqa: E402

from fastapi.testclient import TestClient                                # noqa: E402
from runtime_contracts.protocol import (                                 # noqa: E402
    AcquisitionResult, Capability, CostEstimate, EvidenceArtifact, EvidenceRef, ProviderFamily)

import intelligence_gateway.app as gapp                                  # noqa: E402
from intelligence_gateway import ApiKey, KeyRegistry, MeteredGateway, managed_registry   # noqa: E402

from context_runtime.adapters.store_inmemory import InMemoryStore        # noqa: E402
from context_runtime.integrations.intelligence_gateway import (          # noqa: E402
    GatewayCandidateGenerator, GatewayClient, GatewayRetriever, NeedBook, gateway_source)
from context_runtime.planner.candidates_registry import RegistryCandidateGenerator   # noqa: E402
from context_runtime.runtime.runtime import ContextRuntime               # noqa: E402
from context_runtime.types import Constraints, Goal                      # noqa: E402


@dataclass
class _Paid:
    provider_id: str = "managed:firmographics"
    family: ProviderFamily = ProviderFamily.EXTERNAL_DATA
    price: float = 0.40

    def capabilities(self): return (Capability.COMPANY_IDENTITY,)
    def estimate_cost(self, r): return CostEstimate(money=self.price, latency_ms=20)
    def check_entitlement(self, t, c): return True

    def acquire(self, request):
        art = EvidenceArtifact(provider=self.provider_id, family=self.family, capability=request.capability, subject="acme",
                               observations=({"lei": "X1"},), retrieved_at="2026-01-01T00:00:00Z",
                               source_refs=(EvidenceRef(ref="m:1", content_hash="h", source=self.provider_id),),
                               confidence=0.9, cost=self.price, license_scope="managed", raw_response_digest="d")
        return AcquisitionResult.found((art,), cost=self.price)


def _rig(*, approved: bool, cap_usd: float = 0.0, synthesize: bool = True):
    keys = KeyRegistry()
    keys.add(ApiKey(key="k1", tenant="acme", orchestration_fee_pct=0.15, monthly_cap_usd=cap_usd))
    reg = managed_registry(_Paid())
    gapp._gw = MeteredGateway(registry=reg, keys=keys, rights=rightskit.approve_all(reg) if approved else None)
    if synthesize:                                         # what a production deployment supplies: the stock default never writes an answer
        gapp._gw.synthesize = lambda need, arts: ("Acme Ltd, LEI X1", max(a.confidence for a in arts), {"artifact_count": len(arts)}, (), ())
    tc = TestClient(gapp.app)

    def transport(method, url, headers, body):
        r = tc.request(method, "/" + url.split("//", 1)[1].split("/", 1)[1], headers=headers, content=body)
        return r.status_code, r.json()
    client = GatewayClient("https://gw.test", lambda: "k1", transport=transport)
    book = NeedBook()
    planner = GatewayCandidateGenerator(RegistryCandidateGenerator(), client, book)
    rt = ContextRuntime.default(candidates=planner)
    rt.retriever = GatewayRetriever(InMemoryStore([{"chunk_id": "d1", "filename": "local.md", "text": "acme local notes"}]), client, book)
    return rt, planner, rt.retriever


def _goal(**cons):
    return Goal("who is acme", sources=(gateway_source("company_identity", subjects=["acme"], purpose="kyc"),), constraints=Constraints(**cons))


def test_a_rights_cleared_provider_flows_through_planning_and_execution_against_the_real_gateway():
    rt, planner, retr = _rig(approved=True)
    plan = rt.plan(_goal(max_cost_usd=1.0))
    assert planner.decisions[0].offered, planner.decisions[0]
    ctx = rt.build_context(plan, _goal(max_cost_usd=1.0))
    assert ctx.hits and ctx.hits[0].source == "intelligence-gateway" and "LEI X1" in ctx.hits[0].text
    assert retr.spent_usd == pytest.approx(0.46, abs=1e-6)                 # 0.40 + 15% fee: the gateway's own meter


def test_with_no_rights_decision_the_real_gateway_serves_nothing_and_the_goal_falls_back():
    rt, planner, retr = _rig(approved=False)
    rt.plan(_goal(max_cost_usd=1.0))
    assert not planner.decisions[0].offered
    assert planner.decisions[0].reason in ("not served or not entitled for this caller", "rights-blocked", "no provider available") or "refused" in planner.decisions[0].reason


def test_the_real_gateways_cap_is_enforced_even_if_the_plan_was_offered():
    rt, planner, retr = _rig(approved=True, cap_usd=0.10)                  # a quote of 0.46 exceeds the 0.10 cap
    ctx = rt.build_context(rt.plan(_goal(max_cost_usd=1.0)), _goal(max_cost_usd=1.0))
    assert ctx.hits == () and retr.spent_usd == 0.0


def test_the_stock_gateway_without_a_synthesizer_charges_and_returns_no_answer_and_this_is_recorded():
    """A gap on the gateway side, pinned here so it cannot go unnoticed: its default synthesizer writes no answer and /v1/resolve relays no raw observations."""
    rt, planner, retr = _rig(approved=True, synthesize=False)
    ctx = rt.build_context(rt.plan(_goal(max_cost_usd=1.0)), _goal(max_cost_usd=1.0))
    assert ctx.hits == () and retr.spent_usd == pytest.approx(0.46, abs=1e-6)
    assert retr.refusals[-1]["reason"] == "empty answer"
