"""The business-module fleet: catalog instantiates, tenants handle/learn/gate generically."""
from __future__ import annotations

from context_runtime.integrations.modules import (
    CATALOG, ModuleTenant, SourceBundle, build_fleet, question_kind, reward,
)


def test_whole_catalog_instantiates():
    fleet = build_fleet()
    assert len(fleet) == len(CATALOG) >= 16
    for name, tenant in fleet.items():
        assert isinstance(tenant, ModuleTenant)
        assert tenant.arms                      # has source bundles to choose from


def test_handle_pulls_source_evidence():
    t = ModuleTenant(CATALOG["support"])
    r = t.handle("customer cannot log in after the latest release")
    assert r.hits and r.context
    assert all(h.source in CATALOG["support"].sources for h in r.hits)


def test_action_questions_recommend_gated_action():
    t = ModuleTenant(CATALOG["billing"])
    r = t.handle("issue a refund for the duplicate charge")
    assert r.kind == "action" and r.recommended_action == "billing.refund"


def test_side_effecting_action_denied_without_approver():
    t = ModuleTenant(CATALOG["billing"], approver=lambda a: False)
    res = t.act("refund", amount=10)
    assert not res.ok and "denied" in res.error
    assert t.registry.audit[-1]["allowed"] is False


def test_action_allowed_with_approver_but_dry_run():
    t = ModuleTenant(CATALOG["billing"], approver=lambda a: True)
    res = t.act("refund", amount=10)
    assert res.ok and "dry-run" in res.text


def test_question_kind_classification():
    assert question_kind("issue a refund") == "action"
    assert question_kind("why did revenue fall?") == "analysis"
    assert question_kind("show invoice 42") == "lookup"


def test_reward_prefers_cheaper_bundle():
    assert reward(True, SourceBundle(("a",)), 4) > reward(True, SourceBundle(("a", "b", "c")), 4)
    assert reward(False, SourceBundle(("a",)), 4) == 0.0


def test_tenant_learns_cheapest_sufficient_bundle():
    t = ModuleTenant(CATALOG["control_tower"], epsilon=0.1)
    latent = "warehouse"          # the decisive source for this domain question
    q = "why did revenue fall last quarter?"
    for _ in range(70):
        r = t.handle(q)
        success = latent in r.bundle.sources
        t.record_outcome(q, success)
    chosen = t.policy()[f"control_tower:{question_kind(q)}"]
    assert latent in chosen
