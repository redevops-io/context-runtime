"""Revenue & Intelligence GTM tenant — offline, no API keys.

Checks the load-bearing claims: the runtime learns the decisive provider per need, beats a fixed pipeline on
quality at lower cost, penalises a confident wrong-entity match below a missing value, and denies outreach
under NO_OUTREACH while allowing a draft.
"""
from __future__ import annotations

from context_runtime.integrations.revenue_intelligence import (
    DEFAULT_BUNDLES, DECISIVE_PROVIDER, Fixture, ProviderBundle, RevenueIntelligenceTenant,
    _gtm_bandit, gtm_bucket, outcome_for, reward_enrich,
)


def _fixtures() -> dict[str, Fixture]:
    return {f.account_id: f for f in [
        Fixture("c1", "Acme", "acme.io", "company_identity", "pdl", "Acme Inc", wrong=("apollo",)),
        Fixture("c2", "Beta", "beta.ai", "person_role", "apollo", "VP Eng"),
        Fixture("c3", "Gamma", "gamma.dev", "tech_signal", "builtwith", "K8s + Snowflake"),
        Fixture("c4", "Delta", "delta.co", "contact_verify", "hunter", "cto@delta.co"),
        Fixture("c5", "Epsilon", "eps.com", "firmographics", "apollo", "Series B"),
    ]}


def test_bucket_classifier():
    assert gtm_bucket("who is the decision maker / VP of data") == "person_role"
    assert gtm_bucket("verify this email is deliverable") == "contact_verify"
    assert gtm_bucket("what tech stack does the platform use") == "tech_signal"
    assert gtm_bucket("resolve the canonical company / dedupe") == "company_identity"
    assert gtm_bucket("employee count and funding stage") == "firmographics"


def test_wrong_entity_penalised_below_missing():
    fx = Fixture("x", "X", "x.io", "company_identity", "pdl", "X Inc", wrong=("apollo",))
    only_wrong = ProviderBundle(("apollo",))          # confident wrong entity, no decisive source
    nothing = ProviderBundle(("crm",))                # neither decisive nor wrong → missing
    assert outcome_for(only_wrong, fx) == "wrong"
    assert outcome_for(nothing, fx) == "missing"
    assert reward_enrich("wrong", only_wrong) < reward_enrich("missing", nothing) == 0.0


def test_decisive_present_beats_the_wrong_trap():
    """If the decisive provider is in the bundle, verification wins even when a wrong provider is present."""
    fx = Fixture("x", "X", "x.io", "company_identity", "pdl", "X Inc", wrong=("apollo",))
    assert outcome_for(ProviderBundle(("apollo", "pdl")), fx) == "correct"


def test_runs_offline_and_denies_outreach():
    t = RevenueIntelligenceTenant(_fixtures(), approver=lambda spec: False, no_outreach=True)
    r = t.enrich("c2")
    assert r.outcome in ("correct", "wrong", "missing")
    assert t.prepare_outreach("c2").ok is True                    # a draft is allowed
    sent = t.send_outreach("c2")
    assert sent.ok is False and "NO_OUTREACH" in sent.text        # sending is denied
    assert t.registry.audit[-1]["allowed"] is False


def test_learns_decisive_provider_per_need():
    t = RevenueIntelligenceTenant(_fixtures(), bandit=_gtm_bandit(0.1), approver=lambda spec: False)
    ids = list(_fixtures())
    for i in range(200):
        aid = ids[i % len(ids)]
        t.record_outcome(aid, t.enrich(aid).outcome)
    for need, decisive in DECISIVE_PROVIDER.items():
        if need in t.policy():
            key = t.policy()[need]
            bundle = next(b for b in DEFAULT_BUNDLES if b.key == key)
            assert decisive in bundle.providers, f"{need}: learned {key} lacks decisive {decisive}"


def test_adaptive_beats_fixed_pipeline():
    fx = _fixtures()
    t = RevenueIntelligenceTenant(fx, bandit=_gtm_bandit(0.1), approver=lambda spec: False)
    ids = list(fx)
    for i in range(200):
        aid = ids[i % len(ids)]
        t.record_outcome(aid, t.enrich(aid).outcome)

    fixed = ProviderBundle(("crm", "apollo", "pdl"))              # misses hunter + builtwith
    fixed_correct = sum(outcome_for(fixed, f) == "correct" for f in fx.values())
    adaptive_correct = 0
    adaptive_cost = 0.0
    for f in fx.values():
        bundle = next(b for b in DEFAULT_BUNDLES if b.key == t.policy()[f.need])
        adaptive_correct += int(outcome_for(bundle, f) == "correct")
        adaptive_cost += bundle.cost
    n = len(fx)
    # adaptive resolves every need (fixed pipeline cannot) at a lower average cost than the fixed set
    assert adaptive_correct >= fixed_correct
    assert adaptive_correct == n
    assert adaptive_cost / n <= fixed.cost
