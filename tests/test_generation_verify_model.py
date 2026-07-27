"""Reasoning Plane — Phase 4 (Verification Optimizer) + Phase 5 (model competence).

Phase 4: correctness-sensitive mission classes also get a self-checked (verify + one retry) variant of
each strategy as a DISTINCT arm, so the bandit learns where self-verification pays off.
Phase 5: the reasoning arm already carries the model tier; here we warm-start + surface which model
actually succeeds per mission class ("DeepSeek here, Qwen there"), measured at oracle.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from context_runtime import ContextRuntime
from context_runtime.optimizer.online import plan_key
from context_runtime.reasoner import strategies as gs
from context_runtime.reasoner.strategy_reasoner import StrategyReasoner
from context_runtime.types import Candidate, Goal, Constraints, ModelResult, ReasonRequest, StepSpec


@pytest.fixture(autouse=True)
def _restore():
    ps, mc = dict(gs._ACTIVE_PRIORS), dict(gs._MODEL_COMPETENCE)
    yield
    gs._ACTIVE_PRIORS.clear(); gs._ACTIVE_PRIORS.update(ps)
    gs._MODEL_COMPETENCE.clear(); gs._MODEL_COMPETENCE.update(mc)


# ═══════════════════════════ Phase 4 — Verification Optimizer ═══════════════════════════
def test_offers_verify_only_for_correctness_sensitive_classes(monkeypatch):
    monkeypatch.setenv("CR_GENSTRATEGY", "1")
    assert gs.offers_verify("multi_hop") and gs.offers_verify("high_risk")
    assert not gs.offers_verify("exact_lookup") and not gs.offers_verify("conceptual")
    monkeypatch.delenv("CR_GENSTRATEGY", raising=False)
    assert not gs.offers_verify("multi_hop")   # gated by the flag


def test_verify_variants_are_distinct_arms(monkeypatch):
    monkeypatch.setenv("CR_GENSTRATEGY", "1")
    rt = ContextRuntime.default([{"chunk_id": "a::0", "filename": "a.md", "text": "auth issues tokens", "created_at": None}])
    g = Goal(text="how does auth relate to the billing outage", constraints=Constraints())
    intent = rt.intent.analyze(g)
    assert intent.bucket == "multi_hop"
    cands = rt.candidates.generate(intent, g)
    verified = [c for c in cands if any(s.type == "reason" and s.params.get("verify") for s in c.steps)]
    plain = [c for c in cands if any(s.type == "reason" and not s.params.get("verify") for s in c.steps)]
    assert verified and plain                      # both offered for a verify-bucket
    # a verified candidate is a distinct bandit arm (+v)
    assert plan_key(verified[0]).count("+v") == 1


def test_plan_key_marks_the_verified_arm():
    def cand(strategy, verify):
        rp = {"strategy": strategy}
        if verify:
            rp["verify"] = True
        return Candidate(steps=(StepSpec("retrieve", {"method": "hybrid"}), StepSpec("reason", rp)), model_tier="cheap")
    assert plan_key(cand("decompose", False)) == "hybrid:decompose:cheap"
    assert plan_key(cand("decompose", True)) == "hybrid:decompose+v:cheap"
    assert plan_key(cand("decompose", True)) != plan_key(cand("decompose", False))


class _SeqModel:
    """Returns queued texts in order (last repeats) — to drive the self-check retry deterministically."""
    def __init__(self, texts):
        self.texts, self.calls = list(texts), 0

    def complete(self, mreq):
        t = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return ModelResult(text=t, model="stub", tier="local")

    def capabilities(self, m):
        from context_runtime.types import ModelCapabilities
        return ModelCapabilities()


def _req(context_text, question):
    ctx = SimpleNamespace(assembled_text=context_text,
                          plan=SimpleNamespace(id="p1", intent=SimpleNamespace(normalized=question)))
    return ReasonRequest(context=ctx, capability="synthesis")


def test_verify_retries_on_a_weak_first_answer():
    # first answer is ungrounded (faithfulness 0), retry is grounded → the retry is kept
    model = _SeqModel(["disk failure firmware bug", "token expiry"])
    r = StrategyReasoner(model, "terse", verify=True)
    res = r.reason(_req("token expiry caused the outage", "why outage"))
    assert res.text == "token expiry" and model.calls == 2


def test_verify_does_not_retry_a_grounded_answer():
    model = _SeqModel(["token expiry", "SHOULD NOT BE USED"])
    r = StrategyReasoner(model, "terse", verify=True)
    res = r.reason(_req("token expiry caused the outage", "why outage"))
    assert res.text == "token expiry" and model.calls == 1     # grounded → no retry


# ═══════════════════════════ Phase 5 — model competence ═══════════════════════════
def _cell(tmp, dataset, model, strategy, oracle):
    (tmp / f"{dataset}__hybrid__{model}__{strategy}.json").write_text(json.dumps(
        {"dataset": dataset, "method": "hybrid", "model": model, "strategy": strategy,
         "n": 10, "acc_oracle": oracle}))


def test_model_competence_from_ablation_ranks_models_per_class(tmp_path):
    # musique (→multi_hop): Qwen succeeds at oracle, DeepSeek over-abstains (the report's finding #5)
    _cell(tmp_path, "musique", "qwen", "decompose", 0.80)
    _cell(tmp_path, "musique", "deepseek", "decompose", 0.00)
    comp = gs.model_competence_from_ablation(str(tmp_path))
    assert comp["multi_hop"] == {"qwen": 0.80, "deepseek": 0.00}
    gs.set_model_competence(comp)
    assert gs.competent_model("multi_hop") == "qwen"       # bigger != better, learned
    assert gs.competent_model("temporal") is None          # unknown class


def test_load_priors_richer_format_applies_strategies_and_competence(tmp_path):
    p = tmp_path / "priors.json"
    p.write_text(json.dumps({
        "strategies": {"multi_hop": ["decompose", "reason"]},
        "model_competence": {"multi_hop": {"qwen": 0.8, "deepseek": 0.0}},
    }))
    gs.load_priors(str(p))
    assert gs.strategies_for("multi_hop") == ("reason", "decompose")   # cost-ordered
    assert gs.competent_model("multi_hop") == "qwen"


def test_explain_block_surfaces_verify_and_competence(monkeypatch):
    monkeypatch.setenv("CR_GENSTRATEGY", "1")
    gs.set_model_competence({"multi_hop": {"qwen": 0.8, "deepseek": 0.0}})
    blk = gs.explain_block("multi_hop", method="hybrid", tier="cheap")
    assert blk["verify_offered"] is True
    assert blk["competent_model"] == "qwen"
    assert blk["model_competence"] == {"qwen": 0.8, "deepseek": 0.0}
    # a non-verify class shows no self-check offer
    assert gs.explain_block("conceptual")["verify_offered"] is False
