"""LibreChat self-learning tenant: learns the retrieval strategy that maximizes
LLM-judged retrieval quality per request-type. Uses a small inline corpus so the test
needs no ingestion backends."""
from __future__ import annotations

from context_runtime.integrations.librechat import (
    DEFAULT_STRATEGIES,
    LibreChatTenant,
    RetrievalStrategy,
    _parse_score,
    heuristic_judge,
    reward_from_judgment,
)


def _corpus(tmp_path):
    docs = {
        "steroid.txt": "Steroid profile: testosterone, cortisol and DHEA measured by LC-MS/MS.",
        "lipid.txt": "Lipid profile: total cholesterol, LDL, HDL and triglycerides panel.",
        "hormones.txt": "Reproductive hormones: FSH, LH and prolactin reference ranges.",
        "chat.txt": "Nutrients.tech group chat: members discussed their lab results and diet.",
    }
    for name, text in docs.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return str(tmp_path)


def test_retrieve_returns_grounded_context(tmp_path):
    t = LibreChatTenant(corpus_dir=_corpus(tmp_path))
    ctx = t.retrieve("what is in the steroid profile testosterone panel")
    assert ctx.hits, "expected retrieval to return hits from the corpus"
    assert "steroid" in ctx.context.lower()
    assert ctx.strategy.key  # a strategy was chosen


def test_heuristic_judge_scores_relevance(tmp_path):
    t = LibreChatTenant(corpus_dir=_corpus(tmp_path))
    good = t.retrieve("steroid profile testosterone cortisol")
    good_score = heuristic_judge(good.request, good.context, good.hits)
    assert good_score > 0.3, f"relevant request should score > 0.3, got {good_score}"
    # a request with no corpus overlap should score low.
    empty_score = heuristic_judge("quantum chromodynamics gluon", "", ())
    assert empty_score == 0.0


def test_reward_prefers_cheaper_strategy_at_equal_quality():
    cheap = RetrievalStrategy("bm25", 3, False)
    pricey = RetrievalStrategy("hybrid", 8, True)
    assert reward_from_judgment(0.9, cheap) > reward_from_judgment(0.9, pricey)
    assert reward_from_judgment(0.0, cheap) == 0.0


def test_parse_score():
    assert _parse_score("0.8") == 0.8
    assert _parse_score("Score: 0.95 out of 1") == 0.95
    assert _parse_score("1.7") == 1.0            # clamped
    assert _parse_score("not a number") == 0.0


def test_tenant_learns_best_strategy_and_persists(tmp_path):
    corpus = _corpus(tmp_path)
    persist = str(tmp_path / "librechat.json")
    t = LibreChatTenant(corpus_dir=corpus, persist_path=persist)
    request = "what were the steroid profile testosterone results"
    latent = DEFAULT_STRATEGIES[2].key  # hybrid:k8:rr — the strategy our judge rewards here

    rewards = []
    for _ in range(60):
        ctx = t.retrieve(request)
        # a controlled judge: the latent strategy yields great context, others mediocre.
        score = 0.92 if ctx.strategy.key == latent else 0.4
        rewards.append(t.record_judgment(request, score))
    assert t.suggest(request) == latent, f"tenant learned {t.suggest(request)}, want {latent}"
    early = sum(rewards[:15]) / 15
    late = sum(rewards[-15:]) / 15
    assert late >= early, f"reward should improve/hold ({early:.2f}→{late:.2f})"

    # reload → learned policy preserved.
    t2 = LibreChatTenant(corpus_dir=corpus, persist_path=persist)
    assert t2.suggest(request) == latent


def test_handle_runs_retrieve_judge_learn_in_one_call(tmp_path):
    t = LibreChatTenant(corpus_dir=_corpus(tmp_path))
    ctx, score, reward = t.handle("lipid profile cholesterol LDL")
    assert ctx.hits and 0.0 <= score <= 1.0 and reward >= 0.0
    assert t.policy(), "handle() should have updated the policy"
