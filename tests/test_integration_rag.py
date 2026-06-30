"""redevops-rag × Context Runtime tenant: config arms, quality reward, tuning loop."""
from __future__ import annotations

from context_runtime.integrations.redevops_rag import (
    ContextRuntimeRetrieverTuner,
    DEFAULT_ARMS,
    RetrievalConfig,
    reciprocal_rank,
    reward_from_quality,
    _rag_bandit,
)


def test_config_kwargs_match_hybrid_search_signature():
    # the kwargs we emit must be exactly redevops-rag hybrid_search params
    kw = RetrievalConfig().kwargs()
    assert set(kw) == {"pool", "limit", "vector_threshold", "recency_half_life_days",
                       "keyword_boost_per_term", "keyword_boost_cap"}


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b"], {"a"}) == 1.0
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_reward_prefers_cheaper_config_at_equal_quality():
    cheap = DEFAULT_ARMS[0]      # pool 20, no rerank
    pricey = DEFAULT_ARMS[3]     # pool 100, rerank
    assert reward_from_quality(0.8, cheap) > reward_from_quality(0.8, pricey)


def test_choose_routes_through_planner():
    tuner = ContextRuntimeRetrieverTuner()
    cfg = tuner.choose("look up error code 429")
    assert isinstance(cfg, RetrievalConfig)
    assert tuner._key("look up error code 429") in tuner._pending


def test_record_outcome_closes_loop_and_calibrates():
    tuner = ContextRuntimeRetrieverTuner()
    before = tuner.runtime.estimator.statistics().fields[0].sample_count
    tuner.choose("how do we rotate api keys")
    r = tuner.record_outcome("how do we rotate api keys", quality=0.9, latency_s=0.3)
    after = tuner.runtime.estimator.statistics().fields[0].sample_count
    assert after == before + 1
    assert 0 < r <= 1


def test_tuner_learns_best_config_for_a_bucket():
    tuner = ContextRuntimeRetrieverTuner(bandit=_rag_bandit(0.1))
    good = DEFAULT_ARMS[2]
    rng = [0xABCDEF]
    for _ in range(60):
        cfg = tuner.choose("what is the difference between RRF and reranking")
        q = 0.95 if cfg.key == good.key else 0.3
        tuner.record_outcome("what is the difference between RRF and reranking", quality=q)
    # the (conceptual) bucket should converge to the high-quality arm
    assert good.key in tuner.policy().values()
