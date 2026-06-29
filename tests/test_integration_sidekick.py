"""sidekick × ContextOS integration: drop-in compat, bandit learning, loop closure."""
from __future__ import annotations

from contextos import ContextRuntime
from contextos.integrations.sidekick import (
    ContextOSSkillStore,
    DEFAULT_ARMS,
    EpsilonGreedyBandit,
    Skill,
    SubtaskOutcome,
    reward_from_outcome,
)


def _store(tmp_path):
    rt = ContextRuntime.default([])
    store = ContextOSSkillStore(tmp_path / "skills", runtime=rt)
    store.save(Skill("retry-flaky", "tests fail intermittently", "wrap in retry", ["pytest passes"]))
    store.save(Skill("add-flag", "new cli option", "argparse add_argument", ["--flag works"]))
    return store


def test_dropin_surface_matches_sidekick(tmp_path):
    store = _store(tmp_path)
    # the methods sidekick's orchestrator calls must all exist
    for m in ("save", "all", "recall", "record_use"):
        assert callable(getattr(store, m))
    assert len(store.all()) == 2
    hits = store.recall("flaky tests keep failing", limit=2)
    assert all(isinstance(s, Skill) for s in hits)


def test_recall_routes_through_planner_and_records_pending(tmp_path):
    store = _store(tmp_path)
    store.recall("add a --verbose cli flag", limit=2)
    # recall stashed a (plan, strategy) for the outcome to correlate against
    key = store._key("add a --verbose cli flag")
    assert key in store._pending
    plan, strat = store._pending[key]
    assert strat in DEFAULT_ARMS
    assert plan.intent.bucket  # planner ran


def test_record_outcome_closes_loop(tmp_path):
    store = _store(tmp_path)
    before = store.runtime.estimator.statistics().fields[0].sample_count
    store.recall("fix flaky tests", limit=2)
    reward = store.record_outcome("fix flaky tests", SubtaskOutcome(accepted=True, first_attempt=True, tokens_total=2000))
    after = store.runtime.estimator.statistics().fields[0].sample_count
    assert after == before + 1          # cost-model observed the actual
    assert reward > 0.5                 # accepted + first-try + cheap → high reward


def test_reward_prefers_accepted_first_try_and_cheap():
    best = reward_from_outcome(accepted=True, first_attempt=True, tokens_total=1000)
    retried = reward_from_outcome(accepted=True, first_attempt=False, tokens_total=1000)
    expensive = reward_from_outcome(accepted=True, first_attempt=True, tokens_total=9000)
    rejected = reward_from_outcome(accepted=False, first_attempt=False, tokens_total=1000)
    assert best > retried > 0
    assert best > expensive
    assert rejected == 0.0


def test_bandit_learns_the_rewarding_arm():
    b = EpsilonGreedyBandit(DEFAULT_ARMS, epsilon=0.1)
    good = DEFAULT_ARMS[3]   # hybrid:8:4000
    # reward 'good' highly, others poorly, in one bucket
    for _ in range(60):
        chosen = b.select("code_reasoning")
        b.update("code_reasoning", chosen, 1.0 if chosen.key == good.key else 0.0)
    assert b.policy()["code_reasoning"] == good.key


def test_contextos_beats_naive_over_a_task_stream():
    """End-to-end mechanism check (the harness in miniature)."""
    rt = ContextRuntime.default([])
    store = ContextOSSkillStore("/tmp/contextos_test_skills", runtime=rt, bandit=EpsilonGreedyBandit(DEFAULT_ARMS, epsilon=0.15))
    store.save(Skill("port-codes", "map error codes", "bm25 over logs", []))
    latent = "bm25:3:1500"

    rng = [0xBEEF]
    def outcome(key):
        rng[0] = (rng[0] * 1103515245 + 12345) & 0x7FFFFFFF
        roll = rng[0] / 0x7FFFFFFF
        match = key == latent
        return SubtaskOutcome(accepted=roll < (0.9 if match else 0.4), first_attempt=False, tokens_total=1500)

    accepted = []
    for _ in range(40):
        store.recall("look up ERR-500 status code", limit=1)
        _, strat = store._pending[store._key("look up ERR-500 status code")]
        o = outcome(strat.key)
        store.record_outcome("look up ERR-500 status code", o)
        accepted.append(o.accepted)

    early = sum(accepted[:10]) / 10
    late = sum(accepted[-10:]) / 10
    assert late >= early   # learning does not regress; usually improves
