"""Learned policy survives a restart: bandit + tenant + vibex planner persist to disk."""
from __future__ import annotations

from context_runtime.integrations.bandit import EpsilonGreedyBandit
from context_runtime.integrations.modules import CATALOG, ModuleTenant
from context_runtime.integrations.vibexgen import DEFAULT_CHAINS, SceneSpec, VibexgenPlanner


class _Arm:
    def __init__(self, k):
        self.key = k


def test_bandit_state_survives_reload(tmp_path):
    arms = (_Arm("a"), _Arm("b"))
    p = str(tmp_path / "b.json")
    b1 = EpsilonGreedyBandit(arms, persist_path=p)
    for _ in range(20):
        b1.update("ctx", _Arm("a"), 1.0)
        b1.update("ctx", _Arm("b"), 0.0)
    assert b1.policy()["ctx"] == "a"
    # fresh instance, same path → reloads the learned policy
    b2 = EpsilonGreedyBandit(arms, persist_path=p)
    assert b2.policy()["ctx"] == "a"
    assert b2.value("ctx", "a")[0] == 20


def test_module_tenant_persists(tmp_path):
    p = str(tmp_path / "billing.json")
    t1 = ModuleTenant(CATALOG["billing"], persist_path=p)
    latent = CATALOG["billing"].sources[0]
    q = "why is the ledger off?"
    for _ in range(40):
        r = t1.handle(q)
        t1.record_outcome(q, latent in r.bundle.sources)
    learned = t1.policy()
    assert learned
    # reload
    t2 = ModuleTenant(CATALOG["billing"], persist_path=p)
    assert t2.policy() == learned


def test_vibex_planner_persists(tmp_path):
    p = str(tmp_path / "vibex.json")
    scene = SceneSpec(("hero",), "noir", "city", "action", "realistic")
    v1 = VibexgenPlanner(persist_path=p)
    latent = DEFAULT_CHAINS[4].key
    good = {c: 0.9 for c in __import__("context_runtime.integrations.vibexgen",
                                       fromlist=["CRITERIA_WEIGHTS"]).CRITERIA_WEIGHTS}
    bad = {c: 0.3 for c in good}
    for _ in range(60):
        ch = v1.plan_chain("trailer", scene)
        v1.record_scores("trailer", scene, good if ch.key == latent else bad)
    assert v1.suggest("trailer", scene) == latent
    # reload → suggestion preserved
    v2 = VibexgenPlanner(persist_path=p)
    assert v2.suggest("trailer", scene) == latent


def test_bandit_tolerates_corrupt_persist_and_backfills_new_arm(tmp_path):
    p = tmp_path / "b.json"
    p.write_text("{ not valid json", encoding="utf-8")
    b = EpsilonGreedyBandit((_Arm("a"), _Arm("b")), persist_path=str(p))   # corrupt file must not crash load
    assert b.value("ctx", "a")[0] == 0
    for _ in range(5):
        b.update("ctx", _Arm("a"), 1.0)
    # reload with an EXTRA arm 'c' on the persisted (a,b) context → 'c' backfills, no KeyError
    b2 = EpsilonGreedyBandit((_Arm("a"), _Arm("b"), _Arm("c")), persist_path=str(p))
    assert b2.value("ctx", "a")[0] == 5             # existing arm reloaded
    assert b2.value("ctx", "c")[0] == 0             # new arm optimistically backfilled at count 0
