"""vibexgen tenant: scenario-before-generation, chain policy, multi-criteria reward, learning."""
from __future__ import annotations

from context_runtime.integrations.vibexgen import (
    DEFAULT_CHAINS, GenerationChain, SceneSpec, VibexgenPlanner,
    reward_from_scores, scenario_key, _vibex_bandit,
)

GOOD = {c: 0.9 for c in ("prompt_adherence", "visual_quality", "motion_coherence",
                         "character_consistency", "lighting_fidelity", "scenery_fidelity",
                         "continuity", "subtitle_accuracy", "voiceover_sync", "pacing", "aesthetic_appeal")}
BAD = {c: 0.3 for c in GOOD}


def test_scenario_selected_before_generation():
    p = VibexgenPlanner()
    choice = p.select_scenario("demo", ["a plain shot",
                                        "a host at a desk, warm studio lighting, slow camera, upbeat mood"])
    assert choice.index == 1                 # the detailed scenario wins, before any generation


def test_reward_is_quality_minus_generation_cost():
    cheap = GenerationChain("wan", "t2v", "base", 30, "720p")
    pricey = GenerationChain("seedance", "t2v", "base", 40, "1080p")
    # same scores → the cheaper chain earns more reward (efficiency frontier)
    assert reward_from_scores(GOOD, cheap) > reward_from_scores(GOOD, pricey)
    # bad scores tank the reward regardless of cost
    assert reward_from_scores(GOOD, cheap) > reward_from_scores(BAD, cheap)


def test_scenario_key_is_template_plus_features():
    k = scenario_key("trailer", SceneSpec(("hero",), "noir", "city night", "action", "realistic"))
    assert k.startswith("trailer::") and "action" in k and "chars" in k


def test_plan_chain_then_record_learns():
    p = VibexgenPlanner()
    scene = SceneSpec(("host",), "studio", "office", "dialogue", "realistic", True)
    chain = p.plan_chain("demo", scene)
    assert chain in DEFAULT_CHAINS
    r = p.record_scores("demo", scene, GOOD, gen_cost_usd=0.5)
    assert r > 0.5


def test_planner_learns_best_chain_for_a_template():
    p = VibexgenPlanner(bandit=_vibex_bandit(0.1))
    scene = SceneSpec(("hero", "villain"), "noir", "city night", "action", "realistic")
    latent = DEFAULT_CHAINS[4].key            # seedance (the action/cinematic best)
    for _ in range(80):
        chain = p.plan_chain("trailer", scene)
        p.record_scores("trailer", scene, GOOD if chain.key == latent else BAD,
                        gen_cost_usd=chain.cost_units() * 0.5)
    assert p.suggest("trailer", scene) == latent


def test_cost_model_calibrates_on_generation_cost():
    p = VibexgenPlanner()
    before = p.runtime.estimator.statistics().fields[0].sample_count
    scene = SceneSpec((), "neon", "street", "pan", "stylized")
    p.plan_chain("social", scene)
    p.record_scores("social", scene, GOOD, gen_cost_usd=0.8, gen_latency_s=40)
    after = p.runtime.estimator.statistics().fields[0].sample_count
    assert after == before + 1
