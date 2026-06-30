"""vibexgen × Context Runtime — learn the best generation chain BEFORE the expensive step.

The planner picks the scenario cheaply (stage 1), then the generation chain from a policy
keyed by (template, scenario features). The user scores the result on multiple criteria;
the planner learns to suggest a better chain next time for that template + scenario type —
WITHOUT wasting generation comparing finished videos.

Hidden truth here: each (template, scenario kind) has a latent-best chain (e.g. realistic
3D scenes → LTX-2.3 i2v; action → seedance; cheap drafts → wan). The planner must discover
it from recorded multi-criteria scores alone.

    python examples/vibexgen_learning.py
"""
from __future__ import annotations

from context_runtime.integrations.vibexgen import (
    DEFAULT_CHAINS, GenerationChain, SceneSpec, VibexgenPlanner, reward_from_scores, scenario_key,
)

# (template, SceneSpec, latent-best chain key) — what the planner must learn.
REQUESTS = [
    ("product-demo", SceneSpec(("host",), "studio", "office desk", "dialogue", "realistic", True),
     "ltx/i2v/LTX-2.3-3DREAL-LoRA/28s/720p"),
    ("cinematic-trailer", SceneSpec(("hero", "villain"), "noir", "city night", "action", "realistic"),
     "seedance/t2v/base/40s/1080p"),
    ("social-short", SceneSpec((), "neon", "street", "pan", "stylized"),
     "wan/t2v/base/30s/720p"),
    ("explainer", SceneSpec(("narrator",), "studio", "white background", "static", "realistic", True),
     "ltx/t2v/LTX-2.3-3DREAL-LoRA/28s/720p"),
]


def _scores_for(chain_key: str, latent: str, rng: list[int]) -> dict:
    """Simulated user multi-criteria scores: the latent-best chain scores high across the
    board; mismatches score lower (esp. on adherence/visual/motion)."""
    rng[0] = (rng[0] * 1103515245 + 12345) & 0x7FFFFFFF
    roll = rng[0] / 0x7FFFFFFF
    match = chain_key == latent
    hi = 0.8 + 0.2 * roll
    lo = 0.35 + 0.3 * roll
    base = hi if match else lo
    return {c: round(min(1.0, base + (0.05 if "fidelity" in c else 0.0)), 3)
            for c in ("prompt_adherence", "visual_quality", "motion_coherence",
                      "character_consistency", "lighting_fidelity", "scenery_fidelity",
                      "continuity", "subtitle_accuracy", "voiceover_sync", "pacing", "aesthetic_appeal")}


def run(rounds: int = 64) -> None:
    planner = VibexgenPlanner()
    rng = [0x1BEEF]
    learned, fixed = [], []
    default_chain = DEFAULT_CHAINS[2]   # "wan/t2v/base" as the naive fixed strategy

    # stage 1 demo: choose a scenario before generating
    cands = ["A plain shot.", "A host at an office desk, warm studio light, slow camera push, upbeat mood."]
    pick = planner.select_scenario("product-demo", cands)
    print(f"stage 1 — scenario chosen BEFORE generation: candidate #{pick.index} (pred {pick.predicted})\n")

    for i in range(rounds):
        template, scene, latent = REQUESTS[i % len(REQUESTS)]
        chain = planner.plan_chain(template, scene)
        scores = _scores_for(chain.key, latent, rng)
        learned.append(planner.record_scores(template, scene, scores,
                                              gen_cost_usd=chain.cost_units() * 0.5))
        # baseline: always the fixed default chain (no learning)
        fixed.append(reward_from_scores(_scores_for(default_chain.key, latent, rng), default_chain))

    w = 16
    print(f"reward = weighted multi-criteria score − generation-cost penalty\n")
    print(f"  Context Runtime (learned chain per template): {sum(learned[-w:]) / w:.3f}")
    print(f"  fixed default chain (wan/t2v):                {sum(fixed[-w:]) / w:.3f}")
    print("\n── learned generation strategy per (template · scenario) ──")
    for (template, scene, latent) in REQUESTS:
        key = scenario_key(template, scene)
        got = planner.suggest(template, scene)
        flag = " ✓" if got == latent else f"  (latent: {latent})"
        print(f"  {key:<46} → {got}{flag}")


if __name__ == "__main__":
    run()
