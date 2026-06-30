"""vibexgen × Context Runtime — video generation as a cost-based planning problem.

Comparing finished videos to learn is wasteful: generation is the expensive step. So
the decision is made BEFORE generating, exactly like a query planner decides a plan
before the costly execution:

  1. SCENARIO SELECTION (cheap) — generate 2-3 candidate scenario *texts*, score them,
     pick one before any generation happens.
  2. CHAIN PLANNING (learned) — choose the generation chain (engine · mode · model ·
     params) from a policy keyed by (template, scenario features). The scene details
     described in the prompt — characters, lighting, scenery, t2v/i2v, model — are the
     CONTEXT the policy keys on.
  3. FEEDBACK — the user scores the result on MULTIPLE criteria; the score is recorded.
     Next time, for that template + scenario description, the planner suggests a better
     chain. Reward = weighted multi-criteria quality − generation cost (the efficiency
     frontier: best result per generation $).

Same shared contextual bandit + cost model as every other Context Runtime tenant; only
the arms (generation chains) and the reward (multi-criteria score) are app-specific.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from ..runtime.runtime import ContextRuntime
from ..types import Goal, Plan, Trace
from .bandit import EpsilonGreedyBandit

# ──────────────────────────── scene context (described in the prompt) ────────────────────────────


@dataclass(frozen=True)
class SceneSpec:
    """Per-scene detail as written in the generation prompt — the CONTEXT, not a choice."""

    characters: tuple[str, ...] = ()
    lighting: str = ""          # "golden hour" · "noir" · "studio" · "neon"
    scenery: str = ""           # "forest" · "office" · "city night"
    motion: str = "static"      # "static" · "pan" · "action" · "dialogue"
    style: str = "realistic"    # "realistic" · "stylized" · "anime"
    has_speech: bool = False     # drives voiceover/lip-sync criteria


# ──────────────────────────── the generation chain (the bandit arm) ────────────────────────────

# engine → rough relative generation cost (the expensive step we must not waste)
_ENGINE_COST = {"grok": 1.0, "wan": 0.6, "hunyuan": 0.8, "seedance": 1.2, "ltx": 0.5}


@dataclass(frozen=True)
class GenerationChain:
    """The CHOSEN strategy for a scene: which engine/model/mode/params to generate with."""

    engine: str                 # grok · wan · hunyuan · seedance · ltx
    mode: str = "t2v"           # t2v (text-to-video) · i2v (image-to-video)
    model: str = "base"         # e.g. "LTX-2.3-3DREAL-LoRA"
    steps: int = 30
    resolution: str = "720p"    # 720p · 1080p

    @property
    def key(self) -> str:
        return f"{self.engine}/{self.mode}/{self.model}/{self.steps}s/{self.resolution}"

    def cost_units(self) -> float:
        return (_ENGINE_COST.get(self.engine, 1.0)
                * (self.steps / 30.0)
                * (1.6 if self.resolution == "1080p" else 1.0)
                * (1.2 if self.mode == "i2v" else 1.0))   # i2v adds a conditioning pass


# A spanning arm set (extend freely). LTX-2.3-3DREAL-LoRA is the fal model the user cited.
DEFAULT_CHAINS: tuple[GenerationChain, ...] = (
    GenerationChain("ltx", "t2v", "LTX-2.3-3DREAL-LoRA", 28, "720p"),    # cheap, realistic 3D
    GenerationChain("ltx", "i2v", "LTX-2.3-3DREAL-LoRA", 28, "720p"),
    GenerationChain("wan", "t2v", "base", 30, "720p"),                   # cheapest engine
    GenerationChain("hunyuan", "t2v", "base", 30, "1080p"),              # higher fidelity
    GenerationChain("seedance", "t2v", "base", 40, "1080p"),             # priciest / motion
    GenerationChain("grok", "t2v", "base", 30, "720p"),
)

# ──────────────────────────── multi-criteria scoring ────────────────────────────
# The user scores the result on these (each 0..1). Add/reweight freely.
CRITERIA_WEIGHTS: dict[str, float] = {
    "prompt_adherence": 1.5,        # does the video match the prompt
    "visual_quality": 1.3,          # fidelity, artifacts, sharpness
    "motion_coherence": 1.2,        # temporal consistency, no flicker/warping
    "character_consistency": 1.0,   # same character across scenes
    "lighting_fidelity": 0.8,       # matches described lighting
    "scenery_fidelity": 0.8,        # matches described scenery
    "continuity": 0.8,              # scene-to-scene consistency
    "subtitle_accuracy": 0.7,       # WER + timing (if subtitles)
    "voiceover_sync": 0.7,          # audio-visual sync / naturalness (if speech)
    "pacing": 0.6,                  # scene rhythm / cut timing
    "aesthetic_appeal": 1.0,        # overall engagement
}
COST_LAMBDA = 0.3   # how much generation cost trades against quality


def reward_from_scores(scores: dict[str, float], chain: GenerationChain,
                       chains: tuple[GenerationChain, ...] = DEFAULT_CHAINS) -> float:
    """Weighted multi-criteria quality minus a normalized generation-cost penalty."""
    num = sum(w * scores.get(c, 0.0) for c, w in CRITERIA_WEIGHTS.items())
    den = sum(CRITERIA_WEIGHTS.values())
    quality = num / den if den else 0.0
    max_cost = max(c.cost_units() for c in chains)
    return round(max(0.0, quality - COST_LAMBDA * (chain.cost_units() / max_cost)), 4)


def scenario_key(template: str, scene: SceneSpec) -> str:
    """The contextual key the policy learns against: template + coarse scenario features —
    'next time, for this template and this kind of scenario, suggest a better chain'."""
    feat = (f"{scene.style}|{scene.motion}|{'chars' if scene.characters else 'nochars'}|"
            f"{'speech' if scene.has_speech else 'silent'}|{(scene.scenery or 'none').split()[0]}")
    return f"{template}::{feat}"


# ──────────────────────────── the planner ────────────────────────────

Scorer = "callable(str) -> float"   # scenario text → cheap predicted quality (LLM judge or heuristic)


@dataclass
class ScenarioChoice:
    index: int
    scenario: str
    predicted: float


def _vibex_bandit(epsilon: float = 0.12) -> EpsilonGreedyBandit:
    return EpsilonGreedyBandit(DEFAULT_CHAINS, epsilon=epsilon)


class VibexgenPlanner:
    """Plans video generation: pick the scenario BEFORE generating, then the chain from a
    learned policy, then learn from the user's multi-criteria scores."""

    def __init__(self, runtime: ContextRuntime | None = None, bandit: EpsilonGreedyBandit | None = None,
                 chains: tuple[GenerationChain, ...] = DEFAULT_CHAINS):
        self.runtime = runtime or ContextRuntime.default([])
        self.chains = chains
        self.bandit = bandit or _vibex_bandit()
        self._pending: dict[str, tuple[str, GenerationChain]] = {}

    # ── stage 1: choose the scenario BEFORE the expensive generation ──
    def select_scenario(self, request: str, candidates: list[str], scorer=None) -> ScenarioChoice:
        """Score 2-3 candidate scenario texts cheaply and pick the best — no generation yet.

        ``scorer`` maps a scenario string → predicted quality (an LLM judge in prod, a
        heuristic offline). This is where the cheap choice is made, so generation $ is
        only ever spent on the chosen scenario."""
        def _default(s: str) -> float:
            # cheap heuristic: prefer scenarios that name concrete detail (chars/light/scenery)
            t = s.lower()
            return min(1.0, 0.2 + 0.1 * sum(k in t for k in
                       ("character", "light", "scen", "camera", "mood", "color", "motion")))
        sc = scorer or _default
        ranked = sorted(((i, c, sc(c)) for i, c in enumerate(candidates)), key=lambda x: x[2], reverse=True)
        i, c, p = ranked[0]
        return ScenarioChoice(index=i, scenario=c, predicted=round(p, 4))

    # ── stage 2: suggest the generation chain from the learned policy ──
    def plan_chain(self, template: str, scene: SceneSpec) -> GenerationChain:
        ctx = scenario_key(template, scene)
        chain = self.bandit.select(ctx)
        self._pending[self._key(template, scene)] = (ctx, chain)
        return chain

    def suggest(self, template: str, scene: SceneSpec) -> str:
        """The chain vibexgen recommends for this (template, scenario) — the learned best."""
        return self.bandit.policy().get(scenario_key(template, scene), "(unlearned)")

    # ── feedback: the user's multi-criteria scores → learning ──
    def record_scores(self, template: str, scene: SceneSpec, scores: dict[str, float],
                      gen_cost_usd: float = 0.0, gen_latency_s: float = 0.0) -> float:
        key = self._key(template, scene)
        if key not in self._pending:
            return 0.0
        ctx, chain = self._pending.pop(key)
        r = reward_from_scores(scores, chain, self.chains)
        self.bandit.update(ctx, chain, r)
        # calibrate the cost model on the real generation cost/latency
        plan = self.runtime.plan(Goal(text=f"generate {template} {scenario_key(template, scene)}"))
        self.runtime.estimator.observe(plan, Trace(
            plan_id=plan.id, goal_text=template, actual_cost_usd=gen_cost_usd,
            actual_latency_seconds=gen_latency_s, actual_tokens=chain.steps,
            verification_passed=r >= 0.6))
        return r

    def policy(self) -> dict[str, str]:
        return self.bandit.policy()

    def scoreboard(self) -> list[dict]:
        """Per (template·scenario) context: chains ranked by learned score — for the UI."""
        rows = []
        for ctx, arms in self.bandit.stats.items():
            ranked = sorted(arms.items(), key=lambda kv: kv[1][1], reverse=True)
            rows.append({
                "context": ctx,
                "best": ranked[0][0] if ranked else None,
                "chains": [{"chain": k, "score": round(v[1], 3), "trials": int(v[0])}
                           for k, v in ranked if v[0] > 0],
            })
        return [r for r in rows if r["chains"]]

    def leaderboard(self) -> list[dict]:
        """Global ranking of generation chains by average score across all contexts."""
        agg: dict[str, list[float]] = {}
        for arms in self.bandit.stats.values():
            for k, (n, mean) in arms.items():
                a = agg.setdefault(k, [0.0, 0.0])
                a[0] += mean * n
                a[1] += n
        out = [{"chain": k, "avg_score": round(s / n, 3) if n else 0.0, "trials": int(n)}
               for k, (s, n) in agg.items() if n > 0]
        return sorted(out, key=lambda x: x["avg_score"], reverse=True)

    @staticmethod
    def _key(template: str, scene: SceneSpec) -> str:
        return hashlib.sha256(f"{template}|{scenario_key(template, scene)}".encode()).hexdigest()[:16]
