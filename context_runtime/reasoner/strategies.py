"""Reasoning strategies + generation-strategy registry (SPEC 4.4) - the Reasoning Plane.

This module holds two complementary layers that the planner composes, reconciled here from the two
lines that developed them in parallel:

1. Concrete multi-shot reasoners (plan-worker-critic, debate, tool-loop) - ``ReasonerPlugin`` arms over
   one or more ``ModelPlugin`` calls, rolling sub-call costs into one ``ModelResult`` so the cost model
   prices a multi-shot strategy correctly and EXPLAIN shows every model that ran. Entry point:
   ``reasoner_for(strategy, model)``.
2. The generation-strategy registry / reasoning-effort control - self-consistency, self-refinement,
   effort-vs-model, warm-start priors from the ablation, and model competence, exposed to the planner
   (``strategies_for``, ``enabled``, ``explain_block``, ``refine_depth``, ...) as another bandit arm
   keyed by intent, so one classification drives both the retrieval and the reasoning decision.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable

from ..plugins.base import ModelPlugin
from ..types import ModelRequest, ModelResult, PluginInfo, ReasonRequest
from .single_shot import SingleShotReasoner, _SYSTEM as _ANSWER_SYSTEM


# --- 1. concrete multi-shot reasoner plugins ------------------------------------------------

def _question(ctx) -> str:
    return ctx.plan.intent.normalized or ctx.plan.id


def _rollup(calls: list[ModelResult], final_text: str) -> ModelResult:
    """Fold N sub-call results into one result carrying the final answer + total cost/tokens."""
    if not calls:
        return ModelResult(text=final_text, model="none", tier="none")
    ptoks = sum(r.prompt_tokens for r in calls)
    ctoks = sum(r.completion_tokens for r in calls)
    cost = round(sum(r.est_cost_usd for r in calls), 6)
    models: tuple[str, ...] = tuple(m for r in calls for m in (r.models_used or (r.model,)))
    last = calls[-1]
    return ModelResult(text=final_text, model=last.model, tier=last.tier,
                       prompt_tokens=ptoks, completion_tokens=ctoks, est_cost_usd=cost,
                       models_used=models)


def _ask(model: ModelPlugin, prompt: str, system: str, capability: str, max_tokens: int = 1024) -> ModelResult:
    return model.complete(ModelRequest(messages=({"role": "user", "content": prompt},),
                                       system=system, capability=capability, max_tokens=max_tokens))


# ──────────────────────────── plan → worker → critic ────────────────────────────
_PLAN_SYSTEM = ("Break the question into 2-4 focused sub-questions that, answered together, resolve it. "
                "One sub-question per line, no numbering, no prose.")
_CRITIC_SYSTEM = ("Synthesize a single, correct answer from the sub-answers and the context. "
                  "Cite sources inline like [1]. Silently drop any sub-answer that the context does "
                  "not support — do not invent facts.")


class PlanWorkerCriticReasoner:
    """Decompose the question, answer each part against the context, then synthesize + self-critique."""

    def __init__(self, model: ModelPlugin, *, max_subquestions: int = 3):
        self.model = model
        self.max_subquestions = max_subquestions

    def reason(self, req: ReasonRequest) -> ModelResult:
        ctx = req.context
        q = _question(ctx)
        calls: list[ModelResult] = []

        plan = _ask(self.model, f"Question: {q}", _PLAN_SYSTEM, req.capability, max_tokens=256)
        calls.append(plan)
        subqs = [ln.strip("-• \t") for ln in plan.text.splitlines() if ln.strip()][: self.max_subquestions]
        if not subqs:
            subqs = [q]

        from ._concurrency import fanout
        # Workers are independent (same context, different sub-question) — overlap them when opted in; the
        # plan (above) and critic (below) are the true fork/join boundaries and stay serial. Order preserved.
        workers = fanout(
            lambda sub: _ask(self.model, f"Context:\n{ctx.assembled_text}\n\nSub-question: {sub}",
                             _ANSWER_SYSTEM, req.capability, max_tokens=512),
            subqs)
        calls.extend(workers)
        sub_answers = [f"Q: {sub}\nA: {w.text}" for sub, w in zip(subqs, workers)]

        synth_prompt = (f"Context:\n{ctx.assembled_text}\n\nQuestion: {q}\n\n"
                        f"Sub-answers:\n" + "\n\n".join(sub_answers))
        critic = _ask(self.model, synth_prompt, _CRITIC_SYSTEM, req.capability, max_tokens=1024)
        calls.append(critic)
        return _rollup(calls, critic.text)

    def info(self) -> PluginInfo:
        return PluginInfo(name="plan_worker_critic", kind="reasoner",
                          capabilities=frozenset({"plan_worker_critic"}))


# ──────────────────────────── debate → judge ────────────────────────────
_DEBATER_SYSTEMS = (
    _ANSWER_SYSTEM + " Argue for the most defensible answer you can support from the context.",
    _ANSWER_SYSTEM + " Independently answer; where the context is ambiguous, prefer the more "
                     "conservative reading and note the uncertainty.",
)
_JUDGE_SYSTEM = ("Two independent answers to the same question are given with the context. Produce the "
                 "single best final answer: keep what the context supports, resolve disagreements in "
                 "favor of the citation-backed claim, and cite sources inline like [1].")


class DebateReasoner:
    """Two independent answers, then a judge merges the stronger, citation-backed one."""

    def __init__(self, model: ModelPlugin, *, rounds: int = 2):
        self.model = model
        self.rounds = max(2, rounds)

    def reason(self, req: ReasonRequest) -> ModelResult:
        ctx = req.context
        q = _question(ctx)
        prompt = f"Context:\n{ctx.assembled_text}\n\nQuestion: {q}"
        from ._concurrency import fanout
        # The N debaters are independent (same prompt, different system persona) — overlap them when opted
        # in; the judge below is the join. Order preserved so the judge sees the same answers.
        debaters = fanout(
            lambda i: _ask(self.model, prompt, _DEBATER_SYSTEMS[i % len(_DEBATER_SYSTEMS)],
                           req.capability, max_tokens=768),
            range(self.rounds))
        calls: list[ModelResult] = list(debaters)
        answers = [f"Answer {i + 1}: {r.text}" for i, r in enumerate(debaters)]
        judge_prompt = f"Context:\n{ctx.assembled_text}\n\nQuestion: {q}\n\n" + "\n\n".join(answers)
        judge = _ask(self.model, judge_prompt, _JUDGE_SYSTEM, req.capability, max_tokens=1024)
        calls.append(judge)
        return _rollup(calls, judge.text)

    def info(self) -> PluginInfo:
        return PluginInfo(name="debate", kind="reasoner", capabilities=frozenset({"debate"}))


# ──────────────────────────── bounded tool loop ────────────────────────────
# Convention (kept text-based so it works over any ModelPlugin without extending ModelResult):
#   the model emits either  ACTION: <tool_name> {"json": "args"}   to call a tool, or
#                           FINAL: <answer>                        to stop.
_ACTION_RE = re.compile(r"^\s*ACTION:\s*(\S+)\s*(\{.*\})?\s*$", re.I | re.M)
_FINAL_RE = re.compile(r"FINAL:\s*(.+)\s*$", re.I | re.S)
_TOOL_SYSTEM = (
    "You may call tools to gather facts before answering. To call a tool, output exactly:\n"
    "ACTION: <tool_name> {\"arg\": \"value\"}\n"
    "When you have enough information, output exactly:\nFINAL: <your answer with inline [1] citations>\n"
    "Answer only from the context and tool observations; do not invent facts."
)


class ToolLoopReasoner:
    """A bounded agentic loop. ``tool_runner(name, args) -> str`` executes an injected tool; without
    one (or when the model answers directly) it degrades to a single grounded answer — so it is safe
    as a default and only *adds* capability when tools are wired."""

    def __init__(self, model: ModelPlugin, *, tool_runner: Callable[[str, dict], str] | None = None,
                 tools: tuple[dict, ...] | None = None, max_iters: int = 4):
        self.model = model
        self.tool_runner = tool_runner
        self.tools = tools
        self.max_iters = max_iters

    def reason(self, req: ReasonRequest) -> ModelResult:
        ctx = req.context
        q = _question(ctx)
        transcript = f"Context:\n{ctx.assembled_text}\n\nQuestion: {q}"
        calls: list[ModelResult] = []
        for _ in range(self.max_iters):
            r = self.model.complete(ModelRequest(
                messages=({"role": "user", "content": transcript},),
                system=_TOOL_SYSTEM, capability=req.capability, tools=self.tools, max_tokens=1024))
            calls.append(r)
            final = _FINAL_RE.search(r.text)
            if final:
                return _rollup(calls, final.group(1).strip())
            action = _ACTION_RE.search(r.text)
            if not action or self.tool_runner is None:
                # no tool requested (or none available) → this reply is the answer
                return _rollup(calls, r.text.strip())
            name = action.group(1)
            try:
                args = json.loads(action.group(2)) if action.group(2) else {}
            except json.JSONDecodeError:
                args = {}
            try:
                obs = self.tool_runner(name, args)
            except Exception as e:  # noqa: BLE001 — a failing tool is an observation, not a crash
                obs = f"tool error: {e}"
            transcript += f"\n\nACTION: {name} {action.group(2) or '{}'}\nOBSERVATION: {obs}"
        # ran out of iterations → one final grounded answer
        last = self.model.complete(ModelRequest(
            messages=({"role": "user", "content": transcript + "\n\nAnswer now."},),
            system=_ANSWER_SYSTEM, capability=req.capability, max_tokens=1024))
        calls.append(last)
        return _rollup(calls, last.text.strip())

    def info(self) -> PluginInfo:
        return PluginInfo(name="tool_loop", kind="reasoner", capabilities=frozenset({"tool_loop"}))


# ──────────────────────────── factory ────────────────────────────
def reasoner_for(strategy: str, model: ModelPlugin, **kw):
    """Map a plan's chosen reasoning strategy to a ReasonerPlugin. Unknown → single_shot (safe default)."""
    if strategy == "plan_worker_critic":
        return PlanWorkerCriticReasoner(model, **kw)
    if strategy == "debate":
        return DebateReasoner(model, **kw)
    if strategy == "tool_loop":
        return ToolLoopReasoner(model, **kw)
    return SingleShotReasoner(model)


# --- 2. generation-strategy registry / reasoning-effort control ----------------------------

# Recalibrated abstention — the cure for over-abstention: don't bail when the pieces are present.
_ABSTAIN = ("If the pieces needed to answer are present in the context, reason across them and answer; "
            "say the context is insufficient only if it truly lacks the answer — never invent facts.")


@dataclass(frozen=True)
class GenerationStrategy:
    """One answer-plane arm. ``extractive`` = the model ends with an ``Answer:`` line to pull from a
    reasoning trace; ``cost_units`` is a rough relative prior (calls × tokens) for the cost model and
    the escalation ladder ordering."""
    name: str
    system: str
    thinking: bool
    max_tokens: int
    extractive: bool = False
    cost_units: float = 1.0


# The legacy default keeps the citation prompt + 1024-token budget the SingleShotReasoner used, so
# `single_shot` behaves identically whether routed through here or the original reasoner.
_LEGACY_SYSTEM = ("Answer the question using ONLY the provided context. Cite sources inline like "
                  "[1], [2]. If the context is insufficient, say so plainly — do not invent facts.")

STRATEGIES: dict[str, GenerationStrategy] = {
    "single_shot": GenerationStrategy("single_shot", _LEGACY_SYSTEM, thinking=False, max_tokens=1024, cost_units=1.0),
    # terse — extractive lookup answer, cheapest arm.
    "terse": GenerationStrategy(
        "terse",
        "Answer using ONLY the provided context, in as few words as possible. If the answer is not in "
        "the context, say so — do not invent facts.",
        thinking=False, max_tokens=96, cost_units=0.4),
    # reason — think then a short final answer (single-hop reasoning / synthesis).
    "reason": GenerationStrategy(
        "reason",
        "Answer the question using ONLY the provided context. Think step by step, then give a short "
        "final answer on a line beginning 'Answer:'. " + _ABSTAIN,
        thinking=True, max_tokens=768, extractive=True, cost_units=2.5),
    # decompose — list intermediate facts, answer each, compose (multi-hop).
    "decompose": GenerationStrategy(
        "decompose",
        "Answer the multi-hop question using ONLY the provided context. First list the intermediate "
        "facts needed and answer each from the context, then compose the final answer on a line "
        "beginning 'Answer:'. " + _ABSTAIN,
        thinking=True, max_tokens=1024, extractive=True, cost_units=3.5),
    # mapreduce — extract structured facts per source, then aggregate (counting / temporal aggregation).
    "mapreduce": GenerationStrategy(
        "mapreduce",
        "You aggregate across sources. From the context, extract every relevant fact as a bullet "
        "'- (when, who/what, value)', then compute the answer over those facts and give it on a line "
        "beginning 'Answer:'. " + _ABSTAIN,
        thinking=True, max_tokens=1024, extractive=True, cost_units=4.0),
}

# Warm-start priors: per intent bucket, the strategies to offer (cheapest-capable first). These are
# the escalation-ladder entry points, seeded from the offline oracle ablation (eval_cube2 Phase 0);
# the bandit refines the ordering online. The first entry is the default pick before any learning.
BUCKET_STRATEGIES: dict[str, tuple[str, ...]] = {
    "exact_lookup":   ("terse",),
    "conceptual":     ("reason", "terse"),
    "incident":       ("reason",),
    "code_reasoning": ("reason",),
    "synthesis":      ("reason",),
    "high_risk":      ("reason",),
    "sensitive":      ("reason",),
    "multi_hop":      ("decompose", "reason"),
    "temporal":       ("mapreduce", "reason"),
    "unknown":        ("reason", "terse"),
}

DEFAULT_STRATEGIES = ("reason",)

# The ACTIVE ladders. Defaults to the hand-seeded priors above; a deployment overrides them from the
# benchmark via load_priors / CR_GENSTRATEGY_PRIORS so the warm start is measured, not guessed.
_ACTIVE_PRIORS: dict[str, tuple[str, ...]] = dict(BUCKET_STRATEGIES)

# eval_cube2 datasets → intent buckets (the ablation's regime → CR's classifier bucket).
DATASET_BUCKET = {"popqa": "exact_lookup", "musique": "multi_hop",
                  "longmemeval": "temporal", "tempo": "temporal", "nutrition": "conceptual"}
# the bench names the cheapest arm `direct`; CR calls it `terse`.
_BENCH_ALIAS = {"direct": "terse"}

# ── Verification Optimizer (Phase 4) ──
# Mission classes where a SELF-CHECKED variant of each strategy (verify the answer, retry once if it
# fails the check) is ALSO offered as a distinct arm. The bandit then learns, per class, whether the
# self-check pays off net of its extra cost — verification becomes a learned decision, not always-on.
# Seeded to correctness-sensitive / error-costly classes; refined online like everything else.
VERIFY_BUCKETS = frozenset({"high_risk", "sensitive", "incident", "multi_hop", "temporal"})
VERIFY_FAITHFULNESS_MIN = 0.5   # first attempt below this faithfulness → one retry
VERIFY_COST_MULT = 2.0          # a verified arm's cost prior ≈ generate + check (+ maybe regenerate)


def enabled() -> bool:
    """Generation-strategy layer is opt-in via CR_GENSTRATEGY (mirrors CR_DIVER / CR_NEMOTRON)."""
    return os.getenv("CR_GENSTRATEGY", "").strip().lower() in ("1", "true", "yes", "on")


def strategies_for(bucket: str) -> tuple[str, ...]:
    return _ACTIVE_PRIORS.get(bucket, DEFAULT_STRATEGIES)


def offers_verify(bucket: str) -> bool:
    """Whether to also offer self-checked (verify+retry) variants for this mission class."""
    return enabled() and bucket in VERIFY_BUCKETS


# ── Self-consistency arm (A) — Best@k / majority@k as an effort tier ──
# The strongest inference-scaling lever (Best@k ≫ Pass@1): sample K reasoning traces and return the
# consensus answer. A distinct arm (+sc) the bandit deploys per class where it pays. Opt-in
# (CR_SELFCONSISTENCY) so the default candidate set (verify only) is unchanged.
SC_BUCKETS = frozenset({"high_risk", "sensitive", "incident", "multi_hop", "temporal"})


def self_consistency_enabled() -> bool:
    return os.getenv("CR_SELFCONSISTENCY", "").strip().lower() in ("1", "true", "yes", "on")


def self_consistency_k() -> int:
    """Sample count K (CR_SC_K, default 5, min 2). Best@k rises with K."""
    try:
        k = int(os.getenv("CR_SC_K", "5"))
        return k if k >= 2 else 5
    except ValueError:
        return 5


def self_consistency_temp() -> float:
    """Sampling temperature for the K samples (CR_SC_TEMP, default 0.7) — must be > 0 or the samples
    don't diverge and self-consistency is vacuous."""
    try:
        t = float(os.getenv("CR_SC_TEMP", "0.7"))
        return t if t > 0 else 0.7
    except ValueError:
        return 0.7


def offers_self_consistency(bucket: str) -> bool:
    """Whether to also offer a +sc variant for this class (thinking arms only)."""
    return enabled() and self_consistency_enabled() and bucket in SC_BUCKETS


# ── Self-refinement depth (C) — verify+retry as an N-iteration budget ──
def refine_depth() -> int:
    """Self-check→retry rounds for a verified arm (CR_REFINE_DEPTH, default 1 = the prior single retry).
    Sequential self-refinement scales quality; the bandit still learns whether the extra rounds pay."""
    try:
        d = int(os.getenv("CR_REFINE_DEPTH", "1"))
        return d if d >= 1 else 1
    except ValueError:
        return 1


# ── Effort-up vs model-up (B) ──
def effort_vs_model() -> bool:
    """Offer a cheap-tier bucket's ladder at BOTH its tier and the strong tier, so the bandit weighs
    'more effort, same model' against 'bigger model' (CR_EFFORT_VS_MODEL). Opt-in."""
    return os.getenv("CR_EFFORT_VS_MODEL", "").strip().lower() in ("1", "true", "yes", "on")


def set_priors(priors: dict) -> None:
    """Override the active strategy ladders (from a measured ablation). Unknown strategies are dropped;
    each ladder is re-ordered cheapest-capable first so index 0 stays the escalation entry point."""
    for bucket, strats in (priors or {}).items():
        clean = [_BENCH_ALIAS.get(s, s) for s in strats]
        clean = [s for s in dict.fromkeys(clean) if s in STRATEGIES]
        if clean:
            _ACTIVE_PRIORS[bucket] = tuple(sorted(clean, key=lambda s: get(s).cost_units))


def load_priors(path: str) -> dict:
    """Load + apply a priors file written by benchmarks/build_priors.py. Two shapes are accepted:
    the compact ``{bucket: [strategies]}`` map, or the richer ``{"strategies": {...},
    "model_competence": {...}}``. Returns the parsed object."""
    import json
    data = json.load(open(path))
    if isinstance(data, dict) and ("strategies" in data or "model_competence" in data):
        set_priors(data.get("strategies") or {})
        set_model_competence(data.get("model_competence") or {})
    else:
        set_priors(data)
    return data


def priors_from_ablation(results_dir: str, *, dataset_bucket: dict | None = None,
                         cond: str = "oracle", margin: float = 0.1) -> dict:
    """Compute per-bucket strategy ladders from the eval_cube2 cell JSONs. For each bucket (via
    ``dataset_bucket``), average the cells' ``acc_<cond>`` per strategy, keep every strategy within
    ``margin`` of the bucket's best (so a cheap-but-adequate arm stays the entry point and better
    costlier ones stay on the ladder for escalation), and order cheapest-first. ``cond=oracle``
    isolates generation from retrieval — the right signal to seed a generation prior."""
    import glob
    import json

    dataset_bucket = dataset_bucket or DATASET_BUCKET
    agg: dict[str, dict[str, list]] = {}
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        try:
            cell = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        bucket = dataset_bucket.get(cell.get("dataset"))
        strat = _BENCH_ALIAS.get(cell.get("strategy"), cell.get("strategy"))
        acc = cell.get(f"acc_{cond}")
        if not bucket or strat not in STRATEGIES or acc is None:
            continue
        agg.setdefault(bucket, {}).setdefault(strat, []).append(float(acc))

    priors: dict[str, tuple[str, ...]] = {}
    for bucket, per_strat in agg.items():
        mean = {s: sum(v) / len(v) for s, v in per_strat.items()}
        best = max(mean.values())
        keep = [s for s, a in mean.items() if a >= best - margin] or [max(mean, key=mean.get)]
        priors[bucket] = tuple(sorted(keep, key=lambda s: get(s).cost_units))
    return priors


# ── Model competence (Phase 5) — "bigger ≠ better", learned per mission class ──
# The reasoning arm already includes the model tier (see optimizer.plan_key), so the bandit learns
# model competence per class ONLINE. This is its warm start + its transparency: which model actually
# succeeds on which mission class, measured at oracle context (generation isolated from retrieval).
_MODEL_COMPETENCE: dict[str, dict[str, float]] = {}


def model_competence_from_ablation(results_dir: str, *, dataset_bucket: dict | None = None,
                                   cond: str = "oracle") -> dict:
    """``{bucket: {model: mean_acc}}`` from the eval_cube2 cells — the measured 'DeepSeek here, Qwen
    there' signal. A model that over-abstains on a class shows a low number here and is routed around."""
    import glob
    import json

    dataset_bucket = dataset_bucket or DATASET_BUCKET
    agg: dict[str, dict[str, list]] = {}
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        try:
            cell = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        bucket = dataset_bucket.get(cell.get("dataset"))
        model = cell.get("model")
        acc = cell.get(f"acc_{cond}")
        if not bucket or not model or acc is None:
            continue
        agg.setdefault(bucket, {}).setdefault(model, []).append(float(acc))
    return {b: {m: round(sum(v) / len(v), 4) for m, v in per.items()} for b, per in agg.items()}


def set_model_competence(comp: dict) -> None:
    _MODEL_COMPETENCE.clear()
    _MODEL_COMPETENCE.update({b: dict(m) for b, m in (comp or {}).items()})


def competent_model(bucket: str) -> str | None:
    """The best-measured model for a mission class (None if unknown) — the warm-start model pick."""
    comp = _MODEL_COMPETENCE.get(bucket)
    return max(comp, key=comp.get) if comp else None


def get(name: str) -> GenerationStrategy:
    return STRATEGIES.get(name) or STRATEGIES["single_shot"]


def explain_block(bucket: str, *, method: str = "", tier: str = "", bandit=None) -> dict:
    """The generation-plane 'show your work' for the transparency panel + EXPLAIN — the mirror of the
    retrieval decision block. Lists the intent bucket's strategy ladder, each arm's config (thinking,
    budget, cost prior), the entry point (the default first pick), and — when a generation bandit is
    supplied — the learned value per strategy arm (arm key = ``method:strategy:tier``, matching
    ``optimizer.online.plan_key``). Off → reports the legacy single_shot."""
    if not enabled():
        return {"enabled": False, "strategy": "single_shot",
                "note": "generation-strategy layer off (set CR_GENSTRATEGY=1)"}
    ladder = strategies_for(bucket)
    cands = []
    for i, name in enumerate(ladder):
        spec = get(name)
        arm = f"{method}:{name}:{tier}" if (method or tier) else name
        n, val = 0, 0.0
        if bandit is not None:
            try:
                n, val = bandit.value(bucket, arm)
            except Exception:  # noqa: BLE001 — transparency must never break serving
                pass
        cands.append({"strategy": name, "thinking": spec.thinking, "max_tokens": spec.max_tokens,
                      "cost_units": spec.cost_units, "entry_point": i == 0,
                      "bandit": {"n": int(n), "value": round(float(val), 4)}})
    return {"enabled": True, "bucket": bucket, "ladder": list(ladder), "candidates": cands,
            "verify_offered": bucket in VERIFY_BUCKETS,
            "sc_offered": offers_self_consistency(bucket), "sc_k": self_consistency_k(),
            "effort_menu": _effort_menu(bucket, method=method, tier=tier, bandit=bandit),
            "effort_note": ("cost_units vs learned value — pareto_optimal marks the non-dominated "
                            "effort tier (no cheaper arm scores as high); effort is spent up the frontier."),
            "model_competence": _MODEL_COMPETENCE.get(bucket),
            "competent_model": competent_model(bucket)}


def _effort_menu(bucket: str, *, method: str = "", tier: str = "", bandit=None) -> list[dict]:
    """The full effort menu (D): each strategy × {base, +sc, +v, +sc+v} with its cost prior and learned
    value, flagging the cost/quality Pareto frontier (an arm is pareto-optimal if no other arm has
    ≤ cost AND ≥ value). Makes 'picked the cheapest arm that scores as high' legible in EXPLAIN."""
    offer_v = enabled() and bucket in VERIFY_BUCKETS
    offer_sc = offers_self_consistency(bucket)
    k = self_consistency_k()
    arms = []
    for name in strategies_for(bucket):
        spec = get(name)
        variants = [("", spec.cost_units)]
        if offer_sc and spec.thinking:
            variants.append(("+sc", spec.cost_units * k))
        if offer_v:
            variants.append(("+v", spec.cost_units * VERIFY_COST_MULT))
        if offer_sc and offer_v and spec.thinking:
            variants.append(("+sc+v", spec.cost_units * k * VERIFY_COST_MULT))
        for suffix, cost in variants:
            arm_key = f"{method}:{name}{suffix}:{tier}" if (method or tier) else f"{name}{suffix}"
            n, val = 0, 0.0
            if bandit is not None:
                try:
                    n, val = bandit.value(bucket, arm_key)
                except Exception:  # noqa: BLE001 — transparency must never break serving
                    pass
            arms.append({"arm": f"{name}{suffix}", "strategy": name, "variant": suffix,
                         "cost_units": round(float(cost), 4), "value": round(float(val), 4), "n": int(n)})
    for a in arms:
        a["pareto_optimal"] = not any(
            b is not a and b["cost_units"] <= a["cost_units"] and b["value"] >= a["value"]
            and (b["cost_units"] < a["cost_units"] or b["value"] > a["value"])
            for b in arms)
    return arms


def extract_final(text: str) -> str:
    """Pull the final answer from a reasoning response: strip <think> blocks, take the last
    'Answer:' line if present, else the last non-empty line."""
    import re
    body = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    for line in reversed(body.splitlines()):
        s = line.strip()
        if s.lower().startswith("answer:"):
            return s.split(":", 1)[1].strip()
    return body.splitlines()[-1].strip() if body.strip() else body


# Auto-load measured priors at import when CR_GENSTRATEGY_PRIORS points at a priors file — so a
# deployment's warm start comes from its own ablation without editing this module.
_PRIORS_FILE = os.getenv("CR_GENSTRATEGY_PRIORS", "").strip()
if _PRIORS_FILE:
    try:
        load_priors(_PRIORS_FILE)
    except Exception:  # noqa: BLE001 — a missing/bad priors file must never break import
        pass
