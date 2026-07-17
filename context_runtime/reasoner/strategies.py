"""Generation-strategy registry — the answer-plane arms (SPEC §4.4, generation-strategy layer).

Producing an answer from retrieved context is not one fixed prompt; it is a choice among
*strategies*, each a (system prompt, thinking flag, token budget, cost prior) bundle. The planner
treats the choice as another bandit arm keyed by intent — the same self-optimization it already runs
for retrieval — so one intent classification drives two decisions (retrieval method + generation
strategy). This module holds the strategy definitions and the per-intent warm-start priors; the
bandit refines them from measured reward (Phase 2).

Opt-in: only wired when CR_GENSTRATEGY is set (see planner.candidates); otherwise the planner keeps
emitting the legacy ``single_shot`` reason step and nothing here is used.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

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


def enabled() -> bool:
    """Generation-strategy layer is opt-in via CR_GENSTRATEGY (mirrors CR_DIVER / CR_NEMOTRON)."""
    return os.getenv("CR_GENSTRATEGY", "").strip().lower() in ("1", "true", "yes", "on")


def strategies_for(bucket: str) -> tuple[str, ...]:
    return _ACTIVE_PRIORS.get(bucket, DEFAULT_STRATEGIES)


def set_priors(priors: dict) -> None:
    """Override the active strategy ladders (from a measured ablation). Unknown strategies are dropped;
    each ladder is re-ordered cheapest-capable first so index 0 stays the escalation entry point."""
    for bucket, strats in (priors or {}).items():
        clean = [_BENCH_ALIAS.get(s, s) for s in strats]
        clean = [s for s in dict.fromkeys(clean) if s in STRATEGIES]
        if clean:
            _ACTIVE_PRIORS[bucket] = tuple(sorted(clean, key=lambda s: get(s).cost_units))


def load_priors(path: str) -> dict:
    """Load a compact ``{bucket: [strategies]}`` priors file (written by benchmarks/build_priors.py)
    and apply it. Returns the applied mapping."""
    import json
    priors = json.load(open(path))
    set_priors(priors)
    return priors


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
    return {"enabled": True, "bucket": bucket, "ladder": list(ladder), "candidates": cands}


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
