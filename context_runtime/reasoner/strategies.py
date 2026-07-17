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


def enabled() -> bool:
    """Generation-strategy layer is opt-in via CR_GENSTRATEGY (mirrors CR_DIVER / CR_NEMOTRON)."""
    return os.getenv("CR_GENSTRATEGY", "").strip().lower() in ("1", "true", "yes", "on")


def strategies_for(bucket: str) -> tuple[str, ...]:
    return BUCKET_STRATEGIES.get(bucket, DEFAULT_STRATEGIES)


def get(name: str) -> GenerationStrategy:
    return STRATEGIES.get(name) or STRATEGIES["single_shot"]


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
