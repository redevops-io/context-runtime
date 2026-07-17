"""Generation reward + escalation (Phase 2) — the measured signal that makes the answer plane
self-optimizing.

Retrieval reward is measurable online; generation reward normally needs gold. The cheapest form that
answers the question here is a composite verifier:

  * faithfulness — is the answer grounded in the retrieved context? (a cheap content-overlap proxy by
    default; swap in an NLI/LLM judge via the ``judge`` hook for higher fidelity)
  * abstention — an answer that gives up earns a low floor, so the escalation ladder is favored when a
    costlier strategy might succeed (this is what cures over-abstention)
  * self-consistency — optional agreement across K samples, for reasoning buckets

The reward folds into the SAME outcome loop the retrieval bandit uses (``OutcomeEvent`` →
``bandit.update``), keyed by the plan arm — which now includes the generation strategy. The escalation
helpers pick the next rung when a cheap strategy underperforms; the optimizer runs it off the serving
path (``select(..., shadow=True)``) and feeds its reward back, so latency stays bounded.
"""
from __future__ import annotations

import re
from typing import Callable

from . import strategies

_WORD = re.compile(r"[a-z0-9]+")
_ABSTAIN_MARKERS = ("not found", "insufficient", "cannot answer", "can't answer", "do not know",
                    "don't know", "no answer", "not in the context", "unable to")
_STOP = {"the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are", "was", "were",
         "it", "its", "as", "at", "by", "be", "this", "that", "with", "from"}

# Reward below this → escalate to the next (costlier) strategy on the ladder.
ESCALATE_BELOW = 0.5
# An abstaining answer earns this floor: low enough to lose to any real answer, non-zero so a genuinely
# unanswerable query (every strategy abstains) doesn't drive the arm's value negative.
ABSTAIN_FLOOR = 0.1


def _content_tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall((text or "").lower()) if t not in _STOP and len(t) > 1}


def is_abstention(answer: str) -> bool:
    a = (answer or "").strip().lower()
    return not a or any(m in a for m in _ABSTAIN_MARKERS)


def faithfulness(answer: str, context: str) -> float:
    """Cheap grounding proxy in [0,1]: the fraction of the answer's content tokens that appear in the
    context. High = grounded; low = the answer asserts things the context doesn't support
    (hallucination or partial multi-hop). Empty answer → 0."""
    ans = _content_tokens(answer)
    if not ans:
        return 0.0
    ctx = _content_tokens(context)
    return len(ans & ctx) / len(ans)


def self_consistency(samples: list[str]) -> float:
    """Agreement across K samples in [0,1]: the largest cluster sharing ≥60% content-token overlap,
    normalized by K. 1.0 = all samples agree. Only meaningful for reasoning buckets."""
    samples = [s for s in (samples or []) if s and not is_abstention(s)]
    if len(samples) < 2:
        return 1.0 if samples else 0.0
    toks = [_content_tokens(s) for s in samples]
    best = 1
    for i, a in enumerate(toks):
        agree = 1 + sum(1 for j, b in enumerate(toks) if j != i and a and len(a & b) / max(len(a), 1) >= 0.6)
        best = max(best, agree)
    return best / len(samples)


class GenerationVerifier:
    """Produce a measured reward for a generated answer. ``judge`` (optional) is a
    ``judge(answer, context, question) -> float in [0,1]`` — e.g. an LLM grader — used in place of the
    faithfulness proxy when supplied. Everything else stays cheap and offline."""

    def __init__(self, judge: Callable[[str, str, str], float] | None = None,
                 *, w_faith: float = 0.7, w_consistency: float = 0.3):
        self.judge = judge
        self.w_faith, self.w_consistency = w_faith, w_consistency

    def reward(self, answer: str, context: str, question: str = "",
               samples: list[str] | None = None) -> tuple[float, dict]:
        abstained = is_abstention(answer)
        if abstained:
            return ABSTAIN_FLOOR, {"abstained": True, "faithfulness": 0.0, "consistency": None}
        faith = self.judge(answer, context, question) if self.judge else faithfulness(answer, context)
        cons = self_consistency(samples) if samples else None
        reward = faith if cons is None else (self.w_faith * faith + self.w_consistency * cons)
        return round(float(reward), 4), {"abstained": False, "faithfulness": round(faith, 4),
                                         "consistency": None if cons is None else round(cons, 4)}


# ─────────────────────────── escalation ladder ───────────────────────────
def should_escalate(reward: float, threshold: float = ESCALATE_BELOW) -> bool:
    return reward < threshold


def next_strategy(bucket: str, current: str) -> str | None:
    """The next (costlier) strategy on the bucket's ladder after ``current``; None at the top. The
    ladder order is the warm-start priors (cheapest-capable first)."""
    ladder = strategies.strategies_for(bucket)
    if current in ladder:
        i = ladder.index(current)
        return ladder[i + 1] if i + 1 < len(ladder) else None
    return ladder[0] if ladder else None


# ─────────────────────────── feedback: reward → bandit ───────────────────────────
def apply_feedback(optimizer, plan, answer: str, context: str, *, question: str = "", bucket: str = "",
                   verifier: GenerationVerifier | None = None, samples: list[str] | None = None) -> dict:
    """Score the served answer and fold the reward into the optimizer's learned policy (the same
    ``learn_from_plan`` path retrieval uses), then report whether to escalate. Returns a dict with the
    reward, the verifier signals, the OutcomeEvent, and the next strategy to shadow (or None). Pass
    ``bucket`` (the selected Plan carries ``intent.bucket == 'unknown'`` until the runtime attaches it)
    so escalation walks the right ladder."""
    from ..learning.events import OutcomeEvent

    verifier = verifier or GenerationVerifier()
    reward, signals = verifier.reward(answer, context, question, samples)
    # feed the same loop as retrieval; learn_from_plan reads plan.extra['bandit'] (arm includes strategy)
    if hasattr(optimizer, "learn_from_plan"):
        optimizer.learn_from_plan(plan, reward)
    event = OutcomeEvent.from_plan(plan, reward, verified=not signals["abstained"])
    strat = next((s.params.get("strategy") for s in plan.chosen.steps if s.type == "reason"), None)
    bucket = bucket or (getattr(getattr(plan, "intent", None), "bucket", "") or "")
    escalate_to = next_strategy(bucket, strat) if should_escalate(reward) else None
    return {"reward": reward, "signals": signals, "event": event, "escalate_to": escalate_to}
