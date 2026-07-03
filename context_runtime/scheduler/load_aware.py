"""Load-aware sizing of the expensive stage (DSpark's prefix scheduler, for retrieval).

DSpark sizes each request's *verification length* by cumulative survival probability and
current engine load: when idle it verifies deep; when saturated it prunes the
low-confidence suffix so it doesn't burn batch capacity that could serve other requests.

We port the idea to the retrieval → (rerank / synthesis) boundary. Given calibrated
per-passage P(relevant) in rank order, we compute the DSpark prefix-survival product
`a_j = Π_{i<=j} p_i` and greedily admit passages into the expensive stage until the
marginal survival falls below a load-dependent floor (idle ⇒ tiny floor, admit almost
everything; busy ⇒ high floor, keep only the near-certain prefix), never exceeding the
budget the profiled cost table says fits the latency ceiling.

Crucially this only ever **trims** the bandit's chosen arm — it caps `final_k` and can
turn `rerank` off under load, but never enlarges depth or turns rerank on. The bandit
still owns "what's the best arm for this bucket"; the sizer makes that arm's depth
load-conditioned instead of fixed, so the two cooperate rather than fight.

Pure and side-effect-free ⇒ trivially unit-testable; all I/O (load band, cost table)
is passed in.
"""
from __future__ import annotations

from dataclasses import dataclass

# Survival-product floor per load band: how likely the *whole prefix* up to a passage
# must be to still admit that passage. Idle ⇒ admit almost everything; busy ⇒ keep only
# the near-certain prefix. These are the retrieval analogue of DSpark's SPS-driven cliff.
BAND_FLOOR = {"lo": 0.02, "mid": 0.15, "hi": 0.40}


@dataclass(frozen=True)
class SizingDecision:
    final_k: int          # how many passages to keep for the expensive stage
    rerank: bool          # whether to run the rerank pass
    kept_survival: float   # expected accepted relevance τ of the kept prefix (Σ survival)
    reason: str


def size_expensive_stage(
    probs: list[float],
    *,
    load_band: str = "lo",
    requested_k: int,
    requested_rerank: bool,
    cost_profile=None,
    rerank_stage: str = "rerank",
    max_latency_seconds: float | None = None,
    min_keep: int = 1,
) -> SizingDecision:
    """Choose how deep the expensive stage runs for one request.

    ``probs`` are calibrated P(relevant) in retrieval rank order (descending expected
    relevance). ``requested_k`` / ``requested_rerank`` come from the bandit's chosen arm
    and act as hard upper bounds — the sizer only trims.
    """
    n = min(len(probs), max(0, requested_k))
    if n == 0:
        return SizingDecision(0, False, 0.0, "no-candidates")

    floor = BAND_FLOOR.get(load_band, BAND_FLOOR["lo"])

    # Greedy admit along the survival product: stop at the first passage whose cumulative
    # survival drops below the load floor (DSpark's early-stop on the prefix).
    survival = 1.0
    kept = 0
    accepted = 0.0
    for j in range(n):
        survival *= max(0.0, min(1.0, probs[j]))
        if kept >= min_keep and survival < floor:
            break
        kept += 1
        accepted += survival

    # Budget guard: if a profiled latency curve is available and a ceiling is set, don't
    # admit past the count whose measured cost fits. Reverts to no-op when unprofiled.
    if cost_profile is not None and max_latency_seconds is not None:
        while kept > min_keep:
            lat = cost_profile.latency(rerank_stage, kept)
            if lat is None or lat <= max_latency_seconds:
                break
            kept -= 1

    # Rerank is the expensive pass: keep it when the arm asked AND we're not saturated;
    # drop it under heavy load. Never enable a rerank the arm didn't request.
    rerank = requested_rerank and load_band != "hi"
    reason = f"band={load_band} floor={floor} kept={kept}/{n}"
    return SizingDecision(final_k=kept, rerank=rerank,
                          kept_survival=round(accepted, 4), reason=reason)
