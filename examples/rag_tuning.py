"""redevops-rag × Context Runtime — tune retrieval knobs per query intent (replay harness).

Same shape as the sidekick harness: the FULL Context Runtime machinery is real (planner,
bandit, cost-model), only the *retrieval-quality signal* is simulated via a hidden
latent-best config per intent (a real run measures quality from labels / answer
correctness, needing the [rag] extra + torch). If Context Runtime is tuning well, its
reward (quality − efficiency penalty) should beat the library's fixed default config.

    python examples/rag_tuning.py
"""
from __future__ import annotations

from context_runtime.integrations.redevops_rag import (
    ContextRuntimeRetrieverTuner,
    DEFAULT_ARMS,
    reward_from_quality,
    _rag_bandit,
)

# Hidden truth: each query type is best served by a particular config arm. The tuner
# must discover this from reward alone. (Indices into DEFAULT_ARMS.)
#   exact_lookup → cheap/precise (arm 0);  conceptual/synthesis → thorough+rerank (arm 2)
#   incident → recency-biased (arm 4)
LATENT = {
    "exact_lookup": DEFAULT_ARMS[0].key,
    "conceptual": DEFAULT_ARMS[2].key,
    "synthesis": DEFAULT_ARMS[2].key,
    "incident": DEFAULT_ARMS[4].key,
    "code_reasoning": DEFAULT_ARMS[1].key,
}
DEFAULT_KEY = DEFAULT_ARMS[1].key   # the library default the tuner must beat

QUERIES = [
    "look up error code 429 in the logs",
    "what is the difference between RRF and reranking",
    "explain why the nightly deploy failed last night",
    "how do we rotate API keys",
    "summarize our incident response process",
]


def _quality(chosen_key: str, bucket: str, rng: list[int]) -> float:
    """Simulated retrieval quality in [0,1]: matched config retrieves the gold chunk high
    (MRR≈0.8-1.0); mismatched config buries it (MRR≈0.2-0.5)."""
    rng[0] = (rng[0] * 1103515245 + 12345) & 0x7FFFFFFF
    roll = rng[0] / 0x7FFFFFFF
    latent = LATENT.get(bucket, DEFAULT_KEY)
    if chosen_key == latent:
        return 0.8 + 0.2 * roll
    return 0.2 + 0.3 * roll


def run(rounds: int = 40) -> None:
    tuner = ContextRuntimeRetrieverTuner(bandit=_rag_bandit(0.15))
    rng = [0xC0FFEE]
    tuned: list[float] = []
    fixed: list[float] = []

    for i in range(rounds):
        q = QUERIES[i % len(QUERIES)]
        cfg = tuner.choose(q)
        plan, _ = tuner._pending[tuner._key(q)]
        bucket = plan.intent.bucket

        q_tuned = _quality(cfg.key, bucket, rng)
        r_tuned = tuner.record_outcome(q, quality=q_tuned, latency_s=cfg.cost_units() * 0.3)
        tuned.append(r_tuned)

        # baseline: always the library's fixed default config
        q_fixed = _quality(DEFAULT_KEY, bucket, rng)
        fixed.append(reward_from_quality(q_fixed, DEFAULT_ARMS[1]))

    w = 10
    print(f"reward = retrieval-quality − efficiency-penalty (higher = better & cheaper)\n")
    print(f"  Context Runtime (tuned per intent): {sum(tuned[-w:]) / w:.3f}")
    print(f"  redevops-rag fixed default:   {sum(fixed[-w:]) / w:.3f}")
    print("\n── learned config policy (best knobs per intent bucket) ──")
    for bucket, key in tuner.policy().items():
        flag = " ✓ matches latent best" if key == LATENT.get(bucket) else ""
        print(f"  {bucket:<14} → {key}{flag}")
    print("\n── cost-model calibration after", rounds, "observed retrievals ──")
    for f in tuner.runtime.estimator.statistics().fields:
        print(f"  {f.field:<18} samples={f.sample_count} calibration={f.calibration}")


if __name__ == "__main__":
    run()
