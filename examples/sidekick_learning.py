"""sidekick × ContextOS — prove the learning loop closes (replay harness).

A real benchmark needs headless Claude Code sessions + the redevops-rag retriever
(torch), neither of which runs in a CI sandbox. So this harness keeps the FULL
ContextOS machinery real (planner, retriever, bandit, cost-model statistics) and only
*simulates the acceptance signal* the way sidekick's metrics would report it — via a
hidden "latent best strategy" per task type. If ContextOS is learning, its acceptance
rate should climb toward the baseline-beating ceiling as the bandit discovers that
latent optimum from reward alone.

    python examples/sidekick_learning.py

Output: a learning curve (ContextOS vs. a fixed naive-recall baseline) + the learned
per-bucket recall policy.
"""
from __future__ import annotations

from contextos import ContextRuntime
from contextos.integrations.sidekick import (
    ContextOSSkillStore,
    Skill,
    SubtaskOutcome,
    _sidekick_bandit,
)

# A skill corpus (what sidekick would have distilled from past runs).
SKILLS = [
    Skill("retry-flaky-tests", "tests fail intermittently", "wrap in retry, isolate the seed", ["pytest -q passes 3x"]),
    Skill("add-cli-flag", "expose a new command-line option", "argparse add_argument + thread through", ["--flag works"]),
    Skill("fix-import-cycle", "circular import error", "move shared types to a leaf module", ["imports cleanly"]),
    Skill("port-error-codes", "map error codes to messages", "lookup table + BM25 over logs", ["ERR-500 resolves"]),
    Skill("refactor-function", "split a large function", "extract helpers, keep signature", ["tests green"]),
]

# A stream of tasks. Each maps to an intent bucket; a *hidden* latent-best strategy key
# decides acceptance probability (the thing the bandit must discover from reward only).
TASKS = [
    ("Fix ERR-500 showing up in the logs", "exact_lookup", "bm25:3:1500"),
    ("Refactor the giant handler function into helpers", "code_reasoning", "hybrid:8:4000"),
    ("Add a --verbose CLI flag to the tool", "code_reasoning", "hybrid:8:4000"),
    ("Why does the deploy keep failing intermittently", "incident", "hybrid:5:3000"),
    ("Resolve the circular import in the planner", "code_reasoning", "hybrid:8:4000"),
    ("Look up status code 429 handling", "exact_lookup", "bm25:3:1500"),
]


def _simulate_outcome(chosen_key: str, latent_best: str, rng: list[int]) -> SubtaskOutcome:
    """Faithful stand-in for a sidekick run: matching the latent-best strategy → high
    acceptance + cheaper/first-try; mismatch → often rejected or needs a retry."""
    # tiny deterministic prng
    x = rng[0] ^ (rng[0] << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    rng[0] = x & 0xFFFFFFFF
    roll = (rng[0] & 0xFFFF) / 0xFFFF

    match = chosen_key == latent_best
    accepted = roll < (0.9 if match else 0.45)
    first = accepted and roll < (0.7 if match else 0.2)
    # matched strategy spends its budget well; mismatch wastes tokens
    budget = int(chosen_key.split(":")[2])
    tokens = budget + (500 if match else 3500)
    return SubtaskOutcome(accepted=accepted, first_attempt=first, tokens_total=tokens,
                          cost_usd=tokens / 1000 * 0.002, wall_ms=tokens * 2)


def run(rounds: int = 30) -> None:
    rt = ContextRuntime.default([])
    store = ContextOSSkillStore(".contextos/skills_demo", runtime=rt,
                                bandit=_sidekick_bandit(0.15))
    for sk in SKILLS:
        store.save(sk)

    rng = [0x1234567]
    window = 6
    history: list[int] = []
    base_history: list[int] = []

    print(f"{'round':>5} {'task':<42} {'bucket':<14} {'chosen':<14} {'ok?':<4} reward")
    for i in range(rounds):
        task, _bucket, latent = TASKS[i % len(TASKS)]
        # ── ContextOS path: bandit picks a strategy, recall, simulate, learn ──
        store.recall(task, limit=3)
        plan, strat = store._pending[store._key(task)]
        out = _simulate_outcome(strat.key, latent, rng)
        reward = store.record_outcome(task, out)
        history.append(1 if out.accepted else 0)

        # ── baseline: naive fixed recall (always hybrid:5:3000, no learning) ──
        base_out = _simulate_outcome("hybrid:5:3000", latent, rng)
        base_history.append(1 if base_out.accepted else 0)

        if i < 6 or i >= rounds - 6:
            print(f"{i:>5} {task[:42]:<42} {plan.intent.bucket:<14} {strat.key:<14} "
                  f"{'yes' if out.accepted else 'no':<4} {reward}")
        elif i == 6:
            print(f"{'...':>5}")

    def rate(h):
        return sum(h[-window:]) / min(window, len(h))

    print("\n── acceptance over last", window, "rounds ──")
    print(f"  ContextOS (learning):  {rate(history):.0%}")
    print(f"  naive baseline (fixed): {rate(base_history):.0%}")
    print("\n── learned recall policy (best strategy per intent bucket) ──")
    for bucket, key in store.bandit.policy().items():
        print(f"  {bucket:<14} → {key}")
    print("\n── cost-model calibration after", rounds, "observed runs ──")
    for f in rt.estimator.statistics().fields:
        print(f"  {f.field:<18} samples={f.sample_count} calibration={f.calibration}")


if __name__ == "__main__":
    run()
