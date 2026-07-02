"""agentic-books × Context Runtime — offline ledger/report selection benchmark.

Simulates 72 bookkeeping questions where Context Runtime picks a report bundle per
question bucket, learns from an answer-correctness proxy minus data-pull cost, and
beats a fixed full-books baseline. Run with:

    PYTHONPATH=. python examples/agentic_books.py
"""
from __future__ import annotations

from statistics import mean

from context_runtime.integrations.agentic_books import (
    DECISIVE_BY_BUCKET,
    DEFAULT_BOOKS,
    AgenticBooksTenant,
    BooksReportBundle,
    reward_from_answer,
)

ROUNDS = 72
BASELINE_BUNDLE = DEFAULT_BOOKS[0]  # full_books — always correct, always most expensive

QUESTION_STREAM = [
    ("Who owes us money right now?", "ar"),
    ("Which vendor bills are due this week?", "ap"),
    ("What is our sales tax liability this quarter?", "tax"),
    ("Is the month-end trial balance reconciled?", "close"),
    ("Show AR aging for overdue customers", "ar"),
    ("Total we owe suppliers this month", "ap"),
    ("VAT owed on last quarter's revenue", "tax"),
    ("Ready to close the books for the period?", "close"),
]

_STATE = [0xBEEFCAFE]


def _rand() -> float:
    x = _STATE[0]
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= x >> 17
    x ^= (x << 5) & 0xFFFFFFFF
    _STATE[0] = x & 0xFFFFFFFF
    return _STATE[0] / 0x100000000


def _value(chosen: BooksReportBundle, bucket: str) -> float:
    decisive = DECISIVE_BY_BUCKET[bucket]
    base = 6.5 if getattr(chosen, decisive) else 2.0
    noise = (_rand() - 0.5) * 0.6
    return max(0.0, base + noise)


def run(rounds: int = ROUNDS) -> None:
    tenant = AgenticBooksTenant(epsilon=0.15)
    learned_rewards: list[float] = []
    baseline_rewards: list[float] = []

    print("First few books decisions (bundle → value → reward):\n")

    for i in range(rounds):
        question, bucket = QUESTION_STREAM[i % len(QUESTION_STREAM)]
        chosen = tenant.choose(question, bucket=bucket)
        value = _value(chosen, bucket)
        reward = tenant.record_outcome(question, value)
        learned_rewards.append(reward)

        baseline_value = _value(BASELINE_BUNDLE, bucket)
        baseline_rewards.append(reward_from_answer(baseline_value, BASELINE_BUNDLE))

        if i < 6:
            print(f"  {question[:44]:<44} → {chosen.key:<12} value={value:4.1f} reward={reward:5.2f}")
        elif i == 6:
            print("  ...")

    avg_learned = mean(learned_rewards[-18:])
    avg_baseline = mean(baseline_rewards[-18:])

    print("\nreward = answer correctness − data pull cost\n")
    print(f"Context Runtime (learned): {avg_learned:.3f}")
    print(f"baseline ({BASELINE_BUNDLE.key}): {avg_baseline:.3f}")

    print("\nlearned policy per bucket:\n")
    policy = tenant.policy()
    if not policy:
        print("  (unlearned)")
    else:
        for bucket_key in sorted(policy):
            print(f"  {bucket_key:<10} → {policy[bucket_key]}")


if __name__ == "__main__":
    run()
