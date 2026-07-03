"""chat-memory × Context Runtime — prove CR learns which memory index to read.

A 3-index agent memory (recency / semantic / entity) exposes three recall MODES.
Reading all three every turn always finds the answer but pays the full read cost;
the right SINGLE index for a given kind of question finds it far cheaper. Context
Runtime should learn, per query bucket, which single index to read — and beat the
fixed "read all three" baseline on value − cost.

The Context Runtime machinery is REAL (the ChatMemoryStore retrieves for real; the
per-bucket EpsilonGreedyBandit learns from reward). Retrieval quality drives the
reward: value if the needed turn is recalled, minus the mode's read cost.

    python examples/chat_memory.py

Output: a learning curve (CR vs the read-all-three baseline) + the learned
per-bucket recall policy.
"""
from __future__ import annotations

from context_runtime.integrations.chat_memory import (
    FULL_BUNDLE,
    ChatMemoryStore,
    ChatMemoryTenant,
    Turn,
    extract_entities,
    memory_bucket,
)

VALUE = 3.0   # value of recalling the needed turn


def build_memory() -> ChatMemoryStore:
    """A support conversation. `ts` is a monotonic turn index (larger = more recent)."""
    raw = [
        # older turns (out of the recency window) — recalled by semantic/entity, not recency
        ("t01", "user", "We evaluated the Starter and Growth plans for the rollout."),
        ("t02", "assistant", "For pricing we picked the Enterprise tier with annual billing."),
        ("t03", "user", "Alice will own the migration and is the primary account contact."),
        ("t04", "assistant", "Bob handles the SSO integration and the SAML metadata."),
        ("t05", "user", "The data residency requirement is EU-only for this customer."),
        ("t06", "assistant", "We agreed the SLA is 99.9% with a 4-hour response window."),
        ("t07", "user", "Refunds follow a prorated model on annual downgrades."),
        ("t08", "assistant", "The onboarding call is scheduled and the checklist is shared."),
        # recent turns (inside the recency window) — recalled by recency
        ("t09", "user", "Okay that all sounds good to me."),
        ("t10", "assistant", "Great — I'll draft the summary and send it over shortly."),
        ("t11", "user", "Let's also confirm the launch date before we wrap."),
        ("t12", "assistant", "Launch is set for the 15th, pending the final sign-off."),
    ]
    turns = [Turn(tid, role, text, ts=float(i + 1), entities=extract_entities(text))
             for i, (tid, role, text) in enumerate(raw)]
    return ChatMemoryStore(turns)


# Each probe: the query, and the turn the answer lives in. Grouped by the bucket whose
# decisive index should recall it cheaply.
PROBES = [
    # followup → recency: generic continuation; the needed context is simply recent.
    ("and then what did we say", "t12", "followup"),
    ("keep going", "t11", "followup"),
    ("continue please", "t12", "followup"),
    # factual → semantic: distinctive tokens matching an older turn.
    ("what pricing tier did we choose for billing", "t02", "factual"),
    ("what response window did we agree on", "t06", "factual"),
    ("what is the data residency requirement", "t05", "factual"),
    # entity → entity: a named entity whose turn shares few query tokens.
    ("who is Alice and what does she own", "t03", "entity"),
    ("what is Bob's role here", "t04", "entity"),
]


def reward(store: ChatMemoryStore, mode, query: str, needed: str) -> float:
    hits = store.search(query, k=mode.k, method=mode.methods[0] if len(mode.methods) == 1 else "all")
    recalled = any(h.chunk_id == needed for h in hits)
    return (VALUE if recalled else 0.0) - mode.cost_units()


def main() -> None:
    store = build_memory()
    tenant = ChatMemoryTenant(epsilon=0.12, seed=13)

    rounds = 72
    cr_rewards, base_rewards = [], []
    for r in range(rounds):
        query, needed, _bucket = PROBES[r % len(PROBES)]
        # Context Runtime: bandit picks the recall mode, learns from the outcome.
        mode = tenant.choose(query)
        rw = reward(store, mode, query, needed)
        tenant.record_outcome(query, mode, rw)
        cr_rewards.append(rw)
        # Baseline: always read all three indices.
        base_rewards.append(reward(store, FULL_BUNDLE, query, needed))

    # steady-state = the last third, after the bandit has committed.
    tail = rounds // 3
    cr = sum(cr_rewards[-tail:]) / tail
    base = sum(base_rewards[-tail:]) / tail
    print(f"Context Runtime (learned): {cr:.3f}")
    print(f"baseline (read all three): {base:.3f}")
    print(f"lift: {cr - base:+.3f}  ({'BEATS' if cr > base else 'does not beat'} baseline)")
    print("\nlearned per-bucket recall policy:")
    for bucket, mode_key in sorted(tenant.policy().items()):
        print(f"  {bucket:9s} -> {mode_key}")
    # sanity: the classifier routes each probe to its intended bucket
    intended = {q: b for q, _, b in PROBES}
    assert all(memory_bucket(q) == b for q, b in intended.items()), "bucket routing drift"


if __name__ == "__main__":
    main()
