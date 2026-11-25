# What learns in the Context Runtime, and how to see, stop and undo it

**The rule.** A learner chooses among alternatives somebody declared, or keeps a statistic that informs such a choice. No
learner changes policy, approval, authorization, egress or verification code or configuration. The list of every learner,
with what it may change, is `context_runtime/learning/registry.py`; `tests/test_learner_registry.py` fails when a class that
learns is not on it. Nothing here optimizes the optimizer, and nothing promotes a learned value into configuration.

Two things the registry says plainly:

* Most learners go through the shared bandit and so have its switch, reset, rollback and ledger. The cost model, the quality
  ledger, pattern memory and the zoning population governor have **no kill switch and no ledger**.
* `LearningAggregator` is the one learner whose set of arms is not fixed in code: an outcome event naming an unknown plan
  shape adds it. Whoever can publish to its bus can add an arm.

## The shared bandit

**Provenance.** Every observation of a persisted policy is appended to `<policy>.ledger.jsonl` (mode 0600) with who or what
produced it: only `source`, `run_id`, `principal`, `plan_hash`. It takes no request or customer text. Attribute without
changing the learners in between:

    from context_runtime.learning.ledger import provenance
    with provenance(source="verifier", run_id=run.id, principal=user):
        tenant.record_outcome(question, success)

An observation nobody attributed is recorded as `unattributed`. A policy learned before the ledger existed starts it as a
baseline. The ledger is bounded (20 MB, then one older file; history restarts from a baseline).

**Stop.** `CR_LEARNING_FROZEN=1` (or `bandit.frozen = True`): nothing is learned and nothing is explored; what was already
learned still decides. To stop the learned values deciding, reset them.

**Undo.** With the service stopped:

    python -m context_runtime.learning.admin status   policy.json
    python -m context_runtime.learning.admin rollback policy.json --by NAME --principal P          # preview
    python -m context_runtime.learning.admin rollback policy.json --by NAME --principal P --apply
    python -m context_runtime.learning.admin reset    policy.json --by NAME --context CTX --apply

A rollback is applied only if the ledger reproduces the live policy. If an entry was ever lost (a full disk, a deleted
line) it refuses, because it could not say what the policy would have been.

**A damaged policy file** is renamed to `<policy>.corrupt-<time>` with one warning and learning starts blank. If it cannot be
renamed it is never overwritten.

## Not built

A record of lessons separate from the learned values; a promotion gate for anything stronger than a choice among declared
arms; a held-out regression guard per learner; tenant identity in the bandit's key (isolation is each tenant owning its
instance and file).
