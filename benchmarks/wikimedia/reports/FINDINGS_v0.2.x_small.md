# Evidence-Native Runtime Benchmark — Stage-1 findings (v0.2.x, small corpus)

**Baseline correctness run of the frozen v0.2.x evidence-native foundation on real Wikimedia history.**
This establishes that the current release has no correctness bugs in the evidence-native invariants,
and is the reference the later v0.3.0 / accelerated runs compare against.

- **Date:** 2026-08-22
- **Corpus (frozen):** strategywiki dump `20260801`, full revision history + logging. 24,038 pages /
  80,092 revisions / 4,788 reverts / 171 protection events. See `../dataset-manifest.json`.
- **Slice under test:** 12 real (revision A → revision B) pairs from distinct content pages, selected
  deterministically (ascending page_id). No synthetic mutations.
- **Runtimes (installed from git tags):** runtime-contracts 0.3.0 · redevops-rag **0.2.1** ·
  context-runtime **0.2.0** · agentic-os **0.2.4** · discovery-runtime 0.1.10.
- **Embedder:** deterministic hashed bag-of-words stub (no torch/network) — correctness invariants
  don't depend on embedding quality; this guarantees byte-identical reruns.
- **Runs:** 3 clean runs; semantic outputs **identical across all 3** (`reproducible_across_runs: true`).

## Result: EXIT GATE PASSED

Every arm passes; every hard gate holds. All arms are reproducible across the 3 runs.

| Arm | Capability | Result |
|---|---|---|
| **A** | point-in-time context + exact replay + re-evaluate | ✅ PASS |
| **B** | incremental Discovery == full rescan | ✅ PASS |
| **C** | evidence lineage integrity | ✅ PASS |
| **D** | freshness / REFRESH sourced from evidence | ✅ PASS |
| **G** | deterministic-first | ✅ PASS |

### Hard gates (plan §5/§7/§13) — all satisfied

- wrong-version substitution = **0** (A)
- silent replay divergence = **0** (A)
- authoritative source/hash mismatch = **0** (C)
- wrong-revision lineage = **0** (C)
- stale-served-as-current = **0** (C)
- incremental/full valid-state mismatch = **none** (B: `valid_state_equivalent: true`)
- model calls in deterministic classification = **0** (G)

### Measured metrics (last run)

**A — replay (12 cases):** exact-revision recovery 12/12, plan fingerprint reproduced 12/12,
ContextEpoch reproduced 12/12, re-evaluation resolved B 12/12, replay latency p50 ≈ 0.45 ms. After the
RAG source advances A→B, exact replay reconstructs A and A stays retrievable by version (point-in-time).

**B — incremental vs full (12 conclusions, 6 changed):** incremental recomputed **6** vs full **12**
(2.0× reduction), incremental model calls **6** vs full **12**, and the incremental valid state equals
the full-rescan valid state.

**C — lineage (14 hits):** 0 hash mismatch, 0 wrong-revision lineage, 0 missing lineage, prior
revision resolvable 12/12. Every current hit traced to revision B's exact rcv1 identity; every prior
revision A stayed addressable with its own identity (`superseded_by` = B).

**D — freshness (48 evaluations):** every stale evaluation (evidence aged to the 2026 dump date)
REFRESHed (24/24); every fresh evaluation (as of one day after the revision) served (24/24); 0
unnecessary refresh, 0 stale-not-refreshed; with the policy disabled, all 24 served unchanged at
freshness 1.0.

**G — deterministic-first (12 conclusions):** all 12 verdicts resolved by the pure `classify()` with
**0** model calls; incremental Discovery recomputed only the 4 STALE conclusions, avoiding a model call
on 8 (4 INVALIDATED + 4 unchanged); content-hash and revision-ordering facts resolved deterministically
for all 12.

## Bugs found

- **Runtime bugs: none.** The frozen v0.2.x runtimes reproduce every evidence-native invariant on real
  data with no defects surfaced.
- **Harness bug (fixed): 1.** An early version of arm A flagged a wrong-version substitution using a
  bare substring test (`b_revid in evidence_refs`), which false-positived when a short revision id
  appeared inside the content-hash hex. Corrected to match the delimited version field (`@{revid}#`);
  after the fix, wrong-version substitution is 0 across all pairs. (This is the benchmark doing its job
  — catching an incorrect assertion before it could mislead.)

## Scope / honesty

- **Arms E and F are intentionally absent.** They require a cross-series trajectory governance engine
  (K revisions / R reverts / window; ALLOW/REFRESH/REQUIRE_REVIEW; OBSERVE→ENFORCE) that does **not**
  exist in the v0.2.x Python or Go runtimes — deferred to the v0.3.0 private release. The corpus
  selection for them (protected-page trajectories with revert counts) is already wired
  (`evidence_corpus.select_protected_page_trajectories`) and ready to drive that engine when it exists.
- This is a **correctness** run. Performance numbers here (latency p50) are incidental; the full-corpus
  performance/scaling comparison and the accelerator crossover benchmarks belong to the v0.3.0 program.
- The stub embedder means retrieval-**quality** is out of scope; identity/replay/lineage/freshness/
  incremental-equivalence — the things v0.2.x claims — are what is validated.

## Reproduce

```bash
cd benchmarks/wikimedia
# install the frozen runtimes from tags into a venv (see dataset-manifest.json for pins), then:
PYTHONPATH=. python -m harness.run_benchmark --runs 3 --pairs 12
# → results/small/run-00{1,2,3}.json + results/small/summary.json ; exit 0 on gate pass
```
