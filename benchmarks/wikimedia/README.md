# Evidence-Native Runtime Benchmark — real Wikimedia history

Validates the shipped **v0.2.x evidence-native** runtime features on a frozen, public,
non-synthetic temporal corpus: the full revision + moderation history of
**strategywiki** (`strategy.wikimedia.org`). Full plan:
`~/Documents/REDEVOPS_WIKIMEDIA_EVIDENCE_NATIVE_BENCHMARK_PLAN.md`.

The thesis (plan §1): *can decisions be bound to the exact evidence that existed at a
point in time, reproduced after the world changes, processed incrementally without
changing valid outcomes, traced through their lineage, kept from serving stale
evidence, and governed against the evidence trajectory that preceded them?*

## Frozen corpus

`dataset-manifest.json` is the freeze (plan §3): project, dump date, filenames, URLs,
sha256, runtime version pins, canonicalization version. Raw dumps live in `data/`
(gitignored) and are pinned by hash. Re-fetch with the URLs in the manifest; verify
against `data/SHA256SUMS`.

Profile (`harness/profile.py`, runs in ~7 s):

| pages | revisions | changed pages | reverts (sha1) | protection events | protected pages | span |
|---|---|---|---|---|---|---|
| 24,038 | 80,092 | 8,411 | 4,788 | 171 | 158 | 2001–2026 |

## Capability matrix — plan arms vs shipped runtimes

Grounded against the pinned runtimes (rc 0.3.0, redevops-rag 0.2.0, discovery-runtime
0.1.10, contextos 0.2.0, agentic-os 0.2.4). Honesty first (plan §18/§22): an arm only
claims what the code actually exercises.

| Test | What it proves | Status | Basis |
|---|---|---|---|
| **A** point-in-time context + exact replay | mission bound to state A survives evidence advancing to B; `rehydrate` reproduces or fails closed; `re_evaluate` is distinct | ✅ **shipped** | agentic-os `ContextView`/`epoch_from_refs`, `plan_fingerprint`, `rehydrate`→`ReplayError`/`ReplayDivergence`/`UnrecoverableAuthority`, `re_evaluate`; real restart via `EventStore(path)` |
| **B** incremental vs full Discovery | `discover_incremental` reaches the same valid state as a full rescan, examining less | ✅ **shipped** | discovery-runtime `discover_incremental`/`discover_full`, `DiscoveryCheckpoint`, `classify`, `IncrementalReport` |
| **G** deterministic-first | deterministic facts resolve without a model call | ✅ **shipped (structural)** | no `deterministic` flag exists; measured via plan shape (no `reason` step ⇒ no model call) + `classify` + `IncrementalReport.model_calls` |
| **C** evidence lineage integrity | derived artifacts trace back to the exact revision identity | ⚠️ **needs runtime wiring** | rc `EvidenceRef`/lineage + mission `evidence_refs` are real, but **redevops-rag does not consume runtime-contracts** (chunk id = bare sha256, no version). Wiring RAG→`EvidenceRef` is a prerequisite PR |
| **D** freshness / REFRESH | stale evidence is penalized and can produce REFRESH | ⚠️ **needs runtime wiring** | contextos `PlanScore.freshness` is scored + `AbstentionGate(min_freshness)`→`"refresh"`, but nothing derives freshness from evidence (`freshness_prior` unused). Wiring evidence-`observed_at`→freshness is a prerequisite PR |
| **E** evidence/action governance trajectories | cross-series rule (K revs / R reverts / window) flags trajectories a per-event baseline misses | ⛔ **engine does not exist** | no cross-series trajectory engine in Python (agentic-os: enterprise overlay, out of repo) **or Go** (CR-enterprise/go is policy+trust, not trajectory; no K/R/window, no dispositions, no OBSERVE→ENFORCE). Must be built before it can be benchmarked |
| **F** OBSERVE→ENFORCE lifecycle | promotion changes disposition, not detection | ⛔ **depends on E** | same missing engine; there is no shadow→enforce rule lifecycle in either runtime |

**Decisions taken (Phase 0):** Stage-1 corpus = strategywiki; suite lives here in
`context-runtime-bench` (no benchmark-only code added to production runtimes — the
harness bridges via public APIs only).

**Open:** E/F require a real cross-series trajectory governance engine that neither the
Python nor the Go runtime currently ships. Building it (vs deferring E/F) is a pending
go/no-go. C/D require small real wiring PRs to redevops-rag and contextos.

## Layout

```
dataset-manifest.json     # the freeze (plan §3)
data/                     # gitignored raw dumps (+ SHA256SUMS)
harness/
  corpus.py               # stream-parse MediaWiki XML → Page/Revision/LogItem
  profile.py              # S2 corpus profile
results/small/            # per-run result bundles
reports/                  # findings
```

## Run

```bash
cd benchmarks/wikimedia
PYTHONPATH=. python -m harness.profile      # S2 profile (no runtime deps)
```
