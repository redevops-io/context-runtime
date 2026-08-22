"""v0.3.0 accelerator crossover arms (H/I/J) for the Wikimedia evidence-native benchmark.

These arms answer the roadmap's accelerator questions on **real strategywiki data**:

  H  cuDF   — does GPU evidence/temporal processing (the incremental-Discovery change-set) preserve
              arm-B semantics and reduce total work above a crossover size?
  I  cuVS   — does GPU ANN retrieval preserve arm-C identity/lineage (same EvidenceRefs as CPU-exact)
              and improve throughput/latency above a crossover corpus size?
  J  cuOpt  — at what candidate/constraint size does GPU optimization of context-assembly token-budget
              packing beat the CPU knapsack, and where does it lose to setup overhead?

Design invariants (roadmap §2):
  * CPU is the reference + fallback path; every GPU result must be **semantically equal** to the CPU
    reference before any timing is reported (correctness gates performance).
  * An accelerator NEVER changes canonical identity/semantics — only latency/throughput/scale.
  * The GPU is a *costed capability that can lose*: total accelerator latency is measured as
    transfer + exec + result-copy (not kernel time alone), so small problems honestly favour the CPU.
  * Correctness and performance are reported separately — no blended "X× faster" headline.

The CPU reference paths import cleanly with no GPU; the GPU paths lazily import cudf/cuvs/cuopt/cupy so
the same modules run on the dev box (prepare inputs, unit-test the reference paths) and on the GPU host
(run the crossover sweep).
"""
