"""Accelerator backends — optional GPU implementations of runtime operations (v0.3.0).

The runtime stays **accelerator-aware, not GPU-dependent**: the CPU path is always the reference and the
fallback, and an accelerator is a *costed capability* the runtime selects only when a measured crossover
says it pays. Nothing here is on by default — with no ``CR_ACCEL`` env flag, or no GPU, or a corpus below
the crossover, every call runs exactly the CPU code it always did.

Backends (benchmarked in `context-runtime` `benchmarks/wikimedia/accelerators`, FINDINGS_v0.3.0):
  * ``cuvs_ann``      — cuVS CAGRA ANN for semantic retrieval. Crossover ~12k vectors; 36× query speedup
                        at 200k. Approximate (recall ~0.96 vs exact), so it is opt-in for large corpora.
  * ``cudf_temporal`` — cuDF for temporal/evidence change-set processing. Crossover ~10^6 rows; exact
                        (byte-identical to the pandas/Python reference).

The seam is ``selector.decide(kind, n)`` → an ``AccelDecision``; a store calls it, and on ``use_gpu`` routes
to the backend, otherwise runs its existing CPU path. Every decision carries a ``reason`` for EXPLAIN.
"""
from .selector import AccelDecision, decide, gpu_available  # noqa: F401
