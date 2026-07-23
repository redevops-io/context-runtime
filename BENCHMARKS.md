# Benchmarks

Context Runtime's benchmark results live in two places you can inspect directly:

- **[`benchmarks.html`](./benchmarks.html)** — the rendered results: the capability ladder and
  per-version drill-downs (v1→v2 calibration in Python & Go, v3 drift, v4 knowledge routing,
  v5 DIVER reasoning retrieval), the standalone studies (heterogeneous shards, chat-memory,
  parallel fusion, tenants/governance), and methodology.
- The **[Benchmarks section of the README](./README.md#benchmarks)** — the headline numbers.

Every result is **reproducible** from the runnable examples:
`PYTHONPATH=. python examples/<name>.py`.
