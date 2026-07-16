# Benchmarks

> **Consolidated.** All Context Runtime benchmark results now live in one canonical document:
> **[`BENCHMARKS.md`](https://github.com/redevops-io/context-runtime/blob/main/BENCHMARKS.md)** — the
> comparable **capability ladder** (v1→v5 on a single mixed stream, end-to-end accuracy + tokens),
> then a **drill-down per version** (v1→v2 calibration in Python & Go, v3 drift, v4 knowledge
> routing, v5 DIVER reasoning retrieval), the standalone studies (heterogeneous shards, chat-memory,
> parallel fusion, tenants/governance), and full methodology.

Everything that used to live in this file, in `docs/BENCHMARK-REPORT.md`, and in
`docs/BENCHMARKS-v3-preliminary.md` has been merged into that one document. The runnable examples
are unchanged — reproduce any result with `PYTHONPATH=. python examples/<name>.py`.
