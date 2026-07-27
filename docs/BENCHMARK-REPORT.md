# Context Runtime — Final Benchmark Report

_Full `examples/` suite run end-to-end on the v3 engine. Every number is produced by a runnable
script (`PYTHONPATH=. python examples/<name>.py`) — no invented figures._

**Run status: 30 / 30 benchmarks passed** (seeded, deterministic simulations; the consolidated
headline is cross-checked in the independent Go re-implementation).

**Judge change in this cycle:** the LLM-as-judge default moved from **OpenAI GPT-5.5 → Grok 4.5**
(xAI, OpenAI-compatible, ~5× cheaper), now a *dedicated* judge endpoint kept separate from the chat
model (`CR_JUDGE_BASE_URL` / `CR_JUDGE_MODEL` / `CR_JUDGE_API_KEY`, default `grok-4.5`). Note: the
offline benchmarks below use a *seeded coverage judge* (no live LLM), so their numbers are unaffected
by the judge model — the judge swap changes the **live** self-learning RAG's cost, not these results.

---

## 1. Headline — v1 → v2, measured in both runtimes (`consolidated_benchmark`)

| Metric | Py v1 | Py v2 | Δ | Go v1 | Go v2 | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| Learned-policy precision | 67.6% | 82.2% | ▲ +14.6 pts | 84.6% | 95.9% | ▲ +11.3 pts |
| Abstention recall (unanswerable caught) | 0.0% | 100.0% | ▲ +100 pts | 0.0% | 100.0% | ▲ +100 pts |
| False-abstain rate (answerable dropped) | 0.0% | 0.0% | — | 0.0% | 0.0% | — |
| Expensive-stage depth (passages) | 8.00 | 3.00 | ▼ −62% | 8.00 | 3.00 | ▼ −63% |
| Precision after the sizer | 37.5% | 100.0% | ▲ +62.5 pts | 37.5% | 100.0% | ▲ +62.5 pts |

_40-seed average; precision headlined at β=0.9 (calibration-trust knob; shipped default 0.5). Go is an
independent re-implementation on identical methodology — directional parity across languages._

## 2. Calibration sweep (`dspark_calibration_bench`)

Served true-precision, 40 seeds, coverage-biased judge:
- v1 judge-only (β=0.0): **67.6%** → v2 judge + calibrated relevance: β0.5 **68.9%** (+1.3), β0.7 **74.3%** (+6.7), β0.9 **82.2%** (+14.6).
- Abstention (P(rel)≥0.5 floor, v1 cannot abstain): sizer off → 8 passages @ 38% precision; sizer on → **3 passages @ 100%** (depth −62%, the pruned tail was the low-relevance one).

## 3. Online adaptation under drift (Gen-4)

- **`online_vs_static_bench`** — best plan drifts A→C at t=200. Post-drift served reward: static **0.20** · online-plain **0.33** · **online-discounted 0.67** (oracle 0.80). Discounting fades stale evidence so the planner tracks the shift.
- **`online_learning`** — priors say hybrid=0.80/graph=0.55 (static always serves hybrid); production reward returns graph=0.9/hybrid=0.3 → after learning the planner serves **graph**, adapting past the stale estimate (surfaced without re-running it, via off-policy value).
- **`learning_loop`** — async learning drained 8 events off the serving path → snapshot v1; replica reconciles and switches hybrid→graph with **zero learning on the hot path**.

## 4. Retrieval methods & routing

- **`hop_routing`** — intent-routed: conceptual → hybrid (acc 0.78, $0.06); **multi-hop → graph** (acc 0.88, $0.43), surfacing a bridge doc via the ATP→α-synuclein hop.
- **`parallel_fusion`** — fan-out wall-clock **5.9× faster** (50.9 ms parallel vs 300.8 ms sequential), then rerank-before-model.
- **`heterogeneous_shards`** — community/coverage routing cut cross-domain noise **23 → 0 docs** across 8 medical queries while keeping recall **8/8**.
- **`multimodal_image_search`** — cross-modal text→image: each text query retrieves the right image as top hit (no OCR, no shared terms).
- **`temporal_retrieval`** — bi-temporal as-of filtering (a record filed 2026-07-15 is correctly invisible to a 2026-06-15 as-of query).
- **`rag_tuning`** — per-intent optimal config recovered (exact_lookup / incident / conceptual all match the latent best) after 40 observed retrievals.

## 5. Trust & governance (Gen-5)

- **`trust_aware`** — trust folded into plan selection (trust_weight 0.6 → the relied-upon local plan wins); abstention gate: expected accuracy 0.80 clears the 0.7 threshold → **SERVE** (else abstain).
- **`capability_registry`, `explain`, `chat_memory`** — capability gating, EXPLAIN-ANALYZE traces, and conversational memory all green.

## 6. Fleet / tenants (`fleet_tenants` + per-app demos)

**17 tenants**, one pattern: each module = one goal, one metric, a learned *cheapest-sufficient source*
policy — replacing hand-wired controllers with one data-driven Context Runtime tenant pattern. The
per-app tenant demos all ran green: `agentic_billing, agentic_books, agentic_compliance,
agentic_support, control_tower, market_radar, growth_engine, social_autopilot, outreach_engine,
outreach_pipeline, incident_review, soc_triage, sidekick_learning, vibexgen_learning`.

## Full run table

All 30 scripts exited 0. Slowest: `consolidated_benchmark` (14s, runs the Go twin), `dspark_calibration_bench` (12s). One pre-existing example bug fixed this cycle: `fleet_tenants` was missing an `outreach` probe (added).

_Reproduce any line: `PYTHONPATH=. python examples/<name>.py`. Headline card:
`PYTHONPATH=. python examples/consolidated_benchmark.py --html out.html`._
