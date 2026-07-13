<!-- Detailed lab report. Synthesis: ../RESULTS.md · Article: https://redevops.io/blog/better-context-beats-more-context -->

# Can better context management beat a better model? (nutrition corpus)

**Dataset:** Russian nutrition-consultation corpus (private) — 107 documents → ~10k chunks,
80 grok-generated grounded QA, hybrid retrieval over a BAAI/bge-m3 DuckDB store.
**Models (CPU / GGUF, comparable footprint):** Qwen3-Coder-Next 35B/3B · Nemotron-3-Super
121B/13B · DeepSeek-V4-Flash 284B/13B. **Judge:** grok-4.5 (off-box, self-contained rows).
**Arms:** `full_dump` (native long-context, ~17–25k tok) · `naive_rag` (fixed top-K) ·
`context_runtime` (the **full `ContextRuntime.run()` pipeline** — plan → retrieve → compress →
reason → verify, via `RedevopsRagRetriever`). Pollution = gold + N distractor docs.

## Judged accuracy (grok-4.5, 80 Q/arm)

| model | full_dump (pol 0 / 100) | naive_rag | **context_runtime** |
|---|---|---|---|
| Qwen3-Coder-Next (35B/3B) | 88 / 89 | 89 / 84 | 84 / 85 |
| Nemotron-3-Super (121B/13B) | **90 / 91** | 91 / 84 | 84 / 76 |
| DeepSeek-V4-Flash (284B/13B) | 48 / 36 | 69 / 59 | **89 / 88** |

Median latency (pol-independent): full_dump ~100–207 s · naive ~17–37 s · CR ~21–44 s.

## The headline: it depends on whether the model can exploit a large context

The variable that decides whether context management wins is **not** pollution level or model
size — it is the model's ability to *use* a large context.

- **DeepSeek drowns in it → Context Runtime rescues it.** With the full ~17–25k-token context,
  DeepSeek returns **"NOT FOUND" on 42/80 questions despite 99% retrieval hit** (the answer is
  in the context; the model can't extract it — classic lost-in-large-context). On the *same*
  questions with CR's compressed context it answers correctly: **89% vs 48%**, flat under
  pollution, and **5× faster** (44 s vs 207 s). This is the regime where context management
  decisively beats native long-context.
- **Nemotron exploits it → compression hurts.** Nemotron handles the full context best
  (90–91%); CR's compression removes signal it doesn't need and *lowers* accuracy (84→76%
  under pollution). Here a capable long-context model beats context management.
- **Qwen is in between** — it handles all three arms at ~84–89%; CR is neutral.

**So the answer to "can context management beat a better model?" is: yes — dramatically —
but only when the model cannot exploit the large context on its own.** When it can, native
long-context wins and compression is a net negative. The crossover is a property of the
model, not of the pollution axis we originally varied.

## Caveats (honest)

- **This is a CPU / GGUF harness.** DeepSeek-V4-Flash's poor full_dump reflects its long-context
  behavior *on this MXFP4 / CPU serving*; a GPU / full-precision deployment may differ. The
  comparison is apples-to-apples across models on one box, which is the point — but "DeepSeek
  can't use long context" is a statement about this setup, not the model in the abstract.
- **DeepSeek full_dump @ pollution 100** lost 20/80 rows to context-overflow errors even at
  ctx 32768 (Cyrillic tokenizes ~1.4× denser than the char estimate); the 36% there is a floor.
  Pollution-0 (48%, zero errors) is the clean number.
- **Clean quants only.** All models use plain quants (bartowski / equivalent), *not* Unsloth
  "UD" dynamic quants. Unsloth quants score higher because they carry their own built-in
  context management — which would confound a benchmark whose entire subject is context
  management. Removing them is what makes this a fair test of Context Runtime specifically.
- **Retrieval is strong** (hit 88–99%), so `naive_rag` is a tough baseline; where CR wins on
  DeepSeek it wins *despite* the answer already being retrievable — the gain is from
  compression letting the model find it, not from better retrieval.

## Reproduce

Serve each model via `scripts/serve_model.sh <gguf> <name> <port> <ctx> <threads> off`, then
`DATASET=nutri python -m harness.run --model-name <m> --base-url … --store nutri.duckdb
--pollution 0,100 --limit 80 --max-context-tokens 12000`. Judge off-box with
`scripts/judge_regrade.py --model grok-4.5 --base-url https://api.x.ai/v1`. The nutrition
corpus is private; the harness is dataset-agnostic (`DATASET` switch).
