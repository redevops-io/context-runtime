# FinanceBench × LibreQB demo

The "crown jewel" demo: Context Runtime + the LibreChat **Libre Query Board** on data that
is genuinely **hard to ingest and query** — real SEC 10-K/10-Q filings from
[FinanceBench](https://github.com/patronus-ai/financebench) (Patronus AI), a recognized
"RAG fails here" benchmark.

It shows the thesis directly: **no single retrieval method is enough**, and the runtime
serves the best one while the Query Board makes every method's answer visible next to the
chat.

## Why this dataset is hard

- **Hard to ingest** — the numbers live in dense financial **tables** buried in long PDFs.
  Naive text extraction flattens `Revenues 66,608 62,286` into word-soup. We use
  `PdfTableExtractor` (pdfplumber) to keep each row intact: `Revenues | $66,608 | $62,286`.
- **Hard to query** — figures sit in tables (vector & BM25 both miss), questions need
  cross-filing aggregation (→ community) and multi-hop numerical reasoning (→ graph /
  iterative). E.g. *"What drove 3M's FY2022 operating margin change?"*

## Run it

```bash
export KIMI_API_KEY=... KIMI_BASE_URL=https://api.moonshot.ai/v1 KIMI_MODEL=kimi-k2.6
cd deploy/financebench
./download.sh                                   # Q&A + 6 filing PDFs → ../../.financebench/
uv run --with pdfplumber build_corpus.py        # PDFs → tables-preserved corpus (~5k passages)
./serve.sh                                       # control plane on :8092 (all 5 methods)
python3 demo.py                                  # print the Query Board for the demo questions
```

Then open LibreChat (`deploy/librechat`, the "Context Runtime (Python)" endpoint points at
:8092) and ask a financial question — the answer plus the **Libre Query Board** appear,
showing BM25 / vector / hybrid / community / graph side by side with the served strategy.

## What the demo shows (verified)

For *"Is 3M a capital-intensive business (FY2022)?"*:
- **BM25** grabs **AMD / American Express** — a lexical false match on "capital".
- **Vector** correctly stays on **3M** (semantic).
- **Hybrid** (served) fuses them; **graph** surfaces cross-doc bridges; **community**
  returns query-conditioned clusters.

The chat answer matches ground truth — e.g. *"3M FY2022 operating margin declined 1.7pp to
19.1%, driven mostly by cost of sales"*.

## Notes

- Data (`../../.financebench/`, PDFs + generated corpus) is gitignored; these scripts +
  `context_runtime/ingest/pdf_tables.py` are the reusable pieces.
- English corpus → do **not** set `CR_QUERY_LANGS`.
- Community detection is query-conditioned above `CR_COMMUNITY_MAX_NODES` (1500) so it stays
  fast on the ~5k-passage corpus.
