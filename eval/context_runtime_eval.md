# Context Runtime RAG Evaluation Report

_Run date: 2024-?? (python eval/context_runtime_eval.py)_

## Python backend results

| Question | Top source | Verdict | Note |
| --- | --- | --- | --- |
| стероидный профиль тестостерон кортизол | 0001_pdf.txt | PASS | Steroid profile sheet with clear testosterone/cortisol hits; rich endocrine keywords in the returned chunk and context. |
| липидный профиль холестерин ЛПНП | 0008_2024-03-13-1-pdf.txt | PASS | Lipid panel document covering LDL/triacylglycerides and cholesterol; direct keyword overlap. |
| ФСГ ЛГ пролактин гормоны | 0002_pdf.txt | PASS | Gonadotropin reference chunk mentions FSH/LH/prolactin explicitly and in context. |
| результаты анализа крови | — | WEAK | Only generic "общий анализ крови" context surfaced; no specific CBC panel snippet returned. |
| витамин Д дефицит | — | WEAK | Context contains "витамин D" deficiency mentions but lacks a concrete lab interpretation chunk. |
| ТТГ щитовидная железа | — | WEAK | Context mentions TSH/thyroid concepts yet no focused thyroid panel source was retrieved. |
| глюкоза сахар крови | — | WEAK | Context references glucose/diabetes terms but no explicit lab result or guidance chunk provided. |
| ферритин железо | — | WEAK | Context briefly mentions anemia/iron/ferritin without linking to a specific ferritin report. |

## Go backend results

| Question | Top source | Verdict | Note |
| --- | --- | --- | --- |
| стероидный профиль тестостерон кортизол | 0001_pdf.txt | PASS | Same steroid profile file as Python; includes androgen/testosterone coverage. |
| липидный профиль холестерин ЛПНП | 0008_2024-03-13-1-pdf.txt | PASS | Lipid profile document retrieved with LDL/холестерин terms highlighted. |
| ФСГ ЛГ пролактин гормоны | 0002_pdf.txt | PASS | Gonadotropin lab sheet returned, matching required endocrine keywords. |
| результаты анализа крови | — | WEAK | Only high-level CBC terminology observed in context, without a concrete result excerpt. |
| витамин Д дефицит | — | WEAK | Vitamin D insufficiency mentioned but no dedicated lab snippet provided. |
| ТТГ щитовидная железа | — | WEAK | Thyroid-related terms present, yet no targeted TSH panel document surfaced. |
| глюкоза сахар крови | — | WEAK | Context references glucose monitoring but lacks a specific lab report extract. |
| ферритин железо | — | WEAK | Context includes iron/anemia vocabulary (including English "iron"), but no ferritin lab chunk retrieved. |

## Python vs Go comparison

- Both backends produced identical verdict distributions: 3 PASS, 5 WEAK, 0 MISS. No queries outright failed.
- The top retrieved sources matched where available (0001/0002/0008 documents). For WEAK cases, neither backend exposed a high-confidence top chunk; responses were assembled from suggestive context only.
- Minor textual divergence: the Go backend adds an English "iron" token for the ferritin query, hinting at slightly different embedding space, but it did not translate into a stronger verdict.

## Overall assessment

Retrieval quality is adequate for endocrine-specific questions (three solid PASSes) but underperforms on broader metabolic and hematology requests. Half of the questions land in WEAK because the backends return diffuse context rather than precise lab snippets. Consistency between Python and Go suggests shared corpus/embedding behavior; improvements must target the shared retrieval pipeline rather than per-backend quirks.

Summary counts:
- Python backend — PASS: 3, WEAK: 5, MISS: 0
- Go backend — PASS: 3, WEAK: 5, MISS: 0

## Recommendations

1. **Improve corpus chunking/granularity**: ensure CBC, thyroid, glucose, ferritin reports are chunked into focused passages so the retriever can surface a concrete lab entry instead of generic explanations.
2. **Enhance embedding/reranking**: introduce a semantic reranker (e.g., cross-encoder) to prioritize chunks with exact lab panels matching the query keywords, especially for high-overlap medical terminology.
3. **Query expansion for Russian medical terms**: expand user queries with synonym lists (e.g., "глюкоза", "гликемия", "гликированный гемоглобин") to better align with varied corpus phrasing.
4. **Threshold tuning**: adjust retrieval score thresholds so weak hits fall back to broader search strategies or trigger secondary retrieval (e.g., using metadata filters) rather than returning generic context.
