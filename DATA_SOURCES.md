# Data sources & licensing

Most evals/examples in this repo use small synthetic, self-authored corpora (no external data). Two demos use public datasets, downloaded on demand into gitignored dirs (we ship the downloader, not the data):

## FinanceBench (Patronus AI) - deploy/financebench/
- Source: https://github.com/patronus-ai/financebench - expert Q&A over real SEC 10-K/10-Q filings.
- License: CC-BY-4.0. Attribution: Patronus AI.
- Used by: examples/heterogeneous_shards.py.

## PubMedQA (Jin et al., EMNLP 2019) - deploy/medical/
- Source: https://github.com/pubmedqa/pubmedqa - expert-labeled biomedical Q&A over PubMed abstracts.
- License: MIT.
- Citation: Jin et al., "PubMedQA: A Dataset for Biomedical Research Question Answering," EMNLP 2019.

## Synthetic / self-authored (no external data)
- eval/context_runtime_eval.py and most examples/ build small hand-written corpora inline. The Russian medical-lab PDFs referenced in eval/context_runtime_eval.md are a private/personal corpus, not a public dataset, and are not redistributed.
