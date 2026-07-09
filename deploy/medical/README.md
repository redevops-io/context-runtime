# MedicalBench corpus — PubMedQA

The medical half of the public chat demo (`chat.redevops.io`, the "Context Runtime ·
MedicalBench" and "· Mixed" models). The medical analogue of `deploy/financebench`: real
domain documents + expert Q&A, from a public dataset.

**Source — PubMedQA** (Jin et al., EMNLP 2019, https://github.com/pubmedqa/pubmedqa, MIT):
1 000 expert-labeled biomedical questions over real PubMed abstract passages, each with gold
contexts and a long-form answer. Its clinical vocabulary (discharge, statement, balance,
chronic/acute, …) deliberately collides with financial 10-K language — which is what makes
the **Mixed** model's coverage-routing win (cross-domain noise → 0) visible.

```bash
./download.sh              # → ../../.medical/pqal.json   (curl, no HF datasets dep)
python3 build_corpus.py    # → ../../.medical/corpus/*.txt (~3.3k passages) + qa.jsonl
```

`build_corpus.py` writes one normalized `.txt` passage per (PMID, abstract section) — the same
pre-normalized shape the FinanceBench corpus uses, so the control plane ingests it directly
(mounted RO at `/medical`, tenant `context-runtime-medical`). No PDF/OCR step.

Data lands in `../../.medical/` (gitignored). We ship the downloader, not the data.
