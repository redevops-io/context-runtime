# Data sources & licensing

Benchmarks here use the public datasets below. Loaders fetch data at runtime; we do not redistribute raw data.

## How this data is used

These datasets are used solely to develop and benchmark the Context Runtime software itself (internal R&D and regression testing of stack improvements). They are NOT redistributed, NOT embedded in or shipped with the product, and NOT part of any customer deployment. Context Runtime customers apply the software to their own data under their own data licenses.

## LiveRAG Benchmark (default for context-vs-model)
- Source: HuggingFace `LiveRAG/Benchmark`. Built by DataMorgana over FineWeb-10BT web pages. (SIGIR'25.)
- License: the underlying FineWeb corpus is ODC-BY 1.0 (https://huggingface.co/datasets/HuggingFaceFW/fineweb); attribute FineWeb + the LiveRAG benchmark authors. Used for non-commercial research/benchmarking.

## MuSiQue (multi-hop QA)
- Source: https://github.com/StonyBrookNLP/musique . License: CC-BY-4.0. Citation: Trivedi et al., TACL 2022.

## PopQA
- Source: https://github.com/AlexTMallen/adaptive-retrieval (HF `akariasai/PopQA`). License: MIT. Citation: Mallen et al., ACL 2023.

## LongMemEval / LongMemEval-S
- Source: https://github.com/xiaowu0162/LongMemEval (HF `xiaowu0162/longmemeval`). License: MIT. Citation: Wu et al., ICLR 2025.

## FinanceBench (Patronus AI)
- Source: https://github.com/patronus-ai/financebench . License: CC-BY-4.0. Attribution: Patronus AI.

## PopQA / LongMemEval / MuSiQue via routing studies
- The PopQA and LongMemEval "routing study" runs reported in reports/ use the same sources above; the unified JSONL was produced by an external prep step. Same licenses apply.
