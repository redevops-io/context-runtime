"""Context-vs-model benchmark harness.

Question under test: at a fixed memory budget, can a smaller model + Context Runtime
(execution planning over retrieval) match or beat a bigger model that manages context
natively — especially as the retrieval corpus gets polluted?

Everything here is dependency-light on purpose (stdlib + the importable core of
``context_runtime`` + matplotlib for the one plot). No torch, no vector DB — the
retriever is a compact BM25 so the harness is reproducible on any box.
"""
