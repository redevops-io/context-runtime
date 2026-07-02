"""TurboVecRetriever — quantized ANN index for the `vector` method AT SCALE (SPEC §4.5).

Our default `vector` retriever (SemanticRetriever) keeps full float32 embeddings and does
brute-force cosine — exact and fast up to ~10^4–10^5 vectors, but a memory/latency wall
beyond that. TurboVec (github.com/RyanCodrai/turbovec, MIT) implements Google Research's
TurboQuant (arxiv 2504.19874): data-oblivious quantization (random rotation → Beta-
distributed coords → Lloyd-Max scalar quant → length-renormalized scoring) giving ~16×
memory reduction at 2-bit, SIMD ANN that matches/beats FAISS, and — crucially — ZERO
training/indexing time (data-oblivious), which fits our no-heavy-setup ethos.

It is a drop-in for the `vector` method: same RetrieverPlugin contract as SemanticRetriever,
same embedding model (reuses store_semantic._embed), so the planner can't tell them apart.
Opt-in for scale via the [turbovec] extra; absent it, `available` is False. This is a
storage/search-index swap, NOT a new retrieval paradigm — quality ≈ semantic, the win is
memory + speed once the corpus is large.

    pip install "context_runtime[turbovec]"     # turbovec + fastembed + numpy
    CR_TURBOVEC_BITS=2|4                          # quantization bit-width (default 4)
"""
from __future__ import annotations

import importlib.util
import os

from ..types import Hit, PluginInfo, Retrieval
from .store_semantic import _embed


def _installed() -> bool:
    return bool(importlib.util.find_spec("turbovec") and importlib.util.find_spec("fastembed"))


class TurboVecRetriever:
    def __init__(self, docs: list[dict] | None = None, *, bit_width: int = 4,
                 source: str = "turbovec"):
        self.docs = list(docs or [])
        self.bit_width = int(os.getenv("CR_TURBOVEC_BITS", bit_width))
        self.source = source
        self._index = None
        self._built_n = -1

    @property
    def available(self) -> bool:
        return _installed()

    def _build(self):
        if self._index is not None and self._built_n == len(self.docs):
            return self._index
        if not self.docs or not self.available:
            self._index, self._built_n = None, len(self.docs)
            return None
        import numpy as np
        from turbovec import TurboQuantIndex
        mat = np.vstack(_embed([d["text"][:1200] for d in self.docs])).astype("float32")
        idx = TurboQuantIndex(dim=mat.shape[1], bit_width=self.bit_width)
        idx.add(mat)
        self._index, self._built_n = idx, len(self.docs)
        return idx

    def search(self, query: str, k: int, method: Retrieval = "vector") -> list[Hit]:
        idx = self._build()
        if idx is None:
            return []
        import numpy as np
        qv = np.asarray(_embed([query])[0], dtype="float32").reshape(1, -1)
        scores, indices = idx.search(qv, k=max(k, 1))
        out: list[Hit] = []
        for s, i in zip(np.ravel(scores), np.ravel(indices), strict=False):
            i = int(i)
            if i < 0 or i >= len(self.docs):
                continue
            d = self.docs[i]
            out.append(Hit(chunk_id=d["chunk_id"], filename=d["filename"], text=d["text"],
                           score=float(s), created_at=d.get("created_at"), source=self.source))
        return out

    def index(self, path: str) -> dict:
        from pathlib import Path
        p = Path(path).expanduser()
        n = 0
        for fp in sorted(p.rglob("*")):
            if fp.suffix.lower() in (".md", ".txt", ".rst") and fp.is_file():
                self.docs.append({"chunk_id": f"{fp.name}::0", "filename": fp.name,
                                  "text": fp.read_text(errors="ignore"), "created_at": None})
                n += 1
        self._index, self._built_n = None, -1  # invalidate
        return {"files": n, "chunks": n}

    def info(self) -> PluginInfo:
        return PluginInfo(name="turbovec_retriever", kind="retriever", version="0.1",
                          capabilities=frozenset({"vector", "quantized", "ann", "scale"}))
