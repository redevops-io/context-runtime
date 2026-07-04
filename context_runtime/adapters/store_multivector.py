"""Multi-vector late-interaction retrieval (ColPali / ColQwen) — Phase 2b of multimodal.

Single-vector cross-modal search (``store_image``) pools a whole image into ONE vector — great
for "find the diagram that means X", weak for dense document pages (a 10-K page, a slide, a form)
where the answer is one region among many. ColPali/ColQwen embed a page as a *set* of patch
vectors and score a query as a *set* of token vectors via **late interaction (MaxSim)**:

    score(Q, D) = Σ_i  max_j  (q_i · d_j)      # each query token matches its best page patch

so a query term can light up the exact patch that answers it — OCR-free, best-on-ViDoRe. The
cost is that a doc is now MANY vectors, so this is a genuinely different index from the quantized
single-vector ANN (TurboVec): it wants a MaxSim-capable store (Qdrant multivector + binary
quantization) at scale. This module implements the retrieval math with an exact in-memory backend
for correctness + tests, and leaves Qdrant/ColPali as gated, injectable backends so the base
install and the test suite need neither a VLM nor a running Qdrant.

    ret = MultiVectorRetriever(doc_embed=..., query_embed=...)   # injected for tests
    ret.index("/pages")                                          # each page → a patch matrix
    hits = ret.search("total lease liability in 2023", k=5)      # MaxSim over pages

Real backend (opt-in, ``[multimodal-colpali]`` extra): ColPali via ``colpali_engine`` for the
embedders; ``qdrant-client`` for a MaxSim server index. Absent either, the class still works
with injected embedders (tests) and degrades to empty otherwise — base install untouched.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from ..types import Hit, PluginInfo

_PAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".tif", ".pdf"}
_COLPALI_MODEL = os.getenv("CR_COLPALI_MODEL", "vidore/colpali-v1.3")


def colpali_available() -> bool:
    """True only if a real ColPali backend can be constructed (the VLM engine is installed)."""
    return importlib.util.find_spec("colpali_engine") is not None


def qdrant_available() -> bool:
    return importlib.util.find_spec("qdrant_client") is not None


def maxsim(query_mat, doc_mat) -> float:
    """Late-interaction score: Σ_i max_j (q_i · d_j) for L2-normalized query/doc patch sets.

    ``query_mat`` is [n_tokens, dim]; ``doc_mat`` is [n_patches, dim]. Pure numpy, exact — the
    reference the quantized/Qdrant path must agree with.
    """
    import numpy as np
    q = np.asarray(query_mat, dtype="float32")
    d = np.asarray(doc_mat, dtype="float32")
    if q.ndim == 1:
        q = q.reshape(1, -1)
    if d.ndim == 1:
        d = d.reshape(1, -1)
    if q.size == 0 or d.size == 0:
        return 0.0
    sims = q @ d.T                     # [n_tokens, n_patches] cosine (inputs normalized)
    return float(sims.max(axis=1).sum())


def _l2_rows(mat):
    import numpy as np
    m = np.asarray(mat, dtype="float32")
    if m.ndim == 1:
        m = m.reshape(1, -1)
    norms = (m * m).sum(axis=1, keepdims=True) ** 0.5
    norms[norms == 0] = 1.0
    return m / norms


class MultiVectorRetriever:
    """Exact in-memory MaxSim over per-page patch matrices.

    ``doc_embed(paths) -> list[patch_matrix]`` and ``query_embed(text) -> token_matrix`` default
    to ColPali (when installed) but are injectable for deterministic tests. Rows are L2-normalized
    on ingest so MaxSim is a cosine late-interaction.
    """

    def __init__(self, docs: list[dict] | None = None, *, source: str = "colpali",
                 doc_embed=None, query_embed=None):
        self.source = source
        self.docs: list[dict] = list(docs or [])   # each: {chunk_id, filename, path, meta, mat}
        self._doc_embed = doc_embed or _default_doc_embed
        self._query_embed = query_embed or _default_query_embed

    @property
    def available(self) -> bool:
        return (self._doc_embed is not _default_doc_embed) or colpali_available()

    def path_for(self, chunk_id: str) -> str | None:
        for d in self.docs:
            if d["chunk_id"] == chunk_id:
                return d.get("path")
        return None

    def index(self, path: str) -> dict:
        """Embed each page/document into a patch matrix. Non-page files are ignored so a mixed
        corpus dir is safe. Idempotent-ish: re-indexing appends, matching the other stores."""
        p = Path(path).expanduser()
        if not self.available:
            return {"files": 0, "pages": 0}
        files = [fp for fp in (sorted(p.rglob("*")) if p.is_dir() else [p])
                 if fp.is_file() and fp.suffix.lower() in _PAGE_EXTS]
        if not files:
            return {"files": 0, "pages": 0}
        mats = self._doc_embed([str(fp) for fp in files])
        n = 0
        for fp, mat in zip(files, mats):
            cid = f"{fp.name}::page"
            self.docs.append({
                "chunk_id": cid, "filename": fp.name, "path": str(fp),
                "mat": _l2_rows(mat),
                "meta": {"type": "page_image", "source_id": fp.name, "page": None,
                         "bbox": None, "embedding_id": cid, "path": str(fp),
                         "late_interaction": True},
            })
            n += 1
        return {"files": n, "pages": n}

    def search(self, query: str, k: int, method: str = "colpali") -> list[Hit]:
        if not self.docs or not query.strip():
            return []
        qvs = self._query_embed(query)
        if qvs is None or len(qvs) == 0:
            return []
        qmat = _l2_rows(qvs)
        scored = [(maxsim(qmat, d["mat"]), i) for i, d in enumerate(self.docs)]
        scored.sort(reverse=True)
        out: list[Hit] = []
        for score, i in scored[: max(k, 0) or len(scored)]:
            d = self.docs[i]
            out.append(Hit(chunk_id=d["chunk_id"], filename=d["filename"],
                           text=f"[page: {d['filename']}]", score=float(score),
                           source=self.source, meta=dict(d["meta"])))
        return out

    def info(self) -> PluginInfo:
        return PluginInfo(name="multivector_store", kind="retriever",
                          capabilities=frozenset({"search", "colpali", "late-interaction",
                                                  "multi-vector", "visual-document"}))


# ──────────────────────────── real ColPali backend (gated) ────────────────────────────

_engine = None
_tried = False


def _load_colpali():
    global _engine, _tried
    if not _tried:
        _tried = True
        try:  # pragma: no cover - needs the VLM + torch, never in CI
            from colpali_engine.models import ColPali, ColPaliProcessor
            model = ColPali.from_pretrained(_COLPALI_MODEL)
            proc = ColPaliProcessor.from_pretrained(_COLPALI_MODEL)
            _engine = (model, proc)
        except Exception:
            _engine = None
    return _engine


def _default_doc_embed(paths: list[str]):  # pragma: no cover - needs the VLM
    eng = _load_colpali()
    if eng is None:
        return []
    from PIL import Image
    model, proc = eng
    batch = proc.process_images([Image.open(p).convert("RGB") for p in paths])
    return [e.float().cpu().numpy() for e in model(**batch)]


def _default_query_embed(text: str):  # pragma: no cover - needs the VLM
    eng = _load_colpali()
    if eng is None:
        return None
    model, proc = eng
    batch = proc.process_queries([text])
    return model(**batch)[0].float().cpu().numpy()
