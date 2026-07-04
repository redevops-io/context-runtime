"""Cross-modal image retrieval — a text query retrieves IMAGES (Phase 2a of multimodal).

The `vector` method finds documents that MEAN a query; this finds *images* that mean it —
screenshots, diagrams, chart frames — with no OCR and no shared terms. It embeds each image
with a CLIP/SigLIP **vision** tower at index time and the text query with the matching **text**
tower at query time; both land in one joint space, so cosine ranks images by a text description.

Backend: fastembed's ImageEmbedding + the matching CLIP text model (ONNX runtime, no torch —
same stack as store_semantic). Optional: with no fastembed/model, image search degrades to an
empty result and the base install is unaffected. The embedders are injectable, so the retrieval
logic is testable without downloading a model.

    ret = ImageRetriever()
    ret.index("/path/with/screenshots")          # embeds every image
    hits = ret.search("bar chart with a revenue drop after Q2", k=5)   # → the right image

Each hit carries the multimodal **evidence-segment** schema in Hit.meta (type/source_id/page/
bbox/embedding_id/path), so a result is actionable, not just a filename.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..types import Hit, PluginInfo

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
# CLIP vision + the MATCHING text tower (same joint space). SigLIP-2 ONNX is a drop-in upgrade
# once packaged; CLIP-ViT-B-32 ships in fastembed today and proves the primitive.
_VISION_MODEL = os.getenv("CR_IMAGE_EMBED_MODEL", "Qdrant/clip-ViT-B-32-vision")
_TEXT_MODEL = os.getenv("CR_IMAGE_TEXT_MODEL", "Qdrant/clip-ViT-B-32-text")

_vision = None
_text = None
_tried = False


def _load():
    global _vision, _text, _tried
    if not _tried:
        _tried = True
        try:
            from fastembed import ImageEmbedding, TextEmbedding
            _vision = ImageEmbedding(_VISION_MODEL)
            _text = TextEmbedding(_TEXT_MODEL)
        except Exception:
            _vision, _text = None, None
    return _vision, _text


def image_embeddings_available() -> bool:
    v, t = _load()
    return v is not None and t is not None


def _l2(vecs):
    import numpy as np
    out = []
    for v in vecs:
        v = np.asarray(v, dtype="float32")
        n = float((v @ v) ** 0.5) or 1.0
        out.append(v / n)   # L2-normalize so dot == cosine
    return out


def _default_image_embed(paths: list[str]):
    v, _ = _load()
    return _l2(v.embed(paths)) if v is not None else []


def _default_text_embed(texts: list[str]):
    _, t = _load()
    return _l2(t.embed(texts)) if t is not None else []


class ImageRetriever:
    """Embeds every image once (cached) and ranks images by cosine to a text query embedding.

    ``image_embed(paths)->vecs`` and ``text_embed(texts)->vecs`` default to fastembed's CLIP
    towers but are injectable for deterministic tests.
    """

    def __init__(self, docs: list[dict] | None = None, *, source: str = "image",
                 image_embed=None, text_embed=None, use_turbovec: bool | None = None,
                 bit_width: int = 4):
        self.source = source
        self.docs: list[dict] = list(docs or [])   # each: {chunk_id, filename, path, text, meta}
        self._image_embed = image_embed or _default_image_embed
        self._text_embed = text_embed or _default_text_embed
        self._emb = None
        self._emb_n = -1
        # optional quantized ANN over the image vectors (scale); None → auto (use if installed).
        self._use_tv = use_turbovec
        self.bit_width = int(os.getenv("CR_TURBOVEC_BITS", bit_width))
        self._ann = None
        self._ann_n = -1

    def _turbovec_ok(self) -> bool:
        if self._use_tv is False:
            return False
        try:
            import importlib.util
            return bool(importlib.util.find_spec("turbovec"))
        except Exception:
            return False

    def path_for(self, chunk_id: str) -> str | None:
        """Resolve an image chunk_id to its file path — only indexed images are servable."""
        for d in self.docs:
            if d["chunk_id"] == chunk_id:
                return d.get("path")
        return None

    @property
    def available(self) -> bool:
        # injected embedders are always "available"; the default path needs fastembed.
        return (self._image_embed is not _default_image_embed) or image_embeddings_available()

    def index(self, path: str) -> dict:
        """Index a folder of images (or a single image). Non-image files are ignored, so this
        can share a mixed corpus dir with the text stores."""
        p = Path(path).expanduser()
        n = 0
        for fp in sorted(p.rglob("*")) if p.is_dir() else [p]:
            if fp.is_file() and fp.suffix.lower() in _IMG_EXTS:
                cid = f"{fp.name}::img"
                self.docs.append({
                    "chunk_id": cid, "filename": fp.name, "path": str(fp),
                    "text": f"[image: {fp.name}]", "created_at": None,
                    "meta": {"type": "image", "source_id": fp.name, "page": None,
                             "bbox": None, "embedding_id": cid, "path": str(fp)},
                })
                n += 1
        self._emb = None   # invalidate cache
        return {"files": n, "images": n}

    def _matrix(self):
        import numpy as np
        if self._emb is not None and self._emb_n == len(self.docs):
            return self._emb
        if not self.docs or not self.available:
            self._emb, self._emb_n = None, len(self.docs)
            return None
        vecs = self._image_embed([d["path"] for d in self.docs])
        self._emb = np.vstack(vecs) if len(vecs) else None
        self._emb_n = len(self.docs)
        return self._emb

    def _ann_index(self):
        """Build a quantized ANN (TurboVec) over the IMAGE vectors — for scale."""
        import numpy as np
        if self._ann is not None and self._ann_n == len(self.docs):
            return self._ann
        if not self.docs or not self.available:
            self._ann, self._ann_n = None, len(self.docs)
            return None
        from turbovec import TurboQuantIndex
        mat = np.vstack(self._image_embed([d["path"] for d in self.docs])).astype("float32")
        idx = TurboQuantIndex(dim=mat.shape[1], bit_width=self.bit_width)
        idx.add(mat)
        self._ann, self._ann_n = idx, len(self.docs)
        return idx

    def _hit(self, i: int, score: float) -> Hit:
        d = self.docs[i]
        return Hit(chunk_id=d["chunk_id"], filename=d["filename"], text=d["text"],
                   score=float(score), created_at=d.get("created_at"),
                   source=self.source, meta=dict(d["meta"]))

    def search(self, query: str, k: int, method: str = "image") -> list[Hit]:
        if not self.docs or not query.strip():
            return []
        import numpy as np
        qvs = self._text_embed([query])
        if not len(qvs):
            return []
        qv = np.asarray(qvs[0], dtype="float32")
        # TurboVec quantized ANN over the image vectors when installed (drop-in for scale);
        # otherwise exact numpy cosine (fine up to ~10^4-10^5 images).
        if self._turbovec_ok():
            idx = self._ann_index()
            if idx is None:
                return []
            scores, indices = idx.search(qv.reshape(1, -1), k=max(k, 1))
            out = []
            for s, i in zip(np.ravel(scores), np.ravel(indices), strict=False):
                i = int(i)
                if 0 <= i < len(self.docs):
                    out.append(self._hit(i, float(s)))
            return out
        mat = self._matrix()
        if mat is None:
            return []
        sims = mat @ qv
        order = np.argsort(-sims)[: max(k, 0) or len(sims)]
        return [self._hit(int(i), float(sims[int(i)])) for i in order]

    def info(self) -> PluginInfo:
        return PluginInfo(name="image_store", kind="retriever",
                          capabilities=frozenset({"search", "image", "cross-modal", "embeddings"}))
