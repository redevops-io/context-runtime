"""Phase 2a — cross-modal image retrieval + multimodal method routing. Embedders are
injected (deterministic), so no model download is needed."""
from __future__ import annotations

from context_runtime.adapters.store_image import ImageRetriever, image_embeddings_available
from context_runtime.adapters.store_multimodal import MultimodalRetriever
from context_runtime.types import Hit

# a tiny shared "concept space" so a text query and an image land in the same axes
AXES = ["chart", "diagram", "cat"]


def _cvec(s: str):
    import numpy as np
    v = np.array([1.0 if a in s.lower() else 0.0 for a in AXES], dtype="float32")
    if v.sum() == 0:
        v = np.ones(len(AXES), dtype="float32")
    return v / (float((v @ v) ** 0.5) or 1.0)   # L2-normalized (store assumes cosine)


def _img_embed(paths):
    return [_cvec(p) for p in paths]


def _txt_embed(texts):
    return [_cvec(t) for t in texts]


def _make_images(tmp_path, names):
    for n in names:
        (tmp_path / n).write_bytes(b"\x89PNG\r\n")   # dummy bytes; the stub embeds by path
    return tmp_path


def test_image_retriever_cross_modal(tmp_path):
    _make_images(tmp_path, ["revenue-chart.png", "system-diagram.png", "cute-cat.png"])
    ret = ImageRetriever(image_embed=_img_embed, text_embed=_txt_embed)
    rep = ret.index(str(tmp_path))
    assert rep["images"] == 3
    hits = ret.search("a bar chart showing a revenue drop", 3, "image")
    assert hits and hits[0].filename == "revenue-chart.png"      # text query → the right IMAGE
    assert hits[0].meta["type"] == "image" and hits[0].meta["source_id"] == "revenue-chart.png"
    # a different query routes to a different image
    assert ret.search("an architecture diagram", 1, "image")[0].filename == "system-diagram.png"


def test_image_retriever_degrades_without_backend(tmp_path):
    # no injected embedder + no fastembed installed → empty (base install unaffected)
    if image_embeddings_available():
        return
    ret = ImageRetriever()
    _make_images(tmp_path, ["x.png"])
    ret.index(str(tmp_path))
    assert ret.search("anything", 3, "image") == []


class _StubText:
    def search(self, query, k, method):
        return [Hit(chunk_id="t0", filename="doc.txt", text=f"text::{method}", score=1.0)]

    def index(self, path):
        return {"files": 1}


def test_multimodal_router_dispatch(tmp_path):
    _make_images(tmp_path, ["revenue-chart.png"])
    img = ImageRetriever(image_embed=_img_embed, text_embed=_txt_embed)
    img.index(str(tmp_path))
    mm = MultimodalRetriever(text=_StubText(), image=img)
    # image method → image store
    assert mm.search("chart", 1, "image")[0].filename == "revenue-chart.png"
    # text method → text store
    assert mm.search("q", 1, "bm25")[0].text == "text::bm25"
    # index fans to both
    rep = mm.index(str(tmp_path))
    assert "text" in rep and rep["image"]["images"] == 1
