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


def test_path_for_resolves_only_indexed(tmp_path):
    _make_images(tmp_path, ["revenue-chart.png"])
    ret = ImageRetriever(image_embed=_img_embed, text_embed=_txt_embed)
    ret.index(str(tmp_path))
    cid = ret.docs[0]["chunk_id"]
    assert ret.path_for(cid).endswith("revenue-chart.png")
    assert ret.path_for("not-indexed::img") is None   # no arbitrary file read


def test_compare_shows_image_column_with_url(tmp_path):
    from context_runtime.integrations.librechat import (
        DEFAULT_STRATEGIES, IMAGE_STRATEGY, LibreChatTenant,
    )
    _make_images(tmp_path, ["revenue-chart.png", "system-diagram.png"])
    img = ImageRetriever(image_embed=_img_embed, text_embed=_txt_embed)
    mm = MultimodalRetriever(text=_StubText(), image=img)
    mm.index(str(tmp_path))
    t = LibreChatTenant(retriever=mm, strategies=DEFAULT_STRATEGIES + (IMAGE_STRATEGY,))
    out = t.compare("a bar chart", k=3)
    assert "image" in out["methods"]                       # the image column is shown
    hits = out["methods"]["image"]
    assert hits and hits[0]["image_url"].startswith("/librechat/image?chunk_id=")
    # a text-only tenant must NOT show an image column
    t2 = LibreChatTenant(retriever=_StubText())
    assert "image" not in t2.compare("q", k=2)["methods"]
    # image_path resolves through the tenant → for the serving endpoint
    assert t.image_path(hits[0]["chunk_id"]).endswith(".png")


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


# ── Phase 3 video: channel fusion + transcript windowing (was zero-coverage, P0) ──

def test_video_search_channel_fusion():
    import numpy as np
    from context_runtime.adapters.store_video import VideoRetriever
    qv = np.array([1.0, 0.0], dtype="float32")
    docs = [
        {"chunk_id": "v::A", "filename": "v.mp4", "text": "A", "start": 0.0, "end": 1.0,
         "fvec": np.array([1.0, 0.0], dtype="float32"), "tvec": np.array([0.0, 1.0], dtype="float32"),
         "meta": {"type": "video_segment"}},                                  # both channels present
        {"chunk_id": "v::B", "filename": "v.mp4", "text": "B", "start": 1.0, "end": 2.0,
         "fvec": None, "tvec": np.array([1.0, 0.0], dtype="float32"), "meta": {"type": "video_segment"}},  # spoken only
        {"chunk_id": "v::C", "filename": "v.mp4", "text": "C", "start": 2.0, "end": 3.0,
         "fvec": np.array([1.0, 0.0], dtype="float32"), "tvec": None, "meta": {"type": "video_segment"}},  # visual only
    ]
    r = VideoRetriever(docs, text_embed=lambda texts: [qv], spoken_weight=0.5)
    hits = {h.chunk_id: h for h in r.search("q", k=5)}
    # both present → weighted blend (0.5*vis + 0.5*spo = 0.5), labelled by the stronger channel (visual)
    assert abs(hits["v::A"].score - 0.5) < 1e-6 and hits["v::A"].meta["channel"] == "visual"
    # a missing channel (score -1) → the present channel wins via max()
    assert abs(hits["v::B"].score - 1.0) < 1e-6 and hits["v::B"].meta["channel"] == "spoken"
    assert abs(hits["v::C"].score - 1.0) < 1e-6 and hits["v::C"].meta["channel"] == "visual"


def test_video_transcript_window_overlap_and_empty():
    from context_runtime.adapters.store_video import VideoRetriever
    r = VideoRetriever([])
    segs = [(0.0, 5.0, "hello"), (5.0, 10.0, "world"), (10.0, 15.0, "later")]
    assert r._transcript_for(segs, 4.0, 6.0) == "hello world"   # overlaps first two (te>start and ts<end)
    assert r._transcript_for(segs, 20.0, 25.0) == ""            # overlaps none → empty


def test_video_index_builds_segments_with_stubs(tmp_path):
    import numpy as np
    from context_runtime.adapters.store_video import VideoRetriever
    vid = tmp_path / "clip.mp4"
    vid.write_bytes(b"x")
    scenes = [(0.0, 5.0, "/kf0.jpg"), (5.0, 10.0, "/kf1.jpg")]
    transcript = [(0.0, 5.0, "spoken words")]                   # only the first scene has speech
    r = VideoRetriever(
        scene_extractor=lambda p: scenes,
        transcriber=lambda p: transcript,
        frame_embed=lambda paths: [np.array([1.0, 0.0], dtype="float32") for _ in paths],
        text_embed=lambda texts: [np.array([0.0, 1.0], dtype="float32") for _ in texts],
    )
    res = r.index(str(vid))
    assert res == {"files": 1, "segments": 2}
    seg0, seg1 = r.docs
    assert seg0["tvec"] is not None and seg0["text"] == "spoken words"
    assert seg1["tvec"] is None and seg1["text"].startswith("[clip.mp4")   # empty transcript → placeholder
    assert seg0["fvec"] is not None and seg1["fvec"] is not None
