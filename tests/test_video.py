"""Phase 3 — timestamped video-segment retrieval + two-channel (visual/spoken) fusion.
Scene detection, ASR, and embedders are injected, so no ffmpeg/scenedetect/whisper/model."""
from __future__ import annotations

from context_runtime.adapters.store_video import VideoRetriever, video_available

AXES = ["merger", "product", "cat"]


def _cvec(s: str):
    import numpy as np
    v = np.array([1.0 if a in s.lower() else 0.0 for a in AXES], dtype="float32")
    if v.sum() == 0:
        v = np.ones(len(AXES), dtype="float32")
    return v / (float((v @ v) ** 0.5) or 1.0)


def _make_video(tmp_path, name="talk.mp4"):
    (tmp_path / name).write_bytes(b"\x00\x00\x00\x18ftypmp4")
    return tmp_path


def test_video_segments_retrieve_by_timestamp(tmp_path):
    _make_video(tmp_path)

    # scene 0 [0-10] shows/says "merger"; scene 1 [10-20] shows/says "product"
    def scenes(path):
        return [(0.0, 10.0, "kf_5.jpg"), (10.0, 20.0, "kf_15.jpg")]

    def transcribe(path):
        return [(0.0, 9.0, "we announced the merger"), (11.0, 19.0, "the new product launch")]

    def frame_embed(paths):
        # keyframe visuals mirror the scene topic (5→merger, 15→product)
        return [_cvec("merger") if "5" in p else _cvec("product") for p in paths]

    ret = VideoRetriever(scene_extractor=scenes, transcriber=transcribe,
                         frame_embed=frame_embed, text_embed=lambda ts: [_cvec(t) for t in ts])
    rep = ret.index(str(tmp_path))
    assert rep["segments"] == 2

    hits = ret.search("when do they talk about the merger", k=2)
    top = hits[0]
    assert top.meta["type"] == "video_segment"
    assert top.meta["start"] == 0.0 and top.meta["end"] == 10.0   # the RIGHT clip
    assert top.meta["deep_link"] == "talk.mp4#t=0"
    # a different query lands on the other segment
    assert ret.search("the product launch", k=1)[0].meta["start"] == 10.0
    assert ret.path_for(top.chunk_id).endswith("talk.mp4")


def test_visual_channel_matches_when_unspoken(tmp_path):
    _make_video(tmp_path)

    def scenes(path):
        return [(0.0, 5.0, "kf_2.jpg")]

    def transcribe(path):
        return []                                    # nothing spoken — visual channel only

    ret = VideoRetriever(scene_extractor=scenes, transcriber=transcribe,
                         frame_embed=lambda ps: [_cvec("merger")],
                         text_embed=lambda ts: [_cvec(t) for t in ts])
    ret.index(str(tmp_path))
    hits = ret.search("merger", k=1)                 # matched purely on the keyframe
    assert hits and hits[0].meta["channel"] == "visual"


def test_degrades_without_backend(tmp_path):
    if video_available():
        return
    _make_video(tmp_path)
    ret = VideoRetriever()                            # no injected backend, no scenedetect
    assert ret.index(str(tmp_path))["segments"] == 0
    assert ret.search("anything", 3) == []
