"""Video retrieval by timestamped segment — Phase 3 of multimodal.

Video is two evidence channels bolted to a clock: what's **on screen** (frames) and what's
**said** (transcript). Prior ingest only pulled the audio track, so a query could match a
transcript but never point at *where in the video* the answer is, and never match something only
shown (a slide, a chart, a face) but not spoken. This retriever indexes a video into
**timestamped segments** — scene-cut boundaries, each with a representative keyframe embedding
(visual, CLIP vision tower) and a transcript-window embedding (CLIP text tower) — so a text query
can retrieve the exact clip: ``score = max(visual, spoken)`` fuses the two channels, and each hit
carries ``{start, end}`` for deep-linking. Same joint CLIP space as ``store_image``, one axis up.

Everything heavy is gated + injectable:
  • scene_extractor(path) -> [(start, end, keyframe_path)]   (PySceneDetect + ffmpeg)
  • transcriber(path)     -> [(start, end, text)]            (faster-whisper — already a dep)
  • frame_embed / text_embed                                 (CLIP towers, shared with store_image)
So the retrieval + fusion logic is unit-testable with stubs, and the base install needs no ffmpeg,
no scenedetect, no model. Absent a backend, video search degrades to empty.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from ..types import Hit, PluginInfo
from .store_image import _default_image_embed, _default_text_embed

_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
# how much to weight the spoken channel vs the visual one in the fused score (0.5 = balanced).
_SPOKEN_WEIGHT = float(os.getenv("CR_VIDEO_SPOKEN_WEIGHT", "0.5"))


def video_available() -> bool:
    """True only if the scene-detection backend is installed (ffmpeg is assumed on PATH)."""
    return importlib.util.find_spec("scenedetect") is not None


def _fmt_ts(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60}:{s % 60:02d}"


class VideoRetriever:
    """Index videos into timestamped segments; retrieve the best clip for a text query.

    Injected components default to real backends (gated) but are overridable for tests:
    ``scene_extractor``, ``transcriber``, ``frame_embed(paths)->vecs``, ``text_embed(texts)->vecs``.
    Visual and spoken vectors live in the same joint CLIP space as the query embedding, so a
    text query can match either channel of a segment.
    """

    def __init__(self, docs: list[dict] | None = None, *, source: str = "video",
                 scene_extractor=None, transcriber=None, frame_embed=None, text_embed=None,
                 spoken_weight: float | None = None):
        self.source = source
        self.docs: list[dict] = list(docs or [])   # each segment: {chunk_id, filename, path, start, end, text, fvec, tvec, meta}
        self._scenes = scene_extractor or _default_scene_extractor
        self._transcribe = transcriber or _default_transcriber
        self._frame_embed = frame_embed or _default_image_embed
        self._text_embed = text_embed or _default_text_embed
        self.spoken_weight = _SPOKEN_WEIGHT if spoken_weight is None else float(spoken_weight)

    @property
    def available(self) -> bool:
        injected = (self._scenes is not _default_scene_extractor
                    or self._transcribe is not _default_transcriber)
        return injected or video_available()

    def path_for(self, chunk_id: str) -> str | None:
        for d in self.docs:
            if d["chunk_id"] == chunk_id:
                return d.get("path")
        return None

    def _transcript_for(self, segments, start: float, end: float) -> str:
        """Concatenate transcript windows overlapping a scene's [start, end]."""
        parts = [t for (ts, te, t) in segments if te > start and ts < end and t]
        return " ".join(parts).strip()

    def index(self, path: str) -> dict:
        """Split each video into scene segments; embed each segment's keyframe (visual) and
        transcript window (spoken). Non-video files are ignored (mixed corpus safe)."""
        p = Path(path).expanduser()
        if not self.available:
            return {"files": 0, "segments": 0}
        vids = [fp for fp in (sorted(p.rglob("*")) if p.is_dir() else [p])
                if fp.is_file() and fp.suffix.lower() in _VIDEO_EXTS]
        import numpy as np
        n_seg = 0
        for fp in vids:
            scenes = list(self._scenes(str(fp)))            # [(start, end, keyframe_path)]
            transcript = list(self._transcribe(str(fp)))     # [(start, end, text)]
            if not scenes:
                continue
            frame_paths = [kf for (_, _, kf) in scenes]
            fvecs = self._frame_embed(frame_paths) if frame_paths else []
            texts = [self._transcript_for(transcript, s, e) for (s, e, _) in scenes]
            # embed only non-empty transcripts; empty windows get a None spoken vector.
            nonempty = [(i, t) for i, t in enumerate(texts) if t]
            tvec_map: dict[int, object] = {}
            if nonempty:
                tv = self._text_embed([t for _, t in nonempty])
                for (i, _), v in zip(nonempty, tv):
                    tvec_map[i] = np.asarray(v, dtype="float32")
            for i, (start, end, kf) in enumerate(scenes):
                cid = f"{fp.name}::{int(start)}-{int(end)}"
                self.docs.append({
                    "chunk_id": cid, "filename": fp.name, "path": str(fp),
                    "start": float(start), "end": float(end),
                    "text": texts[i] or f"[{fp.name} {_fmt_ts(start)}–{_fmt_ts(end)}]",
                    "fvec": np.asarray(fvecs[i], dtype="float32") if i < len(fvecs) else None,
                    "tvec": tvec_map.get(i),
                    "meta": {"type": "video_segment", "source_id": fp.name,
                             "start": float(start), "end": float(end), "keyframe": kf,
                             "path": str(fp), "embedding_id": cid,
                             "deep_link": f"{fp.name}#t={int(start)}"},
                })
                n_seg += 1
        return {"files": len(vids), "segments": n_seg}

    def search(self, query: str, k: int, method: str = "video") -> list[Hit]:
        if not self.docs or not query.strip():
            return []
        import numpy as np
        qvs = self._text_embed([query])
        if not len(qvs):
            return []
        qv = np.asarray(qvs[0], dtype="float32")
        scored = []
        for i, d in enumerate(self.docs):
            vis = float(d["fvec"] @ qv) if d.get("fvec") is not None else -1.0
            spo = float(d["tvec"] @ qv) if d.get("tvec") is not None else -1.0
            # channel fusion: the better-matching channel wins, blended toward spoken by weight.
            fused = max(vis, spo) if min(vis, spo) < 0 else (
                (1 - self.spoken_weight) * vis + self.spoken_weight * spo)
            scored.append((fused, i, "spoken" if spo >= vis else "visual"))
        scored.sort(reverse=True)
        out: list[Hit] = []
        for score, i, channel in scored[: max(k, 0) or len(scored)]:
            d = self.docs[i]
            meta = dict(d["meta"]); meta["channel"] = channel
            out.append(Hit(chunk_id=d["chunk_id"], filename=d["filename"],
                           text=d["text"], score=float(score), source=self.source, meta=meta))
        return out

    def info(self) -> PluginInfo:
        return PluginInfo(name="video_store", kind="retriever",
                          capabilities=frozenset({"search", "video", "temporal",
                                                  "cross-modal", "segment"}))


# ──────────────────────────── real backends (gated) ────────────────────────────


def _default_scene_extractor(video_path: str):  # pragma: no cover - needs ffmpeg + scenedetect
    """Detect scene cuts and dump one keyframe per scene next to the video (cached)."""
    try:
        from scenedetect import detect, ContentDetector
    except Exception:
        return []
    import subprocess
    scenes = detect(video_path, ContentDetector())
    out = []
    kf_dir = Path(video_path).with_suffix("")
    kf_dir.mkdir(parents=True, exist_ok=True)
    for start, end in scenes:
        s, e = start.get_seconds(), end.get_seconds()
        mid = (s + e) / 2.0
        kf = kf_dir / f"kf_{int(mid)}.jpg"
        if not kf.exists():
            subprocess.run(["ffmpeg", "-y", "-ss", str(mid), "-i", video_path,
                            "-frames:v", "1", str(kf)],
                           capture_output=True, check=False)
        out.append((s, e, str(kf)))
    return out


def _default_transcriber(video_path: str):  # pragma: no cover - needs faster-whisper
    """ASR with faster-whisper → [(start, end, text)] windows."""
    try:
        from faster_whisper import WhisperModel
    except Exception:
        return []
    model = WhisperModel(os.getenv("CR_WHISPER_MODEL", "base"))
    segments, _ = model.transcribe(video_path)
    return [(seg.start, seg.end, seg.text) for seg in segments]
