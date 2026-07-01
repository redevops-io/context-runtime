"""Multimodal ingestion for Context Runtime — turn a folder tree of mixed assets
(PDF, DOCX, HTML, images, audio, video, text) into a normalized text corpus that any
runtime (Python or the Go port, via a shared corpus dir) can index and retrieve.

Context Runtime is a text-context planner: every asset is reduced to a searchable
TEXT surface, so a user request can retrieve the right document regardless of its
original modality. Extraction degrades gracefully — text-layer formats (PDF/DOCX/HTML)
need only light deps; image OCR and audio/video ASR activate when their optional
backends are installed, so a machine without them still ingests the bulk of a corpus.

Backends (all optional, lazily imported):
  * PDF   → pypdf                     (text layer)
  * DOCX  → python-docx
  * HTML  → beautifulsoup4 + lxml
  * image → rapidocr-onnxruntime      (OCR, userland — no system tesseract)
  * audio → faster-whisper            (ASR, userland — no torch)
  * video → ffmpeg (audio track) → faster-whisper

`build_corpus()` writes one `<id>.txt` per extracted asset plus a `manifest.jsonl`,
so both the Python `InMemoryStore.index(dir)` and the Go control-plane `POST /index`
ingest the exact same corpus.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# pypdf is chatty about malformed dictionaries in real-world PDFs — silence it.
logging.getLogger("pypdf").setLevel(logging.ERROR)

TEXT_EXTS = {".txt", ".md", ".rst", ".csv", ".log"}
PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx"}
HTML_EXTS = {".html", ".htm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".webp"}
AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".ogg", ".flac", ".aac"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}

_MIN_CHARS = 24  # below this, an extraction is treated as empty (e.g. a scanned PDF)


@dataclass
class Extractors:
    """Lazily-resolved optional backends; each is None when its dep is absent."""

    _ocr: object | None = None
    _asr: object | None = None
    _ocr_tried: bool = False
    _asr_tried: bool = False

    def ocr(self):
        if not self._ocr_tried:
            self._ocr_tried = True
            try:
                from rapidocr_onnxruntime import RapidOCR
                self._ocr = RapidOCR()
            except Exception:
                self._ocr = None
        return self._ocr

    def asr(self):
        if not self._asr_tried:
            self._asr_tried = True
            try:
                from faster_whisper import WhisperModel
                self._asr = WhisperModel(os.getenv("CR_ASR_MODEL", "base"), compute_type="int8")
            except Exception:
                self._asr = None
        return self._asr


@dataclass
class Availability:
    pdf: bool = False
    docx: bool = False
    html: bool = False
    ocr: bool = False
    asr: bool = False
    ffmpeg: bool = False

    def as_dict(self) -> dict:
        return {"pdf": self.pdf, "docx": self.docx, "html": self.html,
                "ocr": self.ocr, "asr": self.asr, "ffmpeg": self.ffmpeg}


def _availability(ex: Extractors) -> Availability:
    import shutil
    a = Availability(ffmpeg=bool(shutil.which("ffmpeg")))
    try:
        import pypdf  # noqa: F401
        a.pdf = True
    except Exception:
        pass
    try:
        import docx  # noqa: F401
        a.docx = True
    except Exception:
        pass
    try:
        import bs4  # noqa: F401
        a.html = True
    except Exception:
        pass
    a.ocr = ex.ocr() is not None
    a.asr = ex.asr() is not None
    return a


# ──────────────────────────── per-modality extractors ────────────────────────────


def _extract_pdf(path: str) -> str:
    from pypdf import PdfReader
    reader = PdfReader(path)
    return "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()


def _extract_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()


def _extract_html(path: str) -> str:
    from bs4 import BeautifulSoup
    with open(path, encoding="utf-8", errors="ignore") as fh:
        soup = BeautifulSoup(fh.read(), "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return re.sub(r"\n{3,}", "\n\n", soup.get_text(" ", strip=True)).strip()


def _extract_image(path: str, ex: Extractors) -> str:
    ocr = ex.ocr()
    if ocr is None:
        return ""
    result, _ = ocr(path)
    if not result:
        return ""
    return " ".join(line[1] for line in result).strip()


def _extract_audio(path: str, ex: Extractors) -> str:
    asr = ex.asr()
    if asr is None:
        return ""
    segments, _ = asr.transcribe(path)
    return " ".join(seg.text for seg in segments).strip()


def _extract_video(path: str, ex: Extractors) -> str:
    if ex.asr() is None:
        return ""
    import shutil
    if not shutil.which("ffmpeg"):
        return ""
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "audio.wav")
        proc = subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-ac", "1", "-ar", "16000", "-vn", wav],
            capture_output=True)
        if proc.returncode != 0 or not os.path.exists(wav):
            return ""
        return _extract_audio(wav, ex)


def extract_text(path: str, ex: Extractors | None = None) -> tuple[str, str]:
    """Return (text, kind) for one asset. kind ∈ {pdf,docx,html,image,audio,video,text}.
    Text may be empty when the required backend is missing (e.g. a scanned PDF without OCR)."""
    ex = ex or Extractors()
    suffix = Path(path).suffix.lower()
    try:
        if suffix in PDF_EXTS:
            return _extract_pdf(path), "pdf"
        if suffix in DOCX_EXTS:
            return _extract_docx(path), "docx"
        if suffix in HTML_EXTS:
            return _extract_html(path), "html"
        if suffix in IMAGE_EXTS:
            return _extract_image(path, ex), "image"
        if suffix in AUDIO_EXTS:
            return _extract_audio(path, ex), "audio"
        if suffix in VIDEO_EXTS:
            return _extract_video(path, ex), "video"
        if suffix in TEXT_EXTS:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                return fh.read().strip(), "text"
    except Exception:
        return "", "error"
    return "", "unsupported"


# ──────────────────────────── corpus builder ────────────────────────────


@dataclass
class CorpusStats:
    out_dir: str
    written: int = 0
    skipped_empty: int = 0
    by_kind: dict = field(default_factory=dict)
    ocr_used: int = 0
    asr_used: int = 0
    availability: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"out_dir": self.out_dir, "written": self.written,
                "skipped_empty": self.skipped_empty, "by_kind": self.by_kind,
                "ocr_used": self.ocr_used, "asr_used": self.asr_used,
                "availability": self.availability}


def _safe_id(rel: str, n: int) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", rel).strip("-").lower() or "doc"
    return f"{n:04d}_{stem[:80]}"


def chunk_text(text: str, size: int, overlap: int = 120) -> list[str]:
    """Split text into ~size-char passages on paragraph boundaries (deterministic).
    Focused passages let the retriever surface the right lab panel instead of a whole
    multi-panel document. size <= 0 disables chunking (one passage = the whole doc)."""
    text = text.strip()
    if size <= 0 or len(text) <= size:
        return [text] if text else []
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for para in paras:
        if len(para) > size:  # a huge paragraph — hard-split it
            for i in range(0, len(para), size - overlap):
                piece = para[i:i + size]
                if piece.strip():
                    chunks.append(piece.strip())
            continue
        if cur and len(cur) + len(para) + 2 > size:
            chunks.append(cur.strip())
            cur = (cur[-overlap:] + "\n\n" + para) if overlap else para  # carry a little context
        else:
            cur = (cur + "\n\n" + para) if cur else para
    if cur.strip():
        chunks.append(cur.strip())
    return chunks or ([text] if text else [])


def build_corpus(sources: list[str], out_dir: str, *, follow_symlinks: bool = True,
                 limit: int | None = None, chunk_chars: int = 900, verbose: bool = False) -> CorpusStats:
    """Walk `sources`, extract each asset to text, split into ~chunk_chars passages, and
    write one `<out_dir>/<id>.txt` per passage (with a provenance header) +
    `<out_dir>/manifest.jsonl`. chunk_chars<=0 keeps one passage per document."""
    ex = Extractors()
    stats = CorpusStats(out_dir=out_dir, availability=_availability(ex).as_dict())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # manifest lives BESIDE the corpus dir, not inside it, so a folder-index (Python or
    # the Go store) ingests only the extracted .txt docs — never the manifest.
    manifest = (out.parent / f"{out.name}.manifest.jsonl").open("w", encoding="utf-8")

    files: list[tuple[str, str]] = []  # (abs_path, provenance label)
    for src in sources:
        base = Path(src)
        if base.is_file():
            files.append((str(base), base.name))
            continue
        for p in sorted(base.rglob("*")):
            try:
                if p.is_file() or (follow_symlinks and p.is_symlink() and p.resolve().is_file()):
                    files.append((str(p), str(p.relative_to(base))))
            except OSError:
                continue
    if limit:
        files = files[:limit]

    n = 0
    for abs_path, label in files:
        text, kind = extract_text(abs_path, ex)
        if kind in ("unsupported",):
            continue
        stats.by_kind[kind] = stats.by_kind.get(kind, 0) + 1
        if len(text) < _MIN_CHARS:
            stats.skipped_empty += 1
            if verbose:
                print(f"  skip (empty {kind}): {label}")
            continue
        if kind == "image":
            stats.ocr_used += 1
        elif kind in ("audio", "video"):
            stats.asr_used += 1
        n += 1
        doc_id = _safe_id(label, n)
        passages = chunk_text(text, chunk_chars)
        multi = len(passages) > 1
        for pi, passage in enumerate(passages):
            cid = f"{doc_id}_p{pi:02d}" if multi else doc_id
            tag = f" · passage {pi + 1}/{len(passages)}" if multi else ""
            header = f"[source: {label} · kind: {kind}{tag}]\n\n"
            (out / f"{cid}.txt").write_text(header + passage, encoding="utf-8")
            manifest.write(json.dumps({
                "id": cid, "source": label, "kind": kind, "passage": pi,
                "chars": len(passage), "path": abs_path,
            }, ensure_ascii=False) + "\n")
            stats.written += 1
        stats.by_kind[kind + "_chunks"] = stats.by_kind.get(kind + "_chunks", 0) + len(passages)
        if verbose:
            print(f"  {kind:6} {len(text):>7} chars → {len(passages):>3} passage(s)  {label}")

    manifest.close()
    return stats


def _main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Build a normalized text corpus from mixed assets.")
    ap.add_argument("sources", nargs="+", help="files or folders to ingest")
    ap.add_argument("--out", required=True, help="output corpus directory")
    ap.add_argument("--limit", type=int, default=None, help="cap the number of assets")
    ap.add_argument("--chunk-chars", type=int, default=900,
                    help="passage size in chars (0 = one passage per document)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    stats = build_corpus(args.sources, args.out, limit=args.limit,
                         chunk_chars=args.chunk_chars, verbose=args.verbose)
    print(json.dumps(stats.as_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
