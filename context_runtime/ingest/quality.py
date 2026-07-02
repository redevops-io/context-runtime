"""QualityPlugins — the optional pre-index gate (SPEC §4.8).

Between extraction and chunking, a QualityPlugin may clean, normalize, or reject a
passage. Two implementations:

  * HeuristicQuality — dependency-free, deterministic: strips control characters,
    normalizes whitespace, drops near-empty text, and de-duplicates identical passages
    across a run. Safe default.
  * LLMQuality — model-driven review: an LLM (the same seam sidekick uses) cleans OCR
    noise / boilerplate or votes to drop low-value passages. Best-effort and fail-open
    — any error or empty response falls back to the input text, so a flaky judge never
    drops good data.

The gate only makes sense for batch ingestion (materialize → clean → index). For live
structured lookups you'd query the source directly through a tool/retriever instead.
"""
from __future__ import annotations

import hashlib
import re

from ..types import PluginInfo, RawAsset

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RUN = re.compile(r"[ \t]{2,}")
_BLANKS = re.compile(r"\n{3,}")


class HeuristicQuality:
    def __init__(self, min_chars: int = 24, dedup: bool = True):
        self.min_chars = min_chars
        self.dedup = dedup
        self._seen: set[str] = set()

    def review(self, text: str, asset: RawAsset) -> str | None:
        text = _CTRL.sub("", text)
        text = _WS_RUN.sub(" ", text)
        text = _BLANKS.sub("\n\n", text).strip()
        if len(text) < self.min_chars:
            return None
        if self.dedup:
            h = hashlib.sha1(re.sub(r"\s+", " ", text).lower().encode("utf-8")).hexdigest()
            if h in self._seen:
                return None
            self._seen.add(h)
        return text

    def info(self) -> PluginInfo:
        return PluginInfo(name="heuristic_quality", kind="quality", version="0.1",
                          capabilities=frozenset({"normalize", "dedup", "min_length"}))


class LLMQuality:
    """Model-driven cleaning/filtering. `model` is any ModelPlugin (SPEC §4.3). mode:
    'clean' rewrites the passage (fixing OCR noise, stripping boilerplate); 'filter'
    keeps it verbatim but drops passages the model judges as junk. Fail-open."""

    def __init__(self, model, model_name: str = "", *, mode: str = "clean",
                 max_tokens: int = 1024, min_chars: int = 24):
        self.model = model
        self.model_name = model_name
        self.mode = mode
        self.max_tokens = max_tokens
        self.min_chars = min_chars

    def _complete(self, prompt: str) -> str:
        from ..types import ModelRequest
        try:
            res = self.model.complete(ModelRequest(
                model=self.model_name, prompt=prompt, max_tokens=self.max_tokens))
            return (getattr(res, "text", "") or "").strip()
        except Exception:
            return ""

    def review(self, text: str, asset: RawAsset) -> str | None:
        if len(text.strip()) < self.min_chars:
            return None
        if self.mode == "filter":
            verdict = self._complete(
                "Reply with exactly KEEP or DROP. DROP only if the following text is "
                "navigation/boilerplate/garbage with no informational value.\n\n" + text[:4000]
            ).upper()
            return None if verdict.startswith("DROP") else text
        cleaned = self._complete(
            "Clean the following extracted text for a search index: fix obvious OCR "
            "errors, remove boilerplate and repeated headers/footers, preserve all facts, "
            "numbers and tables. Return ONLY the cleaned text.\n\n" + text[:8000]
        )
        return cleaned if len(cleaned) >= self.min_chars else text

    def info(self) -> PluginInfo:
        return PluginInfo(name="llm_quality", kind="quality", version="0.1",
                          capabilities=frozenset({"clean", "filter", "llm"}))
