"""Multimodal ingestion for Context Runtime: mixed assets → normalized text corpus."""
from .multimodal import Availability, CorpusStats, build_corpus, extract_text

__all__ = ["build_corpus", "extract_text", "CorpusStats", "Availability"]
