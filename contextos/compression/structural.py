"""Structural compression — a port of sidekick ``context_budget.clip`` (SPEC §5.5).

Lossy but provenance-preserving: keeps head+tail with an elision marker, records what
it was derived from and roughly what it omitted. The semantic compressor (LLMLingua-2)
is the v0.1 optional second stage; this structural pass is always available offline.
"""
from __future__ import annotations

from ..types import Compressed, Hit

_CHARS_PER_TOKEN = 4


def clip(text: str, max_chars: int = 4000) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n... [clipped {omitted} chars] ...\n{text[-tail:]}"


class StructuralCompressor:
    def compress(self, text: str, target_tokens: int) -> Compressed:
        max_chars = max(200, target_tokens * _CHARS_PER_TOKEN)
        clipped = clip(text, max_chars)
        return Compressed(
            text=clipped,
            tokens=max(1, len(clipped) // _CHARS_PER_TOKEN),
            omitted=("clipped-middle",) if len(clipped) < len(text) else (),
        )

    def assemble(self, hits: list[Hit], target_tokens: int) -> Compressed:
        """Pack ranked hits into a citation-numbered context within a token budget."""
        budget_chars = max(400, target_tokens * _CHARS_PER_TOKEN)
        parts: list[str] = []
        used = 0
        derived: list[str] = []
        omitted: list[str] = []
        for i, h in enumerate(hits, 1):
            block = f"[{i}] {h.filename}: {h.text}"
            if used + len(block) > budget_chars and parts:
                omitted.append(h.chunk_id)
                continue
            block = clip(block, budget_chars - used) if used + len(block) > budget_chars else block
            parts.append(block)
            derived.append(h.chunk_id)
            used += len(block)
        text = "\n\n".join(parts)
        return Compressed(
            text=text, tokens=max(1, len(text) // _CHARS_PER_TOKEN),
            derived_from=tuple(derived), omitted=tuple(omitted),
        )
