"""Regression: the context packer must TRUNCATE a single over-budget chunk, not drop it.

Dropping it silently produced an EMPTY context for long single passages (e.g. LongMemEval
sessions), making an arm look like it scored 0 when it simply received no context.
"""
import sys
from pathlib import Path

_HARNESS = Path(__file__).resolve().parents[1] / "benchmarks" / "context-vs-model"
sys.path.insert(0, str(_HARNESS))
from harness.arms import _chunks_to_context, _CHARS_PER_TOK  # noqa: E402


class _Chunk:
    def __init__(self, doc_id, text):
        self.doc_id = doc_id
        self.text = text


def test_oversized_single_chunk_is_truncated_not_dropped():
    budget = 100 * _CHARS_PER_TOK
    ctx = _chunks_to_context([_Chunk("s1", "x" * (budget * 50))], max_tokens=100)
    assert ctx, "an over-budget single chunk must yield a truncated (non-empty) context"
    assert len(ctx) <= budget + 32  # bounded by budget (+ small header slack)


def test_normal_chunks_fit():
    ctx = _chunks_to_context([_Chunk("a", "hello"), _Chunk("b", "world")], max_tokens=1000)
    assert "hello" in ctx and "world" in ctx
