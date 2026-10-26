"""The model-visible contract of a transcript compactor: how removed content is marked, and how a caller can ask what the model can still see.

Context Runtime OWNS the marker formats. A harness (agent-harness) must not hard-code them: it asks the manager through ``is_placeholder`` and
``visibility`` (both are part of ``agent_harness.context.ContextManager``). Everything here is pure text logic with no dependencies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Stubs written by TranscriptCompactor (see transcript.py: _compact / admit / on_context_error)
OUTPUT_STUB = re.compile(r"^\[archived (A\d+)")                                      # a whole tool output replaced by a stub (also the legacy argument stub)
ARGUMENT_PLACEHOLDER = re.compile(r"^\s*<<ARCHIVED-PLACEHOLDER (A\d+)")             # v2 stub that replaces a long tool-call ARGUMENT
LEGACY_ARGUMENT_STUB = re.compile(r"^\s*\[archived (A\d+)")
CLIPPED_NOTE = re.compile(r"archived as (A\d+)")                                      # admit(): head and tail kept, middle archived
STATE_VISIBLE, STATE_CLIPPED, STATE_ARCHIVED, STATE_UNAVAILABLE = "visible", "clipped", "archived", "unavailable"


def is_placeholder(value: Any) -> bool:
    """True for text that is a stand-in for removed content, never real content (it must not be reused as a tool argument)."""
    return isinstance(value, str) and bool(ARGUMENT_PLACEHOLDER.match(value) or LEGACY_ARGUMENT_STUB.match(value))


@dataclass(frozen=True)
class ResultView:
    """What the model can currently see of one tool result."""
    state: str                       # visible | clipped | archived | unavailable
    archive_id: Optional[str] = None
    chars: int = 0

    @property
    def recoverable(self) -> bool:
        return self.state in (STATE_ARCHIVED, STATE_CLIPPED) and self.archive_id is not None


def result_view(messages: List[Dict[str, Any]], tool_call_id: str) -> ResultView:
    """Look the tool result up in the CURRENT message list (what would be sent), not in any history."""
    for m in reversed(messages):
        if m.get("role") == "tool" and m.get("tool_call_id") == tool_call_id:
            c = m.get("content")
            if not isinstance(c, str):
                return ResultView(STATE_VISIBLE, None, 0)
            a = OUTPUT_STUB.match(c)
            if a:
                return ResultView(STATE_ARCHIVED, a.group(1), len(c))
            k = CLIPPED_NOTE.search(c)
            if k:
                return ResultView(STATE_CLIPPED, k.group(1), len(c))
            return ResultView(STATE_VISIBLE, None, len(c))
    return ResultView(STATE_UNAVAILABLE)
