"""Rule tables — the v0.1 planner's knowledge (SPEC §4.2, ARCHITECTURE §6).

Two tables: intent classification (keyword → bucket) and bucket → retrieval/reasoning
defaults. Deliberately simple and inspectable; this is the "80% of the value without
over-engineering" layer that precedes CP-SAT and the learning loop.
"""
from __future__ import annotations

import re

from ..types import IntentBucket, Retrieval

# Ordered: first match wins. (compiled pattern, bucket, risk)
INTENT_RULES: list[tuple[re.Pattern, IntentBucket, str]] = [
    (re.compile(r"\b(error|exception|stack ?trace|code|status)\s*[:#]?\s*\w*\d", re.I), "exact_lookup", "low"),
    (re.compile(r"\b[A-Z]{2,}-\d+\b"), "exact_lookup", "low"),                 # JIRA-123, ERR-500
    (re.compile(r"\b(deploy|incident|outage|failed|failure|rollback|postmortem)\b", re.I), "incident", "medium"),
    (re.compile(r"\b(terraform|kubectl|production|prod|migration|delete|drop)\b", re.I), "high_risk", "high"),
    (re.compile(r"\b(secret|password|api[_ ]?key|private|pii|credential)\b", re.I), "sensitive", "high"),
    (re.compile(r"\b(function|class|refactor|bug|patch|implement|stack)\b", re.I), "code_reasoning", "low"),
    (re.compile(r"\b(summari[sz]e|compare|synthesi[sz]e|overview|explain why)\b", re.I), "synthesis", "low"),
    (re.compile(r"\b(what|why|how|concept|difference|mean)\b", re.I), "conceptual", "low"),
]

# bucket → (retrieval methods to consider, default reasoning strategy, verify?)
BUCKET_DEFAULTS: dict[IntentBucket, tuple[tuple[Retrieval, ...], str, bool]] = {
    "exact_lookup":   (("bm25", "hybrid"), "single_shot", False),
    "conceptual":     (("vector", "hybrid"), "single_shot", False),
    "incident":       (("hybrid",), "single_shot", True),
    "code_reasoning": (("hybrid", "code"), "single_shot", True),
    "synthesis":      (("hybrid",), "single_shot", False),
    "high_risk":      (("hybrid",), "single_shot", True),
    "sensitive":      (("hybrid",), "single_shot", True),
    "unknown":        (("hybrid",), "single_shot", False),
}

# bucket → preferred model tiers, cheapest first (router still has final say)
BUCKET_TIERS: dict[IntentBucket, tuple[str, ...]] = {
    "exact_lookup":   ("local", "cheap"),
    "conceptual":     ("local", "cheap"),
    "incident":       ("cheap", "premium"),
    "code_reasoning": ("cheap", "premium"),
    "synthesis":      ("cheap", "premium"),
    "high_risk":      ("premium",),
    "sensitive":      ("local",),          # restricted data stays local
    "unknown":        ("local", "cheap"),
}

_ENTITY = re.compile(r"\b([A-Z]{2,}-\d+|[A-Za-z]+\d{2,}|\d{3,})\b")
_STOP = {"the", "a", "an", "why", "how", "what", "did", "is", "of", "to", "in", "for"}


def classify(text: str) -> tuple[IntentBucket, str]:
    for pat, bucket, risk in INTENT_RULES:
        if pat.search(text):
            return bucket, risk
    return "unknown", "low"


def extract_entities(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(m.group(1) for m in _ENTITY.finditer(text)))


def normalize(text: str) -> str:
    """Deterministic canonical form → Plan-Cache semantic key half (SPEC §2.2)."""
    toks = [t for t in re.findall(r"\w+", text.lower()) if t not in _STOP and len(t) > 1]
    return " ".join(sorted(set(toks)))
