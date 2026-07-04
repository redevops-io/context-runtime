"""Multimodal query classifier — a bandit *context* feature, not a new selector.

Once the corpus is mixed (text passages + charts + diagrams + scanned pages + video), the
right retrieval method depends on what the query is *asking for*: "what did revenue do after
Q2" wants a chart; "show the architecture" wants a diagram; "where does she mention the merger"
wants a video timestamp. The runtime already learns method-per-intent with a contextual bandit
keyed on ``plan.intent.bucket``; this adds a second, cheap axis to that key — the query's
**modality intent** — so the policy can learn, e.g., that ``chart`` queries in the ``lookup``
bucket are best served by the ``image`` arm while plain ``lookup`` queries are best served by
``hybrid``. Same seam as the load band already folded into the context: no new machinery, just a
richer key. If the tenant has no visual/video arms this simply never changes which arm wins, so
it is safe to leave on.

Rule-based and dependency-free (a keyword/regex prior); it is a *routing hint*, not a
ground-truth label — the bandit still measures reward and can override a mis-hint. A learned
classifier can drop in behind the same ``classify_query`` signature later.
"""
from __future__ import annotations

import re

# Modality-intent labels. Ordered by specificity: the first bucket whose cue fires wins, so
# a "chart" cue beats a generic "visual" cue. `text` is the default (no visual/temporal cue).
CHART = "chart"
DIAGRAM = "diagram"
TABLE = "table"
VISUAL = "visual"
TIMESTAMP = "timestamp"
TEXT = "text"

QUERY_TYPES = (CHART, DIAGRAM, TABLE, VISUAL, TIMESTAMP, TEXT)

# Word-boundary cue sets, most specific first. Kept small and high-precision on purpose — a
# noisy cue would pollute the bandit key and split learning across spurious contexts.
_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (CHART, ("chart", "graph", "plot", "trend", "trending", "axis", "y-axis", "x-axis",
             "bar chart", "line chart", "pie chart", "histogram", "curve", "slope",
             "went up", "went down", "grew", "declined", "over time")),
    (DIAGRAM, ("diagram", "architecture", "flowchart", "flow chart", "schematic", "topology",
               "block diagram", "pipeline diagram", "wiring", "sequence diagram", "org chart",
               "boxes and arrows", "how it connects", "system design")),
    (TABLE, ("table", "spreadsheet", "row", "column", "cell", "line item", "line-item",
             "balance sheet", "income statement", "cash flow statement")),
    (TIMESTAMP, ("video", "clip", "footage", "recording", "timestamp", "at minute",
                 "scene where", "frame where", "says at", "moment when", "part where he",
                 "part where she", "seconds in", "minute mark")),
    (VISUAL, ("image", "picture", "photo", "photograph", "screenshot", "logo", "icon",
              "figure", "illustration", "what does it look like", "looks like", "color of",
              "colour of", "shows a", "show me the", "visual")),
)

# Precompiled: match whole phrases (with word boundaries for single tokens).
_COMPILED: list[tuple[str, re.Pattern]] = []
for _label, _phrases in _CUES:
    _alts = "|".join(re.escape(p) for p in _phrases)
    _COMPILED.append((_label, re.compile(rf"(?<!\w)(?:{_alts})(?!\w)", re.IGNORECASE)))


def classify_query(text: str) -> str:
    """Return the modality-intent label for a query (one of ``QUERY_TYPES``).

    Deterministic, allocation-light, and safe on empty input. First matching cue set wins
    (specific → general); no cue ⇒ ``text``.
    """
    if not text or not text.strip():
        return TEXT
    for label, pat in _COMPILED:
        if pat.search(text):
            return label
    return TEXT


# Map a modality-intent to the retrieval method it *hints* at — used only for documentation /
# a cold-start prior surfaced in the panel; the bandit remains the actual decision-maker.
QTYPE_METHOD_HINT: dict[str, str] = {
    CHART: "image", DIAGRAM: "image", VISUAL: "image",
    TABLE: "colpali", TIMESTAMP: "video", TEXT: "hybrid",
}
