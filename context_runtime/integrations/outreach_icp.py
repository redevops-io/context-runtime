# SPDX-License-Identifier: AGPL-3.0-or-later
"""ICP scoring + the EXPLAIN-teardown message templates for outreach-engine.

Who to sell the *pilot* to, how to rank the signals the connectors produce, and the actual copy to
send. The wedge (from the research): technical buyers convert on **proof of effort** — so the
opener leads with a real finding from running our own **EXPLAIN / redevops-rag** over the
prospect's public RAG artifact, not a mail-merge.
"""
from __future__ import annotations

from .signals import AccountSignal

# ──────────────────────────── ICP ────────────────────────────

ICP_TITLES = (
    "Head of AI Platform", "Head of ML Platform", "ML Infrastructure Lead", "AI Engineering Lead",
    "Staff/Principal AI Engineer", "Head of Data/Platform", "CTO (AI-native, <200)",
)
ICP_COMPANY = ("Building or operating **production RAG / agents** with visible context pain "
               "(hybrid search, reranking, eval, cost/latency) — ideally recently funded.")

# how much each signal predicts a pilot (research: tech-pain + funding + hiring are the strong ones)
SIGNAL_PRIORITY = {"tech_pain": 1.0, "funding": 0.9, "hiring": 0.8, "leadership": 0.7, "cold": 0.15}
# how trustworthy the source is for that signal
SOURCE_WEIGHT = {"github": 1.0, "hn_hiring": 0.9, "manual": 0.85, "hn": 0.6}


def score_signal(s: AccountSignal) -> float:
    """Rank an account signal for outreach: ICP fit (signal × source) + artifact bonus + traction."""
    base = SIGNAL_PRIORITY.get(s.signal, 0.15) * SOURCE_WEIGHT.get(s.source, 0.5)
    if s.artifact:                       # a public RAG artifact to tear down is the whole wedge
        base += 0.25
    base += min(0.20, s.meta.get("stars", 0) / 5000.0)
    return round(min(1.0, base), 3)


def rank_accounts(signals: list[AccountSignal], top: int = 25) -> list[tuple[AccountSignal, float]]:
    ranked = sorted(((s, score_signal(s)) for s in signals), key=lambda x: -x[1])
    return ranked[:top]


# ──────────────────────────── message templates (the teardown) ────────────────────────────

_SIGNAL_HOOK = {
    "tech_pain": "saw {account} is deep in production RAG",
    "funding": "congrats on the raise — the post-round build window is the right time for this",
    "hiring": "saw {account} is hiring for ML/RAG",
    "leadership": "saw you just took over AI at {account}",
    "cold": "noticed {account} is building on LLMs",
}


def teardown_email(s: AccountSignal, explain_finding: str = "", planner_url: str = "https://redevops.io/planner") -> dict:
    """The proof-of-effort email: signal hook + a real EXPLAIN finding on their artifact + pilot CTA."""
    hook = _SIGNAL_HOOK.get(s.signal, _SIGNAL_HOOK["cold"]).format(account=s.account)
    finding = explain_finding or (
        f"I ran our EXPLAIN over {('your repo ' + s.artifact) if s.artifact else 'a public RAG path of yours'} — "
        "retrieval looks like it leans on a single method; a calibrated, quality-routed plan would "
        "change what actually gets served, and you'd see exactly why.")
    subject = f"{s.account}: a 3-min EXPLAIN of your RAG retrieval"
    body = (
        f"Hi — {hook}.\n\n"
        f"{finding}\n\n"
        f"Context Runtime is an open-source, self-hostable query planner for LLM context — it decides "
        f"what the model sees, learns which retrieval strategy wins per query, and shows its work "
        f"(EXPLAIN: {planner_url}). We benchmark v1→v2 in Python and Go.\n\n"
        f"Worth a 20-min pilot scoping call? I'll bring the full EXPLAIN of your stack.\n")
    return {"channel": "email", "subject": subject, "body": body, "account": s.account,
            "artifact": s.artifact, "cta": "pilot scoping call"}


def teardown_linkedin(s: AccountSignal, explain_finding: str = "") -> dict:
    """A shorter LinkedIn opener (send from the founder profile for tier-1 accounts)."""
    hook = _SIGNAL_HOOK.get(s.signal, _SIGNAL_HOOK["cold"]).format(account=s.account)
    msg = (f"{hook.capitalize()}. I ran our open-source EXPLAIN over "
           f"{'your repo' if s.artifact else 'a public RAG path of yours'} and found something worth 3 minutes — "
           f"a calibrated, quality-routed retrieval plan changes what gets served. Open to a quick pilot chat?")
    return {"channel": "linkedin", "body": msg[:600], "account": s.account, "artifact": s.artifact}
