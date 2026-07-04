# SPDX-License-Identifier: AGPL-3.0-or-later
"""Signal connectors — the account-intel feed for market-radar + outreach-engine.

An outbound motion is only as good as its signals: the research is unambiguous that *signal-
triggered* outreach (a company hiring for RAG, discussing retrieval pain, or just funded) converts
5–10× better than a static list. This module pulls those signals from **free, real sources** and
normalizes them to one shape the outreach tenant buckets on (``tech_pain`` / ``hiring`` /
``funding`` / ``leadership``), attaching a **public RAG artifact** when it can — the thing the
EXPLAIN teardown personalizes against.

Sources shipped (no paid API, no key required for the basics):
  • **GitHub** — orgs/repos building RAG (LangChain/LlamaIndex/vector usage) → ``tech_pain`` + the
    repo as the teardown artifact. (Optional ``GITHUB_TOKEN`` raises the rate limit.)
  • **Hacker News** (Algolia API) — RAG/retrieval pain discussions → ``tech_pain`` evidence.
  • **HN "Who is hiring"** comments mentioning ML/RAG roles → ``hiring``.
  • **manual** — funding/leadership (Crunchbase/PitchBook are paid) load from a YAML/JSON you keep.

Every connector takes an injectable ``fetch`` so it is unit-tested offline; the default fetcher is
stdlib urllib (no new deps). Live calls are opt-in by the caller.
"""
from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

# the buckets the outreach tenant routes on (mirrors outreach_engine.outreach_bucket)
BUCKETS = ("tech_pain", "hiring", "funding", "leadership", "cold")


@dataclass(frozen=True)
class AccountSignal:
    account: str                       # org / company / handle
    signal: str                        # one of BUCKETS
    source: str                        # "github" | "hn" | "hn_hiring" | "manual"
    evidence: str                      # the human-readable why
    url: str = ""                      # link to the evidence
    artifact: str | None = None        # a public RAG artifact (repo/doc) for the EXPLAIN teardown
    meta: dict = field(default_factory=dict)

    def as_query(self) -> str:
        """The text the outreach tenant buckets on (account + signal + evidence)."""
        return f"{self.account} — {self.evidence}"


def _default_fetch(url: str, headers: dict | None = None, timeout: float = 15.0):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (trusted GitHub/HN hosts)
        return json.loads(r.read().decode("utf-8"))


# ──────────────────────────── GitHub: orgs building RAG (tech_pain + artifact) ────────────────────────────

_GH_DEFAULT_Q = "rag retrieval langchain OR llamaindex OR pgvector in:readme"


def github_signals(query: str = _GH_DEFAULT_Q, *, limit: int = 10, fetch=_default_fetch,
                   token: str | None = None) -> list[AccountSignal]:
    """Repos/orgs building RAG → tech_pain signals; the repo is the teardown artifact."""
    token = token or os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "context-runtime-signals"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = ("https://api.github.com/search/repositories?sort=updated&order=desc&per_page="
           + str(max(1, min(limit, 50))) + "&q=" + urllib.parse.quote(query))
    try:
        data = fetch(url, headers)
    except Exception:
        return []
    out: list[AccountSignal] = []
    for repo in (data.get("items") or [])[:limit]:
        owner = (repo.get("owner") or {})
        acct = owner.get("login") or repo.get("full_name", "").split("/")[0]
        if not acct:
            continue
        out.append(AccountSignal(
            account=acct, signal="tech_pain", source="github",
            evidence=f"builds RAG in {repo.get('full_name')} "
                     f"({repo.get('stargazers_count', 0)}★): {(repo.get('description') or '')[:120]}",
            url=repo.get("html_url", ""), artifact=repo.get("html_url"),
            meta={"stars": repo.get("stargazers_count", 0), "org": owner.get("type") == "Organization"}))
    return out


# ──────────────────────────── Hacker News (Algolia): RAG pain + hiring ────────────────────────────

def _hn_search(query: str, tags: str, limit: int, fetch) -> list[dict]:
    url = ("https://hn.algolia.com/api/v1/search_by_date?tags=" + tags
           + "&hitsPerPage=" + str(max(1, min(limit, 50))) + "&query=" + urllib.parse.quote(query))
    try:
        return (fetch(url, {"User-Agent": "context-runtime-signals"}).get("hits") or [])[:limit]
    except Exception:
        return []


def hn_signals(query: str = "RAG retrieval hallucination reranking", *, limit: int = 10,
               fetch=_default_fetch) -> list[AccountSignal]:
    """HN comments discussing RAG/retrieval pain → tech_pain evidence (surface who's in the weeds)."""
    out = []
    for h in _hn_search(query, "comment", limit, fetch):
        text = re.sub(r"<[^>]+>", " ", h.get("comment_text") or "")[:160].strip()
        if not text:
            continue
        out.append(AccountSignal(
            account=h.get("author") or "hn-user", signal="tech_pain", source="hn",
            evidence=f"discussing RAG pain: “{text}”",
            url=f"https://news.ycombinator.com/item?id={h.get('objectID')}"))
    return out


_HIRING_RE = re.compile(r"\b(ml|machine learning|rag|retrieval|llm|ai infra|ai infrastructure|ai platform)\b", re.I)


def hn_hiring_signals(query: str = "hiring machine learning RAG LLM", *, limit: int = 10,
                      fetch=_default_fetch) -> list[AccountSignal]:
    """HN 'Who is hiring' comments mentioning ML/RAG roles → hiring signals."""
    out = []
    for h in _hn_search(query, "comment", limit, fetch):
        text = re.sub(r"<[^>]+>", " ", h.get("comment_text") or "")
        if not _HIRING_RE.search(text):
            continue
        # company name is usually the first token(s) of a who-is-hiring post
        acct = (text.strip().split("|")[0].split("(")[0].strip()[:48]) or (h.get("author") or "unknown")
        out.append(AccountSignal(
            account=acct, signal="hiring", source="hn_hiring",
            evidence=f"hiring for ML/RAG: “{text[:140].strip()}”",
            url=f"https://news.ycombinator.com/item?id={h.get('objectID')}"))
    return out


# ──────────────────────────── manual (funding/leadership — paid-API replacements) ────────────────────────────

def manual_signals(rows: list[dict]) -> list[AccountSignal]:
    """Load funding/leadership signals you maintain by hand (or from a paid provider) — one dict per
    row: {account, signal, evidence, url?, artifact?}. Keeps the pipeline honest where free data ends."""
    out = []
    for r in rows:
        sig = r.get("signal", "cold")
        out.append(AccountSignal(
            account=r.get("account", "unknown"), signal=sig if sig in BUCKETS else "cold",
            source="manual", evidence=r.get("evidence", ""), url=r.get("url", ""),
            artifact=r.get("artifact")))
    return out


# ──────────────────────────── aggregate + dedup ────────────────────────────

def collect_signals(*groups: list[AccountSignal]) -> list[AccountSignal]:
    """Merge connector outputs and dedup by (account, signal); prefer a row that has an artifact."""
    best: dict[tuple[str, str], AccountSignal] = {}
    for group in groups:
        for s in group:
            key = (s.account.lower(), s.signal)
            cur = best.get(key)
            if cur is None or (s.artifact and not cur.artifact):
                best[key] = s
    return list(best.values())
