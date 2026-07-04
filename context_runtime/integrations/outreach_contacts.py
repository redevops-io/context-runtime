# SPDX-License-Identifier: AGPL-3.0-or-later
"""Contact discovery — turn an ``AccountSignal`` (a *company* + why-now) into a **person to reach**.

The signal connectors ([[signals.py]]) find *which company* and *why now*; they do NOT find the
individual or their email. This module closes that last mile — the "who do I actually email, and
what's their address" step — with two complementary strategies, in ascending cost:

  1. **GitHub maintainers (free).** For ``tech_pain`` accounts sourced from a public RAG repo, the
     repo's top contributors ARE the buyers — and GitHub exposes their public profile email / commit
     author email through the API. No paid key. Best coverage for the open-source / builder ICP.
  2. **Enrichment waterfall (key-gated).** For any corporate account (funding/leadership/cold), a
     pluggable waterfall of providers (Hunter.io → Apollo → Clay) resolves name+domain → a verified
     work email. Each provider is skipped unless its API key is set, so the waterfall degrades
     gracefully to "no email found" with zero config and lights up as keys are added.

Everything is offline-testable: connectors take an injectable ``fetch`` and the waterfall takes an
injectable list of provider callables. LinkedIn is deliberately NOT automated (ToS-risky) — its
contacts stay a manual, human-in-the-loop step.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .signals import AccountSignal, _default_fetch


@dataclass(frozen=True)
class Contact:
    """A person to reach at an account, with as much as we could resolve."""
    account: str
    name: str
    email: str | None = None
    title: str | None = None
    source: str = ""                 # "github" | "hunter" | "apollo" | "clay" | "manual"
    confidence: float = 0.0          # 0..1 — email verified/high vs guessed/absent
    profile_url: str = ""            # GitHub/LinkedIn/company profile
    meta: dict = field(default_factory=dict)

    @property
    def reachable(self) -> bool:
        return bool(self.email)


# ──────────────────────────── strategy 1: GitHub maintainers (free) ────────────────────────────

def _gh_headers(token: str | None) -> dict:
    h = {"Accept": "application/vnd.github+json", "User-Agent": "context-runtime-contacts"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _repo_full_name(signal: AccountSignal) -> str | None:
    """Pull owner/repo out of a GitHub artifact URL like https://github.com/acme/support-rag."""
    art = signal.artifact or signal.url or ""
    if "github.com/" not in art:
        return None
    path = art.split("github.com/", 1)[1].strip("/")
    parts = path.split("/")
    return f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else None


def github_maintainers(repo_full_name: str, *, limit: int = 5, fetch: Callable = _default_fetch,
                       token: str | None = None) -> list[Contact]:
    """Top contributors of a repo → Contacts, resolving each one's public profile email.

    GitHub exposes a user's public email on ``GET /users/{login}`` when they've made it public;
    contributors who haven't get a Contact with no email (still useful — name + profile for a manual
    or waterfall follow-up). Contributors are ranked by contribution count (the maintainers first).
    """
    owner = repo_full_name.split("/", 1)[0]
    token = token or os.getenv("GITHUB_TOKEN")
    headers = _gh_headers(token)
    url = (f"https://api.github.com/repos/{repo_full_name}/contributors"
           f"?per_page={max(1, min(limit, 30))}")
    try:
        contributors = fetch(url, headers)
    except Exception:
        return []
    if not isinstance(contributors, list):
        return []
    out: list[Contact] = []
    for c in contributors[:limit]:
        login = c.get("login")
        if not login or c.get("type") == "Bot":
            continue
        name, email, company, title = login, None, None, None
        try:
            prof = fetch(f"https://api.github.com/users/{login}", headers)
            name = prof.get("name") or login
            email = prof.get("email") or None          # public profile email (opt-in by the user)
            company = prof.get("company") or None
            title = prof.get("bio") or None
        except Exception:
            prof = {}
        out.append(Contact(
            account=owner, name=name, email=email, title=title, source="github",
            confidence=0.85 if email else 0.3,
            profile_url=c.get("html_url") or f"https://github.com/{login}",
            meta={"contributions": c.get("contributions", 0), "login": login, "company": company}))
    return out


# ──────────────────────────── strategy 2: enrichment waterfall (key-gated) ────────────────────────────

# A provider is (account, name, domain) -> Contact | None. It returns None when it has no key or no
# hit, so the waterfall just moves to the next provider. Signatures are uniform for easy injection.
Provider = Callable[[str, str, str], "Contact | None"]


def _post_json(url: str, payload: dict | None, headers: dict, timeout: float = 15.0):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310
        return json.loads(r.read().decode("utf-8"))


def hunter_provider(fetch_get: Callable = _default_fetch) -> Provider:
    """Hunter.io email-finder — needs HUNTER_API_KEY. name+domain → verified work email."""
    def _p(account: str, name: str, domain: str) -> Contact | None:
        key = os.getenv("HUNTER_API_KEY")
        if not key or not domain:
            return None
        parts = (name or "").split()
        q = {"domain": domain, "api_key": key}
        if len(parts) >= 2:
            q["first_name"], q["last_name"] = parts[0], parts[-1]
        url = "https://api.hunter.io/v2/email-finder?" + urllib.parse.urlencode(q)
        try:
            d = (fetch_get(url) or {}).get("data") or {}
        except Exception:
            return None
        if not d.get("email"):
            return None
        return Contact(account=account, name=name or d.get("email"), email=d["email"],
                       title=(d.get("position") or None), source="hunter",
                       confidence=min(1.0, (d.get("score") or 80) / 100),
                       profile_url=d.get("linkedin_url") or "", meta={"domain": domain})
    return _p


def apollo_provider(post: Callable = _post_json) -> Provider:
    """Apollo.io people-match — needs APOLLO_API_KEY."""
    def _p(account: str, name: str, domain: str) -> Contact | None:
        key = os.getenv("APOLLO_API_KEY")
        if not key or not domain:
            return None
        parts = (name or "").split()
        body = {"domain": domain, "reveal_personal_emails": False}
        if len(parts) >= 2:
            body["first_name"], body["last_name"] = parts[0], parts[-1]
        try:
            d = post("https://api.apollo.io/v1/people/match",
                     body, {"Content-Type": "application/json", "X-Api-Key": key}) or {}
        except Exception:
            return None
        person = d.get("person") or {}
        if not person.get("email"):
            return None
        return Contact(account=account, name=person.get("name") or name, email=person["email"],
                       title=person.get("title"), source="apollo", confidence=0.8,
                       profile_url=person.get("linkedin_url") or "", meta={"domain": domain})
    return _p


def clay_provider(post: Callable = _post_json) -> Provider:
    """Clay waterfall webhook — needs CLAY_WEBHOOK_URL (Clay runs its own multi-provider waterfall)."""
    def _p(account: str, name: str, domain: str) -> Contact | None:
        hook = os.getenv("CLAY_WEBHOOK_URL")
        if not hook or not domain:
            return None
        try:
            d = post(hook, {"account": account, "name": name, "domain": domain},
                     {"Content-Type": "application/json"}) or {}
        except Exception:
            return None
        if not d.get("email"):
            return None
        return Contact(account=account, name=d.get("name") or name, email=d["email"],
                       title=d.get("title"), source="clay", confidence=float(d.get("confidence", 0.75)),
                       profile_url=d.get("linkedin_url") or "", meta={"domain": domain})
    return _p


def default_waterfall() -> list[Provider]:
    """Hunter → Apollo → Clay, in ascending order of cost/complexity. Each self-skips without a key."""
    return [hunter_provider(), apollo_provider(), clay_provider()]


def enrich_email(account: str, name: str, domain: str, *,
                 providers: Iterable[Provider] | None = None) -> Contact | None:
    """Run the waterfall: first provider to return a Contact wins. None if all miss/unconfigured."""
    for provider in (providers if providers is not None else default_waterfall()):
        try:
            hit = provider(account, name, domain)
        except Exception:
            hit = None
        if hit and hit.email:
            return hit
    return None


# ──────────────────────────── orchestration: signal → contacts ────────────────────────────

def _domain_for(signal: AccountSignal) -> str | None:
    """Best-effort company domain for the waterfall (explicit meta wins; else a .com guess)."""
    dom = signal.meta.get("domain")
    if dom:
        return str(dom).replace("https://", "").replace("http://", "").strip("/")
    acct = (signal.account or "").strip().lower()
    # Only guess for plausible single-token company handles (not HN usernames / sentences).
    if acct and " " not in acct and "." not in acct and acct.isascii() and len(acct) <= 30:
        return f"{acct}.com"
    return None


def find_contacts(signal: AccountSignal, *, github_token: str | None = None,
                  providers: Iterable[Provider] | None = None,
                  fetch: Callable = _default_fetch, limit: int = 5) -> list[Contact]:
    """Discover reachable people for one account signal.

    Strategy by source: a GitHub-sourced signal (repo artifact) → free maintainer resolution; then,
    for the top maintainer without a public email, fall through to the waterfall on a guessed domain.
    A non-GitHub signal → straight to the waterfall (needs a provider key to yield an email).
    Returns Contacts best-first; callers filter on ``.reachable`` before sending.
    """
    contacts: list[Contact] = []
    repo = _repo_full_name(signal)
    if repo:
        contacts = github_maintainers(repo, limit=limit, fetch=fetch, token=github_token)
    domain = _domain_for(signal)
    # For any top contact still missing an email, try the waterfall to fill it in.
    if domain:
        filled: list[Contact] = []
        for c in contacts:
            if c.reachable:
                filled.append(c)
                continue
            hit = enrich_email(c.account, c.name, domain, providers=providers)
            filled.append(hit if hit else c)
        contacts = filled
        if not contacts:                       # no repo → pure waterfall on the account
            hit = enrich_email(signal.account, "", domain, providers=providers)
            if hit:
                contacts = [hit]
    contacts.sort(key=lambda c: (c.reachable, c.confidence, c.meta.get("contributions", 0)), reverse=True)
    return contacts
