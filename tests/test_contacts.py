"""Contact discovery — GitHub maintainer resolution + enrichment waterfall (offline, injected)."""
from __future__ import annotations

from context_runtime.integrations.outreach_contacts import (
    Contact, enrich_email, find_contacts, github_maintainers,
)
from context_runtime.integrations.signals import AccountSignal


def _gh_fetch(contributors, profiles):
    """Fake GitHub API: /contributors returns a list, /users/{login} returns a profile dict."""
    def fetch(url, headers=None):
        if "/contributors" in url:
            return contributors
        login = url.rsplit("/", 1)[-1]
        return profiles.get(login, {})
    return fetch


def test_github_maintainers_resolves_public_email_and_confidence():
    contributors = [
        {"login": "jane", "type": "User", "contributions": 120, "html_url": "https://github.com/jane"},
        {"login": "cibot", "type": "Bot", "contributions": 99},
        {"login": "sam", "type": "User", "contributions": 30, "html_url": "https://github.com/sam"},
    ]
    profiles = {
        "jane": {"name": "Jane Dev", "email": "jane@acme.io", "company": "Acme", "bio": "RAG lead"},
        "sam": {"name": "Sam ML", "email": None},
    }
    cs = github_maintainers("acme/support-rag", fetch=_gh_fetch(contributors, profiles))
    assert [c.name for c in cs] == ["Jane Dev", "Sam ML"]      # bot dropped, order preserved
    jane, sam = cs
    assert jane.reachable and jane.email == "jane@acme.io" and jane.confidence > 0.8
    assert not sam.reachable and sam.confidence < 0.5          # no public email → low confidence


def test_waterfall_first_hit_wins_and_skips_unconfigured():
    calls = []

    def miss(account, name, domain):
        calls.append("miss"); return None

    def hit(account, name, domain):
        calls.append("hit")
        return Contact(account=account, name=name, email=f"{name.lower()}@{domain}",
                       source="apollo", confidence=0.8)

    def never(account, name, domain):
        calls.append("never"); return None

    c = enrich_email("acme", "Jane", "acme.com", providers=[miss, hit, never])
    assert c and c.email == "jane@acme.com" and c.source == "apollo"
    assert calls == ["miss", "hit"]                            # stops at first hit; never() not called


def test_waterfall_all_miss_returns_none():
    assert enrich_email("acme", "Jane", "acme.com", providers=[lambda *a: None]) is None


def test_find_contacts_github_then_waterfall_fills_missing_email():
    contributors = [
        {"login": "jane", "type": "User", "contributions": 120, "html_url": "https://github.com/jane"},
        {"login": "sam", "type": "User", "contributions": 30, "html_url": "https://github.com/sam"},
    ]
    profiles = {"jane": {"name": "Jane Dev", "email": "jane@acme.io"}, "sam": {"name": "Sam ML", "email": None}}
    sig = AccountSignal("acme", "tech_pain", "github", "builds RAG",
                        artifact="https://github.com/acme/support-rag", meta={"domain": "acme.io"})

    def waterfall(account, name, domain):     # fills Sam's missing email
        return Contact(account=account, name=name, email="sam@acme.io", source="hunter", confidence=0.7)

    cs = find_contacts(sig, fetch=_gh_fetch(contributors, profiles), providers=[waterfall])
    emails = {c.name: c.email for c in cs}
    assert emails["Jane Dev"] == "jane@acme.io"               # kept the free public email
    assert emails["Sam ML"] == "sam@acme.io"                  # waterfall backfilled the gap
    assert all(c.reachable for c in cs)


def test_find_contacts_non_github_goes_straight_to_waterfall():
    sig = AccountSignal("nimbus", "funding", "manual", "just raised a Series A", meta={"domain": "nimbus.ai"})

    def waterfall(account, name, domain):
        return Contact(account=account, name="VP Eng", email=f"vp@{domain}", title="VP Eng",
                       source="apollo", confidence=0.8)

    cs = find_contacts(sig, providers=[waterfall])
    assert len(cs) == 1 and cs[0].email == "vp@nimbus.ai" and cs[0].title == "VP Eng"


def test_domain_guess_skips_usernames_and_sentences():
    # an HN-username or free-text account should NOT produce a bogus domain → no waterfall email
    sig = AccountSignal("some hn user", "tech_pain", "hn", "griping about reranking")
    assert find_contacts(sig, providers=[lambda *a: Contact("x", "y", email="z@z.com")]) == []


# ── network-failure / malformed-response degradation added for the v2 release audit ──

def test_github_maintainers_ratelimited_dict_returns_empty():
    ratelimited = {"message": "API rate limit exceeded"}      # GitHub returns a DICT, not a list
    assert github_maintainers("acme/x", fetch=lambda u, h=None: ratelimited) == []


def test_github_maintainers_fetch_raises_degrades_to_empty():
    def boom(u, h=None):
        raise RuntimeError("network down")
    assert github_maintainers("acme/x", fetch=boom) == []


def test_maintainer_profile_fetch_raises_keeps_login_no_email():
    contributors = [{"login": "jane", "type": "User", "contributions": 5, "html_url": "https://github.com/jane"}]

    def fetch(u, h=None):
        if "/contributors" in u:
            return contributors
        raise RuntimeError("profile 500")                     # /users/{login} fails per-user
    cs = github_maintainers("acme/x", fetch=fetch)
    assert len(cs) == 1 and cs[0].name == "jane" and cs[0].email is None and cs[0].confidence == 0.3


def test_waterfall_provider_that_raises_is_skipped():
    def boom(a, n, d):
        raise RuntimeError("hunter 500")

    def hit(a, n, d):
        return Contact(account=a, name=n, email=f"{n}@{d}", source="apollo", confidence=0.8)
    c = enrich_email("acme", "Jane", "acme.com", providers=[boom, hit])
    assert c and c.email == "Jane@acme.com" and c.source == "apollo"
