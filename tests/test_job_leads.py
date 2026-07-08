"""Job-lead outreach: internal-vs-consulting classification, tailored pitch, dedup, and lead-finding."""
from __future__ import annotations

from context_runtime.integrations.job_leads import (
    DEFAULT_TEMPLATE, JobListing, OutreachLedger, PitchTemplate, StaticJobSource, classify,
    find_leads, write_pitch,
)


def _l(company, title, desc="", url="", **kw):
    return JobListing(company=company, title=title, description=desc, url=url, **kw)


def test_classify_internal_ai_role_is_a_lead():
    c = classify(_l("Acme Retail", "AI Data Engineer", "Build our in-house RAG platform for production."))
    assert c.is_lead and c.kind == "internal" and c.score >= 0.9   # explicit in-house signals


def test_classify_consultancy_is_skipped():
    c = classify(_l("BigConsulting", "AI Engineer", "Deliver forward-deployed AI for our clients."))
    assert not c.is_lead and c.kind == "consulting"


def test_classify_non_ai_role_is_other():
    assert classify(_l("Acme", "Frontend Developer", "React and CSS.")).kind == "other"


def test_write_pitch_substitutes_and_flags_resume():
    listing = _l("Acme", "AI Developer", "You will build RAG. Attach your CV/resume.")
    p = write_pitch(listing, DEFAULT_TEMPLATE, resume_path="/docs/resume.pdf")
    assert "your AI Developer opening" in p["body"]
    assert p["attach"] == "/docs/resume.pdf"          # 'resume' in the description → attach
    assert p["subject"].startswith("Re: your AI Developer opening")
    # a listing that doesn't ask for a resume → no attachment
    assert write_pitch(_l("Acme", "AI Engineer", "Build stuff."), DEFAULT_TEMPLATE,
                       resume_path="/docs/resume.pdf")["attach"] == ""


def test_ledger_dedupes_reappearing_listings(tmp_path):
    ledger = OutreachLedger(path=str(tmp_path / "led.jsonl"))
    listing = _l("Acme", "AI Data Engineer", url="https://acme/jobs/1")
    assert not ledger.already_pitched(listing)
    ledger.record(listing)
    assert ledger.already_pitched(listing)                         # same key won't be pitched again
    # a fresh ledger loading the same file still sees it (persistent)
    assert OutreachLedger(path=str(tmp_path / "led.jsonl")).already_pitched(listing)


def test_find_leads_filters_dedupes_and_ranks(tmp_path):
    source = StaticJobSource([
        _l("Acme Retail", "AI Data Engineer", "Build our in-house AI platform.", url="u1"),   # lead (high)
        _l("Startup", "AI Engineer", "Ship AI features.", url="u2"),                            # lead (base)
        _l("BigConsulting", "AI Engineer", "Forward-deployed AI for clients.", url="u3"),       # skip
        _l("Shop", "Store Manager", "Retail ops.", url="u4"),                                   # skip
    ])
    ledger = OutreachLedger(path=str(tmp_path / "led.jsonl"))
    leads = find_leads(source, ledger, queries=("x",))
    companies = [d["listing"].company for d in leads]
    assert companies == ["Acme Retail", "Startup"]                 # consultancy + non-AI dropped, best first
    # after pitching Acme, it's excluded next time
    ledger.record(leads[0]["listing"])
    assert [d["listing"].company for d in find_leads(source, ledger, queries=("x",))] == ["Startup"]


def test_adzuna_source_parses_and_degrades():
    from context_runtime.integrations.job_leads import AdzunaSource

    assert AdzunaSource(app_id="", app_key="").search("ai") == []      # unconfigured → empty, no crash

    class _Resp:
        def json(self):
            return {"results": [{"title": "AI Data Engineer", "company": {"display_name": "Acme"},
                                 "redirect_url": "https://x/1", "description": "Build our in-house AI platform.",
                                 "location": {"display_name": "NYC"}, "created": "2026-07-01"}]}

    class _Client:
        def get(self, url, params=None):
            return _Resp()

    got = AdzunaSource(app_id="a", app_key="k", client=_Client()).search("AI Data Engineer", 10)
    assert len(got) == 1 and got[0].company == "Acme" and got[0].source == "adzuna"
    assert classify(got[0]).is_lead


def test_profile_store_owner_vs_generic_and_isolated(tmp_path):
    from context_runtime.integrations.job_leads import DEFAULT_TEMPLATE, GENERIC_TEMPLATE, ProfileStore

    store = ProfileStore(dir=str(tmp_path), owner="alex")
    owner = store.load("alex")
    assert "AI Data Engineer" in owner.include and owner.template == DEFAULT_TEMPLATE   # seeded owner config
    other = store.load("bob")
    assert other.include == [] and other.template == GENERIC_TEMPLATE                    # blank for others

    other.include = ["DevOps Engineer"]
    other.template = "Hi {company}, about {title}. {match}"
    store.save(other)
    assert store.load("bob").include == ["DevOps Engineer"]                              # persisted per user
    assert store.load("alex").include != store.load("bob").include                       # isolated


def test_classify_for_uses_profile_targets():
    from context_runtime.integrations.job_leads import LeadProfile, classify_for

    bob = LeadProfile("bob", include=["DevOps"], exclude=["consult"])
    assert classify_for(_l("Acme", "Senior DevOps Engineer", "Run our k8s."), bob).is_lead
    assert not classify_for(_l("Acme", "AI Data Engineer", "Build AI."), bob).is_lead    # not bob's target
    assert classify_for(_l("BigCo", "DevOps consultant", "for our clients"), bob).kind == "excluded"
    assert classify_for(_l("x", "AI Engineer", ""), LeadProfile("new")).kind == "unset"  # no targets set


def test_find_leads_for_scopes_to_profile(tmp_path):
    from context_runtime.integrations.job_leads import LeadProfile, OutreachLedger, StaticJobSource, find_leads_for

    src = StaticJobSource([_l("A", "DevOps Engineer", "run our infra", url="u1"),
                           _l("B", "AI Data Engineer", "build our AI", url="u2")])
    leads = find_leads_for(src, OutreachLedger(path=str(tmp_path / "l.jsonl")),
                           LeadProfile("bob", include=["DevOps"]), queries=("x",))
    assert [d["listing"].company for d in leads] == ["A"]   # only bob's targeted role


def test_pitch_template_is_shared_and_editable(tmp_path):
    tpl = PitchTemplate(path=str(tmp_path / "tpl.txt"))
    assert tpl.load() == DEFAULT_TEMPLATE                           # default until edited
    tpl.save("Hi, {company} — about your {title} role. {match}")
    assert PitchTemplate(path=str(tmp_path / "tpl.txt")).load().startswith("Hi, {company}")   # all users see it
    tpl.reset()
    assert tpl.load() == DEFAULT_TEMPLATE
