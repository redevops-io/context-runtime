"""Email-ingest collector (source #6) — bid-match notification emails (e.g. BidNet Direct).

BidNet's sanctioned automated channel is the match-alert emails it sends (its robots.txt forbids scraping
the search UI). This collector reads those from an inbox, read-only and FROM-filtered to the alert sender,
so on a shared mailbox it can only ever touch the alert mail — never other correspondence. The BidNet
email PARSER is provisional until locked against a real sample; the plumbing here (MIME extraction, the
FROM-filter guardrail, IMAP credential handling) is backend-independent and verified.
"""
from __future__ import annotations

import json
from email.message import EmailMessage

from context_runtime.integrations.procurement_sources import (
    ImapFetcher, MboxFetcher, SourceMethod, _extract_email, collect, parse_bidnet_email, parse_email_alerts,
    parse_source, ProcurementSource,
)

_BIDNET_HTML = """<html><body>
<p>3 new bids match your saved search:</p>
<ul>
<li><a href="https://www.bidnetdirect.com/florida/solicitations/notice/1234567">
    Professional Engineering Services - Roadway Design</a> — Miami-Dade County</li>
<li><a href="https://www.bidnetdirect.com/florida/solicitations/notice/1234599">
    Structural Engineering Peer Review Services</a> — City of Hollywood</li>
</ul>
<p><a href="https://www.bidnetdirect.com/public/registration/">Manage your account</a></p>
</body></html>"""


def _write_eml(path, *, frm, subject, html):
    m = EmailMessage()
    m["From"] = frm
    m["To"] = "alex@joinredevops.com"
    m["Subject"] = subject
    m["Date"] = "Fri, 19 Sep 2026 08:15:00 -0400"
    m.set_content("plain-text fallback")
    m.add_alternative(html, subtype="html")
    path.write_bytes(bytes(m))


def test_extract_email_pulls_html_and_headers():
    m = EmailMessage()
    m["From"] = "BidNet Direct <noreply@bidnetdirect.com>"
    m["Subject"] = "Your bid matches"
    m.set_content("text part")
    m.add_alternative("<p>html part</p>", subtype="html")
    d = _extract_email(m)
    assert d["from"].endswith("bidnetdirect.com>") and d["subject"] == "Your bid matches"
    assert "html part" in d["html"] and "text part" in d["text"]


def test_parse_bidnet_email_extracts_solicitation_links_only():
    rows = parse_bidnet_email({"from": "noreply@bidnetdirect.com", "html": _BIDNET_HTML})
    # two solicitation links; the /public/registration/ account link is NOT a solicitation → skipped
    assert len(rows) == 2
    assert rows[0]["event_id"] == "1234567" and rows[0]["title"].startswith("Professional Engineering")
    assert all(r["record_kind"] == "bidnet_email" and r["response_due_at"] == "" for r in rows)  # never inferred
    assert all("bidnetdirect.com" in r["link"] for r in rows)


def test_mbox_fetcher_from_filter_only_touches_alert_sender(tmp_path):
    _write_eml(tmp_path / "a.eml", frm="noreply@bidnetdirect.com", subject="matches", html=_BIDNET_HTML)
    _write_eml(tmp_path / "b.eml", frm="prospect@somecompany.com",   # an outreach reply — must be ignored
               subject="Re: your cold email", html="<p>secret reply content</p>")
    res = MboxFetcher(str(tmp_path), from_filter="bidnetdirect.com").fetch()
    assert res.status == 200
    recs = parse_email_alerts(res.text)
    assert len(recs) == 2                                             # only the BidNet email was parsed
    assert "secret reply content" not in res.text                    # the outreach mail was never included


def test_collect_via_email_source(tmp_path):
    _write_eml(tmp_path / "m.eml", frm="alerts@bidnetdirect.com", subject="matches", html=_BIDNET_HTML)
    src = ProcurementSource("src-bidnet-fl", "fl-statewide", "BidNet Direct — FL match alerts",
                            "imap://alex@joinredevops.com/INBOX?from=bidnetdirect.com", SourceMethod.EMAIL)
    obs = collect([src], MboxFetcher(str(tmp_path)))
    assert len(obs) == 2 and all(o.raw["record_kind"] == "bidnet_email" for o in obs)
    assert all(o.content_hash.startswith("sha256:") for o in obs)     # evidence preserved with hashes


def test_real_bidnet_sender_and_domain_are_recognized():
    # BidNet actually mails from noreply@bidnet.com with links that may be bidnet.com or bidnetdirect.com
    html = ('<a href="https://www.bidnet.com/florida/solicitations/notice/900123">'
            'Civil Engineering On-Call Services</a>')
    recs = parse_email_alerts(json.dumps([{"from": "BidNet Direct <noreply@bidnet.com>", "html": html}]))
    assert len(recs) == 1 and recs[0]["event_id"] == "900123"
    assert recs[0]["title"].startswith("Civil Engineering")


def test_imap_fetcher_without_creds_is_error_not_crash():
    f = ImapFetcher.from_env(prefix="NONEXISTENT_IMAP_PREFIX")
    assert not f.configured and f.host == "imap.gmail.com"
    r = f.fetch()                                                     # no network: returns before connecting
    assert r.status == 0 and "credentials not set" in r.content_type
