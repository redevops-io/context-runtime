"""Hermetic tests for the vuln-DB read client — no Doris, no network."""
from __future__ import annotations

from context_runtime.integrations.vuln_db import Principal, VulnDB

_ROWS = [
    {"cve_id": "CVE-A", "package": "requests", "source": "osv", "owner": "osv", "cvss": 9.1, "refs": "u1", "summary": "a"},
    {"cve_id": "CVE-B", "package": "jinja2", "source": "nvd", "owner": "nvd", "cvss": 7.0, "refs": "u2", "summary": "b"},
]


def _db(monkeypatch, rows=None):
    db = VulnDB(host="x")
    seen = {}
    monkeypatch.setattr(db, "_query", lambda sql, params=(): (seen.update(sql=sql, params=params), rows if rows is not None else _ROWS)[1])
    return db, seen


def test_lookup_builds_filtered_sql(monkeypatch):
    db, seen = _db(monkeypatch)
    db.lookup(package="requests", min_cvss=7.0, limit=5)
    assert "AND package=%s" in seen["sql"] and "AND cvss>=%s" in seen["sql"]
    assert seen["params"] == ("requests", 7.0, 5)


def test_policy_admin_sees_all_columns(monkeypatch):
    db, _ = _db(monkeypatch)
    out = db.lookup(principal=Principal(roles=frozenset({"admin"})))
    assert len(out) == 2 and all("refs" in r for r in out)


def test_policy_row_scope_and_column_mask(monkeypatch):
    db, _ = _db(monkeypatch)
    out = db.lookup(principal=Principal(roles=frozenset({"analyst"}), owns_rows_of=frozenset({"osv"})))
    assert [r["cve_id"] for r in out] == ["CVE-A"]        # only the owned source
    assert all("refs" not in r for r in out)             # sensitive column masked


def test_policy_no_scope_sees_nothing(monkeypatch):
    db, _ = _db(monkeypatch)
    assert db.lookup(principal=Principal(roles=frozenset({"guest"}))) == []


def test_enrich_groups_by_cve_and_filters(monkeypatch):
    db, _ = _db(monkeypatch)
    findings = [{"id": "CVE-A"}, {"id": "CVE-B"}, {"id": "CVE-Z"}]
    got = db.enrich(findings, principal=Principal(roles=frozenset({"security"})))
    assert set(got) == {"CVE-A", "CVE-B"} and got["CVE-A"][0]["package"] == "requests"
