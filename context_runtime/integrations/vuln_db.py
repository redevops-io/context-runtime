"""VulnDB — read-only client for the shared vulnerability database (Apache Doris).

Consumes the NVD/OSV/Snyk vuln DB that ``vuln_feeds`` populates (table ``vulns``). This is the
READ path for apps like Edge Sentinel: enrich scan findings and look up advisories/patched pins,
with the SAME row-scope + column-mask permissioning the enterprise ``DorisStore`` applies — a
principal only sees the sources it owns, and sensitive columns (``refs``) are masked for
non-privileged callers. The full policy engine lives in context-runtime-v3; the minimal
``Principal`` here mirrors it for the read side.

pymysql is an OPTIONAL, deferred import (like the store adapters); the client degrades to
``available() == False`` when it's missing or Doris is unreachable — callers stay functional.
Connection comes from the environment (Doris FE, exposed to the host via a NodePort):

    DORIS_MYSQL_HOST / DORIS_FE_HOST (127.0.0.1)   DORIS_QUERY_PORT (9030)
    DORIS_USER (root)   DORIS_PASSWORD ('')   DORIS_DATABASE (context_runtime)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

_SENSITIVE = ("refs",)   # columns masked for non-privileged principals


@dataclass(frozen=True)
class Principal:
    """Who's asking — mirrors the enterprise policy engine's Identity (read side)."""
    roles: frozenset = field(default_factory=frozenset)
    owns_rows_of: frozenset = field(default_factory=frozenset)   # sources/owners this principal may read

    @property
    def privileged(self) -> bool:
        return bool({"admin", "security"} & set(self.roles))


def _pymysql():
    try:
        import pymysql
        return pymysql
    except Exception as e:  # noqa: BLE001
        raise ImportError("VulnDB needs the optional `pymysql` dependency — `pip install pymysql`.") from e


class VulnDB:
    def __init__(self, *, host: str | None = None, port: int | None = None, user: str | None = None,
                 password: str | None = None, database: str | None = None, timeout: float = 8.0):
        self.host = host or os.getenv("DORIS_MYSQL_HOST") or os.getenv("DORIS_FE_HOST", "127.0.0.1")
        self.port = int(port or os.getenv("DORIS_QUERY_PORT", "9030"))
        self.user = user or os.getenv("DORIS_USER", "root")
        self.password = password if password is not None else os.getenv("DORIS_PASSWORD", "")
        self.database = database or os.getenv("DORIS_DATABASE", "context_runtime")
        self.timeout = timeout

    # ---- connection (deferred, degrades) -------------------------------------
    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        pymysql = _pymysql()
        conn = pymysql.connect(host=self.host, port=self.port, user=self.user, password=self.password,
                               database=self.database, connect_timeout=self.timeout,
                               cursorclass=pymysql.cursors.DictCursor)
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return list(cur.fetchall())
        finally:
            conn.close()

    def available(self) -> bool:
        try:
            self._query("SELECT 1 AS ok")
            return True
        except Exception:  # noqa: BLE001
            return False

    # ---- permissioning (same shape as the enterprise DorisStore) -------------
    @staticmethod
    def _apply_policy(rows: list[dict], principal: Principal | None) -> list[dict]:
        if principal is None or principal.privileged:
            return rows
        owns = principal.owns_rows_of
        out = []
        for r in rows:
            if not owns or not ({r.get("owner"), r.get("source")} & set(owns)):
                continue                                   # row scope: only owned sources
            out.append({k: v for k, v in r.items() if k not in _SENSITIVE})   # column mask
        return out

    # ---- reads ---------------------------------------------------------------
    def lookup(self, *, package: str | None = None, cve: str | None = None, min_cvss: float | None = None,
               principal: Principal | None = None, limit: int = 50) -> list[dict]:
        sql = "SELECT * FROM vulns WHERE 1=1"
        params: list = []
        if package:
            sql += " AND package=%s"
            params.append(package)
        if cve:
            sql += " AND cve_id=%s"
            params.append(cve)
        if min_cvss is not None:
            sql += " AND cvss>=%s"
            params.append(min_cvss)
        sql += " ORDER BY cvss DESC LIMIT %s"
        params.append(int(limit))
        return self._apply_policy(self._query(sql, tuple(params)), principal)

    def count(self, principal: Principal | None = None) -> int:
        rows = self.lookup(principal=principal, limit=100000)
        return len(rows)

    def enrich(self, findings, principal: Principal | None = None) -> dict:
        """Given scan findings (objects/dicts with cve_id or id), return which are corroborated by our
        vuln DB: {cve_id: [our records]}. Batched by the CVE ids present, then policy-filtered."""
        ids = []
        for f in findings or []:
            cid = getattr(f, "id", None) or (f.get("id") if isinstance(f, dict) else None) or (f.get("cve_id") if isinstance(f, dict) else None)
            if cid:
                ids.append(cid)
        ids = list(dict.fromkeys(ids))[:200]
        if not ids:
            return {}
        placeholders = ",".join(["%s"] * len(ids))
        try:
            rows = self._query(f"SELECT * FROM vulns WHERE cve_id IN ({placeholders})", tuple(ids))
        except Exception:  # noqa: BLE001
            return {}
        rows = self._apply_policy(rows, principal)
        out: dict = {}
        for r in rows:
            out.setdefault(r["cve_id"], []).append(r)
        return out
