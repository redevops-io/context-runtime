"""Procurement source collectors — the pluggable layer that turns official government sources into
observations, then into qualified `revenue-handoff/v1` opportunities.

The plan's collector split: collection (fetch a portal / SAM.gov / an RSS feed) is separate from
Discovery's semantic contract. A `Fetcher` does the retrieval (real over HTTP with robots + rate-limit
respect, or a fixture in tests); a `Collector` parses one source's response into raw solicitations;
`collect` preserves every `Observation` (raw + content hash + timestamp = the evidence); and
`run_region` ties it to the geospatial jurisdiction resolution + business qualification, emitting the
handoff records the Mission Runtime consumes.

Nothing here fabricates data: without a live `Fetcher` (or with an empty response) it yields nothing.
Pointing a live fetcher at a specific portal is an operational choice — respect each source's terms.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Protocol
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

from .local_gov import (
    BusinessProfile, JurisdictionRegistry, RevenueOpportunity, SolicitationType, default_registry,
    qualify, to_handoff,
)


# ──────────────────────────── fetching (real or fixture) ────────────────────────────

@dataclass
class FetchResult:
    status: int
    text: str
    content_type: str = ""


class Fetcher(Protocol):
    def fetch(self, url: str, *, timeout: float = 20.0) -> FetchResult: ...


def _robots_disallows(txt: str, user_agent: str) -> list[str]:
    """The Disallow rules that apply to US, parsed by user-agent GROUP (RFC-style records).

    A robots.txt record is one or more ``User-agent`` lines followed by rules; rules apply only to the
    agents in their own record. The naive "collect every Disallow line" approach is wrong — it lets one
    bot's ``Disallow: /`` (e.g. Baiduspider/Yandex blanket blocks common on CivicPlus sites) block a
    compliant bot it was never addressed to. We match our product token, falling back to the ``*`` group."""
    token = user_agent.split("/", 1)[0].strip().lower()
    groups: dict[str, list[str]] = {}
    agents: list[str] = []
    seen_rule = False
    for raw in txt.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        field, val = (s.strip() for s in line.split(":", 1))
        field = field.lower()
        if field == "user-agent":
            if seen_rule:                     # a rule already closed the previous record → start a new one
                agents, seen_rule = [], False
            agents.append(val.lower())
            groups.setdefault(val.lower(), [])
        elif field == "disallow":
            seen_rule = True
            for a in agents:
                groups.setdefault(a, []).append(val)
        elif field in ("allow", "crawl-delay", "sitemap", "host"):
            seen_rule = True
    # prefer a group that matches our token (prefix either way), else the wildcard group
    for agent, rules in groups.items():
        if agent != "*" and (agent in token or token in agent):
            return rules
    return groups.get("*", [])


@dataclass
class FixtureFetcher:
    """A deterministic fetcher for tests/offline runs — maps url → canned response. The response bodies
    are clearly test fixtures, never real government data presented as real. ``detail_responses`` maps an
    INFORMS detail action id (e.g. "SCP_COSP_WK_FL_DESCR$1") → the canned detail page, letting the detail
    enrichment stage run fully offline."""
    responses: dict[str, FetchResult] = field(default_factory=dict)
    detail_responses: dict[str, FetchResult] = field(default_factory=dict)

    def fetch(self, url: str, *, timeout: float = 20.0) -> FetchResult:
        return self.responses.get(url, FetchResult(404, "", ""))

    def informs_detail(self, list_url: str, action_id: str, *, timeout: float = 20.0) -> Optional[FetchResult]:
        return self.detail_responses.get(action_id)


class UrllibFetcher:
    """A real HTTP fetcher with a declared User-Agent, a per-host rate limit, and robots.txt respect.
    Used only when a caller explicitly opts into live collection."""

    def __init__(self, user_agent: str = "ReDevOps-ProcurementBot/0.1 (+https://redevops.io)",
                 min_interval_s: float = 2.0):
        self.user_agent = user_agent
        self.min_interval_s = min_interval_s
        self._last: dict[str, float] = {}
        self._robots: dict[str, str] = {}

    def _allowed(self, url: str) -> bool:
        import urllib.request
        p = urlparse(url)
        host = f"{p.scheme}://{p.netloc}"
        if host not in self._robots:
            try:
                req = urllib.request.Request(host + "/robots.txt", headers={"User-Agent": self.user_agent})
                with urllib.request.urlopen(req, timeout=10.0) as r:  # noqa: S310 (explicit opt-in)
                    self._robots[host] = r.read().decode("utf-8", "ignore")
            except Exception:  # noqa: BLE001
                self._robots[host] = ""      # no robots reachable → default allow, still rate-limited
        disallows = _robots_disallows(self._robots[host], self.user_agent)
        return not any(d and p.path.startswith(d) for d in disallows)

    def _opener(self):
        # A per-instance opener with a cookie jar so a session handshake (e.g. PeopleSoft's 302 +
        # PSJSESSIONID cookie on INFORMS) is followed correctly across redirects.
        if not hasattr(self, "_op"):
            import http.cookiejar
            import urllib.request
            self._op = urllib.request.build_opener(
                urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        return self._op

    def fetch(self, url: str, *, timeout: float = 20.0) -> FetchResult:
        import urllib.request
        if not self._allowed(url):
            return FetchResult(999, "", "blocked-by-robots")
        host = urlparse(url).netloc
        wait = self.min_interval_s - (time.monotonic() - self._last.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
            with self._opener().open(req, timeout=timeout) as r:  # noqa: S310 (explicit opt-in)
                body = r.read().decode("utf-8", "ignore")
                self._last[host] = time.monotonic()
                return FetchResult(getattr(r, "status", 200), body, r.headers.get("Content-Type", ""))
        except Exception as e:  # noqa: BLE001
            return FetchResult(0, "", f"error:{e!r}")

    def informs_detail(self, list_url: str, action_id: str, *, timeout: float = 30.0) -> Optional[FetchResult]:
        """Fetch one INFORMS solicitation's detail page by replaying its PeopleSoft ICAction postback.

        There is no GET-able detail URL: the list grid's per-row "open" control posts back to the same
        component with ``ICAction=<action_id>``. We GET the list first (for the live ICSID/ICStateNum
        form state), then POST the row action. Read-only — this only opens the publicly-exposed detail
        page; it never submits, registers, or bids. Returns None on any failure (→ graceful list-only)."""
        import urllib.parse
        import urllib.request
        if not self._allowed(list_url):
            return None
        try:
            host = urlparse(list_url).netloc
            wait = self.min_interval_s - (time.monotonic() - self._last.get(host, 0.0))
            if wait > 0:
                time.sleep(wait)
            req = urllib.request.Request(list_url, headers={"User-Agent": self.user_agent})
            html_text = self._opener().open(req, timeout=timeout).read().decode("utf-8", "ignore")
            fields = dict(re.findall(
                r"<input[^>]*type='hidden'[^>]*name='([^']+)'[^>]*value='([^']*)'", html_text))
            if "ICStateNum" not in fields:
                return None
            fields.update({"ICAJAX": "1", "ICNAVTYPEDD": "1", "ICAction": action_id})
            comp = list_url.split("?", 1)[0]
            data = urllib.parse.urlencode(fields).encode()
            post = urllib.request.Request(comp, data=data, headers={"User-Agent": self.user_agent})
            with self._opener().open(post, timeout=timeout) as r:  # noqa: S310 (explicit opt-in)
                body = r.read().decode("utf-8", "ignore")
                self._last[host] = time.monotonic()
                return FetchResult(getattr(r, "status", 200), body, r.headers.get("Content-Type", ""))
        except Exception:  # noqa: BLE001
            return None


def _find_chrome() -> Optional[str]:
    """Locate a Chromium/Chrome binary for headless rendering (system install or a Playwright download)."""
    import glob
    import shutil
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        p = shutil.which(name)
        if p:
            return p
    for pat in (os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux/chrome"),
                os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux/headless_shell")):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


class BrowserFetcher:
    """A headless-browser fetcher for portals that block plain HTTP (bot protection) or render bids only
    after JavaScript (SPAs). It renders the page with headless Chromium and returns the resulting DOM, so
    a per-portal parser can extract from the rendered HTML.

    Same politeness as the HTTP fetcher: robots.txt is honoured (per user-agent group) and a per-host rate
    limit applies. Rendering is heavier, so the default interval is larger. Read-only — it only renders the
    publicly-exposed page; it never submits anything."""

    def __init__(self, user_agent: str = "ReDevOps-ProcurementBot/0.1 (+https://redevops.io)",
                 min_interval_s: float = 3.0, chrome: Optional[str] = None,
                 virtual_time_ms: int = 12000, timeout_s: float = 90.0):
        self.user_agent = user_agent
        self.min_interval_s = min_interval_s
        self.chrome = chrome or _find_chrome()
        self.virtual_time_ms = virtual_time_ms
        self.timeout_s = timeout_s
        self._last: dict[str, float] = {}
        self._robots: dict[str, str] = {}

    def _allowed(self, url: str) -> bool:
        import urllib.request
        p = urlparse(url)
        host = f"{p.scheme}://{p.netloc}"
        if host not in self._robots:
            try:
                req = urllib.request.Request(host + "/robots.txt", headers={"User-Agent": self.user_agent})
                with urllib.request.urlopen(req, timeout=10.0) as r:  # noqa: S310 (explicit opt-in)
                    self._robots[host] = r.read().decode("utf-8", "ignore")
            except Exception:  # noqa: BLE001
                self._robots[host] = ""
        return not any(d and p.path.startswith(d)
                       for d in _robots_disallows(self._robots[host], self.user_agent))

    def fetch(self, url: str, *, timeout: float = 0.0) -> FetchResult:
        import subprocess
        if self.chrome is None:
            return FetchResult(0, "", "error:no-chrome-binary")
        if not self._allowed(url):
            return FetchResult(999, "", "blocked-by-robots")
        host = urlparse(url).netloc
        wait = self.min_interval_s - (time.monotonic() - self._last.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        cmd = [self.chrome, "--headless=new", "--no-sandbox", "--disable-gpu",
               f"--virtual-time-budget={self.virtual_time_ms}", f"--user-agent={self.user_agent}",
               "--dump-dom", url]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=timeout or self.timeout_s)
            self._last[host] = time.monotonic()
            dom = out.stdout or ""
            if not dom.strip():
                return FetchResult(0, "", f"error:empty-render:{out.returncode}")
            return FetchResult(200, dom, "text/html")
        except subprocess.TimeoutExpired:
            return FetchResult(0, "", "error:render-timeout")
        except Exception as e:  # noqa: BLE001
            return FetchResult(0, "", f"error:{e!r}")


def _extract_email(msg) -> dict:
    """Flatten an email.message.Message into {subject, from, date, text, html}."""
    def hdr(k):
        return str(msg.get(k, "") or "")
    text, html = "", ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_maintype() == "multipart":
                continue
            ctype = part.get_content_type()
            payload = part.get_payload(decode=True)
            if payload is None:
                continue
            body = payload.decode(part.get_content_charset() or "utf-8", "ignore")
            if ctype == "text/plain" and not text:
                text = body
            elif ctype == "text/html" and not html:
                html = body
    else:
        payload = msg.get_payload(decode=True)
        body = payload.decode(msg.get_content_charset() or "utf-8", "ignore") if payload else ""
        if msg.get_content_type() == "text/html":
            html = body
        else:
            text = body
    return {"subject": hdr("Subject"), "from": hdr("From"), "date": hdr("Date"), "text": text, "html": html}


@dataclass
class MboxFetcher:
    """Read emails from an mbox file or a directory of .eml files — the offline/testing path for the EMAIL
    source (and the fallback when a live inbox is exported by hand). Filters to messages FROM the alert
    sender so nothing else in the mailbox is ever touched."""
    path: str
    from_filter: str = "bidnet"

    def fetch(self, url: str = "", *, timeout: float = 20.0) -> FetchResult:
        import email as _email
        import glob
        import mailbox
        msgs = []
        try:
            if os.path.isdir(self.path):
                for f in sorted(glob.glob(os.path.join(self.path, "*.eml"))):
                    with open(f, "rb") as fh:
                        msgs.append(_email.message_from_binary_file(fh))
            else:
                msgs = list(mailbox.mbox(self.path))
        except Exception as e:  # noqa: BLE001
            return FetchResult(0, "", f"error:{e!r}")
        recs = [_extract_email(m) for m in msgs]
        if self.from_filter:
            recs = [r for r in recs if self.from_filter.lower() in r["from"].lower()]
        return FetchResult(200, json.dumps(recs), "application/x-emails")


class ImapFetcher:
    """Read bid-match notification emails from an inbox over IMAP — read-only and SERVER-SIDE filtered to
    ``FROM from_filter`` (e.g. bidnetdirect.com), so on a shared mailbox the collector can only ever touch
    the alert mail, never other correspondence. Credentials come from the environment; they are never
    stored here or logged. Google Workspace needs an app password (imap.gmail.com)."""

    def __init__(self, host: str, user: str, password: str, *, from_filter: str = "bidnet",
                 folder: str = "INBOX", limit: int = 100):
        self.host = host
        self.user = user
        self._password = password
        self.from_filter = from_filter
        self.folder = folder
        self.limit = limit

    @classmethod
    def from_env(cls, prefix: str = "BIDNET_IMAP", **kw) -> "ImapFetcher":
        """Build from ``<prefix>_HOST`` (default imap.gmail.com), ``_USER``, ``_PASSWORD`` (an app
        password for Google Workspace). The password is read here and never echoed."""
        return cls(os.environ.get(f"{prefix}_HOST", "imap.gmail.com"),
                   os.environ.get(f"{prefix}_USER", ""), os.environ.get(f"{prefix}_PASSWORD", ""), **kw)

    @property
    def configured(self) -> bool:
        return bool(self.user and self._password)

    def fetch(self, url: str = "", *, timeout: float = 30.0) -> FetchResult:
        import email as _email
        import imaplib
        if not self.configured:
            return FetchResult(0, "", "error:IMAP credentials not set (BIDNET_IMAP_USER/PASSWORD)")
        try:
            M = imaplib.IMAP4_SSL(self.host, timeout=timeout)
            M.login(self.user, self._password)
            M.select(self.folder, readonly=True)                      # read-only: never mutate the mailbox
            typ, data = M.search(None, "FROM", f'"{self.from_filter}"')  # server-side scope guardrail
            ids = data[0].split() if data and data[0] else []
            if self.limit:
                ids = ids[-self.limit:]
            recs = []
            for i in ids:
                typ, d = M.fetch(i, "(RFC822)")
                if d and d[0]:
                    recs.append(_extract_email(_email.message_from_bytes(d[0][1])))
            M.logout()
            return FetchResult(200, json.dumps(recs), "application/x-emails")
        except Exception as e:  # noqa: BLE001
            return FetchResult(0, "", f"error:{e!r}")


# ──────────────────────────── the source registry ────────────────────────────

class SourceMethod(str, Enum):
    RSS = "rss"
    JSON = "json"
    HTML = "html"
    SAM_API = "sam_api"
    INFORMS = "informs"          # Miami-Dade PeopleSoft public bidding grid (portal-specific extractor)
    MDC_FUTURE = "mdc_future"    # Miami-Dade Strategic Procurement "Future Solicitations" (forecast) JSON
    CIVICPLUS = "civicplus"      # CivicPlus municipal Bids module (server-rendered HTML; many FL cities)
    BONFIRE = "bonfire"          # Bonfire (bonfirehub.com) open-opportunities portal (needs a browser render)
    EMAIL = "email"              # bid-match notification emails (e.g. BidNet Direct) read from an inbox


# Methods whose pages block plain HTTP or render only after JS → they need a BrowserFetcher.
BROWSER_METHODS = frozenset({SourceMethod.BONFIRE})


class SourceHealth(str, Enum):
    UNKNOWN = "unknown"
    OK = "ok"
    STALE = "stale"
    ERROR = "error"


@dataclass
class ProcurementSource:
    """An official procurement endpoint for one jurisdiction, with provenance + health."""
    source_id: str
    jurisdiction_id: str
    name: str
    url: str
    method: SourceMethod
    discovered_at: str = ""
    last_ok_at: str = ""
    health: SourceHealth = SourceHealth.UNKNOWN
    provenance: str = "seed"                 # "seed" | "linked" | "source_discovery"


# Bump when the parsing/normalization changes materially — recorded on every observation so the durable
# lineage can tell "the source changed" from "our extractor changed".
COLLECTOR_VERSION = "procurement-collector/0.2"


@dataclass
class Observation:
    """Every fetch is preserved as an observation — the replayable evidence behind a discovery.

    ``kind`` distinguishes a *list*/discovery observation from a *detail*/enrichment observation: they are
    separate immutable observations with their own hashes and provenance, and a normalized opportunity
    references BOTH. ``observation_id`` is the stable, content-addressed identity used as an evidence ref
    and as the durable-store dedup key (immutable: same content ⇒ same id)."""
    source_id: str
    url: str
    fetched_at: str
    status: int
    content_hash: str
    raw: dict                                # the parsed raw solicitation record
    kind: str = "list"                       # "list" (discovery) | "detail" (enrichment)
    collector_version: str = COLLECTOR_VERSION
    known_at: str = ""                       # when this observation entered OUR knowledge (bi-temporal)

    def __post_init__(self):
        if not self.known_at:
            self.known_at = self.fetched_at

    @property
    def observation_id(self) -> str:
        return f"obs:{self.source_id}:{self.kind}:{self.content_hash.split(':')[-1]}"


def _seed_sources() -> list[ProcurementSource]:
    """Real official procurement URLs for the seeded 33180-area jurisdictions (#36). Methods are best
    known; a live source-discovery pass would confirm/refresh them."""
    return [
        ProcurementSource("src-miami-dade-informs", "fl-miami-dade-county", "Miami-Dade County — INFORMS Public Bidding",
                          "https://supplier.miamidade.gov/psc/EXTSUPP/SUPPLIER/ERP/c/"
                          "SCP_PUBLIC_MENU_FL.SCP_PUB_BID_CMP_FL.GBL?PAGE=SCP_PUB_BIDLIST_FL",
                          SourceMethod.INFORMS, provenance="official-verified"),
        ProcurementSource("src-miami-dade-future", "fl-miami-dade-county",
                          "Miami-Dade County — Future Solicitations (forecast)",
                          "https://www.miamidade.gov/apps/ISD/stratproc/Home/FutureSolicitationsList",
                          SourceMethod.MDC_FUTURE, provenance="official-verified"),
        ProcurementSource("src-aventura", "fl-aventura", "City of Aventura — Bids",
                          "https://www.cityofaventura.com/bids.aspx", SourceMethod.CIVICPLUS,
                          provenance="official-verified"),
        ProcurementSource("src-hallandale-beach", "fl-hallandale-beach", "City of Hallandale Beach — Bids",
                          "https://www.cohb.org/bids.aspx", SourceMethod.CIVICPLUS,
                          provenance="official-verified"),
        ProcurementSource("src-broward-bonfire", "fl-broward-county", "Broward County — Purchasing (Bonfire)",
                          "https://broward.bonfirehub.com/portal/?tab=openOpportunities",
                          SourceMethod.BONFIRE, provenance="official-verified"),
        ProcurementSource("src-mdcps", "fl-mdcps", "Miami-Dade County Public Schools — Procurement",
                          "https://procurement.dadeschools.net/", SourceMethod.HTML),
        ProcurementSource("src-fort-lauderdale", "fl-fort-lauderdale", "City of Fort Lauderdale — Bids",
                          "https://www.fortlauderdale.gov/departments/finance/procurement-services", SourceMethod.HTML),
        ProcurementSource("src-sam", "*", "SAM.gov federal opportunities",
                          "https://api.sam.gov/opportunities/v2/search", SourceMethod.SAM_API),
    ]


def default_source_registry() -> dict[str, ProcurementSource]:
    return {s.source_id: s for s in _seed_sources()}


# ──────────────────────────── parsing ────────────────────────────

def parse_rss(text: str) -> list[dict]:
    """Parse an RSS/Atom procurement feed into raw records (title/description/link/pubDate)."""
    out: list[dict] = []
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return out
    for item in root.iter():
        tag = item.tag.lower().rsplit("}", 1)[-1]
        if tag in ("item", "entry"):
            rec = {}
            for child in item:
                ctag = child.tag.lower().rsplit("}", 1)[-1]
                if ctag in ("title", "description", "summary", "link", "pubdate", "updated", "guid", "id"):
                    rec[ctag] = (child.text or child.attrib.get("href", "")).strip()
            if rec:
                out.append(rec)
    return out


def parse_json_items(text: str, items_key: str = "items") -> list[dict]:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    items = data.get(items_key) or data.get("opportunitiesData") or data.get("results") or []
    return [d for d in items if isinstance(d, dict)]


import html as _html  # noqa: E402

_INFORMS_SPAN = re.compile(r"id='SCP_PUB_AUC_VW_([A-Z0-9_]+)\$(\d+)'\s*>([^<]*)</span>")
# The issuing department is rendered per row under a different view prefix on the same grid.
_INFORMS_DEPT = re.compile(r"id='BUS_UNIT_AUC_VW_DESCR\$(\d+)'\s*>([^<]*)</span>")
# Each row's "open detail" control is a PeopleSoft ICAction postback, not a GET-able URL. The action id
# is deterministic from the row index; the enrichment stage replays it to fetch that row's detail page.
_INFORMS_DETAIL_ACTION = "SCP_COSP_WK_FL_DESCR"


def parse_informs(text: str) -> list[dict]:
    """Extract rows from the Miami-Dade INFORMS public-bidding grid (a PeopleSoft Fluid page).

    The grid renders each field as ``<span id='SCP_PUB_AUC_VW_<FIELD>$<row>'>value</span>`` — AUC_ID,
    AUC_NAME, AUC_FORMAT, AUC_TYPE — plus the department under ``BUS_UNIT_AUC_VW_DESCR$<row>``. We group
    by row index and return {title, event_id, format, type, department, detail_action}. ``detail_action``
    is the ICAction id the enrichment stage replays to fetch that row's detail page.
    """
    from collections import defaultdict
    rows: dict[int, dict] = defaultdict(dict)
    for fld, idx, val in _INFORMS_SPAN.findall(text):
        rows[int(idx)][fld] = _html.unescape(val).strip()
    depts = {int(idx): _html.unescape(val).strip() for idx, val in _INFORMS_DEPT.findall(text)}
    out = []
    for i in sorted(rows):
        r = rows[i]
        if not (r.get("AUC_ID") or r.get("AUC_NAME")):
            continue
        out.append({"title": r.get("AUC_NAME", ""), "event_id": r.get("AUC_ID", ""),
                    "format": r.get("AUC_FORMAT", ""), "type": r.get("AUC_TYPE", ""),
                    "department": depts.get(i, ""), "link": "",
                    "detail_action": f"{_INFORMS_DETAIL_ACTION}${i}"})
    return out


# ── INFORMS detail (enrichment) — the PeopleSoft solicitation detail page ──
# The detail page renders deterministic fields as <span id='SCP_P_AUCDTL_VW_<FIELD>$0'>value</span>.
_INFORMS_DTL_SPAN = re.compile(r"id='SCP_P_AUCDTL_VW_([A-Z0-9_]+)\$0'[^>]*>(.*?)</span>", re.DOTALL)
_INFORMS_DTL_DEPT = re.compile(r"id='BUS_UNIT_AUC_VW_DESCR\$0'[^>]*>([^<]*)</span>")
_INFORMS_DTL_PYMT = re.compile(r"id='PYMT_TR_EFF_VW_DESCR\$0'[^>]*>([^<]*)</span>")
_MDY_DATE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")


def _mdy_to_iso(text: str) -> str:
    """"09/21/2026 02:00 PM EST" → "2026-09-21". Returns "" if no M/D/Y date is present — we never
    invent a date the source did not state (the plan's "don't infer missing deadlines")."""
    m = _MDY_DATE.search(text or "")
    if not m:
        return ""
    mm, dd, yyyy = m.groups()
    return f"{yyyy}-{int(mm):02d}-{int(dd):02d}"


def _clean(val: str) -> str:
    v = _html.unescape(re.sub(r"<[^>]+>", " ", val)).replace("\xa0", " ").strip()
    return "" if v in ("", "&nbsp;") else re.sub(r"\s+", " ", v)


def parse_informs_detail(text: str) -> dict:
    """Extract the deterministic structure the INFORMS detail page actually exposes for one solicitation.

    Only fields the page states are returned — response_due_at/pre_bid/question deadlines are populated
    solely when present, honoring "don't infer missing deadlines". Returns a flat record the enrichment
    stage merges into the normalized opportunity: title, event_id, status, format, type, department,
    contact, start/end (response_due) dates (ISO + original text), long description, payment terms,
    multiple-bids flag, and document presence.
    """
    f = {_html.unescape(k): _clean(v) for k, v in _INFORMS_DTL_SPAN.findall(text)}
    dept = _INFORMS_DTL_DEPT.search(text)
    pymt = _INFORMS_DTL_PYMT.search(text)
    end_text = f.get("SCP_END_DATE_CHAR", "")
    rec: dict = {
        "record_kind": "informs_detail",
        "title": f.get("AUC_NAME", ""),
        "event_id": f.get("AUC_ID", ""),
        "status": f.get("AUC_STATUS", ""),
        "format": f.get("AUC_FORMAT", ""),
        "type": f.get("AUC_TYPE", ""),
        "round": f.get("AUC_ROUND", ""),
        "version": f.get("AUC_VERSION", ""),
        "multiple_bids": f.get("MULTIPLE_BIDS_FLG", ""),
        "contact": f.get("NAME1", ""),
        "department": _clean(dept.group(1)) if dept else "",
        "payment_terms": _clean(pymt.group(1)) if pymt else "",
        "description": f.get("DESCRLONG", ""),
        "posted_at": _mdy_to_iso(f.get("SCP_STRT_DATE_CHAR", "")),
        "start_at_text": f.get("SCP_STRT_DATE_CHAR", ""),
        "response_due_at": _mdy_to_iso(end_text),      # the real deadline — "" if the page states none
        "response_due_at_text": end_text,
    }
    # documents/addenda: the page renders download controls when attachments exist
    n_docs = len(re.findall(r"Download", text))
    rec["has_documents"] = n_docs > 0
    rec["document_count"] = n_docs
    return rec


def parse_sam(text: str) -> list[dict]:
    """Parse a SAM.gov Opportunities v2 response (``opportunitiesData``) into raw records.

    SAM is a FEDERAL, region-agnostic source: each notice carries a place of performance (state), which
    qualification uses instead of local-jurisdiction membership. We map the documented fields; NAICS is the
    authoritative category (municipal-style title keyword tagging is a poor fit for federal notices)."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []
    items = data.get("opportunitiesData") if isinstance(data, dict) else (data if isinstance(data, list) else [])
    out = []
    for d in items or []:
        if not isinstance(d, dict) or not (d.get("title") or d.get("noticeId")):
            continue
        poc = d.get("pointOfContact") or []
        contact = ""
        if isinstance(poc, list) and poc and isinstance(poc[0], dict):
            contact = (poc[0].get("fullName") or "").strip()
            if poc[0].get("email"):
                contact = f"{contact} <{poc[0]['email']}>".strip()
        pop = d.get("placeOfPerformance") or {}
        st = pop.get("state") if isinstance(pop, dict) else None
        state = ((st.get("code") or st.get("name")) if isinstance(st, dict) else (st or "")) or ""
        naics = str(d.get("naicsCode") or "").strip()
        dept = (d.get("fullParentPathName") or "")
        out.append({
            "record_kind": "sam",
            "title": (d.get("title") or "").strip(),
            "event_id": (d.get("noticeId") or d.get("solicitationNumber") or "").strip(),
            "format": (d.get("type") or d.get("baseType") or "").strip(),
            "posted_at": str(d.get("postedDate") or "")[:10],
            "response_due_at": str(d.get("responseDeadLine") or "")[:10],
            "department": dept.split(".")[-1].strip() if dept else "",
            "contact": contact,
            "set_aside": (d.get("typeOfSetAsideDescription") or "").strip(),
            "naics": naics,
            "categories": [naics] if naics else [],
            "place_of_performance_state": str(state).strip(),
            "link": (d.get("uiLink") or d.get("additionalInfoLink") or "").strip(),
        })
    return out


def sam_gov_url(base_url: str, api_key: str, *, posted_from: str, posted_to: str,
                naics: tuple[str, ...] = (), limit: int = 25,
                ptypes: tuple[str, ...] = ("o", "k", "p"), state: str = "") -> str:
    """Build a SAM.gov Opportunities v2 query URL. ``api_key`` is a secret from the environment — it lives
    only in the URL handed to the fetcher, never in the source registry or the evidence store."""
    from urllib.parse import urlencode
    params = {"api_key": api_key, "postedFrom": posted_from, "postedTo": posted_to, "limit": str(limit)}
    if ptypes:
        params["ptype"] = ",".join(ptypes)
    if naics:
        params["ncode"] = ",".join(naics)
    if state:
        params["state"] = state
    sep = "&" if "?" in base_url else "?"
    return f"{base_url}{sep}{urlencode(params)}"


def resolve_fetch_url(source: ProcurementSource, *, profile: Optional[BusinessProfile] = None,
                      now: Optional[str] = None, window_days: int = 14):
    """The effective URL to fetch for a source. Identity for most sources; for SAM_API it builds the keyed
    query URL from env ``SAM_API_KEY`` over a recent posted window + the profile's NAICS. Returns
    ``(url, error)`` — url is None (with an error string) when SAM has no key, so the caller records a
    source-reliability failure rather than silently skipping (the key is never logged or stored)."""
    if source.method is not SourceMethod.SAM_API:
        return source.url, ""
    import os
    from datetime import datetime, timedelta, timezone
    key = os.environ.get("SAM_API_KEY", "").strip()
    if not key:
        return None, "SAM_API_KEY not set"
    end = datetime.strptime(now[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc) if now else datetime.now(timezone.utc)
    start = end - timedelta(days=window_days)
    naics = tuple(profile.naics) if profile else ()
    return sam_gov_url(source.url, key, posted_from=start.strftime("%m/%d/%Y"),
                       posted_to=end.strftime("%m/%d/%Y"), naics=naics), ""


def parse_civicplus(text: str) -> list[dict]:
    """Parse a CivicPlus municipal Bids module (server-rendered HTML; the same platform hundreds of US
    cities run — e.g. Aventura, Hallandale Beach). Each bid is a ``listItemsRow bid`` block with a title
    link to ``bids.aspx?bidID=N``, an optional "Bid No.", and a status block that renders labels
    ("Status:", "Closes:") and their values in parallel spans. We take the non-label span values, so a
    closing value that is not a date (e.g. "Upon Contract") yields no deadline — we do not infer one. An
    empty list is a legitimate "no open bids", not a broken collector."""
    out = []
    for b in re.split(r'<div class="listItemsRow bid', text)[1:]:
        bid = re.search(r'bids?\.aspx\?bidID=(\d+)', b, re.I)
        if not bid:
            continue
        title = re.search(r'bidID=\d+"[^>]*>([^<]+)</a>', b, re.I)
        no = re.search(r'Bid No\.</strong>\s*([^<]+?)\s*<', b)
        # within the status region, the label spans end with ":"; the remaining spans are the values
        si = b.find("bidStatus")
        vals = []
        if si >= 0:
            vals = [_html.unescape(v).strip()
                    for v in re.findall(r'<span[^>]*>([^<]*)</span>', b[si:si + 700])]
            vals = [v for v in vals if v and not v.endswith(":")]
        status = vals[0] if len(vals) > 0 else ""
        closes = vals[1] if len(vals) > 1 else ""
        out.append({
            "record_kind": "civicplus",
            "title": _html.unescape(title.group(1)).strip() if title else "",
            "event_id": (no.group(1).strip() if no else f"bid{bid.group(1)}"),
            "status": status,
            "response_due_at": _mdy_to_iso(closes),      # "" for non-date closings (don't infer)
            "link": f"bids.aspx?bidID={bid.group(1)}",   # relative; normalize resolves against the source
            "format": "",
        })
    return out


_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def _bonfire_date_to_iso(text: str) -> str:
    """"Sep 21st 2026, 2:00 PM EDT" → "2026-09-21". "" if no such date — never inferred."""
    m = re.search(r'([A-Za-z]{3})[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})', text or "")
    if not m:
        return ""
    mon = _MONTHS.get(m.group(1).lower())
    return f"{m.group(3)}-{mon:02d}-{int(m.group(2)):02d}" if mon else ""


def parse_bonfire(html: str) -> list[dict]:
    """Parse a Bonfire (bonfirehub.com) open-opportunities portal — the rendered DOM (needs a browser to
    produce; the page is a JS/DataTables app). Bonfire powers many public agencies, so this one parser
    serves them all. Columns: Status, Ref#, Project (title), Department, Close Date, Days Left, Action
    (a link to /opportunities/N). The underscore.js template row (containing ``<%``) is skipped."""
    out = []
    def _txt(x):
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", x)).strip()
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL):
        if "<%" in row or "/opportunities/" not in row:
            continue
        link = re.search(r"/opportunities/\d+", row)
        tds = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
        if not link or len(tds) < 5:
            continue
        ref = _txt(tds[1])
        title = _txt(tds[2])
        if not (ref or title):
            continue
        out.append({
            "record_kind": "bonfire",
            "title": _html.unescape(title),
            "event_id": _html.unescape(ref) or link.group(0).split("/")[-1],
            "status": _txt(tds[0]),
            "department": _html.unescape(_txt(tds[3])),
            "response_due_at": _bonfire_date_to_iso(_txt(tds[4])),
            "response_due_at_text": _txt(tds[4]),
            "link": link.group(0),                       # relative; normalize resolves against the source
            "format": "",
        })
    return out


def parse_bidnet_email(msg: dict) -> list[dict]:
    """Extract matched solicitations from a BidNet Direct bid-match notification email.

    PROVISIONAL until locked against a real sample: BidNet match emails list each matched solicitation as
    a link to its bidnetdirect.com detail page with the title as the link text; this pulls those (title +
    link + any numeric id in the URL). Agency and exact close date will be mapped once we have a real
    email — we do not invent a deadline. This is link-based extraction, so it degrades safely (a layout we
    don't recognise yields nothing rather than garbage)."""
    body = msg.get("html") or ""
    out, seen = [], set()
    for m in re.finditer(
            r'<a[^>]+href="(https?://[^"]*bidnet(?:direct)?\.com[^"]*)"[^>]*>(.*?)</a>', body, re.I | re.DOTALL):
        href, title = m.group(1), re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", m.group(2))).strip()
        if not title or len(title) < 6 or href in seen:
            continue
        if not re.search(r"(solicitation|/bid|notice|supplier|opportunit|/tenders?/)", href, re.I):
            continue
        seen.add(href)
        idm = re.search(r"(\d{5,})", href)
        out.append({
            "record_kind": "bidnet_email",
            "title": _html.unescape(title),
            "event_id": idm.group(1) if idm else "bn-" + hashlib.sha256(href.encode()).hexdigest()[:10],
            "response_due_at": "",                            # locked from a real sample; never inferred
            "link": href,
            "format": "",
        })
    return out


def parse_email_alerts(text: str) -> list[dict]:
    """Parse a batch of alert emails (JSON list from Mbox/ImapFetcher). Routes by sender: BidNet emails go
    to parse_bidnet_email. The FROM-filter already restricts the batch to the alert sender."""
    try:
        emails = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []
    out = []
    for e in emails if isinstance(emails, list) else []:
        if not isinstance(e, dict):
            continue
        # BidNet Direct mails from noreply@bidnet.com (bid links may be bidnet.com or bidnetdirect.com)
        if "bidnet" in (e.get("from", "") or "").lower():
            out.extend(parse_bidnet_email(e))
    return out


def parse_future_solicitations(text: str) -> list[dict]:
    """Parse Miami-Dade's "Future Solicitations" forecast feed (a DataTables JSON endpoint).

    Each row is an intended *future* procurement (a forecast, not yet biddable): documentTitle, a
    webPostingCounter id, release/removal dates, and a submitting contact. We map removalDate to
    response_due_at only as the forecast's own posted window — it is marked record_kind=forecast so
    normalization types it GOV_FORECAST, never a live solicitation."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []
    rows = data if isinstance(data, list) else (data.get("data") or data.get("items") or [])
    out = []
    for d in rows:
        if not isinstance(d, dict) or not d.get("documentTitle"):
            continue
        out.append({
            "record_kind": "forecast",
            "title": str(d.get("documentTitle", "")).strip(),
            "event_id": f"FUT-{d.get('webPostingCounter')}",
            "format": "FORECAST",
            "posted_at": _mdy_to_iso(str(d.get("releaseDate", ""))),
            "response_due_at": _mdy_to_iso(str(d.get("removalDate", ""))),
            "contact": str(d.get("sendFeedBack", "")).strip(),
            "contact_email": str(d.get("emailAddress", "")).strip(),
            "document_count": int(d.get("attachmentCount") or 0),
            "link": "https://www.miamidade.gov/apps/ISD/stratproc/",
        })
    return out


# Keyword → service-category tagger. Municipal solicitation titles carry no NAICS, so we derive coarse
# service tags from the title text; qualification then matches them against the business's services.
_CATEGORY_KEYWORDS = {
    "engineering": ("engineering", "engineer"),
    "it": ("cisco", "software", "hardware", "adobe", "license", "technology", " it ", "network", "systems"),
    "professional services": ("professional", "svcs", "services", "consult", "legislative", "legal", "title company"),
    "maintenance": ("maintenance", "repair"),
    "hvac": ("hvac", "air condition", "chiller", "mechanical", "rooftop"),
    "construction": ("construction", "build", "renovation", "roofing"),
    "accounting": ("1099", "irs", "payroll", "retirement"),
    "equipment": ("clock", "equipment", "furniture", "vehicle"),
}


def tag_categories(title: str) -> tuple[str, ...]:
    t = f" {title.lower()} "
    tags = [cat for cat, kws in _CATEGORY_KEYWORDS.items() if any(k in t for k in kws)]
    return tuple(sorted(set(tags)))


# ──────────────────────────── collect ────────────────────────────

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_source(method: SourceMethod, text: str) -> list[dict]:
    """Dispatch a source's response body to its parser. One place so `collect` and the soak collector
    agree on what a source's records are (generic HTML is portal-specific → no records here)."""
    if method is SourceMethod.RSS:
        return parse_rss(text)
    if method is SourceMethod.SAM_API:
        return parse_sam(text)
    if method is SourceMethod.JSON:
        return parse_json_items(text)
    if method is SourceMethod.INFORMS:
        return parse_informs(text)
    if method is SourceMethod.MDC_FUTURE:
        return parse_future_solicitations(text)
    if method is SourceMethod.CIVICPLUS:
        return parse_civicplus(text)
    if method is SourceMethod.BONFIRE:
        return parse_bonfire(text)
    if method is SourceMethod.EMAIL:
        return parse_email_alerts(text)
    return []


def collect(sources: list[ProcurementSource], fetcher: Fetcher, *, now: Optional[str] = None) -> list[Observation]:
    """Fetch and parse every source; preserve one Observation per raw record (with a content hash)."""
    ts = now or _now_iso()
    out: list[Observation] = []
    for src in sources:
        res = fetcher.fetch(src.url)
        if res.status != 200 or not res.text:
            continue
        for rec in parse_source(src.method, res.text):
            out.append(_observe(src.source_id, src.url, ts, res.status, rec, kind="list"))
    return out


def _observe(source_id: str, url: str, ts: str, status: int, rec: dict, *, kind: str) -> Observation:
    """Build an immutable, content-addressed Observation from a parsed raw record."""
    h = "sha256:" + hashlib.sha256(
        json.dumps(rec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    return Observation(source_id, url, ts, status, h, rec, kind=kind)


# ──────────────────────────── detail enrichment (a stage separate from discovery) ────────────────────────────
# List collection is discovery; fetching a solicitation's detail page is a SEPARATE enrichment stage. The
# detail is preserved as its own immutable Observation (kind="detail", its own hash/provenance); the
# normalized opportunity then references BOTH the list and the detail observation.

def enrich_informs(list_obs: Observation, source: ProcurementSource, fetcher: Fetcher, *,
                   now: Optional[str] = None) -> Optional[Observation]:
    """Fetch one INFORMS row's detail page and return it as a separate detail Observation, or None if the
    fetcher cannot perform the PeopleSoft detail postback (offline fixtures without a detail canned in).

    The detail navigation is an ICAction postback, not a GET — so the fetcher must expose
    ``informs_detail(list_url, action_id)``. Degrades gracefully: a fetcher without it (or a failed
    postback) leaves the opportunity list-only rather than fabricating detail."""
    action = list_obs.raw.get("detail_action")
    getter = getattr(fetcher, "informs_detail", None)
    if not action or not callable(getter):
        return None
    res = getter(source.url, action)
    if not res or res.status != 200 or not res.text:
        return None
    rec = parse_informs_detail(res.text)
    if not (rec.get("event_id") or rec.get("title")):
        return None
    return _observe(source.source_id, source.url, now or _now_iso(), 200, rec, kind="detail")


def merge_detail(opp: RevenueOpportunity, detail_obs: Observation) -> RevenueOpportunity:
    """Merge a detail observation into a normalized opportunity, referencing BOTH observations. Detail
    fields (real response deadline, status, department, contact, documents) win over list-only guesses;
    the deadline is only set from a date the detail page actually stated."""
    d = detail_obs.raw
    if d.get("response_due_at"):
        opp.response_due_at = d["response_due_at"]
    opp.status = d.get("status", "") or opp.status
    opp.department = d.get("department", "") or opp.department
    opp.contact = d.get("contact", "") or opp.contact
    opp.payment_terms = d.get("payment_terms", "") or opp.payment_terms
    if d.get("description"):
        opp.description = d["description"]
    if d.get("posted_at"):
        opp.posted_at = d["posted_at"]
    if d.get("document_count"):
        opp.documents = tuple(f"attachment:{i + 1}" for i in range(int(d["document_count"])))
    # refine the solicitation type from the detail page's declared format, if present
    if d.get("format"):
        opp.solicitation_type = _guess_type(d["format"])
    opp.detail_observed = True
    if detail_obs.observation_id not in opp.evidence_ids:
        opp.evidence_ids = opp.evidence_ids + (detail_obs.observation_id,)
    return opp


# ──────────────────────────── normalize a raw record → opportunity ────────────────────────────

_TYPE_HINTS = {"rfp": SolicitationType.RFP, "rfq": SolicitationType.RFQ, "rfi": SolicitationType.RFI,
               "itb": SolicitationType.ITB, "grant": SolicitationType.GRANT,
               "forecast": SolicitationType.FORECAST, "bid": SolicitationType.BID}


def _guess_type(title: str) -> SolicitationType:
    t = title.lower()
    for k, v in _TYPE_HINTS.items():
        if k in t:
            return v
    return SolicitationType.BID


def normalize(obs: Observation, source: ProcurementSource, *,
              categories: tuple[str, ...] = (), zip_code: str = "") -> RevenueOpportunity:
    """Map a raw observation to the normalized RevenueOpportunity (dedup id from source + record hash)."""
    r = obs.raw
    title = r.get("title") or r.get("title_text") or r.get("subject") or "(untitled solicitation)"
    # municipal titles carry no NAICS → derive coarse service tags from the title; federal (SAM) records
    # carry an authoritative NAICS in `categories`/`naics`. Union all available signals.
    cats = tuple(sorted(set(categories) | set(tag_categories(title))
                        | {str(c) for c in r.get("categories", ()) if c}
                        | ({str(r["naics"])} if r.get("naics") else set())))
    contact = r.get("contact", "")
    if r.get("contact_email"):
        contact = f"{contact} <{r['contact_email']}>".strip()
    link = r.get("link") or r.get("id") or ""
    if link and not link.startswith("http"):                 # resolve relative links (CivicPlus, some RSS)
        from urllib.parse import urljoin
        link = urljoin(source.url, link)
    return RevenueOpportunity(
        opportunity_id=f"{source.source_id}:{r.get('event_id') or obs.content_hash.split(':')[-1]}",
        source=source.source_id, issuing_entity=source.name.split(" — ")[0],
        jurisdiction_id=source.jurisdiction_id, title=title,
        solicitation_type=_guess_type(r.get("format", "") + " " + title), place_of_performance_zip=zip_code,
        description=r.get("description") or r.get("summary") or "",
        categories=cats, posted_at=r.get("posted_at") or "",
        response_due_at=r.get("response_due_at") or r.get("pubdate") or "",
        set_aside=r.get("set_aside", ""),
        place_of_performance_state=r.get("place_of_performance_state", ""),
        status=r.get("status", ""),
        department=r.get("department", ""), contact=contact,
        source_url=link or source.url,
        evidence_ids=(obs.observation_id,),
        discovered_at=obs.fetched_at)


# ──────────────────────────── the one-region pipeline ────────────────────────────

@dataclass
class RegionResult:
    monitored_jurisdiction_ids: list[str]
    observations: list[Observation]           # every list AND detail observation preserved
    handoffs: list[dict]                      # revenue-handoff/v1 records for PURSUE / urgent-REVIEW
    changes: list = field(default_factory=list)  # EvidenceChange records when a durable store is attached


def run_region(zip_code: str, radius_miles: float, profile: BusinessProfile, *,
               fetcher: Fetcher, registry: Optional[JurisdictionRegistry] = None,
               sources: Optional[list[ProcurementSource]] = None,
               default_categories: tuple[str, ...] = (),
               enrich: bool = False, store=None) -> RegionResult:
    """Resolve the monitored jurisdictions for a ZIP/radius, collect their sources, normalize + qualify,
    and emit handoff records — the collector → discovery → qualification → handoff pipeline, one region.

    ``enrich=True`` runs the detail-enrichment stage (INFORMS detail postback) so opportunities carry the
    real response deadline/status/department; it degrades to list-only when a detail cannot be fetched.
    ``store`` (an EvidenceStore) durably persists every observation and records an EvidenceChange per
    opportunity — the lineage that survives restarts.
    """
    reg = registry or default_registry()
    monitored = reg.within(zip_code, radius_miles)
    monitored_ids = {m.jurisdiction.id for m in monitored}

    all_sources = sources or list(default_source_registry().values())
    # only poll sources for jurisdictions in this region (plus federal SAM, which is region-agnostic)
    region_sources = [s for s in all_sources if s.jurisdiction_id in monitored_ids or s.jurisdiction_id == "*"]
    by_id = {s.source_id: s for s in region_sources}

    list_observations = collect(region_sources, fetcher)
    observations: list[Observation] = []
    handoffs: list[dict] = []
    changes: list = []
    for obs in list_observations:
        src = by_id[obs.source_id]
        observations.append(obs)
        if store is not None:
            store.put_observation(obs)
        opp = normalize(obs, src, categories=default_categories, zip_code=zip_code)

        if enrich and src.method is SourceMethod.INFORMS:
            detail = enrich_informs(obs, src, fetcher)
            if detail is not None:
                observations.append(detail)
                if store is not None:
                    store.put_observation(detail)
                merge_detail(opp, detail)

        if store is not None:
            changes.append(store.ingest(opp))

        q = qualify(opp, profile, reg)
        if q.decision.value in ("PURSUE", "REVIEW"):
            handoffs.append(to_handoff(opp, q, monitored))
    return RegionResult(sorted(monitored_ids), observations, handoffs, changes)
