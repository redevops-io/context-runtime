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
        # a deliberately conservative check: honour a global "User-agent: * / Disallow: <prefix>".
        disallow = [ln.split(":", 1)[1].strip()
                    for ln in self._robots[host].splitlines()
                    if ln.lower().startswith("disallow:")]
        return not any(d and p.path.startswith(d) for d in disallow)

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


# ──────────────────────────── the source registry ────────────────────────────

class SourceMethod(str, Enum):
    RSS = "rss"
    JSON = "json"
    HTML = "html"
    SAM_API = "sam_api"
    INFORMS = "informs"          # Miami-Dade PeopleSoft public bidding grid (portal-specific extractor)
    MDC_FUTURE = "mdc_future"    # Miami-Dade Strategic Procurement "Future Solicitations" (forecast) JSON


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
        ProcurementSource("src-aventura", "fl-aventura", "City of Aventura — Bids",
                          "https://www.cityofaventura.com/bids.aspx", SourceMethod.HTML),
        ProcurementSource("src-miami-dade-informs", "fl-miami-dade-county", "Miami-Dade County — INFORMS Public Bidding",
                          "https://supplier.miamidade.gov/psc/EXTSUPP/SUPPLIER/ERP/c/"
                          "SCP_PUBLIC_MENU_FL.SCP_PUB_BID_CMP_FL.GBL?PAGE=SCP_PUB_BIDLIST_FL",
                          SourceMethod.INFORMS, provenance="official-verified"),
        ProcurementSource("src-miami-dade-future", "fl-miami-dade-county",
                          "Miami-Dade County — Future Solicitations (forecast)",
                          "https://www.miamidade.gov/apps/ISD/stratproc/Home/FutureSolicitationsList",
                          SourceMethod.MDC_FUTURE, provenance="official-verified"),
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


def collect(sources: list[ProcurementSource], fetcher: Fetcher, *, now: Optional[str] = None) -> list[Observation]:
    """Fetch and parse every source; preserve one Observation per raw record (with a content hash)."""
    ts = now or _now_iso()
    out: list[Observation] = []
    for src in sources:
        res = fetcher.fetch(src.url)
        if res.status != 200 or not res.text:
            continue
        if src.method is SourceMethod.RSS:
            records = parse_rss(res.text)
        elif src.method in (SourceMethod.JSON, SourceMethod.SAM_API):
            records = parse_json_items(res.text)
        elif src.method is SourceMethod.INFORMS:
            records = parse_informs(res.text)
        elif src.method is SourceMethod.MDC_FUTURE:
            records = parse_future_solicitations(res.text)
        else:  # generic HTML is portal-specific — a real deployment plugs a per-portal extractor here.
            records = []
        for rec in records:
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
    # municipal titles carry no NAICS → derive coarse service tags from the title (union with any passed in)
    cats = tuple(sorted(set(categories) | set(tag_categories(title))))
    contact = r.get("contact", "")
    if r.get("contact_email"):
        contact = f"{contact} <{r['contact_email']}>".strip()
    return RevenueOpportunity(
        opportunity_id=f"{source.source_id}:{r.get('event_id') or obs.content_hash.split(':')[-1]}",
        source=source.source_id, issuing_entity=source.name.split(" — ")[0],
        jurisdiction_id=source.jurisdiction_id, title=title,
        solicitation_type=_guess_type(r.get("format", "") + " " + title), place_of_performance_zip=zip_code,
        description=r.get("description") or r.get("summary") or "",
        categories=cats, posted_at=r.get("posted_at") or "",
        response_due_at=r.get("response_due_at") or r.get("pubdate") or "",
        department=r.get("department", ""), contact=contact,
        source_url=r.get("link") or r.get("id") or source.url,
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
