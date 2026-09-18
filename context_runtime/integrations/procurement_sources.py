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
    are clearly test fixtures, never real government data presented as real."""
    responses: dict[str, FetchResult] = field(default_factory=dict)

    def fetch(self, url: str, *, timeout: float = 20.0) -> FetchResult:
        return self.responses.get(url, FetchResult(404, "", ""))


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


# ──────────────────────────── the source registry ────────────────────────────

class SourceMethod(str, Enum):
    RSS = "rss"
    JSON = "json"
    HTML = "html"
    SAM_API = "sam_api"
    INFORMS = "informs"          # Miami-Dade PeopleSoft public bidding grid (portal-specific extractor)


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


@dataclass
class Observation:
    """Every fetch is preserved as an observation — the replayable evidence behind a discovery."""
    source_id: str
    url: str
    fetched_at: str
    status: int
    content_hash: str
    raw: dict                                # the parsed raw solicitation record


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


def parse_informs(text: str) -> list[dict]:
    """Extract rows from the Miami-Dade INFORMS public-bidding grid (a PeopleSoft Fluid page).

    The grid renders each field as ``<span id='SCP_PUB_AUC_VW_<FIELD>$<row>'>value</span>`` — AUC_ID,
    AUC_NAME, AUC_FORMAT, AUC_TYPE. We group by row index and return {title, event_id, format, type}.
    """
    from collections import defaultdict
    rows: dict[int, dict] = defaultdict(dict)
    for field, idx, val in _INFORMS_SPAN.findall(text):
        rows[int(idx)][field] = _html.unescape(val).strip()
    out = []
    for i in sorted(rows):
        r = rows[i]
        if not (r.get("AUC_ID") or r.get("AUC_NAME")):
            continue
        out.append({"title": r.get("AUC_NAME", ""), "event_id": r.get("AUC_ID", ""),
                    "format": r.get("AUC_FORMAT", ""), "type": r.get("AUC_TYPE", ""), "link": ""})
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
        else:  # generic HTML is portal-specific — a real deployment plugs a per-portal extractor here.
            records = []
        for rec in records:
            h = "sha256:" + hashlib.sha256(
                json.dumps(rec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
            out.append(Observation(src.source_id, src.url, ts, res.status, h, rec))
    return out


# ──────────────────────────── normalize a raw record → opportunity ────────────────────────────

_TYPE_HINTS = {"rfp": SolicitationType.RFP, "rfq": SolicitationType.RFQ, "rfi": SolicitationType.RFI,
               "itb": SolicitationType.ITB, "grant": SolicitationType.GRANT, "bid": SolicitationType.BID}


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
    return RevenueOpportunity(
        opportunity_id=f"{source.source_id}:{r.get('event_id') or obs.content_hash.split(':')[-1]}",
        source=source.source_id, issuing_entity=source.name.split(" — ")[0],
        jurisdiction_id=source.jurisdiction_id, title=title,
        solicitation_type=_guess_type(r.get("format", "") + " " + title), place_of_performance_zip=zip_code,
        description=r.get("description") or r.get("summary") or "",
        categories=cats, response_due_at=r.get("response_due_at") or r.get("pubdate") or "",
        source_url=r.get("link") or r.get("id") or source.url,
        evidence_ids=(f"obs:{obs.source_id}:{obs.content_hash.split(':')[-1]}",),
        discovered_at=obs.fetched_at)


# ──────────────────────────── the one-region pipeline ────────────────────────────

@dataclass
class RegionResult:
    monitored_jurisdiction_ids: list[str]
    observations: list[Observation]
    handoffs: list[dict]                      # revenue-handoff/v1 records for PURSUE / urgent-REVIEW


def run_region(zip_code: str, radius_miles: float, profile: BusinessProfile, *,
               fetcher: Fetcher, registry: Optional[JurisdictionRegistry] = None,
               sources: Optional[list[ProcurementSource]] = None,
               default_categories: tuple[str, ...] = ()) -> RegionResult:
    """Resolve the monitored jurisdictions for a ZIP/radius, collect their sources, normalize + qualify,
    and emit handoff records — the collector → discovery → qualification → handoff pipeline, one region.
    """
    reg = registry or default_registry()
    monitored = reg.within(zip_code, radius_miles)
    monitored_ids = {m.jurisdiction.id for m in monitored}

    all_sources = sources or list(default_source_registry().values())
    # only poll sources for jurisdictions in this region (plus federal SAM, which is region-agnostic)
    region_sources = [s for s in all_sources if s.jurisdiction_id in monitored_ids or s.jurisdiction_id == "*"]
    by_id = {s.source_id: s for s in region_sources}

    observations = collect(region_sources, fetcher)
    handoffs: list[dict] = []
    for obs in observations:
        src = by_id[obs.source_id]
        opp = normalize(obs, src, categories=default_categories, zip_code=zip_code)
        q = qualify(opp, profile, reg)
        if q.decision.value in ("PURSUE", "REVIEW"):
            handoffs.append(to_handoff(opp, q, monitored))
    return RegionResult(sorted(monitored_ids), observations, handoffs)
