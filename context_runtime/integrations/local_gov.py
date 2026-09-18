"""Local-Government Opportunity Discovery — geographic revenue signals for SMBs.

The Revenue Missions plan's strongest realism lever: instead of a synthetic tender fixture, resolve a
business owner's ZIP + service radius into the **real** public entities whose procurement a business
near that ZIP could actually bid on, then qualify each opportunity against that specific business with
explicit evidence — `PURSUE` / `REVIEW` / `REJECT`. The output is *"3 opportunities worth reviewing
today"*, not *"137 new bids"*.

This module owns the **geographic + qualification core** and reuses the deterministic geospatial engine
(`geospatial/engine.py`) — the governing rule holds: *do not ask an LLM to infer a spatial relationship
a geometry engine can calculate.* Collection (scraping portals, SAM.gov, PDF acquisition) is a separate,
pluggable concern per the plan; here the geography, jurisdictions and distances are real, and a
solicitation's *content* arrives as an Observation from whatever collector produced it.

What is genuinely new here vs the existing engine:
  * geographic (haversine) distance in miles — the engine's `distance` is planar only;
  * a `JurisdictionRegistry` seeded with real jurisdictions (not the Fulton-County zoning fixture);
  * `jurisdictions_for(zip, radius)` → the monitored public entities, each with WHY-it-was-included
    evidence (containment via `point_in_polygon`, or proximity via haversine);
  * a normalized `RevenueOpportunity` contract + evidence-backed `qualify(...)`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..geospatial.contracts import Coord, Ring
from ..geospatial.engine import point_in_polygon

_EARTH_RADIUS_MILES = 3958.7613


# ──────────────────────────── geographic distance (new capability) ────────────────────────────

def haversine_miles(a: Coord, b: Coord) -> float:
    """Great-circle distance in miles between two lon/lat coordinates.

    The engine's ``distance`` is Euclidean in a planar CRS; "within 50 miles of a ZIP" is a geographic
    question over lon/lat, so it needs haversine. Coordinates are ``(lon, lat)`` per the engine's ``Coord``.
    """
    lon1, lat1 = a
    lon2, lat2 = b
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return _EARTH_RADIUS_MILES * 2 * math.asin(min(1.0, math.sqrt(h)))


# ──────────────────────────── jurisdictions ────────────────────────────

class JurisdictionKind(str, Enum):
    CITY = "city"
    COUNTY = "county"
    SCHOOL_DISTRICT = "school_district"
    SPECIAL_DISTRICT = "special_district"
    UTILITY = "utility"
    STATE = "state"


class InclusionReason(str, Enum):
    CONTAINS = "contains"          # the ZIP falls inside this jurisdiction (polygon or declared)
    WITHIN_RADIUS = "within_radius"  # the jurisdiction is within the owner's service radius


@dataclass(frozen=True)
class Jurisdiction:
    """A public body that issues procurement. ``centroid`` is (lon, lat); ``boundary`` (optional) is a
    real polygon ring used for containment via the geometry engine."""
    id: str
    name: str
    kind: JurisdictionKind
    centroid: Coord
    county: str
    state: str
    boundary: Optional[Ring] = None
    procurement_url: str = ""


@dataclass(frozen=True)
class ZipArea:
    zip_code: str
    centroid: Coord   # (lon, lat)
    county: str
    state: str


@dataclass(frozen=True)
class MonitoredJurisdiction:
    """A jurisdiction included in an owner's monitored area, with the evidence for WHY."""
    jurisdiction: Jurisdiction
    reason: InclusionReason
    distance_miles: float
    detail: str


class JurisdictionRegistry:
    """The set of public bodies known to the system, resolvable to an owner's service area.

    Seed data is real (real jurisdictions, real approximate centroids); a production registry would grow
    it from an authoritative boundary dataset (Census TIGER places/counties + a ZIP-centroid gazetteer).
    """

    def __init__(self, zips: list[ZipArea], jurisdictions: list[Jurisdiction]):
        self._zips = {z.zip_code: z for z in zips}
        self._jurisdictions = list(jurisdictions)

    def zip_area(self, zip_code: str) -> Optional[ZipArea]:
        return self._zips.get(zip_code)

    def jurisdictions(self) -> list[Jurisdiction]:
        return list(self._jurisdictions)

    def within(self, zip_code: str, radius_miles: float) -> list[MonitoredJurisdiction]:
        """The public bodies a business at ``zip_code`` serving ``radius_miles`` should monitor.

        A jurisdiction is included if it **contains** the ZIP (its boundary polygon covers the ZIP
        centroid, or it is the ZIP's declared county/state) or if its centroid is **within the radius**
        (haversine). Every inclusion carries its reason and distance — the "why it was monitored" evidence
        the plan requires. Sorted: containers first, then by distance. Deterministic.
        """
        z = self._zips.get(zip_code)
        if z is None:
            raise KeyError(f"unknown ZIP {zip_code!r}; seed it in the JurisdictionRegistry")

        out: list[MonitoredJurisdiction] = []
        for j in self._jurisdictions:
            dist = haversine_miles(z.centroid, j.centroid)
            contains = False
            detail = ""
            if j.boundary is not None and point_in_polygon(z.centroid, j.boundary):
                contains, detail = True, f"ZIP {zip_code} centroid falls inside {j.name}'s boundary"
            elif j.kind in (JurisdictionKind.COUNTY, JurisdictionKind.STATE) and j.county == z.county and j.state == z.state:
                # A county/state that the ZIP's own county/state names contains it, regardless of the
                # far-off area centroid (a county centroid can sit miles from any populated ZIP).
                contains, detail = True, f"ZIP {zip_code} is in {j.name}"
            if contains:
                out.append(MonitoredJurisdiction(j, InclusionReason.CONTAINS, round(dist, 1), detail))
            elif dist <= radius_miles:
                out.append(MonitoredJurisdiction(
                    j, InclusionReason.WITHIN_RADIUS, round(dist, 1),
                    f"{j.name} is {dist:.1f} mi from ZIP {zip_code} (≤ {radius_miles:g} mi service radius)"))
        out.sort(key=lambda m: (m.reason is not InclusionReason.CONTAINS, m.distance_miles))
        return out


# ──────────────────────────── the normalized opportunity ────────────────────────────

class SolicitationType(str, Enum):
    RFP = "RFP"
    RFQ = "RFQ"
    RFI = "RFI"
    ITB = "ITB"
    BID = "BID"
    GRANT = "GRANT"
    FORECAST = "FORECAST"


@dataclass
class RevenueOpportunity:
    """A discovered public-sector opportunity, normalized across the many ways municipalities describe
    procurement into one contract the Revenue Mission machinery consumes (mirrors the plan's §5 / LG
    normalization). Different sources → one shape."""
    opportunity_id: str
    source: str
    issuing_entity: str
    jurisdiction_id: str
    title: str
    solicitation_type: SolicitationType
    place_of_performance_zip: str
    description: str = ""
    categories: tuple[str, ...] = ()          # NAICS / commodity codes / service tags
    posted_at: str = ""
    response_due_at: str = ""                  # ISO date; "" = unknown
    estimated_value: Optional[float] = None
    set_aside: str = ""
    required_licenses: tuple[str, ...] = ()
    required_certifications: tuple[str, ...] = ()
    documents: tuple[str, ...] = ()
    contact: str = ""
    source_url: str = ""
    evidence_ids: tuple[str, ...] = ()
    discovered_at: str = ""
    geographic_distance_miles: Optional[float] = None


# ──────────────────────────── the business the opportunity is qualified against ────────────────────────────

@dataclass
class BusinessProfile:
    """The specific business qualification is run against — finding *every* nearby tender is not useful;
    finding the ones *this* business should spend time on is."""
    name: str
    service_zip: str
    service_radius_miles: float
    services: tuple[str, ...] = ()             # capability/service tags (match against opportunity categories)
    naics: tuple[str, ...] = ()
    licenses: tuple[str, ...] = ()
    certifications: tuple[str, ...] = ()
    min_contract_value: Optional[float] = None
    max_contract_value: Optional[float] = None


class Decision(str, Enum):
    PURSUE = "PURSUE"
    REVIEW = "REVIEW"
    REJECT = "REJECT"


@dataclass
class Qualification:
    """The evidence-backed verdict. Every result carries explicit reasons — never a bare score."""
    opportunity_id: str
    decision: Decision
    reasons: list[str] = field(default_factory=list)
    service_match: float = 0.0                 # fraction of the opportunity's categories the business covers
    distance_miles: Optional[float] = None


def qualify(opp: RevenueOpportunity, profile: BusinessProfile, registry: JurisdictionRegistry) -> Qualification:
    """Qualify one opportunity against one business, geographically and by capability. Precedence:
    a hard geographic or licensing miss is a REJECT before anything else; an unknown/mismatched
    requirement is a REVIEW; a clean capability+geography+deadline match is a PURSUE. Mirrors the plan's
    worked examples (`REJECT — mandatory license not present`, `REVIEW — insurance requirement unknown`, …).
    """
    reasons: list[str] = []

    # ── geography (hard gate): is the issuing jurisdiction inside the owner's monitored service area? ──
    # Membership (not raw centroid distance) is the right test — a county contains the ZIP even though its
    # area centroid can sit far away, and `within(...)` already resolved containment vs proximity correctly.
    monitored = {m.jurisdiction.id: m for m in registry.within(profile.service_zip, profile.service_radius_miles)}
    m = monitored.get(opp.jurisdiction_id)
    dist: Optional[float] = None
    if m is None:
        j = next((jj for jj in registry.jurisdictions() if jj.id == opp.jurisdiction_id), None)
        zbiz = registry.zip_area(profile.service_zip)
        if j is not None and zbiz is not None:
            d = round(haversine_miles(zbiz.centroid, j.centroid), 1)
            reasons.append(f"{j.name} is outside the {profile.service_radius_miles:g}-mi service area ({d:.0f} mi)")
        else:
            reasons.append(f"jurisdiction {opp.jurisdiction_id!r} is not monitored for this service area")
        return Qualification(opp.opportunity_id, Decision.REJECT, reasons, 0.0, dist)
    dist = m.distance_miles

    # ── mandatory licenses/certifications (hard gate) ──
    have_lic = {s.lower() for s in profile.licenses}
    have_cert = {s.lower() for s in profile.certifications}
    missing_lic = [l for l in opp.required_licenses if l.lower() not in have_lic]
    missing_cert = [c for c in opp.required_certifications if c.lower() not in have_cert]
    if missing_lic:
        reasons.append(f"mandatory {missing_lic[0]} license not present")
        return Qualification(opp.opportunity_id, Decision.REJECT, reasons, 0.0, dist)

    # ── capability match ──
    biz_services = {s.lower() for s in profile.services} | {n.lower() for n in profile.naics}
    opp_cats = [c.lower() for c in opp.categories]
    matched = [c for c in opp_cats if any(c in s or s in c for s in biz_services)]
    service_match = (len(matched) / len(opp_cats)) if opp_cats else 0.0
    if service_match == 0.0 and opp_cats:
        reasons.append(f"no service match: opportunity is {', '.join(opp.categories)}; business does {', '.join(profile.services)}")
        return Qualification(opp.opportunity_id, Decision.REJECT, reasons, 0.0, dist)
    reasons.append(f"service match {service_match:.0%} ({', '.join(matched) or 'none'})")
    if m.reason is InclusionReason.CONTAINS:
        reasons.append(f"issuing jurisdiction ({m.jurisdiction.name}) covers your ZIP")
    else:
        reasons.append(f"within service area — {dist:.0f} mi")

    # ── contract-size fit (soft) ──
    if opp.estimated_value is not None:
        if profile.max_contract_value is not None and opp.estimated_value > profile.max_contract_value:
            reasons.append(f"estimated ${opp.estimated_value:,.0f} above max contract size ${profile.max_contract_value:,.0f}")
            return Qualification(opp.opportunity_id, Decision.REVIEW, reasons, service_match, dist)
        if profile.min_contract_value is not None and opp.estimated_value < profile.min_contract_value:
            reasons.append(f"estimated ${opp.estimated_value:,.0f} below min contract size ${profile.min_contract_value:,.0f}")
            return Qualification(opp.opportunity_id, Decision.REVIEW, reasons, service_match, dist)

    # ── unknown requirements → REVIEW; strong match + deadline → PURSUE ──
    if missing_cert:
        reasons.append(f"certification requirement to confirm: {missing_cert[0]}")
        return Qualification(opp.opportunity_id, Decision.REVIEW, reasons, service_match, dist)
    if service_match >= 0.5:
        if opp.response_due_at:
            reasons.append(f"response due {opp.response_due_at}")
        return Qualification(opp.opportunity_id, Decision.PURSUE, reasons, service_match, dist)
    reasons.append("partial service match — worth a human look")
    return Qualification(opp.opportunity_id, Decision.REVIEW, reasons, service_match, dist)


# ──────────────────────────── the core Revenue-Missions invariant ────────────────────────────

def has_valid_next_state(*, owner: str, next_action: str = "", waiting_condition: str = "",
                         terminal_disposition: str = "") -> bool:
    """The plan's product invariant: an active qualified opportunity must have an owner PLUS either a
    future next action, an explicit waiting condition, or a terminal disposition. Writing a CRM record
    does not count as handling the lead."""
    return bool(owner) and bool(next_action or waiting_condition or terminal_disposition)


# ──────────────────────────── real seed data (South Florida test region around ZIP 33180) ────────────────────────────
# Real jurisdictions, real approximate centroids (lon, lat). Aventura carries a real simplified boundary
# so containment runs through the geometry engine. This is the LG-1/LG-2 seed for a reproducible test region.

_AVENTURA_BOUNDARY: Ring = [
    (-80.160, 25.940), (-80.113, 25.940), (-80.113, 25.972), (-80.160, 25.972), (-80.160, 25.940),
]

SEED_ZIPS: list[ZipArea] = [
    ZipArea("33180", (-80.1390, 25.9565), "Miami-Dade", "FL"),   # Aventura / NE Miami-Dade
    ZipArea("33139", (-80.1300, 25.7907), "Miami-Dade", "FL"),   # Miami Beach
    ZipArea("33009", (-80.1484, 25.9812), "Broward", "FL"),      # Hallandale Beach
]

SEED_JURISDICTIONS: list[Jurisdiction] = [
    Jurisdiction("fl-miami-dade-county", "Miami-Dade County", JurisdictionKind.COUNTY, (-80.5510, 25.6110), "Miami-Dade", "FL", procurement_url="https://www.miamidade.gov/procurement/"),
    Jurisdiction("fl-broward-county", "Broward County", JurisdictionKind.COUNTY, (-80.4000, 26.1000), "Broward", "FL", procurement_url="https://www.broward.org/purchasing/"),
    Jurisdiction("fl-aventura", "City of Aventura", JurisdictionKind.CITY, (-80.1392, 25.9564), "Miami-Dade", "FL", boundary=_AVENTURA_BOUNDARY, procurement_url="https://www.cityofaventura.com/bids"),
    Jurisdiction("fl-sunny-isles-beach", "City of Sunny Isles Beach", JurisdictionKind.CITY, (-80.1223, 25.9290), "Miami-Dade", "FL"),
    Jurisdiction("fl-north-miami-beach", "City of North Miami Beach", JurisdictionKind.CITY, (-80.1625, 25.9331), "Miami-Dade", "FL"),
    Jurisdiction("fl-hallandale-beach", "City of Hallandale Beach", JurisdictionKind.CITY, (-80.1484, 25.9812), "Broward", "FL"),
    Jurisdiction("fl-miami-beach", "City of Miami Beach", JurisdictionKind.CITY, (-80.1300, 25.7907), "Miami-Dade", "FL"),
    Jurisdiction("fl-miami", "City of Miami", JurisdictionKind.CITY, (-80.1918, 25.7617), "Miami-Dade", "FL"),
    Jurisdiction("fl-fort-lauderdale", "City of Fort Lauderdale", JurisdictionKind.CITY, (-80.1373, 26.1224), "Broward", "FL"),
    Jurisdiction("fl-mdcps", "Miami-Dade County Public Schools", JurisdictionKind.SCHOOL_DISTRICT, (-80.2100, 25.7800), "Miami-Dade", "FL"),
    # Out of a 50-mi radius — must be filtered out (proves the geography works, not just includes everything):
    Jurisdiction("fl-key-west", "City of Key West", JurisdictionKind.CITY, (-81.7800, 24.5551), "Monroe", "FL"),
    Jurisdiction("fl-orlando", "City of Orlando", JurisdictionKind.CITY, (-81.3792, 28.5383), "Orange", "FL"),
]


def default_registry() -> JurisdictionRegistry:
    """The seeded South-Florida test registry (ZIP 33180 and neighbours)."""
    return JurisdictionRegistry(SEED_ZIPS, SEED_JURISDICTIONS)
