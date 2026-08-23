"""Geospatial contracts (plan §2, §8-10): typed spatial evidence + the zoning/parcel domain model.

`GeoRef` is the typed geospatial reference that rides alongside evidence — it does NOT replace EvidenceRef,
it extends the evidence-native model with geometry identity, CRS, jurisdiction and spatiotemporal state.
Geometry has a canonical hash so the same parcel boundary is one identity across providers, and reprojection
is explicit and auditable (never a silent CRS mismatch).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum

Coord = tuple[float, float]                    # (lon, lat) or (x, y) in the ring's CRS
Ring = list[Coord]                             # a closed ring (first==last not required; we close it)


# ──────────────────────────── spatial operations (the capabilities) ────────────────────────────

class SpatialOp(str, Enum):
    POINT_IN_POLYGON = "POINT_IN_POLYGON"
    INTERSECTS = "INTERSECTS"
    WITHIN = "WITHIN"
    CONTAINS = "CONTAINS"
    OVERLAPS = "OVERLAPS"
    TOUCHES = "TOUCHES"
    DISTANCE = "DISTANCE"
    NEAREST = "NEAREST"
    BUFFER = "BUFFER"
    CENTROID = "CENTROID"
    SPATIAL_JOIN = "SPATIAL_JOIN"
    BBOX_QUERY = "BBOX_QUERY"
    REPROJECT = "REPROJECT"
    AREA = "AREA"


# ──────────────────────────── typed spatial evidence ────────────────────────────

def geometry_hash(rings: list[Ring], crs: str) -> str:
    """Canonical, stable identity for a geometry — rounded coords + CRS, hashed. Same boundary in the same
    CRS → same id across providers, so parcel identity is not provider-specific."""
    # normalise to float so an int coord (0) and its float twin (0.0) hash identically
    norm = [[[round(float(x), 7), round(float(y), 7)] for x, y in ring] for ring in rings]
    payload = json.dumps({"crs": crs, "rings": norm}, sort_keys=True, separators=(",", ":"))
    return "geo:" + hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()


@dataclass(frozen=True)
class GeoRef:
    """A typed geospatial reference carried with evidence (plan §2). Extends, not replaces, EvidenceRef."""
    geometry_type: str                         # "Polygon" | "Point" | "MultiPolygon"
    geometry_hash: str                         # canonical hashable identity (from geometry_hash())
    crs: str                                   # explicit CRS/SRID, e.g. "EPSG:4326"
    bbox: tuple[float, float, float, float]    # (minx, miny, maxx, maxy)
    centroid: Coord
    jurisdiction: str = ""
    spatial_resolution: float | None = None
    horizontal_accuracy: float | None = None
    valid_time: str = ""                       # world-time validity (ISO); "" = -inf
    observed_at: str = ""                      # when observed/recorded (transaction time)
    source: str = ""
    source_version: str = ""


# ──────────────────────────── zoning / parcel domain (plan §8-10) ────────────────────────────

class UseDisposition(str, Enum):
    """How a use maps against a parcel's rules — never forced into a binary (plan §9)."""
    PERMITTED = "PERMITTED"
    CONDITIONAL = "CONDITIONAL"
    SPECIAL_EXCEPTION = "SPECIAL_EXCEPTION"
    PROHIBITED = "PROHIBITED"
    UNKNOWN = "UNKNOWN"


# Normalized use ontology (plan §9) — a starter set; jurisdictions map their local rules onto these.
USE_ONTOLOGY: tuple[str, ...] = (
    "RESIDENTIAL_SINGLE_FAMILY", "RESIDENTIAL_MULTI_FAMILY", "MIXED_USE", "OFFICE", "RETAIL", "RESTAURANT",
    "HOTEL", "LIGHT_INDUSTRIAL", "HEAVY_INDUSTRIAL", "WAREHOUSE", "LOGISTICS", "AGRICULTURE", "SELF_STORAGE",
    "DATA_CENTER", "MEDICAL", "EDUCATION", "RELIGIOUS", "ENTERTAINMENT", "PARKING", "SOLAR", "EV_CHARGING",
)


class ConstraintType(str, Enum):
    MIN_LOT_AREA = "MIN_LOT_AREA"
    MIN_LOT_WIDTH = "MIN_LOT_WIDTH"
    MAX_HEIGHT = "MAX_HEIGHT"
    MAX_FAR = "MAX_FAR"
    MAX_LOT_COVERAGE = "MAX_LOT_COVERAGE"
    FRONT_SETBACK = "FRONT_SETBACK"
    SIDE_SETBACK = "SIDE_SETBACK"
    REAR_SETBACK = "REAR_SETBACK"
    MAX_DENSITY = "MAX_DENSITY"
    PARKING_MINIMUM = "PARKING_MINIMUM"
    FLOOD_RESTRICTION = "FLOOD_RESTRICTION"
    OVERLAY_RESTRICTION = "OVERLAY_RESTRICTION"


@dataclass(frozen=True)
class LandUseConstraint:
    """A development control tested against a project spec — not just a zoning label (plan §10)."""
    type: ConstraintType
    value: float | str
    unit: str = ""
    operator: str = ">="                       # ">=", "<=", "==" …
    source_evidence: str = ""
    confidence: float = 1.0


@dataclass
class ParcelEntity:
    """Canonical parcel with provenance kept visible — the canonical value is a conclusion; provider records
    remain evidence (plan §8). Provider disagreement is never flattened away."""
    parcel_id: str                             # canonical (geometry_hash-based) id
    geometry_ref: GeoRef
    apn: str = ""
    address: str = ""
    jurisdiction: str = ""
    lot_area: float | None = None              # in the parcel's CRS/area units
    provider_ids: dict[str, str] = field(default_factory=dict)      # provider -> their record id
    zoning_codes: list[str] = field(default_factory=list)          # raw codes seen across providers
    zoning_districts: list[str] = field(default_factory=list)
    overlays: list[str] = field(default_factory=list)
    constraints: list[LandUseConstraint] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
