"""CPU-authoritative deterministic spatial engine (plan §2, §4) — the reference implementation.

Pure-Python, exact, dependency-free geometry: point-in-polygon (ray casting), polygon intersection (bbox
prefilter + vertex containment + edge crossing), area (shoelace), distance, centroid, bbox, spatial join.
This is the authoritative path — an optional shapely/PostGIS/GPU backend may be selected for speed later,
but must return the same relations (the accelerator changes latency, not the answer).

Deterministic-first rules the plan insists on:
  * **No silent CRS mismatch.** Binary operations require operands in the same CRS; otherwise a CrsMismatch
    is raised so the planner inserts an explicit, auditable REPROJECT node (a blocking-SLO invariant).
  * Geometry is assumed to be in a **planar CRS (metres/feet)** for area/distance; geographic lon/lat must be
    reprojected first (the REPROJECT capability). Point-in-polygon is CRS-agnostic (topological).
"""
from __future__ import annotations

from .contracts import Coord, Ring


class CrsMismatch(ValueError):
    """A binary spatial op was asked to combine operands in different CRSs — never done silently."""


def require_same_crs(a_crs: str, b_crs: str) -> None:
    if a_crs != b_crs:
        raise CrsMismatch(f"CRS mismatch: {a_crs} vs {b_crs} — insert a REPROJECT node before this op")


# ──────────────────────────── primitives ────────────────────────────

def bbox(ring: Ring) -> tuple[float, float, float, float]:
    xs = [x for x, _ in ring]
    ys = [y for _, y in ring]
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def signed_area(ring: Ring) -> float:
    """Shoelace signed area (planar CRS units²). Positive = counter-clockwise."""
    s = 0.0
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def area(ring: Ring) -> float:
    return abs(signed_area(ring))


def centroid(ring: Ring) -> Coord:
    """Polygon centroid (planar). Falls back to vertex mean for degenerate/zero-area rings."""
    a = signed_area(ring)
    if abs(a) < 1e-12:
        n = len(ring) or 1
        return (sum(x for x, _ in ring) / n, sum(y for _, y in ring) / n)
    cx = cy = 0.0
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    return (cx / (6 * a), cy / (6 * a))


def distance(a: Coord, b: Coord) -> float:
    """Euclidean distance in planar CRS units (metres/feet). Geographic coords must be reprojected first."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


# ──────────────────────────── point-in-polygon (ray casting; CRS-agnostic) ────────────────────────────

def point_in_ring(pt: Coord, ring: Ring) -> bool:
    """True if pt is strictly inside or on the boundary of the ring (even-odd ray casting)."""
    x, y = pt
    inside = False
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        # boundary check (on an edge counts as inside — deterministic tie-break)
        if min(y1, y2) <= y <= max(y1, y2) and min(x1, x2) - 1e-12 <= x <= max(x1, x2) + 1e-12:
            if abs((x2 - x1) * (y - y1) - (x - x1) * (y2 - y1)) < 1e-9:
                return True
        if (y1 > y) != (y2 > y):
            xint = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xint:
                inside = not inside
    return inside


def point_in_polygon(pt: Coord, outer: Ring, holes: list[Ring] | None = None) -> bool:
    if not point_in_ring(pt, outer):
        return False
    for h in (holes or []):
        if point_in_ring(pt, h):
            return False
    return True


# ──────────────────────────── segment / polygon intersection ────────────────────────────

def _seg_cross(p1: Coord, p2: Coord, p3: Coord, p4: Coord) -> bool:
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return (v > 1e-12) - (v < -1e-12)
    d1, d2 = orient(p3, p4, p1), orient(p3, p4, p2)
    d3, d4 = orient(p1, p2, p3), orient(p1, p2, p4)
    return d1 != d2 and d3 != d4


def polygons_intersect(a: Ring, b: Ring) -> bool:
    """Deterministic polygon-polygon intersection: bbox prefilter, then containment either way, then any
    edge crossing. Sufficient for parcel↔zoning overlap on simple (non-self-intersecting) rings."""
    if not bbox_intersects(bbox(a), bbox(b)):
        return False
    if any(point_in_ring(p, b) for p in a) or any(point_in_ring(p, a) for p in b):
        return True
    na, nb = len(a), len(b)
    for i in range(na):
        for j in range(nb):
            if _seg_cross(a[i], a[(i + 1) % na], b[j], b[(j + 1) % nb]):
                return True
    return False


# ──────────────────────────── spatial join (the workhorse for zoning assignment) ────────────────────────────

def spatial_join_by_centroid(parcels: list[tuple[str, Ring, str]],
                             zones: list[tuple[str, Ring, str]]) -> dict[str, list[str]]:
    """Assign each parcel to the zoning polygons whose area contains the parcel's centroid. Operands must
    share a CRS (checked). Deterministic. parcels/zones are (id, ring, crs). Returns parcel_id -> [zone_id]."""
    out: dict[str, list[str]] = {}
    for pid, pring, pcrs in parcels:
        c = centroid(pring)
        hits = []
        for zid, zring, zcrs in zones:
            require_same_crs(pcrs, zcrs)
            if point_in_polygon(c, zring):
                hits.append(zid)
        out[pid] = hits
    return out
