"""Benchmark arms. Python is the SEMANTIC REFERENCE (unmodified production engine); Shapely is an
optional reality-check baseline (§4 Arm C). Each arm exposes the same three operations over the corpus
data model so the runner + correctness gate treat them uniformly."""
from __future__ import annotations

import os
import sys

# import the unmodified production engine (the reference implementation, §2)
_CTX = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _CTX not in sys.path:
    sys.path.insert(0, _CTX)
from context_runtime.geospatial import engine as _eng  # noqa: E402

Ring = list  # list[[x,y]]


class PythonArm:
    name = "python"
    available = True

    def point_in_polygon(self, pt, ring, holes=None):
        return _eng.point_in_polygon(tuple(pt), [tuple(p) for p in ring],
                                     [[tuple(p) for p in h] for h in holes] if holes else None)

    def polygons_intersect(self, a, b):
        return _eng.polygons_intersect([tuple(p) for p in a], [tuple(p) for p in b])

    def spatial_join(self, parcels, zones):
        p = [(pid, [tuple(v) for v in ring], crs) for pid, ring, crs in parcels]
        z = [(zid, [tuple(v) for v in ring], crs) for zid, ring, crs in zones]
        return _eng.spatial_join_by_centroid(p, z)


class ShapelyArm:
    """Reality-check only — Shapely's boundary/touch semantics differ from the reference epsilon rules,
    so it is NOT a parity target (§8). Reported for context (§4)."""
    name = "shapely"

    def __init__(self):
        try:
            import shapely  # noqa: F401
            from shapely.geometry import Polygon, Point  # noqa: F401
            self.available = True
            self._Polygon, self._Point = Polygon, Point
        except Exception:
            self.available = False

    def _poly(self, ring, holes=None):
        return self._Polygon([tuple(p) for p in ring],
                             [[tuple(p) for p in h] for h in holes] if holes else None)

    def point_in_polygon(self, pt, ring, holes=None):
        poly = self._poly(ring, holes)
        p = self._Point(tuple(pt))
        return bool(poly.covers(p))  # covers = interior OR boundary (closest to the reference tie-break)

    def polygons_intersect(self, a, b):
        return bool(self._poly(a).intersects(self._poly(b)))

    def spatial_join(self, parcels, zones):
        zpolys = [(zid, self._poly(ring)) for zid, ring, _ in zones]
        out = {}
        for pid, ring, _ in parcels:
            c = self._poly(ring).centroid
            out[pid] = [zid for zid, zp in zpolys if zp.covers(c)]
        return out


# Batch interface (uniform across arms). Python/Shapely loop in-process; Mojo runs one subprocess per
# batch (see integration/mojo_bridge.py). run_all times these = the integrated experience (§5.2).
def _add_batch(cls):
    def pip_batch(self, items):
        return [self.point_in_polygon(it["point"], it["ring"]) for it in items]
    def pairs_batch(self, items):
        return [self.polygons_intersect(it["a"], it["b"]) for it in items]
    def join_batch(self, items):
        return [self.spatial_join(it["parcels"], it["zones"]) for it in items]
    cls.pip_batch, cls.pairs_batch, cls.join_batch = pip_batch, pairs_batch, join_batch
    cls.last_kernel_ns = None
    return cls


_add_batch(PythonArm)
_add_batch(ShapelyArm)


def get_arm(name: str):
    if name == "python":
        return PythonArm()
    if name == "shapely":
        return ShapelyArm()
    raise ValueError(f"unknown arm {name}")
