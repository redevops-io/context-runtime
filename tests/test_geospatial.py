"""Geospatial engine + contracts — deterministic CPU spatial operations."""
from __future__ import annotations

import pytest

from context_runtime.geospatial import contracts as C
from context_runtime.geospatial import engine as E

# a unit square [0,10]x[0,10] and a smaller one overlapping it
SQ = [(0, 0), (10, 0), (10, 10), (0, 10)]
SQ2 = [(5, 5), (15, 5), (15, 15), (5, 15)]
FAR = [(100, 100), (110, 100), (110, 110), (100, 110)]


def test_point_in_polygon():
    assert E.point_in_polygon((5, 5), SQ) is True
    assert E.point_in_polygon((0, 0), SQ) is True        # boundary counts as inside (deterministic)
    assert E.point_in_polygon((11, 5), SQ) is False


def test_point_in_polygon_with_hole():
    hole = [(3, 3), (7, 3), (7, 7), (3, 7)]
    assert E.point_in_polygon((1, 1), SQ, [hole]) is True
    assert E.point_in_polygon((5, 5), SQ, [hole]) is False    # inside the hole → outside the polygon


def test_area_and_centroid():
    assert E.area(SQ) == pytest.approx(100.0)                 # 10x10 planar square
    assert E.centroid(SQ) == pytest.approx((5.0, 5.0))


def test_distance():
    assert E.distance((0, 0), (3, 4)) == pytest.approx(5.0)


def test_polygons_intersect():
    assert E.polygons_intersect(SQ, SQ2) is True             # overlapping corners
    assert E.polygons_intersect(SQ, FAR) is False            # bbox-disjoint
    # containment (one inside the other) is an intersection
    inner = [(2, 2), (4, 2), (4, 4), (2, 4)]
    assert E.polygons_intersect(SQ, inner) is True


def test_spatial_join_by_centroid():
    parcels = [("p1", [(1, 1), (2, 1), (2, 2), (1, 2)], "EPSG:2240"),   # centroid (1.5,1.5) ∈ SQ
               ("p2", [(20, 20), (21, 20), (21, 21), (20, 21)], "EPSG:2240")]  # far away
    zones = [("R1", SQ, "EPSG:2240")]
    out = E.spatial_join_by_centroid(parcels, zones)
    assert out["p1"] == ["R1"] and out["p2"] == []


def test_crs_mismatch_is_never_silent():
    parcels = [("p1", [(1, 1), (2, 1), (2, 2)], "EPSG:4326")]
    zones = [("R1", SQ, "EPSG:2240")]
    with pytest.raises(E.CrsMismatch):
        E.spatial_join_by_centroid(parcels, zones)


def test_geometry_hash_is_canonical_and_stable():
    h1 = C.geometry_hash([SQ], "EPSG:2240")
    h2 = C.geometry_hash([[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]], "EPSG:2240")
    assert h1 == h2 and h1.startswith("geo:")                # same boundary+CRS → one identity
    assert C.geometry_hash([SQ], "EPSG:4326") != h1          # CRS is part of identity


def test_georef_and_parcel_construct():
    ref = C.GeoRef(geometry_type="Polygon", geometry_hash=C.geometry_hash([SQ], "EPSG:2240"),
                   crs="EPSG:2240", bbox=E.bbox(SQ), centroid=E.centroid(SQ), jurisdiction="Fulton County")
    parcel = C.ParcelEntity(parcel_id=ref.geometry_hash, geometry_ref=ref, lot_area=E.area(SQ))
    assert parcel.lot_area == pytest.approx(100.0)
    assert parcel.geometry_ref.crs == "EPSG:2240"
