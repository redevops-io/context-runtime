"""Geospatial capabilities (v0.3.x) — typed spatial evidence + planner-selectable spatial operations.

Geospatial support is a **Runtime capability, not a separate Geospatial Runtime**: a parcel's geometry,
CRS, jurisdiction and temporal state become typed evidence (`GeoRef`), and spatial operations
(point-in-polygon, intersects, spatial-join, distance, …) become capabilities the planner selects,
composes, verifies and governs. The governing rule:

    Do not ask an LLM to infer a spatial relationship a deterministic geometry engine can calculate.

The reference engine is pure-Python and deterministic (`engine.py`); heavier backends (shapely/GeoPandas,
PostGIS, DuckDB-Spatial, a GPU implementation) are optional and planner-selectable on measured crossover,
exactly like the cuVS/cuDF accelerators — the *result* never changes, only how fast it is produced.

Reference workload: **zoning intelligence** (`integrations/zoning_intelligence.py`) — parcel-first ("what
can I build here?") and use-first ("find compatible parcels"), reconciling parcel/zoning/ordinance
evidence across providers while preserving provenance, uncertainty and review requirements.
"""
from .contracts import (  # noqa: F401
    GeoRef, SpatialOp, UseDisposition, ConstraintType, ParcelEntity, LandUseConstraint, USE_ONTOLOGY,
)
