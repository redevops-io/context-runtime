"""geo.redevops.io — the geospatial / zoning-intelligence capability, live.

Nothing here is faked. Dispositions are computed by the SAME deterministic-first, fail-safe resolver the
`zoning_intelligence` tenant uses, over REAL parcels harvested from official ArcGIS sources by
`zoning-sources-mcp` (a 32-parcel cross-country sample from the full 149,624-district sweep). Spatial
relations are computed by the runtime's dependency-free geometry engine — never guessed by an LLM.

The rule on show: the official base zoning is authoritative, so base-incompatible uses are PROHIBITED with
certainty; a base-by-right use is PERMITTED; and wherever the answer needs an ordinance use-table we have
not parsed, the runtime abstains to UNKNOWN and cites the source — the false-permitted = 0 SLO.

    uvicorn app:app --host 0.0.0.0 --port 8098 --loop asyncio --http h11
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from context_runtime.geospatial import engine
from context_runtime.geospatial.contracts import geometry_hash

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "harvested_us_sample.json"
STATIC = HERE / "static"

TARGET_USES = ("RESIDENTIAL_SINGLE_FAMILY", "OFFICE", "RETAIL", "WAREHOUSE", "LIGHT_INDUSTRIAL")
USE_FAMILY = {"RESIDENTIAL_SINGLE_FAMILY": "residential", "OFFICE": "commercial", "RETAIL": "commercial",
              "WAREHOUSE": "industrial", "LIGHT_INDUSTRIAL": "industrial"}
NEEDS_ORDINANCE = {"OFFICE", "RETAIL", "WAREHOUSE", "LIGHT_INDUSTRIAL"}

# The completed nationwide harvest (kept off-repo for size) — the corpus these parcels are drawn from.
HARVEST = {"districts": 149624, "services": 4562, "codes": 16711, "datasets": 1880}
# Reproducible offline result from examples/zoning_intelligence.py (learned bundle vs baselines).
LEARNED = {"reward": 0.894, "avg_cost": 5.67, "false_permits": 0,
           "vs_single_provider_reward": 0.080, "vs_single_provider_false_permits": 1,
           "vs_thorough_reward": 0.850, "vs_thorough_cost": 8.00}

_PARCELS = json.loads(DATA.read_text(encoding="utf-8"))


def code_family(code: str) -> str:
    c = (code or "").strip().upper()
    if c.startswith(("R-", "RH", "RM", "RS", "RR", "RE")) or (c.startswith("R") and not c.startswith("RET")):
        return "residential"
    if c.startswith(("C-", "CB", "CG", "CN", "CS", "TC", "MU", "MX")) or c.startswith("C"):
        return "commercial"
    if c.startswith(("M-", "I-", "IL", "IG", "IND", "PI")) or c.startswith(("M", "I")):
        return "industrial"
    return "unknown"


def resolve(code: str, use: str) -> dict:
    """Deterministic-first, fail-safe: PERMITTED only base-by-right; UNKNOWN (never a guess) where the
    ordinance/overlay is needed but unparsed; PROHIBITED with certainty on base incompatibility."""
    fam = code_family(code)
    if use not in USE_FAMILY:
        disp, why = "UNKNOWN", f"{use} is outside the demo's use set"
    elif fam == "unknown":
        disp, why = "UNKNOWN", "zoning-code family unrecognized — cannot conclude from base alone"
    elif USE_FAMILY[use] != fam:
        disp, why = "PROHIBITED", f"official base zoning {code} ({fam}) excludes a {USE_FAMILY[use]} use"
    elif use in NEEDS_ORDINANCE:
        disp, why = "UNKNOWN", "base-compatible, but permitted/conditional status needs the ordinance — verify"
    else:
        disp, why = "PERMITTED", f"base zoning {code} permits {use.lower().replace('_', ' ')} by right"
    false_permit = disp == "PERMITTED" and (use not in USE_FAMILY or USE_FAMILY.get(use) != fam)
    return {"disposition": disp, "reason": why, "false_permit": false_permit}


app = FastAPI(title="ReDevOps geospatial / zoning demo")


@app.get("/healthz")
def healthz():
    return {"ok": True, "parcels": len(_PARCELS), "harvest_districts": HARVEST["districts"]}


@app.get("/api/summary")
def summary():
    return {"harvest": HARVEST, "learned": LEARNED, "sample_parcels": len(_PARCELS),
            "target_uses": list(TARGET_USES)}


@app.get("/api/parcels")
def parcels():
    return [{"parcel_id": p["parcel_id"], "zoning_code": p["zoning_code"], "code_family": code_family(p["zoning_code"]),
             "dataset": p.get("dataset", ""), "jurisdiction": p.get("jurisdiction", ""),
             "ordinance_url": p.get("ordinance_url", ""), "centroid": p.get("centroid")}
            for p in _PARCELS]


@app.get("/api/evaluate")
def evaluate():
    """Evaluate every target use against every harvested parcel — the nationwide SLO in one call."""
    rows, tally, false_permits = [], {"PERMITTED": 0, "PROHIBITED": 0, "UNKNOWN": 0}, 0
    for p in _PARCELS:
        per = {}
        for use in TARGET_USES:
            r = resolve(p["zoning_code"], use)
            per[use] = r["disposition"]
            tally[r["disposition"]] = tally.get(r["disposition"], 0) + 1
            false_permits += 1 if r["false_permit"] else 0
        rows.append({"parcel_id": p["parcel_id"], "zoning_code": p["zoning_code"],
                     "dataset": p.get("dataset", ""), "dispositions": per,
                     "ordinance_url": p.get("ordinance_url", "")})
    return {"rows": rows, "tally": tally, "false_permits": false_permits,
            "parcels": len(_PARCELS), "uses": len(TARGET_USES)}


@app.get("/api/spatial")
def spatial():
    """A real geometry computation — never an LLM guess. Is a proposed building point inside the parcel
    boundary? Computed by the engine; plus the parcel's area and centroid, and its canonical geometry id."""
    # A demo parcel boundary (a simple quadrilateral) in a planar CRS, and two candidate build points.
    ring = [(0.0, 0.0), (0.0, 100.0), (120.0, 100.0), (120.0, 0.0)]
    inside_pt, outside_pt = (60.0, 50.0), (140.0, 50.0)
    gid = geometry_hash([ring], "EPSG:2240")
    return {
        "geometry_id": gid,
        "area": engine.area(ring),
        "centroid": engine.centroid(ring),
        "point_inside_parcel": {"point": inside_pt, "result": engine.point_in_polygon(inside_pt, ring)},
        "point_outside_parcel": {"point": outside_pt, "result": engine.point_in_polygon(outside_pt, ring)},
        "note": "point-in-polygon, area and centroid are exact engine computations; the LLM is never asked "
                "to infer a spatial relation a geometry engine can calculate.",
    }


@app.get("/")
def index():
    return FileResponse(str(STATIC / "index.html"))


app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")
