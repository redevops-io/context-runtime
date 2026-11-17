"""geo.redevops.io — the geospatial / zoning-intelligence capability, live.

Nothing here is faked. Dispositions are computed by the SAME deterministic-first, fail-safe resolver the
`zoning_intelligence` tenant uses, over REAL parcels harvested from official ArcGIS sources by
`zoning-sources-mcp` (a 32-parcel cross-country sample from the full 149,624-district sweep). Spatial
relations are computed by the runtime's dependency-free geometry engine — never guessed by an LLM.

The rule on show: the official base zoning is authoritative, so base-incompatible uses are PROHIBITED with
certainty; a base-by-right use is PERMITTED; and wherever the answer needs an ordinance use-table we have
not parsed, the runtime abstains to UNKNOWN and cites the source — the false-permitted = 0 SLO.

    uvicorn app:app --host 0.0.0.0 --port 8098 --loop asyncio --http h11

URL prefix: by default the app serves at the root of its origin. Set PUBLIC_BASE_PATH (for example `/geo`) to serve
the page, the API, the static files, the interactive docs and the health endpoint under that prefix instead, with no
proxy-side rewrite: every route is registered under the prefix, and nothing is served outside it.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
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

# --- URL prefix ---------------------------------------------------------------------------------------------------
ENV_BASE_PATH = "PUBLIC_BASE_PATH"
MAX_BASE_PATH_LENGTH = 128
_SEGMENT = re.compile(r"[A-Za-z0-9._~-]+")      # RFC 3986 unreserved characters only: nothing needs escaping


class InvalidBasePath(ValueError):
    """PUBLIC_BASE_PATH is not usable as a URL path prefix; the app refuses to start rather than guess."""


def validate_base_path(raw: str) -> str:
    """Return `raw` unchanged if it is a valid prefix, else raise InvalidBasePath.

    Valid: the empty string (serve at the root), or `/seg[/seg...]` with no trailing slash, no empty, `.` or `..`
    segment, and only unreserved URL characters (letters, digits, `.`, `_`, `~`, `-`). The value is not trimmed or
    normalised, so what is configured is exactly what is served.
    """
    if not isinstance(raw, str):
        raise InvalidBasePath(f"{ENV_BASE_PATH} must be a string")
    if raw == "":
        return ""
    if len(raw) > MAX_BASE_PATH_LENGTH:
        raise InvalidBasePath(f"{ENV_BASE_PATH} is longer than {MAX_BASE_PATH_LENGTH} characters")
    if not raw.startswith("/"):
        raise InvalidBasePath(f"{ENV_BASE_PATH} must be empty or start with '/'")
    if raw.endswith("/"):
        raise InvalidBasePath(f"{ENV_BASE_PATH} must not end with '/'")
    if ".." in raw:
        raise InvalidBasePath(f"{ENV_BASE_PATH} must not contain '..'")
    for seg in raw[1:].split("/"):
        if seg == "" or seg == ".":
            raise InvalidBasePath(f"{ENV_BASE_PATH} must not contain an empty or '.' path segment")
        if not _SEGMENT.fullmatch(seg):
            raise InvalidBasePath(f"{ENV_BASE_PATH} may only contain letters, digits and . _ ~ - within a segment")
    return raw


def configured_base_path() -> str:
    return validate_base_path(os.environ.get(ENV_BASE_PATH, ""))


# The page is a template: the two placeholders are filled once per app instance from the validated prefix.
_PAGE_TEMPLATE = (STATIC / "index.html").read_text(encoding="utf-8")
_PLACEHOLDERS = ("__BASE_PATH_JSON__", "__SITE_ROOT__")


def render_page(base_path: str) -> str:
    """The page with its URL prefix filled in. Every URL the page requests is built from `BASE` (the prefix as a
    JSON string); the one link that points at the host's root rather than at this app is relative to the prefix
    depth (`../` per segment), or the plain root path when there is no prefix."""
    site_root = "../" * len(base_path.split("/")[1:]) if base_path else "/"
    page = _PAGE_TEMPLATE.replace("__BASE_PATH_JSON__", json.dumps(base_path)).replace("__SITE_ROOT__", site_root)
    assert not any(ph in page for ph in _PLACEHOLDERS)
    return page


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


router = APIRouter()


@router.get("/healthz")
def healthz():
    return {"ok": True, "parcels": len(_PARCELS), "harvest_districts": HARVEST["districts"]}


@router.get("/api/summary")
def summary():
    return {"harvest": HARVEST, "learned": LEARNED, "sample_parcels": len(_PARCELS),
            "target_uses": list(TARGET_USES)}


@router.get("/api/parcels")
def parcels():
    return [{"parcel_id": p["parcel_id"], "zoning_code": p["zoning_code"], "code_family": code_family(p["zoning_code"]),
             "dataset": p.get("dataset", ""), "jurisdiction": p.get("jurisdiction", ""),
             "ordinance_url": p.get("ordinance_url", ""), "centroid": p.get("centroid")}
            for p in _PARCELS]


@router.get("/api/evaluate")
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


@router.get("/api/spatial")
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


def create_app(base_path: str | None = None) -> FastAPI:
    """Build the app. `base_path=None` reads PUBLIC_BASE_PATH (empty = serve at the root, as before).

    With a prefix, every route (page, API, health, docs, static files) exists only under it; any other path is a 404,
    `<prefix>` redirects to `<prefix>/`, and the redirect target keeps the prefix. `/healthz` is then served at
    `<prefix>/healthz` only, so a probe must use the prefixed path."""
    base = configured_base_path() if base_path is None else validate_base_path(base_path)
    app = FastAPI(title="ReDevOps geospatial / zoning demo",
                  docs_url=f"{base}/docs", redoc_url=f"{base}/redoc", openapi_url=f"{base}/openapi.json",
                  swagger_ui_oauth2_redirect_url=f"{base}/docs/oauth2-redirect")
    app.include_router(router, prefix=base)
    page = render_page(base)

    def index():
        return HTMLResponse(page)

    app.add_api_route(f"{base}/", index, methods=["GET"], response_class=HTMLResponse)
    app.add_api_route(f"{base}/index.html", index, methods=["GET"], response_class=HTMLResponse,
                      include_in_schema=False)
    if base:
        # The bare prefix needs an explicit, path-only redirect: the framework's own slash redirect builds an absolute
        # URL from the request's Host header, which behind a proxy can be the internal origin. (Everything below the
        # prefix is handled by the static mount registered last, so no other path reaches that automatic redirect.)
        def to_prefix_root(request: Request):
            q = request.url.query
            return RedirectResponse(f"{base}/" + (f"?{q}" if q else ""), status_code=307)

        app.add_api_route(base, to_prefix_root, methods=["GET"], include_in_schema=False)
    app.mount(base or "/", StaticFiles(directory=str(STATIC), html=True), name="static")
    return app


app = create_app()
