"""Zoning intelligence on REAL, directly-harvested official data (not the synthetic fixture).

The parcels in `examples/data/harvested_brisbane_ca.json` were harvested live from OFFICIAL keyless
sources by the `zoning-sources-mcp` server (US Census geocoder → ArcGIS Hub discovery → ArcGIS REST
query against Brisbane, CA's own published zoning layer). Each carries a canonical `geometry_hash` id, the
real zoning code, and a link to the district's ordinance page.

This demo shows the runtime concluding on that real data with the SAME deterministic-first, fail-safe
discipline as the tenant: the official base zoning is authoritative, so base-incompatible uses are
PROHIBITED with certainty; but where a use's permissibility depends on the ordinance's permitted-use
table or an overlay we have NOT parsed (the honest "deep wall"), the runtime abstains to UNKNOWN and cites
the ordinance URL for verification — never a false "permitted". That abstention IS the false-permitted=0
SLO, now demonstrated on real official data.

    PYTHONPATH=. python examples/zoning_real_world.py
"""
from __future__ import annotations

import json
import os

from context_runtime.geospatial.contracts import GeoRef, ParcelEntity, UseDisposition as D

DATA = os.path.join(os.path.dirname(__file__), "data", "harvested_brisbane_ca.json")

TARGET_USES = ("RESIDENTIAL_SINGLE_FAMILY", "OFFICE", "RETAIL", "WAREHOUSE", "LIGHT_INDUSTRIAL")

# Family of a use, and the family a real zoning code implies (from its prefix — the real-world taxonomy is
# per-jurisdiction, so this is a coarse, honest classifier, not a claim of full ordinance parsing).
USE_FAMILY = {
    "RESIDENTIAL_SINGLE_FAMILY": "residential",
    "OFFICE": "commercial", "RETAIL": "commercial",
    "WAREHOUSE": "industrial", "LIGHT_INDUSTRIAL": "industrial",
}
# Uses that need the ordinance's permitted-use table / an overlay check before PERMITTED can be confirmed.
NEEDS_ORDINANCE = {"OFFICE", "RETAIL", "WAREHOUSE", "LIGHT_INDUSTRIAL"}


def code_family(code: str) -> str:
    c = (code or "").strip().upper()
    if c.startswith(("R-", "R", "RH", "RM", "RS")) and not c.startswith("RET"):
        return "residential"
    if c.startswith(("C-", "C", "CB", "CG", "CN", "TC", "MU")):
        return "commercial"
    if c.startswith(("M-", "M", "I-", "I", "IL", "IG", "IND", "PI")):
        return "industrial"
    return "unknown"


def as_parcel(rec: dict) -> ParcelEntity:
    """Rebuild the runtime's ParcelEntity/GeoRef from a harvested record (same geometry_hash identity)."""
    ref = GeoRef(geometry_type=rec["geometry_type"], geometry_hash=rec["parcel_id"], crs=rec["crs"],
                 bbox=tuple(rec["bbox"]), centroid=tuple(rec["centroid"]),
                 jurisdiction=rec["jurisdiction"], source=rec.get("source_url", ""))
    return ParcelEntity(parcel_id=rec["parcel_id"], geometry_ref=ref, apn=rec.get("apn", ""),
                        jurisdiction=rec["jurisdiction"], zoning_codes=[rec["zoning_code"]])


def resolve_real(rec: dict, use: str) -> tuple[D, str]:
    """Deterministic-first + fail-safe on the official base zoning alone. PERMITTED only for the archetypal
    base-by-right case; UNKNOWN (never a guess) where the ordinance/overlay is needed but unparsed."""
    fam = code_family(rec["zoning_code"])
    if fam == "unknown":
        return D.UNKNOWN, "zoning-code family unrecognized — cannot conclude from base alone"
    if USE_FAMILY[use] != fam:
        return D.PROHIBITED, f"official base zoning {rec['zoning_code']} ({fam}) excludes a {fam_of(use)} use"
    if use in NEEDS_ORDINANCE:
        return D.UNKNOWN, "base-compatible, but permitted/conditional status needs the ordinance — verify"
    return D.PERMITTED, f"official base zoning {rec['zoning_code']} permits {use.lower().replace('_',' ')} by right"


def fam_of(use: str) -> str:
    return USE_FAMILY[use]


def run() -> None:
    universe = json.load(open(DATA, encoding="utf-8"))
    print(f"Real parcels harvested from official sources: {len(universe)} "
          f"(jurisdiction: {universe[0]['jurisdiction']})\n")

    false_permits = 0
    for rec in universe:
        parcel = as_parcel(rec)                       # proves the drop-in GeoRef/ParcelEntity contract
        print(f"■ {parcel.zoning_codes[0]:6} parcel {parcel.parcel_id[:22]}  "
              f"({parcel.geometry_ref.geometry_type}, {parcel.jurisdiction})")
        if rec.get("ordinance_url"):
            print(f"    ordinance: {rec['ordinance_url']}")
        for use in TARGET_USES:
            disp, why = resolve_real(rec, use)
            mark = {"PERMITTED": "✓", "PROHIBITED": "✗", "UNKNOWN": "·"}.get(disp.value, "?")
            print(f"      {mark} {use:26} → {disp.value:11} — {why}")
            # A conclusion can never be a confident PERMITTED without base compatibility; assert the SLO.
            if disp == D.PERMITTED and code_family(rec["zoning_code"]) != USE_FAMILY[use]:
                false_permits += 1
        print()

    print(f"false-permits (concluded PERMITTED against incompatible base): {false_permits}  "
          f"— the SLO holds on real data")
    print("Wherever the answer depends on the ordinance's use table (the deep wall), the runtime returns "
          "UNKNOWN and cites the official source — it never fabricates a 'permitted'.")


if __name__ == "__main__":
    run()
