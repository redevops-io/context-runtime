# Harvested zoning data — provenance

These fixtures are **real** parcels/districts harvested from OFFICIAL keyless sources by the
`zoning-sources-mcp` server (US Census/FCC jurisdiction resolution → ArcGIS Hub discovery → ArcGIS REST
query against each jurisdiction's own published zoning layer). No API-provider data; no keys.

- `harvested_brisbane_ca.json` — 3 parcels from Brisbane, CA's published zoning layer.
- `harvested_us_sample.json` — a 32-parcel, cross-country sample (one parcel per distinct dataset:
  Phoenix, Tucson, LA County, San Diego, SF, Oakland, Denver, Miami, Las Vegas, Charlotte, Raleigh,
  Cincinnati, Columbus, Cleveland, Philadelphia, San Antonio, Salt Lake, Soldotna, …).

Drawn from the full nationwide sweep (kept off-repo for size):

> **DONE: 149,624 districts from 4,562 official ArcGIS services, 16,711 distinct zoning codes,
> 1,880 datasets** — the ArcGIS Hub `zoning` index swept to end-of-index (page 103).

Each record carries the canonical `geometry_hash` id (`parcel_id`), CRS, bbox/centroid, the real zoning
code, and — where published — a link to the district's ordinance page. `examples/zoning_real_world.py
nationwide` runs the deterministic-first, fail-safe tenant over the sample: **false-permits = 0**, with
honest `UNKNOWN` abstention wherever the answer depends on an ordinance use-table we have not parsed.
