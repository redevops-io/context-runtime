# Changelog

## 0.3.x — geospatial capabilities + zoning intelligence (unreleased)

Geospatial support as a **Runtime capability, not a separate Geospatial Runtime**: a parcel's geometry,
CRS, jurisdiction and temporal state become typed evidence, and spatial operations become capabilities
the planner selects, composes, verifies and governs. Fully additive and offline; no existing path changes.

**What is now true**

- **Typed spatial evidence** (`context_runtime.geospatial.contracts`): `GeoRef` (geometry identity via a
  canonical, hashable `geometry_hash`; explicit CRS; jurisdiction; valid/observed time) extends — never
  replaces — the evidence-native model. Domain vocabulary: `ParcelEntity`, `UseDisposition`
  (PERMITTED / CONDITIONAL / SPECIAL_EXCEPTION / PROHIBITED / **UNKNOWN**), `LandUseConstraint`, a
  normalized use ontology.
- **CPU-authoritative spatial engine** (`context_runtime.geospatial.engine`): pure-Python, exact,
  dependency-free point-in-polygon, polygon intersection, area, centroid, distance, and centroid spatial
  join. Binary ops **refuse a silent CRS mismatch** (raise so the planner inserts an explicit REPROJECT).
  Heavier backends (shapely/PostGIS/DuckDB-Spatial/GPU) are planner-selectable on measured crossover —
  the accelerator changes latency, never the answer.
- **Zoning-intelligence tenant** (`context_runtime.integrations.zoning_intelligence` +
  `examples/zoning_intelligence.py`): the geospatial reference benchmark. The Runtime learns *which
  evidence to acquire* (Regrid · ATTOM · official GIS · ordinance) per use-difficulty bucket. A
  deterministic-first, fail-safe resolver reconciles evidence and only concludes PERMITTED when the
  evidence is sufficient — making **false-permitted a structural impossibility** for reconciled bundles
  (the blocking SLO). Measured: learned **+0.894 reward at 1.4× lower evidence cost** than
  always-thorough, **0 false-permits vs 1** for single-provider. Includes use-first land search,
  dependency-scoped incremental recomputation, and population-governance drift detection.

## 0.2.0 — freshness sourced from evidence (v0.2.x final stabilization)

Correctness/completeness fix to functionality v0.2.x already advertised. The freshness *scoring*, the
`REFRESH` verdict, and the EXPLAIN lineage *renderer* shipped in Slice 5 — but freshness was never
**sourced** from the evidence being evaluated, the REFRESH gate was off the normal serving path, and
the EXPLAIN lineage was not populated by any production path. This release wires all three end-to-end.
Entirely opt-in: with no `FreshnessPolicy`, freshness is `1.0` and every path is byte-for-byte the
legacy one.

**What is now true**

- **`Hit` carries evidence identity** — `version`, `content_hash`, `observed_at` flow from the
  retriever. The redevops-rag binding (`store_redevops`) supplies them from the canonical `EvidenceRef`
  that redevops-rag 0.2.1 now emits; `store_inmemory` carries them when the corpus provides them.
- **Freshness derived from evidence** (`context_runtime.freshness`): `FreshnessPolicy`
  (`age_decay` | `ttl`) turns a hit's `observed_at` (vs an `as_of` reference) into a `[0,1]` score;
  `score_hits` aggregates (worst-evidence governs). Unknown timestamps → `1.0` (never penalized).
- **REFRESH on the normal serving path** — `ContextRuntime(..., freshness_policy=...)` computes
  freshness from the retrieved evidence in `run()`/`execute()`, records it on `PlanScore.freshness` and
  `RunResult.freshness`, and — when the evidence is staler than `min_freshness` — declines to serve,
  returning `RunResult(refresh=True, refresh_reason=...)` before the reason step. No longer a
  test-only `BanditOptimizer` configuration.
- **EXPLAIN populated, not merely rendered** — `freshness.lineage_from_hits(...)` /
  `explain_hit_row(...)` build the lineage from the actual served evidence, so EXPLAIN names the exact
  source ref / version / content-hash / freshness used.

**Backward compatibility** — `Hit`/`RunResult` gained fields with defaults; `run`/`execute` gained an
optional `as_of`. With no `FreshnessPolicy` the runtime behaves identically to 0.2.0 (verified). New
tests: `tests/test_freshness_sourced.py` (the Slice-5 primitive tests remain in
`tests/test_freshness_slice5.py`).
