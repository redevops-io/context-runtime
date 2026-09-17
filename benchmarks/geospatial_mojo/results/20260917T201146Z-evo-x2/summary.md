# Geospatial Python vs Mojo — dev run

- Linux-7.0.0-31-generic-x86_64-with-glibc2.43 · Python 3.14.4 · Mojo Mojo 1.1.0 (8189361e) · commit `fe040f1324`
- Mojo clean optimized build (T6): 0.835s

## Correctness (vs Python reference)

| Arm | General | Adversarial | Mismatches |
|---|---:|---:|---:|
| shapely | 1485 | 12 | 20 |
| mojo | 1485 | 12 | 0 |

## End-to-end performance (median ms; speedup = python/other)

| Workload | Case | python | shapely | mojo | mojo speedup |
|---|---|---:|---:|---:|---:|
| point_in_polygon | v4 | 0.116 | 1.977 | 7.628 | 0.02× |
| point_in_polygon | v16 | 0.295 | 2.474 | 9.929 | 0.03× |
| point_in_polygon | v64 | 0.983 | 4.137 | 18.223 | 0.05× |
| point_in_polygon | v256 | 3.547 | 10.883 | 53.362 | 0.07× |
| point_in_polygon | v1024 | 15.345 | 39.425 | 196.591 | 0.08× |
| polygon_intersection | v4 | 0.255 | 1.556 | 7.483 | 0.03× |
| polygon_intersection | v16 | 3.681 | 2.076 | 10.199 | 0.36× |
| polygon_intersection | v64 | 48.634 | 4.267 | 22.557 | 2.16× |
| polygon_intersection | v256 | 741.999 | 12.544 | 83.536 | 8.88× |
| spatial_join | L100_R10_r50 | 0.442 | 2.180 | 7.442 | 0.06× |
| spatial_join | L200_R10_r10 | 0.794 | 4.305 | 7.662 | 0.10× |
| spatial_join | L200_R50_r90 | 3.898 | 15.728 | 8.106 | 0.48× |
| spatial_join | L300_R50_r0 | 4.989 | 23.594 | 9.019 | 0.55× |
| spatial_join | L300_R100_r100 | 11.118 | 44.134 | 9.040 | 1.23× |
| runtime_batch | L100_R10_r50 | 0.503 | 2.206 | 7.324 | 0.07× |
| runtime_batch | L200_R10_r10 | 0.816 | 4.282 | 7.821 | 0.10× |
| runtime_batch | L200_R50_r90 | 3.838 | 15.712 | 8.172 | 0.47× |
| runtime_batch | L300_R50_r0 | 5.061 | 23.392 | 8.748 | 0.58× |
| runtime_batch | L300_R100_r100 | 11.218 | 45.023 | 9.502 | 1.18× |

## Mojo: native kernel (T1) vs integrated (T3)  — exposes the Python↔Mojo boundary cost

| Workload | Case | Python (ms) | Mojo native kernel (ms) | Mojo integrated (ms) | Boundary (ms) | Kernel speedup |
|---|---|---:|---:|---:|---:|---:|
| point_in_polygon | v4 | 0.116 | 0.020 | 7.628 | 7.607 | 5.69× |
| point_in_polygon | v16 | 0.295 | 0.038 | 9.929 | 9.891 | 7.72× |
| point_in_polygon | v64 | 0.983 | 0.090 | 18.223 | 18.133 | 10.95× |
| point_in_polygon | v256 | 3.547 | 0.308 | 53.362 | 53.055 | 11.53× |
| point_in_polygon | v1024 | 15.345 | 1.073 | 196.591 | 195.517 | 14.30× |
| polygon_intersection | v4 | 0.255 | 0.024 | 7.483 | 7.458 | 10.52× |
| polygon_intersection | v16 | 3.681 | 0.111 | 10.199 | 10.088 | 33.03× |
| polygon_intersection | v64 | 48.634 | 1.235 | 22.557 | 21.322 | 39.37× |
| polygon_intersection | v256 | 741.999 | 17.504 | 83.536 | 66.031 | 42.39× |
| spatial_join | L100_R10_r50 | 0.442 | 0.030 | 7.442 | 7.411 | 14.65× |
| spatial_join | L200_R10_r10 | 0.794 | 0.051 | 7.662 | 7.611 | 15.60× |
| spatial_join | L200_R50_r90 | 3.898 | 0.180 | 8.106 | 7.926 | 21.67× |
| spatial_join | L300_R50_r0 | 4.989 | 0.263 | 9.019 | 8.756 | 19.00× |
| spatial_join | L300_R100_r100 | 11.118 | 0.480 | 9.040 | 8.559 | 23.16× |
| runtime_batch | L100_R10_r50 | 0.503 | 0.030 | 7.324 | 7.294 | 16.74× |
| runtime_batch | L200_R10_r10 | 0.816 | 0.055 | 7.821 | 7.766 | 14.96× |
| runtime_batch | L200_R50_r90 | 3.838 | 0.176 | 8.172 | 7.997 | 21.83× |
| runtime_batch | L300_R50_r0 | 5.061 | 0.244 | 8.748 | 8.504 | 20.73× |
| runtime_batch | L300_R100_r100 | 11.218 | 0.548 | 9.502 | 8.954 | 20.47× |

## Integrated crossover (§12)

| Workload | First case Mojo wins integrated |
|---|---|
| point_in_polygon | none observed |
| polygon_intersection | v64 |
| spatial_join | L300_R100_r100 |
| runtime_batch | L300_R100_r100 |
