"""Deterministic seeded corpus generation (§6, §7). Persists identical inputs for every arm.

Emits corpus/{point_in_polygon,polygon_pairs,spatial_join,adversarial}.jsonl + manifest.json.
Inputs only — each arm computes its own outputs; Python is the reference for parity.
"""
from __future__ import annotations

import json
import math
import os
import random
import subprocess

SEED = 0x5EED2026
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.abspath(os.path.join(HERE, "..", "corpus"))
EPS = 1e-9  # the engine's on-edge tolerance (point_in_ring uses 1e-9 / 1e-12)


def ngon(cx, cy, r, n, rot=0.0):
    return [[cx + r * math.cos(rot + 2 * math.pi * i / n),
             cy + r * math.sin(rot + 2 * math.pi * i / n)] for i in range(n)]


def gen_point_in_polygon(rng, out):
    # sizes per §6 W1 (queries capped for a tractable default; run_all scales by profile)
    for verts in (4, 16, 64, 256, 1024):
        poly = ngon(0.0, 0.0, 100.0, verts, rot=rng.random())
        for k in range(200):  # per-(verts) query block; run_all replicates blocks for larger N
            cat = k % 5
            if cat == 0:      # clearly inside
                pt = [rng.uniform(-30, 30), rng.uniform(-30, 30)]
            elif cat == 1:    # clearly outside
                pt = [rng.uniform(200, 300), rng.uniform(200, 300)]
            elif cat == 2:    # near an edge
                v = poly[rng.randrange(verts)]
                pt = [v[0] * 0.999, v[1] * 0.999]
            elif cat == 3:    # exactly on a vertex
                pt = list(poly[rng.randrange(verts)])
            else:             # within epsilon of a boundary vertex
                v = poly[rng.randrange(verts)]
                pt = [v[0] + EPS * 0.5, v[1] - EPS * 0.5]
            out.append({"case": f"v{verts}", "verts": verts, "ring": poly, "point": pt})


def gen_polygon_pairs(rng, out):
    for verts in (4, 16, 64, 256):
        for k in range(120):
            a = ngon(0.0, 0.0, 50.0, verts, rot=rng.random())
            cat = k % 6
            if cat == 0:      # disjoint (far)
                b = ngon(1000.0, 1000.0, 50.0, verts)
            elif cat == 1:    # bbox overlap, no true intersection (offset diagonally into a corner gap)
                b = ngon(95.0, 95.0, 50.0, verts)
            elif cat == 2:    # edge crossing
                b = ngon(60.0, 0.0, 50.0, verts)
            elif cat == 3:    # containment (b inside a)
                b = ngon(0.0, 0.0, 10.0, verts)
            elif cat == 4:    # touching edge
                b = ngon(100.0, 0.0, 50.0, verts)
            else:             # near-touching at epsilon
                b = ngon(100.0 + EPS, 0.0, 50.0, verts)
            out.append({"case": f"v{verts}", "verts": verts, "a": a, "b": b})


def gen_spatial_join(rng, out):
    # §6 W3 matrix; default sizes kept modest, run_all scales via profile
    for (nl, nr, rate) in [(100, 10, 0.5), (200, 10, 0.1), (200, 50, 0.9), (300, 50, 0.0), (300, 100, 1.0)]:
        # right zones on a grid; each a square
        zones = []
        side = math.ceil(math.sqrt(nr))
        for zi in range(nr):
            gx, gy = (zi % side) * 100.0, (zi // side) * 100.0
            zones.append([f"z{zi}", [[gx, gy], [gx + 90, gy], [gx + 90, gy + 90], [gx, gy + 90]], "EPSG:3857"])
        parcels = []
        for pi in range(nl):
            if rng.random() < rate and zones:  # place centroid inside a random zone
                z = zones[rng.randrange(len(zones))][1]
                cx = (z[0][0] + z[1][0]) / 2 + rng.uniform(-20, 20)
                cy = (z[0][1] + z[2][1]) / 2 + rng.uniform(-20, 20)
            else:                               # place well outside the grid
                cx, cy = rng.uniform(-500, -100), rng.uniform(-500, -100)
            parcels.append([f"p{pi}", [[cx - 1, cy - 1], [cx + 1, cy - 1], [cx + 1, cy + 1], [cx - 1, cy + 1]], "EPSG:3857"])
        out.append({"case": f"L{nl}_R{nr}_r{int(rate*100)}", "nl": nl, "nr": nr, "parcels": parcels, "zones": zones})


def gen_adversarial(rng, out):
    """§8 adversarial determinism: values around the 1e-9/1e-12 epsilon behavior."""
    ulp = 2.220446049250313e-16
    sq = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]
    fixtures = [
        ("on_vertex", {"ring": sq, "point": [0.0, 0.0]}),
        ("on_edge_mid", {"ring": sq, "point": [5.0, 0.0]}),
        ("eps_minus_ulp_inside", {"ring": sq, "point": [5.0, 0.0 + (1e-9 - ulp)]}),
        ("eps_exact", {"ring": sq, "point": [5.0, 0.0 + 1e-9]}),
        ("eps_plus_ulp", {"ring": sq, "point": [5.0, 0.0 + (1e-9 + ulp)]}),
        ("just_outside_edge", {"ring": sq, "point": [5.0, -1e-9]}),
        ("collinear_x", {"ring": sq, "point": [10.0 + 1e-12, 5.0]}),
        ("large_magnitude", {"ring": [[1e7, 1e7], [1e7 + 10, 1e7], [1e7 + 10, 1e7 + 10], [1e7, 1e7 + 10]], "point": [1e7 + 5, 1e7]}),
        ("negative_coords", {"ring": [[-10, -10], [-1, -10], [-1, -1], [-10, -1]], "point": [-5, -10]}),
        ("reversed_ring", {"ring": sq[::-1], "point": [5.0, 5.0]}),
        ("repeated_last", {"ring": sq + [sq[0]], "point": [5.0, 5.0]}),
        ("tiny_polygon", {"ring": [[0, 0], [1e-6, 0], [1e-6, 1e-6], [0, 1e-6]], "point": [5e-7, 5e-7]}),
    ]
    for cid, f in fixtures:
        out.append({"case": cid, "op": "point_in_polygon", **f})


def write_jsonl(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")


def main():
    os.makedirs(CORPUS, exist_ok=True)
    rng = random.Random(SEED)
    sets = {
        "point_in_polygon": [], "polygon_pairs": [], "spatial_join": [], "adversarial": [],
    }
    gen_point_in_polygon(rng, sets["point_in_polygon"])
    gen_polygon_pairs(rng, sets["polygon_pairs"])
    gen_spatial_join(rng, sets["spatial_join"])
    gen_adversarial(rng, sets["adversarial"])
    for name, rows in sets.items():
        write_jsonl(os.path.join(CORPUS, f"{name}.jsonl"), rows)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        commit = "unknown"
    manifest = {"seed": hex(SEED), "generator_commit": commit, "epsilon": {"on_edge": 1e-9, "collinear": 1e-12},
                "counts": {k: len(v) for k, v in sets.items()}}
    with open(os.path.join(CORPUS, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print("corpus written:", manifest["counts"])


if __name__ == "__main__":
    main()
