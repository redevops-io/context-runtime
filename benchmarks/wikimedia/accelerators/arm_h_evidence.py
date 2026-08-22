"""Arm H — cuDF evidence/temporal processing (roadmap §5.3, §6.1; benchmark question H).

The runtime operation being accelerated is the one arm B exercises: deriving the **current valid state**
of a body of evidence as of a point in time, and re-deriving it incrementally when a change set arrives.
Over a real revision history this is a dataframe workload — a temporal filter, a per-entity "latest
revision as-of T" reduction, and a content-hash correlation (how many entities share identical content):

    valid_state(as_of) = evidence[ts <= as_of]  ->  per ref keep max-ts row  ->  {ref: current revid}
    content_correlation = distinct refs per content_hash          (evidence dedup / correlation signal)

Both pandas (CPU reference) and cudf (GPU) express this with the *same* API, so the two paths are the
same computation on two backends. The GPU result must be byte-identical to the CPU result (the arm-B
"valid_state(incremental) == valid_state(full)" gate, now at scale) before any timing counts.

Inputs are real strategywiki revision metadata produced by ``prepare_inputs.py``: (ref=page_id,
revid, content_hash, ts_unix) for up to ~80k real revisions.
"""
from __future__ import annotations

import numpy as np

from .common import time_median

ARM = "H"
NAME = "cuDF evidence/temporal processing (incremental-Discovery change-set)"


# --- the operation, written once against a dataframe module (pandas or cudf) -------------------------

def _valid_state(xf, ref, revid, sha, ts, as_of: int):
    """Current valid state as of ``as_of`` + content correlation, computed with dataframe module ``xf``.

    Frame construction is inside the callable on purpose: for cudf this moves the columns host->device,
    so the measured region includes the transfer the roadmap insists on counting.
    """
    df = xf.DataFrame({"ref": ref, "revid": revid, "sha": sha, "ts": ts})
    df = df[df["ts"] <= as_of]                       # temporal window
    df = df.sort_values(["ref", "ts"])               # order within each entity
    cur = df.drop_duplicates(subset=["ref"], keep="last")   # latest revision per ref as-of T
    corr = int(cur.groupby("sha")["ref"].count().sum())     # evidence correlation over content hashes
    return cur[["ref", "revid"]], corr


def _to_state_dict(cur_frame) -> dict:
    """Materialise a {ref: revid} dict on the host from a pandas or cudf frame (for equivalence)."""
    try:
        cur_frame = cur_frame.to_pandas()            # cudf -> pandas
    except AttributeError:
        pass
    return dict(zip(cur_frame["ref"].tolist(), cur_frame["revid"].tolist()))


# --- CPU reference and GPU paths ---------------------------------------------------------------------

def cpu_valid_state(arrays, as_of):
    import pandas as pd
    return _valid_state(pd, *arrays, as_of)


def gpu_valid_state(arrays, as_of):
    import cudf
    import cupy as cp
    dev = [cp.asarray(a) for a in arrays]            # explicit host->device (counted by the timer)
    return _valid_state(cudf, *dev, as_of)


# --- incremental == full, at scale (the arm-B semantic invariant, preserved by the accelerator) ------

def incremental_equals_full(arrays, split_ts: int, later_ts: int) -> bool:
    """Full recompute at ``later_ts`` must equal (base state at ``split_ts``) merged with the change set
    of refs that advanced between the two times. Pure-CPU check that the *operation itself* (not the
    backend) preserves the incremental-Discovery invariant on the real data."""
    import pandas as pd
    ref, revid, sha, ts = arrays
    full_cur, _ = _valid_state(pd, ref, revid, sha, ts, later_ts)
    full = _to_state_dict(full_cur)

    base_cur, _ = _valid_state(pd, ref, revid, sha, ts, split_ts)
    base = _to_state_dict(base_cur)
    # change set: rows in (split_ts, later_ts] -> refs that advanced
    m = (ts > split_ts) & (ts <= later_ts)
    changed = _to_state_dict(_valid_state(pd, ref[m], revid[m], sha[m], ts[m], later_ts)[0]) if m.any() else {}
    incremental = {**base, **changed}
    return incremental == full and len(changed) == len(set(ref[m].tolist()))


# --- corpus scaling (tile real rows into more distinct entities, for rows beyond the real dump) ------

def extend_table(arrays, n: int) -> tuple[tuple, bool]:
    """Return an n-row evidence table. Up to the real row count it is the real dump; beyond, real rows are
    tiled with per-tile offsets on ref/revid (distinct entities — a larger evidence history / a fleet of
    wikis) while content hashes and timestamps are reused (realistic dedup + temporal distribution).
    Second value = whether scaled."""
    ref, revid, sha, ts = arrays
    m = len(ref)
    if n <= m:
        return (ref[:n], revid[:n], sha[:n], ts[:n]), False
    reps = -(-n // m)                                 # ceil
    ref_off, revid_off = int(ref.max()) + 1, int(revid.max()) + 1
    ref2 = np.concatenate([ref + i * ref_off for i in range(reps)])[:n]
    revid2 = np.concatenate([revid + i * revid_off for i in range(reps)])[:n]
    sha2 = np.tile(sha, reps)[:n]
    ts2 = np.tile(ts, reps)[:n]
    return (ref2, revid2, sha2, ts2), True


# --- crossover sweep ---------------------------------------------------------------------------------

def sweep(arrays, sizes, *, repeats: int = 5) -> list[dict]:
    as_of = int(np.median(arrays[3]))
    rows = []
    for n in sizes:
        sub, scaled = extend_table(arrays, n)
        n = len(sub[0])
        cpu_med, cpu_min, _, cpu_r = time_median(lambda: cpu_valid_state(sub, as_of), repeats=repeats)
        gpu_med, gpu_min, gpu_cold, gpu_r = time_median(
            lambda: gpu_valid_state(sub, as_of), repeats=repeats, sync=True)
        correct = _to_state_dict(cpu_r[0]) == _to_state_dict(gpu_r[0]) and cpu_r[1] == gpu_r[1]
        rows.append({
            "n": n, "scaled": bool(scaled),
            "cpu_ms_median": round(cpu_med, 3), "gpu_ms_median": round(gpu_med, 3),
            "gpu_ms_cold": round(gpu_cold, 3), "speedup": round(cpu_med / gpu_med, 2) if gpu_med else None,
            "n_entities": len(_to_state_dict(cpu_r[0])), "correct": bool(correct),
        })
    return rows
