"""cuDF backend for temporal/evidence change-set processing (the arm-H workload).

``TemporalStore.changes`` answers "what began or ended validity in [since, until)?" over a body of facts.
The reference is a Python loop; past the ~10^6-fact crossover a dataframe reduction wins. This computes the
**identical** result (same records, same order) with a dataframe module — cudf on the GPU, or pandas on the
CPU (used by the equivalence test, so the vectorised logic is checked without a GPU).

Exactness detail: the Python reference appends, per fact in order, a "began" record then an "ended" record,
then stable-sorts by ``at``. We reproduce that by tagging each record with a sequence key (2·i / 2·i+1) and
sorting by ``(at, seq)`` — a stable sort by ``at`` that breaks ties exactly as the loop does. ISO-8601
timestamps compare lexically, matching the reference's string comparisons.
"""
from __future__ import annotations

from typing import Sequence


def _changes_df(xf, valid_from: Sequence[str], valid_to: Sequence, subject: Sequence[str],
                predicate: Sequence[str], obj: Sequence[str], text: Sequence[str],
                *, since: str, until: str, k: int) -> list[dict]:
    """Compute the change list with dataframe module ``xf`` (pandas or cudf)."""
    n = len(valid_from)
    seq = list(range(n))
    df = xf.DataFrame({
        "seq": seq, "vf": list(valid_from), "vt": list(valid_to),
        "subject": list(subject), "predicate": list(predicate), "obj": list(obj), "text": list(text),
    })
    began = df[(df["vf"] >= since) & (df["vf"] < until)].copy()
    began["at"] = began["vf"]
    began["change"] = "began"
    began["order"] = began["seq"] * 2

    ended = df[df["vt"].notnull() & (df["vt"] >= since) & (df["vt"] < until)].copy()
    ended["at"] = ended["vt"]
    ended["change"] = "ended"
    ended["order"] = ended["seq"] * 2 + 1

    cols = ["at", "change", "text", "subject", "predicate", "obj", "order"]
    both = xf.concat([began[cols], ended[cols]], ignore_index=True)
    both = both.sort_values(["at", "order"])            # stable-by-at, tie-break as the loop does
    try:
        both = both.to_pandas()                          # cudf → host
    except AttributeError:
        pass
    out = [{"at": r.at, "change": r.change, "fact": r.text, "subject": r.subject,
            "predicate": r.predicate, "object": r.obj}
           for r in both.itertuples(index=False)]
    return out[:k] if k > 0 else out


def changes_gpu(valid_from, valid_to, subject, predicate, obj, text, *, since, until, k) -> list[dict]:
    import cudf
    return _changes_df(cudf, valid_from, valid_to, subject, predicate, obj, text,
                       since=since, until=until, k=k)


def changes_cpu_df(valid_from, valid_to, subject, predicate, obj, text, *, since, until, k) -> list[dict]:
    """pandas equivalent — the reference for the GPU path, and what the no-GPU equivalence test runs."""
    import pandas as pd
    return _changes_df(pd, valid_from, valid_to, subject, predicate, obj, text,
                       since=since, until=until, k=k)
