"""Fit the retrieval score→P(relevant) calibration map from a logged run.

Reads the JSONL the control plane writes when CR_CALIBRATE=1
(``$CONTEXT_RUNTIME_HOME/librechat_calib_log.jsonl``) and fits a per-method isotonic
map, writing the artifact the control plane loads at startup
(``librechat_calibration.json``). Order-preserving, so it never reorders hits.

Usage:
    python -m context_runtime.scripts.fit_calibration \
        [--log PATH] [--out PATH] [--min-samples N]

Defaults resolve under $CONTEXT_RUNTIME_HOME (or .context-runtime).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..integrations.calibration import CalibrationLog, fit_from_log


def _home() -> Path:
    return Path(os.environ.get("CONTEXT_RUNTIME_HOME")
               or os.environ.get("AGENTIC_OS_HOME", ".context-runtime"))


def main(argv: list[str] | None = None) -> int:
    home = _home()
    ap = argparse.ArgumentParser(description="Fit retrieval score calibration from a log.")
    ap.add_argument("--log", default=str(home / "librechat_calib_log.jsonl"))
    ap.add_argument("--out", default=str(home / "librechat_calibration.json"))
    ap.add_argument("--min-samples", type=int, default=20,
                    help="methods with fewer pairs stay unfit (identity) rather than overfit")
    args = ap.parse_args(argv)

    log = CalibrationLog(args.log)
    rows = log.rows()
    if not rows:
        print(f"no calibration rows at {args.log} — run with CR_CALIBRATE=1 first")
        return 1
    cmap = fit_from_log(log, min_samples=args.min_samples)
    cmap.save(args.out)
    fitted = {m: c["n"] for m, c in cmap.to_dict().items()}
    print(f"read {len(rows)} rows; fitted per-method pairs: {fitted}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
