"""CloudMonitoringReader — a ``TelemetryReader`` over Google Cloud Monitoring (time series).

The GCP implementation of the neutral telemetry seam: read-only operational signals for a post-deploy
monitor loop. ``query(expr, window_s)`` runs a metric filter over the last ``window_s`` seconds and
returns points. ``client`` + ``now`` injectable so it's testable without the SDK or wall-clock.
"""
from __future__ import annotations

import time


class CloudMonitoringReader:
    def __init__(self, session=None, *, client=None, now=time.time):
        self._session = session
        self._client = client
        self._now = now

    def _mon(self):
        if self._client is None:
            self._client = self._session.monitoring_client()
        return self._client

    def query(self, expr: str, window_s: int = 300) -> list[dict]:
        end = int(self._now())
        project = self._session.project if self._session else None
        request = {
            "name": f"projects/{project}",
            "filter": expr,
            "interval": {"start_time": {"seconds": end - int(window_s)}, "end_time": {"seconds": end}},
        }
        series = self._mon().list_time_series(request=request)
        rows: list[dict] = []
        for ts in series:
            metric = dict(getattr(ts, "metric", {}).labels) if hasattr(getattr(ts, "metric", {}), "labels") else {}
            for pt in getattr(ts, "points", []) or []:
                val = getattr(getattr(pt, "value", None), "double_value", None)
                rows.append({"metric": getattr(getattr(ts, "metric", None), "type", ""),
                             "labels": metric, "value": val})
        return rows
