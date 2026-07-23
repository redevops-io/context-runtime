"""CloudWatchReader — a ``TelemetryReader`` over CloudWatch Logs Insights.

The AWS implementation of the neutral telemetry seam: read-only operational signals for a post-deploy
monitor loop (Sidekick's job in agentic-os). ``query(expr, window_s)`` runs a Logs Insights query over
the last ``window_s`` seconds and returns rows. Azure Monitor / GCP Cloud Monitoring implement the same
Protocol. Client + clock are injectable so the poll loop is testable without boto3 or wall-clock.
"""
from __future__ import annotations

import time


class CloudWatchReader:
    def __init__(self, session=None, *, log_group: str | None = None, client=None,
                 poll_interval: float = 0.5, max_wait_s: float = 20.0, now=time.time, sleep=time.sleep):
        self._session = session
        self.log_group = log_group
        self._client = client
        self.poll_interval = poll_interval
        self.max_wait_s = max_wait_s
        self._now = now
        self._sleep = sleep

    def _logs(self):
        if self._client is None:
            self._client = self._session.client("logs")
        return self._client

    def query(self, expr: str, window_s: int = 300) -> list[dict]:
        client = self._logs()
        end = int(self._now())
        start = end - int(window_s)
        kwargs = {"startTime": start, "endTime": end, "queryString": expr}
        if self.log_group:
            kwargs["logGroupName"] = self.log_group
        qid = client.start_query(**kwargs)["queryId"]

        waited = 0.0
        while True:
            res = client.get_query_results(queryId=qid)
            status = res.get("status")
            if status in ("Complete", "Failed", "Cancelled", "Timeout"):
                break
            if waited >= self.max_wait_s:
                raise TimeoutError(f"cloudwatch query {qid} still {status} after {self.max_wait_s}s")
            self._sleep(self.poll_interval)
            waited += self.poll_interval
        if status != "Complete":
            raise RuntimeError(f"cloudwatch query {qid} {status}")

        # each result row is a list of {field, value}; flatten to a dict per row
        return [{c["field"]: c.get("value") for c in row} for row in res.get("results", []) or []]
