"""Concurrent appenders lose no line, on any OS.

A bare ``open(path, "a")`` write is not safe on Windows: the C runtime appends by seeking to the end and then
writing, so two writers that both seek before either writes overwrite each other. These tests release threads
and processes together through a barrier (the schedule that exposes it) and require exactly N*K intact lines.
"""

from __future__ import annotations

import json
import multiprocessing
import threading
from pathlib import Path

import pytest

from context_runtime.observability.exporters import JsonlExporter
from context_runtime.platform_services import locked_append

THREADS = 3
PROCESSES = 3
ROUNDS = 200          # raised until the old bare append fails reliably on Windows (see the pull request)
TIMEOUT = 120


def _worker(barrier, kind: str, path: str, who: str, rounds: int) -> None:
    """Runs in a child process (spawn) or a thread: one append per barrier-released round."""
    exporter = JsonlExporter(path) if kind == "exporter" else None
    for r in range(rounds):
        barrier.wait(TIMEOUT)
        record = {"id": f"{who}-{r}", "pad": "x" * 200}
        if exporter is not None:
            exporter.export(record)
        else:
            locked_append(path, json.dumps(record) + "\n")


def _run(kind: str, path: Path) -> list[str]:
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(THREADS + PROCESSES)            # threads and processes released together
    procs = [ctx.Process(target=_worker, args=(barrier, kind, str(path), f"p{i}", ROUNDS)) for i in range(PROCESSES)]
    for p in procs:
        p.start()
    errors: list[BaseException] = []

    def run_thread(i: int) -> None:
        try:
            _worker(barrier, kind, str(path), f"t{i}", ROUNDS)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=run_thread, args=(i,)) for i in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(TIMEOUT)
    for p in procs:
        p.join(TIMEOUT)
        assert p.exitcode == 0, f"child exited with {p.exitcode}"
    assert not errors, errors
    return path.read_text(encoding="utf-8").splitlines()


def _check(lines: list[str]) -> None:
    expected = {f"{w}-{r}" for w in [f"t{i}" for i in range(THREADS)] + [f"p{i}" for i in range(PROCESSES)]
                for r in range(ROUNDS)}
    assert len(lines) == len(expected), f"{len(lines)} lines, expected {len(expected)}"
    ids = {json.loads(ln)["id"] for ln in lines}                  # every line parses: none torn or merged
    assert ids == expected


@pytest.mark.parametrize("kind", ["exporter", "helper"])
def test_barrier_released_threads_and_processes_lose_no_line(tmp_path, kind):
    path = tmp_path / "out" / "traces.jsonl"
    path.parent.mkdir()
    _check(_run(kind, path))


def test_a_long_record_is_never_interleaved(tmp_path):
    path = tmp_path / "big.jsonl"
    payload = "y" * 100_000                                       # larger than any stdio buffer

    def go(i: int) -> None:
        for r in range(5):
            locked_append(path, json.dumps({"id": f"{i}-{r}", "pad": payload}) + "\n")

    ts = [threading.Thread(target=go, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(TIMEOUT)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 20 and all(len(json.loads(ln)["pad"]) == 100_000 for ln in lines)


def test_durable_false_skips_fsync_and_still_appends(tmp_path, monkeypatch):
    import os
    calls = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real(fd))[1])
    locked_append(tmp_path / "a.log", "one\n", durable=False)
    assert calls == []
    locked_append(tmp_path / "a.log", "two\n")
    assert len(calls) == 1
    assert (tmp_path / "a.log").read_text(encoding="utf-8") == "one\ntwo\n"


def test_exporter_writes_one_newline_terminated_json_line_per_trace(tmp_path):
    exp = JsonlExporter(str(tmp_path / "t.jsonl"))
    exp.export({"id": "a"})
    exp.export({"id": "b"})
    raw = (tmp_path / "t.jsonl").read_bytes()
    assert raw == b'{"id": "a"}\n{"id": "b"}\n'                   # explicit "\n", no CRLF translation on Windows


# ───────────────────────── the exporter must not stall the application ─────────────────────────

HOLDER = """
import sys, time
from context_runtime.platform_services import file_lock
with file_lock(sys.argv[1], timeout=30):
    print("held", flush=True)
    time.sleep(float(sys.argv[2]))
"""


def test_export_gives_up_quickly_when_another_process_holds_the_lock(tmp_path, caplog):
    import logging
    import subprocess
    import sys
    import time

    from context_runtime.observability import exporters
    from context_runtime.platform_services import lock_path_for
    exp = JsonlExporter(str(tmp_path / "t.jsonl"))
    holder = subprocess.Popen([sys.executable, "-c", HOLDER, str(lock_path_for(exp.path)), "30"],
                              stdout=subprocess.PIPE, text=True, encoding="utf-8")
    try:
        assert holder.stdout.readline().strip() == "held"
        with caplog.at_level(logging.WARNING, logger="context_runtime.observability"):
            t0 = time.monotonic()
            for i in range(3):
                exp.export({"id": i})                           # dropped, not raised
            elapsed = time.monotonic() - t0
        assert elapsed < 3 * exporters.EXPORT_LOCK_TIMEOUT + 2.0, elapsed       # three waits of about one second
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warnings) == 1 and "trace dropped" in warnings[0].getMessage()      # ONE warning, not one per trace
        assert not exp.path.exists() or exp.path.read_text(encoding="utf-8") == ""
    finally:
        holder.kill()
        holder.wait(10)
    exp.export({"id": "after"})                                  # the lock is free again: exports resume
    assert [json.loads(ln)["id"] for ln in exp.path.read_text(encoding="utf-8").splitlines()] == ["after"]


def test_export_from_a_thread_that_already_holds_the_lock_returns_after_the_timeout(tmp_path, caplog):
    import logging
    import time

    from context_runtime.observability import exporters
    from context_runtime.platform_services import file_lock, lock_path_for
    exp = JsonlExporter(str(tmp_path / "t.jsonl"))
    with caplog.at_level(logging.WARNING, logger="context_runtime.observability"):
        with file_lock(lock_path_for(exp.path)):
            t0 = time.monotonic()
            exp.export({"id": "reentrant"})                      # same-thread re-entry: bounded, not a deadlock
            elapsed = time.monotonic() - t0
    assert elapsed < exporters.EXPORT_LOCK_TIMEOUT + 2.0
    assert sum("trace dropped" in r.getMessage() for r in caplog.records) == 1


def test_a_non_lock_error_is_logged_once_and_raised_as_before(tmp_path, monkeypatch, caplog):
    import logging

    from context_runtime.observability import exporters
    exp = JsonlExporter(str(tmp_path / "t.jsonl"))

    def boom(*a, **k):
        raise PermissionError(13, "read-only directory")

    monkeypatch.setattr(exporters, "locked_append", boom)
    with caplog.at_level(logging.WARNING, logger="context_runtime.observability"):
        for _ in range(2):
            with pytest.raises(PermissionError):
                exp.export({"id": 1})
    assert sum("failed" in r.getMessage() for r in caplog.records) == 1


def test_threads_of_one_process_queue_on_the_in_process_lock(tmp_path, monkeypatch):
    """Even if the file lock did not exclude threads (some network file systems), appends from threads do not mix."""
    from context_runtime import platform_services as ps

    class NoExclusion:                                           # a file lock that never excludes anything
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(ps, "file_lock", lambda *a, **k: NoExclusion())
    path = tmp_path / "n.log"
    barrier = threading.Barrier(4)

    def go(i):
        for r in range(100):
            barrier.wait(TIMEOUT)
            locked_append(path, f"{i}-{r}\n")

    ts = [threading.Thread(target=go, args=(i,)) for i in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(TIMEOUT)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 400
