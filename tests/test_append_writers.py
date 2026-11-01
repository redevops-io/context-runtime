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
PROCESSES = 2
ROUNDS = 40
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
