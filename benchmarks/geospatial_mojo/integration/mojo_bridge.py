"""Python↔Mojo integration boundary (§5.2). Builds the optimized Mojo runner once (T6), then drives it as
a subprocess per batch: serialize the workload to a flat float file, invoke the native binary, parse
results + the in-kernel KERNEL_NS (T1 native). run_all times the whole round-trip = the integrated number
the Runtime would actually experience; `last_kernel_ns` exposes the native-kernel time for the T1-vs-T3 table.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MOJO_DIR = os.path.abspath(os.path.join(HERE, "..", "mojo"))
PIXI_BIN = os.path.expanduser("~/.pixi/bin")


def _mojo() -> str | None:
    return shutil.which("mojo") or (os.path.join(PIXI_BIN, "mojo") if os.path.exists(os.path.join(PIXI_BIN, "mojo")) else None)


def mojo_version() -> str:
    m = _mojo()
    if not m:
        return "unavailable"
    return subprocess.check_output([m, "--version"], text=True).strip()


class MojoArm:
    name = "mojo"

    def __init__(self):
        self.mojo = _mojo()
        self.available = bool(self.mojo)
        self.last_kernel_ns = None
        self.build_ns = None
        self.binary = None
        if self.available:
            self.binary = os.path.join(tempfile.gettempdir(), "geo_runner_bench")
            t0 = time.perf_counter_ns()
            subprocess.check_call([self.mojo, "build", "runner.mojo", "-o", self.binary],
                                  cwd=MOJO_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.build_ns = time.perf_counter_ns() - t0  # T6

    # ---- flat-protocol serialization (must match mojo/runner.mojo) ----
    @staticmethod
    def _ring_tokens(ring):
        return [str(len(ring))] + [repr(float(c)) for p in ring for c in p]

    def _invoke(self, workload, tokens):
        path = os.path.join(tempfile.gettempdir(), f"geo_in_{workload}_{os.getpid()}.txt")
        with open(path, "w") as fh:
            fh.write(" ".join(tokens))
        try:
            out = subprocess.run([self.binary, workload, path], text=True, capture_output=True, check=True).stdout
        finally:
            try: os.unlink(path)
            except OSError: pass
        lines = out.splitlines()
        kernel_ns = None
        body = []
        for ln in lines:
            if ln.startswith("KERNEL_NS"):
                kernel_ns = int(ln.split()[1])
            elif ln != "":
                body.append(ln)
        self.last_kernel_ns = kernel_ns
        return body

    def pip_batch(self, items):
        toks = [str(len(items))]
        for it in items:
            toks += self._ring_tokens(it["ring"]) + [repr(float(it["point"][0])), repr(float(it["point"][1]))]
        return [ln.strip() == "1" for ln in self._invoke("pip", toks)]

    def pairs_batch(self, items):
        toks = [str(len(items))]
        for it in items:
            toks += self._ring_tokens(it["a"]) + self._ring_tokens(it["b"])
        return [ln.strip() == "1" for ln in self._invoke("pairs", toks)]

    def join_batch(self, items):
        # one join per invocation (each corpus row is a full join); returns list[dict pid->[zid]]
        results = []
        ksum = 0
        for it in items:
            zones, parcels = it["zones"], it["parcels"]
            toks = [str(len(parcels)), str(len(zones))]
            for _zid, ring, _crs in zones:
                toks += self._ring_tokens(ring)
            for _pid, ring, _crs in parcels:
                toks += self._ring_tokens(ring)
            body = self._invoke("join", toks)
            if self.last_kernel_ns:
                ksum += self.last_kernel_ns
            out = {}
            for pi, ln in enumerate(body):
                parts = ln.split()
                k = int(parts[0]) if parts else 0
                out[parcels[pi][0]] = [zones[int(z)][0] for z in parts[1:1 + k]]
            results.append(out)
        self.last_kernel_ns = ksum or self.last_kernel_ns
        return results
