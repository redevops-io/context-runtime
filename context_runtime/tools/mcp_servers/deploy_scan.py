"""deploy_scan — a runtime-DAST MCP server (stdio) for edge-sentinel's deployment plane.

Exposes three tools that scan a **running deployment** (not code/packages) for exploitable
exposure with nmap + nuclei + ZAP. Scans take minutes, but MCP calls must stay well under the
~55s SDK ceiling, so the contract is async — start, poll, fetch — never a blocking scan_and_wait:

  * scan_start(target, profile)  → {job_id, state:"running"}      [SIDE-EFFECTING: sends probe
                                     traffic; gated by the harness ApprovalPolicy]
  * scan_status(job_id)          → {state, tools_done, elapsed_s}  [readOnly]
  * scan_results(job_id)         → {summary, findings[]}           [readOnly]

profile ∈ recon(nmap) | web-baseline(nuclei+zap passive) | web-active(nuclei+zap ACTIVE) | full.
Every scan_start enforces the fail-closed ``SCAN_ALLOWLIST`` scope; an off-scope target is
refused before any scanner subprocess is spawned. Real tools run when installed + SCAN_LIVE=1;
otherwise a faithful simulated report lets the whole loop run offline.

Speaks newline-delimited JSON-RPC 2.0 (MCP stdio) matching ``context_runtime.tools.mcp.MCPClient``.
Background scan threads only mutate the in-memory job table — they never write stdout, so the
JSON-RPC stream stays clean.

Run:  python -m context_runtime.tools.mcp_servers.deploy_scan
"""
from __future__ import annotations

import json
import sys
import threading
import time
import uuid

from ...integrations.runtime_scan_engine import (
    ACTIVE_PROFILES, PROFILES, DeploymentScanner, OutOfScopeError, ScanReport,
    Finding, host_of, in_scope,
)

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _set(job_id: str, **kw) -> None:
    with _LOCK:
        if job_id in _JOBS:
            _JOBS[job_id].update(kw)


def _run_job(job_id: str, target: str, tools: list[str]) -> None:
    scanner = DeploymentScanner()
    findings: list[Finding] = []
    for tool in tools:
        try:
            rep = scanner.run_tool(tool, target)
        except OutOfScopeError as e:      # defence in depth (scan_start already checked)
            _set(job_id, state="error", error=str(e))
            return
        except Exception as e:            # a scanner crash fails the job, not the server
            _set(job_id, state="error", error=f"{type(e).__name__}: {e}")
            return
        findings.extend(rep.findings)
        with _LOCK:
            _JOBS[job_id]["tools_done"].append(tool)
            _JOBS[job_id]["simulated"] = _JOBS[job_id].get("simulated", False) or rep.simulated
    _set(job_id, state="done", findings=[f.as_dict() for f in findings])


# ──────────────────────────── tool implementations ────────────────────────────


def scan_start(target: str, profile: str = "web-baseline") -> dict:
    target = (target or "").strip()
    if profile not in PROFILES:
        return {"error": f"unknown profile {profile!r}; choose one of {sorted(PROFILES)}"}
    if not target:
        return {"error": "target is required"}
    if not in_scope(target):
        return {"error": f"out of scope: {host_of(target)!r} is not in SCAN_ALLOWLIST "
                         f"(fail-closed — set SCAN_ALLOWLIST to your own hosts)"}
    job_id = uuid.uuid4().hex[:12]
    tools = list(PROFILES[profile])
    with _LOCK:
        _JOBS[job_id] = {"state": "running", "target": target, "profile": profile,
                         "tools": tools, "tools_done": [], "findings": None,
                         "started": time.monotonic(), "simulated": False,
                         "active": profile in ACTIVE_PROFILES}
    threading.Thread(target=_run_job, args=(job_id, target, tools), daemon=True).start()
    return {"job_id": job_id, "state": "running", "profile": profile, "tools": tools,
            "target": target, "active": profile in ACTIVE_PROFILES}


def scan_status(job_id: str) -> dict:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return {"error": f"unknown job_id {job_id!r}"}
        elapsed = round(time.monotonic() - job["started"], 1)
        n = len(job["findings"]) if job.get("findings") is not None else None
        return {"job_id": job_id, "state": job["state"], "profile": job["profile"],
                "tools_done": list(job["tools_done"]), "tools": list(job["tools"]),
                "elapsed_s": elapsed, "n_findings": n, "error": job.get("error")}


def scan_results(job_id: str) -> dict:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return {"error": f"unknown job_id {job_id!r}"}
        if job["state"] != "done":
            return {"job_id": job_id, "state": job["state"], "note": "not ready — keep polling scan_status",
                    "error": job.get("error")}
        findings = list(job["findings"] or [])
        simulated = job.get("simulated", False)
        target = job["target"]
    fobjs = [Finding(**f) for f in findings]
    report = ScanReport(True, target, sorted({f.tool for f in fobjs}), findings=fobjs, simulated=simulated)
    return {"job_id": job_id, "state": "done", "summary": report.summary(), "findings": findings}


# ──────────────────────────── JSON-RPC 2.0 stdio plumbing ────────────────────────────

_TOOLS = [
    {
        "name": "scan_start",
        "description": ("Start a DAST scan of a RUNNING deployment (not code/packages). profile: "
                        "recon (nmap surface) | web-baseline (nuclei + ZAP passive) | web-active "
                        "(nuclei + ZAP ACTIVE attack traffic) | full. Returns a job_id; poll "
                        "scan_status then fetch scan_results. Scope-enforced via SCAN_ALLOWLIST."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "URL or host to scan (must be allowlisted)"},
                "profile": {"type": "string", "enum": sorted(PROFILES),
                            "description": "scan profile (default web-baseline)"},
            },
            "required": ["target"],
        },
        # NO readOnlyHint → harness treats it as side-effecting → ApprovalPolicy gate.
        "annotations": {"destructiveHint": False, "openWorldHint": True},
    },
    {
        "name": "scan_status",
        "description": "Poll a scan job's state (running|done|error), tools completed, and elapsed time.",
        "inputSchema": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "scan_results",
        "description": "Fetch a finished scan's findings + severity summary (once scan_status is 'done').",
        "inputSchema": {"type": "object", "properties": {"job_id": {"type": "string"}}, "required": ["job_id"]},
        "annotations": {"readOnlyHint": True},
    },
]

_IMPL = {"scan_start": scan_start, "scan_status": scan_status, "scan_results": scan_results}


def _result(rid, result=None, error=None) -> dict:
    m: dict = {"jsonrpc": "2.0", "id": rid}
    if error is not None:
        m["error"] = error
    else:
        m["result"] = result
    return m


def _call(name: str, args: dict) -> dict:
    fn = _IMPL.get(name)
    if fn is None:
        return {"content": [{"type": "text", "text": f"unknown tool: {name}"}], "isError": True}
    try:
        data = fn(**args)
    except TypeError as e:
        return {"content": [{"type": "text", "text": f"bad arguments: {e}"}], "isError": True}
    except Exception as e:  # noqa: BLE001
        return {"content": [{"type": "text", "text": f"{name} error: {e}"}], "isError": True}
    is_err = isinstance(data, dict) and "error" in data and data.get("error")
    return {"content": [{"type": "text", "text": json.dumps(data, separators=(",", ":"))}],
            "structuredContent": data, "isError": bool(is_err)}


def _handle(req: dict) -> dict | None:
    method = req.get("method")
    rid = req.get("id")
    params = req.get("params") or {}
    if method == "initialize":
        return _result(rid, {"protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                             "serverInfo": {"name": "deploy-scan", "version": "0.1.0"}})
    if method == "tools/list":
        return _result(rid, {"tools": _TOOLS})
    if method == "tools/call":
        return _result(rid, _call(params.get("name"), params.get("arguments") or {}))
    if rid is not None:
        return _result(rid, error={"code": -32601, "message": f"method not found: {method}"})
    return None


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        resp = _handle(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
