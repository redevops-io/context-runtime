"""Runtime-DAST scanner engine — the light half of the deployment-scan plane.

Kept free of ContextRuntime/bandit imports so the bundled ``deploy_scan`` MCP server can
import it in a slim container (stdlib + ``Hit`` only), the way ``web_search`` stays keyless.

Contents: scope allowlist (fail-closed), Finding/ScanReport, the DeploymentScanner (nmap +
nuclei + ZAP as subprocesses, faithful simulated fallback), and output parsers.
The CR tenant / bandit / reward live in ``runtime_scan.py``.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.parse
import uuid
from dataclasses import dataclass, field

# NB: ``Hit`` is imported lazily inside ``Finding.as_hit`` (not at module top) so this engine
# runs standalone in the slim deploy-scan scanner image without the full context_runtime package.

_SEV_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


# ──────────────────────────── scope (fail-closed) ────────────────────────────


def host_of(target: str) -> str:
    """Host portion of a URL / host:port / bare host."""
    t = (target or "").strip()
    if "://" in t:
        return (urllib.parse.urlparse(t).hostname or "").lower()
    return t.split("/")[0].split(":")[0].lower()


def port_of(target: str) -> str | None:
    """Explicit port from a URL / host:port, else None."""
    t = (target or "").strip()
    if "://" in t:
        p = urllib.parse.urlparse(t).port
        return str(p) if p else None
    seg = t.split("/")[0]
    if ":" in seg:
        pt = seg.rsplit(":", 1)[1]
        return pt if pt.isdigit() else None
    return None


def load_allowlist(env: str | None = None) -> list[str]:
    raw = env if env is not None else os.getenv("SCAN_ALLOWLIST", "")
    return [h.strip().lower() for h in raw.split(",") if h.strip()]


def in_scope(target: str, allowlist: list[str] | None = None) -> bool:
    """True iff target's host is explicitly allowlisted. FAIL-CLOSED: empty allowlist ⇒ False.
    Entries are exact hosts/IPs or ``*.domain`` wildcards."""
    host = host_of(target)
    if not host:
        return False
    al = allowlist if allowlist is not None else load_allowlist()
    if not al:
        return False
    for entry in al:
        if entry.startswith("*."):
            base = entry[2:]
            if host == base or host.endswith("." + base):
                return True
        elif host == entry:
            return True
    return False


class OutOfScopeError(RuntimeError):
    """Raised when a scan is requested against a host not in SCAN_ALLOWLIST."""


# ──────────────────────────── findings ────────────────────────────


@dataclass
class Finding:
    severity: str            # info | low | medium | high | critical
    name: str
    target: str
    tool: str                # nmap | nuclei | zap
    evidence: str = ""
    ref: str = ""            # CVE / template-id / plugin-id

    def as_hit(self, idx: int):
        from ..types import Hit   # lazy: only needed by the CR tenant, not the standalone scanner
        return Hit(chunk_id=f"{self.tool}::{idx}", filename=self.tool, source=self.tool,
                   text=f"[{self.severity.upper()}] {self.name} @ {self.target}"
                        + (f" ({self.ref})" if self.ref else ""),
                   score=float(_SEV_RANK.get(self.severity, 0)) / 4.0)

    def as_dict(self) -> dict:
        return {"severity": self.severity, "name": self.name, "target": self.target,
                "tool": self.tool, "evidence": self.evidence, "ref": self.ref}


@dataclass
class ScanReport:
    ok: bool
    target: str
    tools_run: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    note: str = ""
    simulated: bool = False

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for f in self.findings:
            counts[f.severity] = counts.get(f.severity, 0) + 1
        return {"target": self.target, "tools": self.tools_run, "total": len(self.findings),
                "by_severity": counts, "simulated": self.simulated, "note": self.note,
                "max_severity": max((f.severity for f in self.findings),
                                    key=lambda s: _SEV_RANK.get(s, 0), default="none")}


# ──────────────────────────── target classification + sim truth ────────────────────────────


def scan_bucket(target: str) -> str:
    """Target class → decisive exposure kind. Bare host/IP → TLS/network; a URL with a
    path/params → web; login/search/api paths lean injection-y."""
    t = (target or "").lower()
    if re.search(r"\?|=|/(login|search|api|admin|user)\b", t):
        return "injection"
    if t.startswith("http") or "/" in t.split("://")[-1]:
        return "web_cve"
    return "tls"


# Latent decisive tool per target class — the hidden exposure a bundle must include the right
# tool to catch. Offline-simulation only (mirrors soc_triage's one-decisive-source design).
_SIM_LATENT = {
    "tls":       ("nmap",       ("high", "TLS 1.0/1.1 enabled + weak ciphers", "ssl-enum-ciphers")),
    "web_cve":   ("nuclei",     ("critical", "Exposed .git/config or known CVE", "CVE-2021-41773")),
    "injection": ("zap_active", ("high", "Reflected XSS in query parameter", "40012")),
}


class DeploymentScanner:
    """Runs DAST tools as subprocesses (real when installed + SCAN_LIVE=1); otherwise returns a
    faithful simulated report so the tenant runs offline. Every entrypoint enforces ``in_scope``
    first — no scanner process is spawned for an off-scope host."""

    TOOLS = ("nmap", "nuclei", "zap")

    def __init__(self, timeout: float = 120.0, live: bool | None = None,
                 allowlist: list[str] | None = None):
        self.timeout = timeout
        self.live = (os.getenv("SCAN_LIVE", "0") == "1") if live is None else live
        self.allowlist = allowlist

    def available(self) -> dict:
        return {t: bool(shutil.which(t)) for t in self.TOOLS}

    def _guard(self, target: str) -> None:
        if not in_scope(target, self.allowlist):
            raise OutOfScopeError(
                f"{host_of(target)!r} is not in SCAN_ALLOWLIST — refusing to scan (fail-closed)")

    def _run(self, cmd: list[str], timeout: float | None = None) -> tuple[int, str, str]:
        t = timeout or self.timeout
        try:
            p = subprocess.run(cmd, text=True, capture_output=True, timeout=t)
            return p.returncode, p.stdout, p.stderr
        except FileNotFoundError:
            return 127, "", f"{cmd[0]} not found"
        except subprocess.TimeoutExpired:
            return 124, "", f"{cmd[0]} timed out after {t}s"

    def _docker_image(self) -> str | None:
        """Return the ZAP container image if docker + the image are available, else None."""
        if not shutil.which("docker"):
            return None
        image = os.getenv("SCAN_ZAP_IMAGE", "zaproxy/zap-stable")
        rc, _, _ = self._run(["docker", "image", "inspect", image], timeout=20)
        return image if rc == 0 else None

    # ---- individual tools ---------------------------------------------------
    def nmap_scan(self, target: str) -> ScanReport:
        self._guard(target)
        if not (self.live and shutil.which("nmap")):
            return self._sim(target, "nmap")
        # Port selection: default the top 1000 (covers the common service range). SCAN_NMAP_PORTS
        # overrides — a raw nmap flag ("--top-ports 3000", "-p-") or a port spec ("1-10000",
        # "22,80,8230"). An explicit target port (host:PORT / URL) is always included, so services
        # on high ports (e.g. an agent on :8230, outside the top-1000) aren't missed.
        env_ports = os.getenv("SCAN_NMAP_PORTS", "").strip()
        explicit = port_of(target)
        if env_ports:
            port_args = env_ports.split() if env_ports.startswith("-") else ["-p", env_ports]
        elif explicit:
            port_args = ["-p", f"1-1000,{explicit}"]
        else:
            port_args = ["--top-ports", "1000"]
        rc, out, err = self._run(["nmap", "-Pn", "-T4", *port_args, "-sV",
                                  "--script", "ssl-enum-ciphers", host_of(target)])
        if rc != 0:
            return ScanReport(False, target, ["nmap"], note=f"nmap rc={rc}: {err[:140]}")
        return ScanReport(True, target, ["nmap"], findings=_parse_nmap(out, target))

    def nuclei_scan(self, target: str, severity: str = "low,medium,high,critical") -> ScanReport:
        self._guard(target)
        if not (self.live and shutil.which("nuclei")):
            return self._sim(target, "nuclei")
        cmd = ["nuclei", "-u", _as_url(target), "-severity", severity, "-jsonl", "-silent", "-duc"]
        tdir = os.getenv("SCAN_NUCLEI_TEMPLATES")   # pin templates dir (container bakes a fixed path)
        if tdir:
            cmd += ["-t", tdir]
        rc, out, err = self._run(cmd)
        if rc != 0:
            return ScanReport(False, target, ["nuclei"], note=f"nuclei rc={rc}: {err[:140]}")
        return ScanReport(True, target, ["nuclei"], findings=_parse_nuclei(out, target))

    def zap_baseline(self, target: str) -> ScanReport:
        return self._zap(target, active=False)

    def zap_active(self, target: str) -> ScanReport:
        return self._zap(target, active=True)

    def _zap(self, target: str, *, active: bool) -> ScanReport:
        self._guard(target)
        tool = "zap_active" if active else "zap_baseline"
        script = "zap-full-scan.py" if active else "zap-baseline.py"
        kind = "active" if active else "baseline"
        url = _as_url(target)
        zap_timeout = max(self.timeout, float(os.getenv("SCAN_ZAP_TIMEOUT", "600")))
        if not self.live:
            return self._sim(target, tool)
        # 1) native ZAP scripts on PATH. zap-baseline.py writes -J RELATIVE to the ZAP
        #    working dir (/zap/wrk in the ZAP image), not an absolute path — so pass a
        #    basename and read it back from that working dir.
        if shutil.which(script):
            workdir = os.getenv("SCAN_ZAP_WORKDIR", "/zap/wrk")
            name = f"zap_{uuid.uuid4().hex[:8]}.json"
            rc, _, _ = self._run([script, "-t", url, "-J", name, "-I"], timeout=zap_timeout)
            raw = _read(os.path.join(workdir, name)) or _read(name)   # fallback: process cwd
            return ScanReport(bool(raw) or rc == 0, target, ["zap"],
                              findings=_parse_zap(raw, target), note=f"{kind} native (rc={rc})")
        # 2) containerized ZAP (zaproxy/zap-stable) — report lands in the mounted /zap/wrk
        image = self._docker_image()
        if image:
            import tempfile
            workdir = tempfile.mkdtemp(prefix="zap_")
            os.chmod(workdir, 0o777)   # ZAP container's 'zap' user must be able to write the report
            # --network host so the container can reach host-loopback / LAN deployments
            # (a container's 127.0.0.1 is its own, not the host's). Works for public URLs too.
            net = os.getenv("SCAN_ZAP_NETWORK", "host")
            rc, _, err = self._run(
                ["docker", "run", "--rm", "--network", net, "-v", f"{workdir}:/zap/wrk/:rw", image,
                 script, "-t", url, "-J", "report.json", "-I"], timeout=zap_timeout)
            raw = _read(os.path.join(workdir, "report.json"))
            note = f"{kind} docker (rc={rc})" + ("" if raw else f" — no report: {err[:100]}")
            return ScanReport(bool(raw) or rc == 0, target, ["zap"],
                              findings=_parse_zap(raw, target), note=note)
        # 3) neither available → simulate
        return self._sim(target, tool)

    # ---- simulation ---------------------------------------------------------
    def _sim(self, target: str, tool: str) -> ScanReport:
        """Deterministic simulated findings. A latent exposure keyed to the target class is only
        surfaced by its decisive tool — so a bundle missing that tool misses it."""
        cls = scan_bucket(target)
        findings: list[Finding] = []
        latent = _SIM_LATENT.get(cls)
        if latent and latent[0] == tool:
            sev, name, ref = latent[1]
            findings.append(Finding(sev, name, target, tool.split("_")[0], evidence="simulated", ref=ref))
        findings.append(Finding("info", f"{tool} baseline complete", target, tool.split("_")[0],
                                evidence="simulated"))
        return ScanReport(True, target, [tool.split("_")[0]], findings=findings, simulated=True)

    def run_tool(self, tool: str, target: str) -> ScanReport:
        fn = {"nmap": self.nmap_scan, "nuclei": self.nuclei_scan,
              "zap_baseline": self.zap_baseline, "zap_active": self.zap_active}.get(tool)
        if fn is None:
            return ScanReport(False, target, [tool], note=f"unknown tool {tool}")
        return fn(target)


# profile → ordered tool list (used by the MCP server + tenant)
PROFILES: dict[str, list[str]] = {
    "recon": ["nmap"],
    "web-baseline": ["nuclei", "zap_baseline"],
    "web-active": ["nuclei", "zap_active"],
    "full": ["nmap", "nuclei", "zap_baseline"],
}
ACTIVE_PROFILES = {"web-active", "full"}


def _as_url(target: str) -> str:
    return target if "://" in target else f"http://{target}"


def _read(path: str) -> str:
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return ""


def _parse_nmap(raw: str, target: str) -> list[Finding]:
    out: list[Finding] = []
    for line in raw.splitlines():
        m = re.match(r"(\d+)/tcp\s+open\s+(\S+)(?:\s+(.*))?", line.strip())
        if m:
            port, svc, ver = m.group(1), m.group(2), (m.group(3) or "").strip()
            out.append(Finding("info", f"open {svc} on :{port}" + (f" — {ver}" if ver else ""),
                               target, "nmap", evidence=line.strip()))
        if "SSLv3" in line or "TLSv1.0" in line or "TLSv1.1" in line:
            out.append(Finding("medium", "legacy TLS protocol offered", target, "nmap", evidence=line.strip()))
    return out


def _parse_nuclei(raw: str, target: str) -> list[Finding]:
    out: list[Finding] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        info = d.get("info", {})
        out.append(Finding(str(info.get("severity", "info")).lower(),
                           info.get("name", d.get("template-id", "finding")),
                           d.get("matched-at", target), "nuclei",
                           evidence=str(info.get("description", ""))[:200], ref=d.get("template-id", "")))
    return out


def _parse_zap(raw: str, target: str) -> list[Finding]:
    out: list[Finding] = []
    try:
        d = json.loads(raw)
    except ValueError:
        return out
    _risk = {"3": "high", "2": "medium", "1": "low", "0": "info"}
    for site in d.get("site", []):
        for a in site.get("alerts", []):
            out.append(Finding(_risk.get(str(a.get("riskcode", "0")), "info"),
                               a.get("name", "alert"), target, "zap",
                               evidence=str(a.get("desc", ""))[:200], ref=str(a.get("pluginid", ""))))
    return out
