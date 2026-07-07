"""SupplyChainScanner — Context Runtime's software-supply-chain inspection plane.

"Inspect the open source we ship before clients run it." Wraps Trivy (CVEs across OS packages +
language dependencies, IaC misconfig, exposed secrets, and SBOM) and Syft (SBOM) as subprocesses,
normalizes their output, and degrades gracefully when the binaries aren't installed — the same
shell-out pattern edge-sentinel already uses for ``cscli``.

This is the PRE-DEPLOY half of the Security & Compliance block (the runtime half is Wazuh /
CrowdSec / Falco); the Edge Sentinel AgentConsole triages and explains the findings — our
scaled analog of Project Lightwell's AI clearinghouse.

No heavy deps: stdlib subprocess + json. Install ``trivy`` (and optionally ``syft`` / ``cosign``)
on the host to light up the real path.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field

_SEV_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "UNKNOWN": 4}


@dataclass
class Finding:
    id: str          # CVE / advisory id
    pkg: str
    installed: str
    fixed: str       # the version that resolves it — the "patched pin" to move to
    severity: str
    title: str
    target: str = ""


@dataclass
class ScanResult:
    ok: bool
    target: str
    scanner: str
    findings: list[Finding] = field(default_factory=list)
    secrets: int = 0
    misconfigs: int = 0
    note: str = ""

    def summary(self) -> dict:
        by: dict[str, int] = {}
        for f in self.findings:
            by[f.severity] = by.get(f.severity, 0) + 1
        return {
            "ok": self.ok,
            "target": self.target,
            "total": len(self.findings),
            "by_severity": by,
            "fixable": sum(1 for f in self.findings if f.fixed),
            "secrets": self.secrets,
            "misconfigs": self.misconfigs,
            "note": self.note,
        }


class SupplyChainScanner:
    def __init__(self, timeout: float = 300.0):
        self.timeout = timeout

    def available(self) -> dict:
        return {t: bool(shutil.which(t)) for t in ("trivy", "syft", "cosign", "clamscan")}

    def _run(self, cmd: list[str]) -> tuple[int, str, str]:
        try:
            p = subprocess.run(cmd, text=True, capture_output=True, timeout=self.timeout)
            return p.returncode, p.stdout, p.stderr
        except FileNotFoundError:
            return 127, "", "binary not found"
        except subprocess.TimeoutExpired:
            return 124, "", "scan timed out"
        except Exception as e:  # noqa: BLE001
            return 1, "", str(e)

    @staticmethod
    def _parse_trivy(raw: str, target: str = "") -> ScanResult:
        try:
            d = json.loads(raw)
        except Exception:  # noqa: BLE001
            return ScanResult(False, target, "trivy", note="unparseable trivy output")
        findings: list[Finding] = []
        secrets = misconfigs = 0
        for res in d.get("Results", []) or []:
            tgt = res.get("Target", target)
            for v in res.get("Vulnerabilities", []) or []:
                findings.append(Finding(
                    id=v.get("VulnerabilityID", ""),
                    pkg=v.get("PkgName", ""),
                    installed=v.get("InstalledVersion", ""),
                    fixed=v.get("FixedVersion", ""),
                    severity=(v.get("Severity") or "UNKNOWN").upper(),
                    title=(v.get("Title") or v.get("Description") or "")[:200],
                    target=tgt,
                ))
            secrets += len(res.get("Secrets", []) or [])
            misconfigs += len(res.get("Misconfigurations", []) or [])
        findings.sort(key=lambda f: (_SEV_ORDER.get(f.severity, 9), not f.fixed))
        return ScanResult(True, target, "trivy", findings=findings, secrets=secrets, misconfigs=misconfigs)

    def scan_fs(self, path: str = ".") -> ScanResult:
        """Scan a directory tree (dependencies, lockfiles, IaC, secrets)."""
        if not shutil.which("trivy"):
            return ScanResult(False, path, "trivy", note="trivy not installed — deploy it to enable supply-chain scanning")
        rc, out, err = self._run(["trivy", "fs", "--quiet", "--format", "json",
                                  "--scanners", "vuln,secret,misconfig", path])
        if not out:
            return ScanResult(False, path, "trivy", note=f"trivy fs failed (rc={rc}): {err[:140]}")
        return self._parse_trivy(out, path)

    def scan_rootfs(self, path: str = "/") -> ScanResult:
        """Scan an entire root filesystem — OS packages + installed language deps. Run from inside a
        container this inspects the container's FULL supply chain (the base image + everything we
        installed), surfacing CVEs that a lockfile scan of the app dir misses."""
        if not shutil.which("trivy"):
            return ScanResult(False, path, "trivy", note="trivy not installed")
        rc, out, err = self._run(["trivy", "rootfs", "--quiet", "--format", "json", "--scanners", "vuln", path])
        if not out:
            return ScanResult(False, path, "trivy", note=f"trivy rootfs failed (rc={rc}): {err[:140]}")
        return self._parse_trivy(out, path)

    def scan_image(self, ref: str) -> ScanResult:
        """Scan a container image reference for known CVEs."""
        if not shutil.which("trivy"):
            return ScanResult(False, ref, "trivy", note="trivy not installed")
        rc, out, err = self._run(["trivy", "image", "--quiet", "--format", "json", ref])
        if not out:
            return ScanResult(False, ref, "trivy", note=f"trivy image failed (rc={rc}): {err[:140]}")
        return self._parse_trivy(out, ref)

    def sbom(self, path: str = ".") -> dict:
        """Produce a component inventory (SBOM) — Syft if present, else Trivy CycloneDX."""
        if shutil.which("syft"):
            rc, out, _ = self._run(["syft", "-q", "-o", "syft-json", path])
            if out:
                try:
                    arts = json.loads(out).get("artifacts", []) or []
                    return {"ok": True, "tool": "syft", "components": len(arts),
                            "sample": [{"name": a.get("name"), "version": a.get("version"), "type": a.get("type")}
                                       for a in arts[:20]]}
                except Exception:  # noqa: BLE001
                    pass
        if shutil.which("trivy"):
            rc, out, _ = self._run(["trivy", "fs", "--quiet", "--format", "cyclonedx", path])
            if out:
                try:
                    comps = json.loads(out).get("components", []) or []
                    return {"ok": True, "tool": "trivy-cyclonedx", "components": len(comps),
                            "sample": [{"name": c.get("name"), "version": c.get("version")} for c in comps[:20]]}
                except Exception:  # noqa: BLE001
                    pass
        return {"ok": False, "note": "no SBOM tool available (install syft or trivy)"}
