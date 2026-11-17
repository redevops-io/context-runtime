#!/usr/bin/env python3
"""Fail if a built image carries secrets, credential files, deployment config or a root user.

    python3 -I deploy/geo-demo/scan_image.py IMAGE [--forbid-env-prefix PREFIX ...] [--allow-env NAME ...] [--mirror]

Checks (all on the image as built, through the docker CLI; nothing is run inside it):

  1. config    : a non-root numeric user, a HEALTHCHECK, and no environment variable whose NAME is secret-shaped
                 (SECRET, TOKEN, PASSWORD, API_KEY, PRIVATE_KEY, CREDENTIAL, *_KEY, *_B64) or starts with a forbidden
                 prefix (for example MAIL_ or CRM_ settings that a public demo must not carry). Values are never printed.
  2. history   : the build history (every layer's command) mentions no secret-shaped build arg or forbidden prefix,
                 and no `.env`/key file is copied in.
  3. filesystem: the exported root filesystem (every layer, merged) holds no credential-shaped FILE (.env, *.pem, *.key,
                 id_rsa, .netrc, .npmrc, .pypirc, .git-credentials, *credentials*.json, *service-account*.json, kubeconfig,
                 .git, ...) outside the system CA-bundle paths, and no file CONTENT with a private-key block that has a
                 body, a cloud/VCS access-token pattern, or a service-account key.

--mirror is for third-party images we copy but do not build (the registry mirrors): a root user or a missing HEALTHCHECK is then
printed as a NOTE (an upstream property the deployment handles with a pod security level or a runAsUser override), while
credential files, key material and secret-shaped environment names still fail.

Findings print the rule and the path only, never a value. Exit status 0 = clean, 1 = findings, 2 = usage/tooling error.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
import tarfile

SECRET_NAME = re.compile(r"(SECRET|TOKEN|PASSWORD|PASSWD|API_?KEY|PRIVATE_?KEY|CREDENTIAL|_KEY$|_B64$)", re.I)
# Variables the base image or the build legitimately sets and that look secret-shaped by name only.
DEFAULT_ALLOW_ENV = {"GPG_KEY"}        # python:* sets the PUBLIC key id used to verify the interpreter source tarball

FILE_GLOBS = [
    ".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "id_rsa*", "id_ed25519*", "id_ecdsa*", "id_dsa*",
    ".netrc", ".npmrc", ".pypirc", ".git-credentials", ".dockercfg", "*credentials*.json", "*service-account*.json",
    "*service_account*.json", "kubeconfig", "*.kubeconfig", "*.kdbx", "*.tfstate", "terraform.tfvars", "secrets.yml",
    "secrets.yaml", "secrets.json", "token.json", "client_secret*.json",
]
FILE_DIR_NAMES = {".git", ".aws", ".ssh", ".gnupg", ".kube"}
# System CA bundles are public certificates, not credentials.
FILE_ALLOW_PREFIXES = ("etc/ssl", "usr/share/ca-certificates/", "usr/lib/ssl/", "etc/ca-certificates", "usr/share/gnupg/")
# Debian's ssl-cert package ships a public, self-signed "snake-oil" key pair so services can start before real certificates exist.
FILE_ALLOW_EXACT = {"etc/ssl/private/ssl-cert-snakeoil.key", "etc/ssl/certs/ssl-cert-snakeoil.pem"}
FILE_ALLOW_SUFFIXES = ("/certifi/cacert.pem",)

CONTENT_PATTERNS = {
    "private-key block with a body": re.compile(rb"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----\s+[A-Za-z0-9+/=\r\n]{64,}"),
    "AWS access key id": re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,})\b"),
    "Slack token": re.compile(rb"\bxox[abprs]-[A-Za-z0-9-]{20,}"),
    "Google service-account key": re.compile(rb'"private_key"\s*:\s*"-----BEGIN[A-Z ]*PRIVATE KEY-----'),
    "Azure/Stripe/OpenAI-style secret key": re.compile(rb"\b(?:sk_live_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{40,})\b"),
}
# AWS documents this key id in its SDK typings and READMEs; any match ending in EXAMPLE is documentation, not a credential.
CONTENT_ALLOW_MATCH = {"AWS access key id": re.compile(rb"EXAMPLE$")}
CONTENT_MAX_BYTES = 2 * 1024 * 1024
CONTENT_SKIP_PREFIXES = ("usr/share/", "var/lib/dpkg/", "var/cache/", "usr/lib/x86_64-linux-gnu/", "usr/lib/aarch64-linux-gnu/",
                         "lib/x86_64-linux-gnu/", "lib/aarch64-linux-gnu/")


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw)


def check_config(image: str, forbid_prefix: list[str], allow_env: set[str], findings: list[str], notes: list[str], mirror: bool) -> None:
    cfg = json.loads(run(["docker", "image", "inspect", image]).stdout)[0]["Config"]
    user = (cfg.get("User") or "").strip()
    uid = user.split(":")[0]
    sink = notes if mirror else findings
    if not uid or uid in ("0", "root") or not uid.isdigit():
        sink.append(f"config: image user is {user or 'unset (root)'!r}; a numeric non-root user is required")
    if not cfg.get("Healthcheck") or cfg["Healthcheck"].get("Test") in (None, ["NONE"]):
        sink.append("config: no HEALTHCHECK")
    for entry in cfg.get("Env") or []:
        name = entry.split("=", 1)[0]
        if name in allow_env:
            continue
        if SECRET_NAME.search(name):
            findings.append(f"config: environment variable name {name} is secret-shaped (value not shown)")
        elif any(name.startswith(p) for p in forbid_prefix):
            findings.append(f"config: environment variable name {name} matches a forbidden prefix (value not shown)")


def check_history(image: str, forbid_prefix: list[str], findings: list[str]) -> None:
    hist = run(["docker", "history", "--no-trunc", "--format", "{{.CreatedBy}}", image]).stdout.splitlines()
    for line in hist:
        for m in re.finditer(r"\b(?:ARG|ENV)\s+([A-Z0-9_]+)", line):
            name = m.group(1)
            if name == "KERNEL_REF":
                continue
            if SECRET_NAME.search(name) and name not in DEFAULT_ALLOW_ENV:
                findings.append(f"history: a layer sets {name} (secret-shaped name)")
            elif any(name.startswith(p) for p in forbid_prefix):
                findings.append(f"history: a layer sets {name} (forbidden prefix)")
        if re.search(r"--mount=type=secret|COPY\s+\S*\.env\b|--secret\b", line):
            findings.append("history: a layer uses a secret mount or copies an .env file")


def _name_hit(path: str) -> str | None:
    parts = path.strip("/").split("/")
    base = parts[-1]
    if path in FILE_ALLOW_EXACT or any(path.startswith(p) for p in FILE_ALLOW_PREFIXES) or any(path.endswith(s) for s in FILE_ALLOW_SUFFIXES):
        return None
    for d in parts[:-1] if not path.endswith("/") else parts:
        if d in FILE_DIR_NAMES:
            return f"inside {d}/"
    if base in FILE_DIR_NAMES:
        return f"{base} directory"
    for g in FILE_GLOBS:
        if fnmatch.fnmatch(base.lower(), g.lower()):
            return f"matches {g}"
    return None


def check_filesystem(image: str, findings: list[str]) -> int:
    cid = run(["docker", "create", image]).stdout.strip()
    n = 0
    try:
        proc = subprocess.Popen(["docker", "export", cid], stdout=subprocess.PIPE)
        assert proc.stdout is not None
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            for m in tar:
                n += 1
                path = m.name.lstrip("./")
                hit = _name_hit(path + ("/" if m.isdir() else ""))
                if hit:
                    findings.append(f"filesystem: /{path} is a credential-shaped path ({hit})")
                if not m.isreg() or m.size == 0 or m.size > CONTENT_MAX_BYTES:
                    continue
                if path.startswith(CONTENT_SKIP_PREFIXES) or path in FILE_ALLOW_EXACT:
                    continue
                fh = tar.extractfile(m)
                data = fh.read() if fh else b""
                for rule, rx in CONTENT_PATTERNS.items():
                    allow = CONTENT_ALLOW_MATCH.get(rule)
                    if any(not (allow and allow.search(m.group(0))) for m in rx.finditer(data)):
                        findings.append(f"filesystem: /{path} contains a {rule}")
        proc.wait()
        if proc.returncode:
            raise RuntimeError("docker export failed")
    finally:
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--forbid-env-prefix", action="append", default=[], metavar="PREFIX",
                    help="fail if an env var name or build ARG/ENV starts with PREFIX (repeatable)")
    ap.add_argument("--allow-env", action="append", default=[], metavar="NAME", help="secret-shaped env NAME that is acceptable")
    ap.add_argument("--mirror", action="store_true", help="third-party image: root user / missing HEALTHCHECK are notes, not failures")
    args = ap.parse_args()
    findings: list[str] = []
    notes: list[str] = []
    try:
        check_config(args.image, args.forbid_env_prefix, DEFAULT_ALLOW_ENV | set(args.allow_env), findings, notes, args.mirror)
        check_history(args.image, args.forbid_env_prefix, findings)
        n = check_filesystem(args.image, findings)
    except (subprocess.CalledProcessError, FileNotFoundError, RuntimeError) as exc:
        print(f"scan_image: tooling error: {exc}", file=sys.stderr)
        return 2
    for note in notes:
        print(f"scan_image: note: {note}")
    if findings:
        print(f"scan_image: {len(findings)} finding(s) in {args.image}:")
        for f in sorted(set(findings)):
            print(f"  - {f}")
        return 1
    print(f"scan_image: {args.image} clean ({n} filesystem entries scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
