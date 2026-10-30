"""The control-plane image builds, starts, and fails closed.

Builds the repository Dockerfile, starts it on loopback only and checks: GET /health answers, a
protected route answers 503 with no key configured, 401 with a wrong key and 200 with the key.
Skipped when docker (or its daemon) is not available. Synthetic keys only.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
KEY = "container-test-synthetic-key-0001"


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def test_dockerfile_pins_the_same_agent_harness_commit_as_pyproject():
    """Runs without docker: the image installs agent-harness at the commit pyproject.toml pins."""
    pyproject = (ROOT / "pyproject.toml").read_text()
    dockerfile = (ROOT / "Dockerfile").read_text()
    pinned = re.search(r"agent-harness @ git\+https://github\.com/redevops-io/agent-harness\.git@([0-9a-f]{40})", pyproject)
    commit = re.search(r"^ARG AGENT_HARNESS_COMMIT=([0-9a-f]{40})$", dockerfile, re.M)
    digest = re.search(r"^ARG AGENT_HARNESS_SHA256=([0-9a-f]{64})$", dockerfile, re.M)
    assert pinned and commit and digest
    assert commit.group(1) == pinned.group(1)
    assert "#sha256=${AGENT_HARNESS_SHA256}" in dockerfile


def _get(url: str, key: str | None = None) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"X-API-Key": key} if key else {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310 (loopback)
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def _run(image: str, env: dict[str, str], started: list[str]) -> str:
    """Start the image on loopback, register the container for cleanup, wait for /health, return the base URL."""
    args = ["docker", "run", "-d", "-p", "127.0.0.1::8080"]
    for k, v in env.items():
        args += ["-e", f"{k}={v}"]
    cid = subprocess.run([*args, image], capture_output=True, text=True, check=True).stdout.strip()
    started.append(cid)
    port_line = subprocess.run(["docker", "port", cid, "8080/tcp"], capture_output=True, text=True, check=True).stdout
    port = port_line.strip().splitlines()[0].rsplit(":", 1)[1]
    base = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            if _get(base + "/health")[0] == 200:
                return base
        except OSError:
            pass
        if subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", cid],
                          capture_output=True, text=True).stdout.strip() != "true":
            logs = subprocess.run(["docker", "logs", "--tail", "20", cid], capture_output=True, text=True)
            raise AssertionError("container exited: " + (logs.stdout + logs.stderr)[-800:])
        time.sleep(0.5)
    raise AssertionError("container did not answer /health")


@pytest.mark.skipif(not _docker_ready(), reason="docker is not available")
def test_image_builds_starts_and_fails_closed():
    image = f"context-runtime-control-plane-test:{uuid.uuid4().hex[:12]}"
    cids: list[str] = []
    try:
        build = subprocess.run(["docker", "build", "-q", "-t", image, str(ROOT)],
                               capture_output=True, text=True, timeout=900)
        assert build.returncode == 0, build.stderr[-1500:]

        # no key configured: the health route answers, everything else is 503
        base = _run(image, {}, cids)
        assert json.loads(_get(base + "/health")[1]) == {"status": "ok"}
        for path in ("/approvals", "/status", "/api/fleet", "/notify/status", "/agent/policy/control-tower", "/v1/models"):
            assert _get(base + path)[0] == 503, path
        assert _get(base + "/approvals", "any-key")[0] == 503

        # key configured: 401 without or with a wrong key, 200 with the key
        base = _run(image, {"CONTEXT_RUNTIME_API_KEY": KEY}, cids)
        assert _get(base + "/health")[0] == 200
        assert _get(base + "/approvals")[0] == 401
        assert _get(base + "/approvals", "wrong-key")[0] == 401
        status, body = _get(base + "/approvals", KEY)
        assert status == 200 and json.loads(body) == []
        assert _get(base + "/status", KEY)[0] == 200
    finally:
        for c in cids:
            subprocess.run(["docker", "rm", "-f", c], capture_output=True)
        subprocess.run(["docker", "rmi", "-f", image], capture_output=True)
