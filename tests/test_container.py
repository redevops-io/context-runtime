"""The control-plane image builds, and what it does without a plane is the point of the file.

Builds the repository Dockerfile and starts it on loopback only:

* with nothing configured it **refuses to start** — a control plane whose every tool call would come back
  ``missing_explicit_grant`` is not a running service, and a container that exits says so in the one place
  an operator looks;
* declared diagnostic, it boots, says which mode it is on ``/health``, and refuses the routes that would
  make it execute;
* declared locally authoritative, it boots and behaves like the service it is: ``/health`` answers, a
  protected route is 503 with no key configured, 401 with a wrong key and 200 with the key.

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
#: What a container on the operator's own machine says when it has no permissions plane to install. A
#: sentence, not ``1`` — the same discipline the tool-registry seam applies to a local-trust switch.
LOCAL_AUTHORITY = "container test: an operator's own laptop, synthetic tenants, no real vendor reachable"


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        # The image is a Linux image (python:3.12-slim): a daemon in Windows-containers mode (the default on
        # GitHub-hosted Windows runners) cannot build it, so only a Linux engine counts as "ready".
        info = subprocess.run(["docker", "info", "--format", "{{.OSType}}"], capture_output=True, text=True,
                              encoding="utf-8", timeout=30)
        return info.returncode == 0 and info.stdout.strip() == "linux"
    except (OSError, subprocess.TimeoutExpired):
        return False


def test_dockerfile_pins_the_same_agent_harness_commit_as_pyproject():
    """Runs without docker: the image installs agent-harness at the commit pyproject.toml pins."""
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
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


def _post(url: str, key: str | None = None, body=None) -> tuple[int, str]:
    data = json.dumps(body if body is not None else []).encode()
    headers = {"content-type": "application/json", **({"X-API-Key": key} if key else {})}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310 (loopback)
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


@pytest.fixture(scope="module")
def control_plane_image():
    """One build for the three containers this file starts — the image is the subject, not the build."""
    if not _docker_ready():
        pytest.skip("a Linux docker engine is not available (no docker, or a Windows-containers daemon)")
    image = f"context-runtime-control-plane-test:{uuid.uuid4().hex[:12]}"
    build = subprocess.run(["docker", "build", "-q", "-t", image, str(ROOT)],
                           capture_output=True, text=True, encoding="utf-8", timeout=1800)
    assert build.returncode == 0, build.stderr[-1500:]
    yield image
    subprocess.run(["docker", "rmi", "-f", image], capture_output=True)


def _start(image: str, env: dict[str, str], started: list[str]) -> str:
    """``docker run -d`` with the given environment; registers the container for cleanup."""
    args = ["docker", "run", "-d", "-p", "127.0.0.1::8080"]
    for k, v in env.items():
        args += ["-e", f"{k}={v}"]
    cid = subprocess.run([*args, image], capture_output=True, text=True, encoding="utf-8",
                         check=True).stdout.strip()
    started.append(cid)
    return cid


def _base(cid: str) -> str:
    line = subprocess.run(["docker", "port", cid, "8080/tcp"], capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout
    return f"http://127.0.0.1:{line.strip().splitlines()[0].rsplit(':', 1)[1]}"


def _logs(cid: str) -> str:
    out = subprocess.run(["docker", "logs", "--tail", "60", cid], capture_output=True, text=True,
                         encoding="utf-8")
    return (out.stdout or "") + (out.stderr or "")


def _state(cid: str) -> tuple[bool, int]:
    """``(running, exit_code)``."""
    line = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}} {{.State.ExitCode}}", cid],
                          capture_output=True, text=True, encoding="utf-8", check=True).stdout.split()
    return line[0] == "true", int(line[1])


def test_the_image_refuses_to_start_with_nothing_to_ask(control_plane_image):
    """The default posture of a privileged service with no plane is to not be running.

    This is the behaviour the container test used to assert the opposite of: the image came up, answered
    ``/health`` with ``{"status": "ok"}``, and denied every tool call its fleet would ever make. A green
    health check over a fleet that cannot act is the failure this whole seam exists to remove, so the
    container now exits, and says which of the three doors to open.
    """
    started: list[str] = []
    try:
        cid = _start(control_plane_image, {}, started)
        for _ in range(240):
            running, code = _state(cid)
            if not running:
                assert code != 0, f"the container exited 0 having refused to serve: {_logs(cid)[-600:]}"
                logs = _logs(cid)
                assert "refusing to start" in logs, logs[-800:]
                assert "CR_LOCAL_AUTHORITY" in logs and "CR_EXECUTION_MODE" in logs   # names the exits
                return
            time.sleep(0.5)
        raise AssertionError(f"container was still running after 120s: {_logs(cid)[-800:]}")
    finally:
        for cid in started:
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True)


def test_the_image_boots_diagnostic_and_has_nothing_to_execute(control_plane_image):
    """The other door: a read-only instance may boot without a plane, and then has no execution path.

    It says so on the public health route — the one place a dashboard looks — and the routes that would
    make it act answer 503 instead of quietly standing tenants up with no way to authorize them.
    """
    started: list[str] = []
    try:
        cid = _start(control_plane_image, {"CONTEXT_RUNTIME_API_KEY": KEY,
                                           "CR_EXECUTION_MODE": "diagnostic",
                                           "CR_DIAGNOSTIC_REASON": "container test: a read-only console with "
                                                                   "no vendor credentials behind it"}, started)
        base = _wait_ready(cid)
        health = json.loads(_get(base + "/health")[1])
        assert health["status"] == "ok" and health["privileged_execution"] is False
        assert health["mode"] == "diagnostic"
        assert _post(base + "/up", KEY, ["agentic-billing"])[0] == 503
        assert _post(base + "/agent/run", KEY, {"module": "control-tower", "question": "q"})[0] == 503
        assert _post(base + "/dispatch", KEY, {"module": "agentic-billing", "agent": "checkout",
                                               "action": "refund", "prompt": "x"})[0] == 503
        # the read side of the surface is untouched: the fleet is still describable
        assert _get(base + "/status", KEY)[0] == 200
    finally:
        for cid in started:
            subprocess.run(["docker", "rm", "-f", cid], capture_output=True)


def _run(image: str, env: dict[str, str], started: list[str]) -> str:
    """Start the image on loopback, register the container for cleanup, wait for /health, return the base URL."""
    cid = _start(image, env, started)
    return _wait_ready(cid)


def _wait_ready(cid: str) -> str:
    """Wait for the container to answer /health; a container that exits is a failure with its logs attached."""
    base = _base(cid)
    for _ in range(120):
        try:
            if _get(base + "/health")[0] == 200:
                return base
        except OSError:
            pass
        if not _state(cid)[0]:
            raise AssertionError("container exited: " + _logs(cid)[-800:])
        time.sleep(0.5)
    raise AssertionError("container did not answer /health")


@pytest.mark.skipif(not _docker_ready(), reason="a Linux docker engine is not available (no docker, or a Windows-containers daemon)")
def test_image_starts_locally_authoritative_and_fails_closed(control_plane_image):
    """The execution posture, declared: the service comes up and the access rules hold.

    It carries ``CR_LOCAL_AUTHORITY`` because that is what makes a bare container legal to start at all —
    the sentence a laptop demo tells the gate, audited as a wildcard grant rather than assumed. Without it
    the container refuses to start, which the test above asserts on its own.
    """
    cids: list[str] = []
    authority = {"CR_LOCAL_AUTHORITY": LOCAL_AUTHORITY}
    try:
        # no key configured: the health route answers, everything else is 503
        base = _run(control_plane_image, dict(authority), cids)
        assert json.loads(_get(base + "/health")[1]) == {"status": "ok", "mode": "execute",
                                                          "privileged_execution": True}
        for path in ("/approvals", "/status", "/api/fleet", "/notify/status", "/agent/policy/control-tower", "/v1/models"):
            assert _get(base + path)[0] == 503, path
        assert _get(base + "/approvals", "any-key")[0] == 503

        # key configured: 401 without or with a wrong key, 200 with the key
        base = _run(control_plane_image, {"CONTEXT_RUNTIME_API_KEY": KEY, **authority}, cids)
        assert _get(base + "/health")[0] == 200
        assert _get(base + "/approvals")[0] == 401
        assert _get(base + "/approvals", "wrong-key")[0] == 401
        status, body = _get(base + "/approvals", KEY)
        assert status == 200 and json.loads(body) == []
        assert _get(base + "/status", KEY)[0] == 200
    finally:
        for c in cids:
            subprocess.run(["docker", "rm", "-f", c], capture_output=True)
