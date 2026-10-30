"""Static invariants of the geo Kubernetes image (deploy/geo-demo/Dockerfile.aks) and of the workflow that publishes it. No Docker.

The built image is checked by deploy/geo-demo/scan_image.py (run by the workflow before and after the push). Pinned here:
the base image is pinned by digest; the image is non-root with a HEALTHCHECK; the source is the named build context
`context_runtime_src` (a checkout of a pinned release tag), never a local path; the agent-harness pin equals the repository's
own Dockerfile and pyproject.toml; the build context is deny-by-default; the workflow starts only from a version tag or a
manual dispatch, writes only the immutable version tag, signs by digest and attaches SBOM + provenance.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GEO = REPO / "deploy" / "geo-demo"
DOCKERFILE = (GEO / "Dockerfile.aks").read_text()
IGNORE = (GEO / "Dockerfile.aks.dockerignore").read_text()
WF = (REPO / ".github" / "workflows" / "build-geo-image.yml").read_text()
SECRET_SHAPED = re.compile(r"(SECRET|TOKEN|PASSWORD|PASSWD|API_?KEY|PRIVATE_?KEY|CREDENTIAL|_KEY$|_B64$)", re.I)


def _lines(text: str) -> list[str]:
    out, cur = [], ""
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if s.endswith("\\"):
            cur += s[:-1] + " "
            continue
        out.append((cur + s).strip())
        cur = ""
    return out


def test_base_image_is_pinned_by_digest_and_nothing_floats():
    m = re.search(r"^ARG PYTHON_IMAGE=(\S+)$", DOCKERFILE, re.M)
    assert m and re.search(r"@sha256:[0-9a-f]{64}$", m.group(1))
    for ln in _lines(DOCKERFILE):
        f = re.match(r"FROM\s+(\S+)", ln, re.I)
        if f:
            assert f.group(1).startswith("${PYTHON_IMAGE}") or f.group(1) == "venv", f.group(1)
    assert ":latest" not in DOCKERFILE


def test_image_is_nonroot_has_a_healthcheck_and_no_apparmor_workaround():
    final = DOCKERFILE.split("FROM ${PYTHON_IMAGE}\n")[-1]
    users = re.findall(r"^USER\s+(\S+)$", final, re.M)
    assert users and re.fullmatch(r"[1-9][0-9]*(:[1-9][0-9]*)?", users[-1])
    assert re.search(r"^HEALTHCHECK\b(?!.*NONE)", final, re.M)
    cmd = [ln for ln in _lines(final) if ln.startswith("CMD")][0]
    assert "exec uvicorn app:app" in cmd and "--loop asyncio" not in cmd and "--http h11" not in cmd


def test_source_is_the_named_context_not_a_local_path():
    assert "context_runtime_src" in DOCKERFILE
    for ln in _lines(DOCKERFILE):
        if ln.startswith("COPY") and "context_runtime" in ln and "healthcheck" not in ln:
            assert "--from=context_runtime_src" in ln, ln
    assert re.search(r"^COPY --from=context_runtime_src /context_runtime \./context_runtime$", DOCKERFILE, re.M)
    assert "COPY context_runtime " not in DOCKERFILE


def test_nothing_secret_shaped_or_credential_like_is_baked_in():
    for ln in _lines(DOCKERFILE):
        m = re.match(r"(ENV|ARG)\s+(.*)$", ln, re.I)
        if m:
            for n in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=", m.group(2)) or [m.group(2).split()[0]]:
                assert not SECRET_SHAPED.search(n), n
        if ln.upper().startswith(("COPY", "ADD")):
            assert not re.search(r"\.env\b|\.pem\b|\.key\b|credentials|id_rsa", ln), ln
            assert not ln.upper().startswith("ADD "), ln


def test_agent_harness_pin_equals_the_repository_dockerfile_and_pyproject():
    root = (REPO / "Dockerfile").read_text()
    for var in ("AGENT_HARNESS_COMMIT", "AGENT_HARNESS_SHA256"):
        mine = re.search(rf"^ARG {var}=(\S+)$", DOCKERFILE, re.M).group(1)
        assert mine == re.search(rf"^ARG {var}=(\S+)$", root, re.M).group(1), var
    commit = re.search(r"^ARG AGENT_HARNESS_COMMIT=(\S+)$", DOCKERFILE, re.M).group(1)
    assert commit in (REPO / "pyproject.toml").read_text()


def test_build_context_is_deny_by_default():
    lines = [ln.strip() for ln in IGNORE.splitlines() if ln.strip() and not ln.startswith("#")]
    assert lines == ["*", "!deploy/geo-demo/healthcheck.py"]
    assert re.search(r"^COPY deploy/geo-demo/healthcheck\.py ", DOCKERFILE, re.M)


def test_workflow_starts_only_from_a_version_tag_or_a_manual_run():
    on_block = WF.split("\non:\n", 1)[1].split("\npermissions:", 1)[0]
    assert re.findall(r"^  (\w+):", on_block, re.M) == ["push", "workflow_dispatch"]
    assert '"v[0-9]+.[0-9]+.[0-9]+"' in on_block and "branches" not in on_block and "pull_request" not in on_block
    assert re.search(r"publish:\n(?:.*\n)*?\s+default: false", on_block)
    assert 'default: "v0.4.1"' in on_block              # the pinned source release


def test_tag_policy_source_checkout_and_signing():
    for ln in WF.splitlines():
        if re.match(r"^\s*tags:\s*\S", ln):
            assert "latest" not in ln and "stable" not in ln and "github.sha" not in ln, ln
    assert "Version tags are immutable" in WF and "already exists" in WF
    assert "build-contexts: |\n            context_runtime_src=./src" in WF and WF.count("context_runtime_src=./src") == 2
    assert "ref: refs/tags/${{ env.SOURCE_REF }}" in WF
    assert "provenance: mode=max" in WF and "sbom: true" in WF
    assert 'cosign sign --yes "$REGISTRY/$IMAGE_NAME@$DIGEST"' in WF
    assert "cosign attest --yes --type spdxjson" in WF and "cosign verify-attestation" in WF
    assert WF.count("id-token: write") == 1
    assert WF.index("scan_image.py \"$IMAGE_NAME:ci\"") < WF.index("Build and push")
    for name in ("Log in to GHCR", "Build and push", "Sign the image by digest", "Attest the SBOM"):
        block = WF.split(f"- name: {name}", 1)[1].split("- name:", 1)[0]
        assert "if: env.PUBLISH == 'true'" in block, name
