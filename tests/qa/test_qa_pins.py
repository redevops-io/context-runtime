"""Pin declarations and the prepared move from commit pins to release tags.

Everything here runs against a copy of the files. The real files are never rewritten, and nothing touches the network.
"""
from __future__ import annotations

import hashlib
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from qa_subproc import REPO, child_env

SPEC = importlib.util.spec_from_file_location("qa_pins", REPO / "scripts" / "qa_pins.py")
pins = importlib.util.module_from_spec(SPEC)
sys.modules["qa_pins"] = pins
SPEC.loader.exec_module(pins)

FILES = (pins.PYPROJECT, pins.WORKFLOW, pins.MANIFEST, *pins.DOCKERFILES)
CONTRACTS_PIN = "15a6f6cf79f22d27a920f59969dfeb2732a2d1cb"
HARNESS_PIN = "6b8ba3a596a5e2caf0fdf009f2af013517591662"
TAGS = dict(contracts_tag="v1.2.3", harness_tag="v4.5.6", harness_sha256=hashlib.sha256(b"archive of the tag").hexdigest())


def copy_tree(tmp_path: Path) -> Path:
    root = tmp_path / "copy"
    for rel in FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, root / rel)
    return root


def digests() -> dict[str, str]:
    return {rel: hashlib.sha256((REPO / rel).read_bytes()).hexdigest() for rel in FILES}


def cli(root: Path, *args: str):
    return subprocess.run([sys.executable, str(REPO / "scripts" / "qa_pins.py"), "--root", str(root), *args],
                          capture_output=True, text=True, env=child_env(), timeout=60)


def tag_args(**over):
    t = {**TAGS, **over}
    return ["rewrite", "--contracts-tag", t["contracts_tag"], "--harness-tag", t["harness_tag"],
            "--harness-sha256", t["harness_sha256"]]


# ── the declarations in this repository ────────────────────────────────────────────────────────────────────

def test_the_declarations_in_this_repository_agree_with_the_manifest():
    assert pins.check(REPO) == []


def test_the_commit_pins_are_the_expected_ones_and_the_existing_harness_pin_is_untouched():
    m = pins.load_manifest(REPO)
    assert m["runtime-contracts"]["commit"] == CONTRACTS_PIN and m["agent-harness"]["commit"] == HARNESS_PIN
    assert m["runtime-contracts"]["temp_pin"] is True and m["agent-harness"]["temp_pin"] is True
    assert m["runtime-contracts"]["tag"] == "" and m["agent-harness"]["tag"] == ""
    pyproject = (REPO / "pyproject.toml").read_text()
    assert f"agent-harness.git@{HARNESS_PIN}" in pyproject
    assert f"runtime-contracts.git@{CONTRACTS_PIN}\"]  # TEMP-PIN" in pyproject
    for rel in pins.DOCKERFILES:
        text = (REPO / rel).read_text()
        assert f"ARG AGENT_HARNESS_COMMIT={HARNESS_PIN}" in text
        assert "ARG AGENT_HARNESS_SHA256=" + m["agent-harness"]["archive_sha256"] in text


def test_a_release_check_fails_while_the_pins_are_temporary():
    problems = pins.check(REPO, release=True)
    assert any("temporary commit pin" in p for p in problems) and any("marker remains" in p for p in problems)
    out = cli(REPO, "check", "--release")
    assert out.returncode == 1 and "FAILED" in out.stdout


def test_the_check_command_succeeds_offline_on_this_repository():
    out = cli(REPO, "check")
    assert out.returncode == 0, out.stdout + out.stderr


def test_only_the_marked_declarations_carry_the_marker():
    for rel in (pins.PYPROJECT, pins.WORKFLOW):
        assert "TEMP-PIN" in (REPO / rel).read_text()
    assert "TEMP-PIN" not in (REPO / "Dockerfile").read_text()


# ── drift is detected ──────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rel,old,new,fragment", [
    (pins.WORKFLOW, f"RUNTIME_CONTRACTS_PIN: {CONTRACTS_PIN}", "RUNTIME_CONTRACTS_PIN: " + "a" * 40, "RUNTIME_CONTRACTS_PIN"),
    (pins.PYPROJECT, f"runtime-contracts.git@{CONTRACTS_PIN}", "runtime-contracts.git@" + "b" * 40, "qa extra pins"),
    (pins.PYPROJECT, f"agent-harness.git@{HARNESS_PIN}", "agent-harness.git@" + "c" * 40, "agent-harness pins"),
    (pins.PYPROJECT, '"]  # TEMP-PIN', '"]', "marker"),
    (pins.WORKFLOW, "      # TEMP-PIN\n", "", "marker"),
    ("Dockerfile", f"ARG AGENT_HARNESS_COMMIT={HARNESS_PIN}", "ARG AGENT_HARNESS_COMMIT=" + "d" * 40, "ref is"),
    ("deploy/proxmox-demo/Dockerfile.contextruntime", "ARG AGENT_HARNESS_SHA256=", "ARG AGENT_HARNESS_SHA256=00", "ARG lines"),
])
def test_drift_between_declarations_and_the_manifest_is_reported(tmp_path, rel, old, new, fragment):
    root = copy_tree(tmp_path)
    path = root / rel
    text = path.read_text()
    assert text.count(old) >= 1, (rel, old)
    path.write_text(text.replace(old, new, 1))
    problems = pins.check(root)
    assert problems and any(fragment in p for p in problems), problems


def test_a_missing_or_malformed_manifest_fails_closed(tmp_path):
    root = copy_tree(tmp_path)
    (root / pins.MANIFEST).write_text("[runtime-contracts]\ncommit = \"abc\"\n")
    assert pins.check(root)
    (root / pins.MANIFEST).unlink()
    assert any("cannot read" in p for p in pins.check(root))


# ── the move to tags, against a copy ───────────────────────────────────────────────────────────────────────

def test_the_rewrite_is_a_dry_run_by_default_and_changes_nothing(tmp_path):
    root = copy_tree(tmp_path)
    before = {rel: (root / rel).read_bytes() for rel in FILES}
    out = cli(root, *tag_args())
    assert out.returncode == 0 and "dry run; no file was changed" in out.stdout
    assert f"+qa = [\"runtime-contracts @ git+https://github.com/redevops-io/runtime-contracts.git@{TAGS['contracts_tag']}\"]" in out.stdout
    assert {rel: (root / rel).read_bytes() for rel in FILES} == before


def test_writing_needs_the_confirmation_flag(tmp_path):
    root = copy_tree(tmp_path)
    before = {rel: (root / rel).read_bytes() for rel in FILES}
    out = cli(root, *tag_args(), "--write")
    assert out.returncode == 2 and "--confirm-tags-exist" in out.stderr
    assert {rel: (root / rel).read_bytes() for rel in FILES} == before


def test_the_confirmed_rewrite_moves_every_declaration_to_the_tags_and_the_result_passes_the_release_check(tmp_path):
    real_before = digests()
    root = copy_tree(tmp_path)
    out = cli(root, *tag_args(), "--write", "--confirm-tags-exist")
    assert out.returncode == 0, out.stderr
    assert "Follow-ups" in out.stdout and "tests/test_container.py" in out.stdout
    pyproject = (root / "pyproject.toml").read_text()
    assert f"runtime-contracts.git@{TAGS['contracts_tag']}\"]" in pyproject and "TEMP-PIN" not in pyproject
    assert re.search(rf'context-state = \["runtime-contracts @ [^"]*@{TAGS["contracts_tag"]}"\]', pyproject)
    assert f"agent-harness.git@{TAGS['harness_tag']}\"" in pyproject
    assert CONTRACTS_PIN not in pyproject and HARNESS_PIN not in pyproject
    workflow = (root / pins.WORKFLOW).read_text()
    assert f"RUNTIME_CONTRACTS_PIN: {TAGS['contracts_tag']}" in workflow and "TEMP-PIN" not in workflow
    for rel in pins.DOCKERFILES:
        text = (root / rel).read_text()
        assert f"ARG AGENT_HARNESS_REF={TAGS['harness_tag']}" in text
        assert f"ARG AGENT_HARNESS_SHA256={TAGS['harness_sha256']}" in text
        assert "/archive/refs/tags/${AGENT_HARNESS_REF}.tar.gz#sha256=${AGENT_HARNESS_SHA256}" in text
        assert "AGENT_HARNESS_COMMIT" not in text
    manifest = pins.load_manifest(root)
    assert manifest["runtime-contracts"]["tag"] == TAGS["contracts_tag"] and not manifest["runtime-contracts"]["temp_pin"]
    assert manifest["agent-harness"]["tag"] == TAGS["harness_tag"] and not manifest["agent-harness"]["temp_pin"]
    assert pins.check(root) == [] and pins.check(root, release=True) == []
    assert cli(root, "check", "--release").returncode == 0
    assert digests() == real_before, "the real files must never be touched by the tests"


def test_the_rewrite_changes_only_the_pin_lines_and_markers(tmp_path):
    root = copy_tree(tmp_path)
    changes = pins.compute_rewrite(root, **TAGS)
    import difflib

    for rel, (old, new) in changes.items():
        changed = [l for l in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0)
                   if l[:1] in "+-" and not l.startswith(("+++", "---"))]
        assert 0 < len(changed) <= 16, (rel, changed)
        assert all(re.search(r"pin|PIN|tag|AGENT_HARNESS|agent-harness|runtime-contracts|temp_pin|archive_sha256|"
                             r"curl|pip|--no-cache-dir|^[+-]\s*#|^[+-]\s*$", l, re.I) for l in changed), (rel, changed)


def test_the_rewrite_after_the_move_is_a_no_op(tmp_path):
    root = copy_tree(tmp_path)
    assert cli(root, *tag_args(), "--write", "--confirm-tags-exist").returncode == 0
    snapshot = {rel: (root / rel).read_bytes() for rel in FILES}
    again = cli(root, *tag_args())
    assert again.returncode == 2          # the manifest no longer has a temporary pin to move: refused, not repeated
    assert {rel: (root / rel).read_bytes() for rel in FILES} == snapshot


@pytest.mark.parametrize("bad", [dict(contracts_tag="main"), dict(contracts_tag="1.2.3"), dict(contracts_tag="v1.2"),
                                 dict(harness_tag=HARNESS_PIN), dict(harness_tag="v1.2.3-rc1"),
                                 dict(harness_sha256="abc"), dict(harness_sha256="G" * 64)])
def test_malformed_tags_and_digests_are_refused(tmp_path, bad):
    root = copy_tree(tmp_path)
    out = cli(root, *tag_args(**bad), "--write", "--confirm-tags-exist")
    assert out.returncode == 2
    assert pins.check(root) == []         # nothing was written


@pytest.mark.parametrize("missing", ["--contracts-tag", "--harness-tag", "--harness-sha256"])
def test_a_partial_move_is_not_possible(tmp_path, missing):
    root = copy_tree(tmp_path)
    args = tag_args()
    i = args.index(missing)
    del args[i:i + 2]
    assert cli(root, *args).returncode == 2        # argparse refuses: the pins only ever move together
    assert pins.check(root) == []


def test_the_rewrite_refuses_a_file_it_cannot_recognise(tmp_path):
    root = copy_tree(tmp_path)
    text = (root / "Dockerfile").read_text().replace("ARG AGENT_HARNESS_SHA256=", "ARG SOMETHING_ELSE=")
    (root / "Dockerfile").write_text(text)
    with pytest.raises(pins.PinError):
        pins.compute_rewrite(root, **TAGS)


# ── remote verification is isolated and offline here ───────────────────────────────────────────────────────

def test_verify_tags_asks_the_remote_for_exactly_the_tags_and_reports_what_is_missing(monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        tag = cmd[-1]
        out = f"abc\t{tag}\n" if tag.endswith("v1.2.3") else ""
        return subprocess.CompletedProcess(cmd, 0, out, "")

    monkeypatch.setattr(pins.subprocess, "run", fake_run)
    problems = pins.verify_tags("https://example.invalid/a.git", "v1.2.3", "https://example.invalid/b.git", "v9.9.9")
    assert [c[:3] for c in calls] == [["git", "ls-remote", "--tags"]] * 2
    assert len(problems) == 1 and "v9.9.9" in problems[0] and "was not found" in problems[0]


def test_verify_tags_reports_an_unreachable_remote_as_a_problem(monkeypatch):
    def boom(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1)

    monkeypatch.setattr(pins.subprocess, "run", boom)
    problems = pins.verify_tags("https://example.invalid/a.git", "v1.2.3", "https://example.invalid/b.git", "v1.2.3")
    assert len(problems) == 2 and all("could not ask" in p for p in problems)


def test_the_script_never_creates_tags_or_releases():
    text = (REPO / "scripts" / "qa_pins.py").read_text()
    for forbidden in ("git tag", '"tag",', "gh release", "git push", "git commit", "requests", "urllib"):
        assert forbidden not in text, forbidden
    assert "ls-remote" in text          # the only git call, and it only reads
