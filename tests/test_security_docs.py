"""SECURITY.md documents the settings the security behaviour depends on (including the explicit opt-ins)."""
from __future__ import annotations

from pathlib import Path

SECURITY = (Path(__file__).resolve().parent.parent / "SECURITY.md").read_text(encoding="utf-8")


def test_security_md_documents_the_opt_in_and_the_settings():
    for needle in ("CONTEXT_RUNTIME_MCP_INHERIT_ENV=1", "inherit_env=True", "CONTEXT_RUNTIME_API_KEY", "AGENTIC_OS_API_KEY",
                   "CR_TENANT_KEYS", "CR_ALLOW_ANONYMOUS_TENANTS", "CONTEXT_RUNTIME_CORPUS_ROOT", "hmac.compare_digest",
                   "GET /health"):
        assert needle in SECURITY, needle


def test_release_notes_exist_for_the_declared_version():
    import context_runtime
    root = Path(__file__).resolve().parent.parent
    notes = root / "docs" / f"release-{context_runtime.__version__}.md"
    assert notes.is_file()
    assert f"## {context_runtime.__version__}" in (root / "CHANGELOG.md").read_text(encoding="utf-8")
