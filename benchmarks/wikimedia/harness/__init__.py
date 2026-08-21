"""Evidence-native runtime benchmark on real Wikimedia (strategywiki) history.

Validates the shipped v0.2.x evidence-native features against a frozen, public,
non-synthetic temporal corpus. See ../README.md for the plan and ../dataset-manifest.json
for the frozen input identity.

The package is deliberately import-light at module load; each test arm imports the
runtime under test (runtime_contracts / discovery_runtime / agentic_os / context_runtime)
inside the arm, so profiling/parsing works even when a given runtime is not installed.
"""
