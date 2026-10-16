# Contributing

1. Fork and branch from the default branch (`v4`).
2. `pip install -e ".[dev]" numpy httpx && pytest`, and `python scripts/docs_check.py`.
3. Python is the source of truth; new features land here first (see [GO_PORT.md](GO_PORT.md)).
4. Any new number needs a card in `docs/evidence/` labelled *Illustrative/synthetic*, *Internally reproduced*,
   *Externally reproduced* or *Field validated*, plus the command that reproduces it.
5. Security issues: [SECURITY.md](SECURITY.md), not a public issue.

Contributions are under the repository [LICENSE](LICENSE).
