# Status

Last verification date: **2026-10-15**, commit `6e8aa19` on default branch `v4`, Python 3.12, Linux x86_64.

**Maturity: reference implementation (alpha), simulation-backed.** Not a production service. Not independently validated.

## Shipped and tested

| Item | Evidence |
|---|---|
| `ContextRuntime.run / explain / simulate`, offline with stub models | `README.md` tour; `tests/` |
| Learned-policy examples for 11 tenants | re-run 2026-10-15, all matched the README; [cards](docs/evidence/cr_tenant_reward_table.yaml) |
| v1 -> v2 simulated benchmark (Python) | [card](docs/evidence/cr_benchmark_v1_v2_headline.yaml) |
| Test suite | 349 collected: 338 passed, 11 skipped, with `pip install -e ".[dev]" numpy httpx && pytest`. [card](docs/evidence/cr_test_count.yaml) |
| Spec conformance checklist (SPEC section 10) | `tests/test_conformance.py` |

## Proposed / roadmap (not shipped)

OR-Tools CP-SAT optimizer, full execution-graph shapes, semantic plan cache, mixture reasoners, neural cost
model (see [ROADMAP.md](ROADMAP.md)). "Production runtime" is a **future** ROADMAP phase (v1.0), not a current state.

## Limitations

- All reward and benchmark numbers come from seeded simulators written by the same authors as the learner.
  Label for all of them: *Illustrative/synthetic*. See [docs/evidence/](docs/evidence/).
- One sidekick example *simulates the acceptance signal*.
- The Go port is in a separate private repository; parity claims are not checkable here.
- Heterogeneous-shard benchmark needs a FinanceBench corpus that is not in the repo.
- The ContractNLI 90-case figures are reported only, not independently reproduced; the source run is not in these repositories. See
  [docs/evidence/cr_contractnli_90case.yaml](docs/evidence/cr_contractnli_90case.yaml).

## Known gaps

- CI runs the unit suite (`tests.yml`, with the `dev`, `test`, `control-plane` and `duckdb` extras) and a gitleaks scan. The plain `[dev]` extra is not enough to run the suite locally:
  `numpy` and `httpx` are needed (the `test` extra installs them).
- No GitHub releases for most tags; not on PyPI.

## Versions

| Identifier | Meaning |
|---|---|
| `0.4.0` | `pyproject.toml` and `context_runtime.__version__` (from 2026-10-27; earlier tags carried older numbers, for example `v0.3.0` declared `0.2.0`) |
| tags `v0.1.0` to `v0.4.0` | semver release tags; `v*` is reserved for releases. `v0.4.0` marks the merge of the `w4` line into `main` |
| tags `w3`, `w4-snapshot-2026-07-11`, `w5`, `w7` | whitepaper tags (`w` and the whitepaper number) |
| branch `w4` (formerly `v4`) | the whitepaper-4 line, fully merged into `main` on 2026-10-27; kept unchanged for reproducing earlier results. The default branch is `main` |
| "v7.0" | a `git describe` string recorded in `runtime-contracts/adoption/implementations.yaml` |

No tags are changed by this documentation. For reproducibility, cite a commit hash.

## Evidence

[docs/evidence/](docs/evidence/) · [BENCHMARKS.md](BENCHMARKS.md) · [examples/](examples/)
