# QA planning

`context_runtime.qa` plans a verification run. Given a declared graph of tests, fixtures, environment profiles and
resource claims, a set of selected tests, declared estimates and budgets, it produces a **`QAPlan`**: a fully resolved,
declarative plan with static lanes, gates and a coverage contract. It then explains the plan in a machine-readable record.
It decides *order, lanes, budgets and evidence level* and nothing else. It never drives a browser, never reads or edits an
assertion, and never executes the plan.

Requires the `qa` extra, which brings in `runtime-contracts` (the `qa/v1` schemas). Importing `context_runtime.qa` without
it fails with an install hint; there is no fallback.

## The data-only boundary

Context Runtime consumes `runtime-contracts` and emits a canonical `QAPlan` built with the contracts' own classes. It
**never imports `agent_harness.qa`**, and the QA planning package imports no browser driver, no runner and nothing from the
agent harness. Whatever executes a plan reads it as data and converts it on its own side. This is enforced by
`tests/qa/test_qa_import_boundary.py`, which scans every module and checks what importing the package loads.

## What it does

| Module | Role |
|---|---|
| `qa/closure.py` | Resolves the selected tests over the declared graph into the minimal closure, in a deterministic dependency order, with a reason for every included node. Refuses an unknown reference, a pin that does not match, a cycle (named) or a duplicate declaration, listing every finding. Derives each test's effective resource claims; a claim on an exclusive resource is a conflict key. |
| `qa/cost.py` | A deterministic cost model: declared per-test duration, cost and risk, with the basis of the figures. Inputs only; nothing is measured. A test without an estimate is refused. |
| `qa/planner.py` | `CostBasedQAPlanner`: list scheduling into static lanes for each candidate lane count, picks the lowest estimated makespan (fewest lanes on a tie), and builds the `QAPlan`. `QAPlanner` is the interface any planner can implement to return the same plan type. `verify_plan` re-checks a plan independently of how it was built. |
| `qa/explain.py` | `explain_plan`: why each test is included and placed, which constraints bound the choice, the critical path, the estimated makespan with its basis, and every rejected candidate with its reason. Hashed with the contracts' canonical hashing and attachable to the plan as `rationale_ref`. |
| `qa/policy.py` | The `ui_action` policy phase bound to the canonical `QAPolicyDecision`, and the default action policy whose hash a plan can carry as `action_policy_ref`. |
| `policy/plane.py` | The `ui_action` phase itself, next to the existing phases. |

`context_runtime.scheduler.TopoScheduler` builds Kahn waves for a retrieval graph with one concurrency number. It has no
durations, claims, conflict keys or capacities and no notion of static lanes with gates, so the QA planner has its own
list scheduler. Nothing outside `context_runtime/qa/` and the policy plane is changed.

## The plan and what is guaranteed

```python
from context_runtime.qa import (CostBasedQAPlanner, CostModel, Estimate, PlanBudget, PlanRequest, RunCaps,
                                plan_with_explanation)

request = PlanRequest(graph, targets, CostModel(estimates), PlanBudget(max_parallel=4, wall_time_ms=600_000),
                      RunCaps(deadline_ms=120_000, max_recovery_attempts=2, recovery_ceiling=2))
explained = plan_with_explanation(CostBasedQAPlanner(), request)
plan, explanation = explained.plan, explained.explanation   # a runtime_contracts.qa.QAPlan, and its hashed explanation
```

Structural guarantees, which hold whatever the real durations turn out to be:

- **FULL coverage.** Every test of the closure is planned exactly once. If the budget cannot cover them all, the planner
  refuses with `InfeasibleBudget` and names the binding constraint; it never drops a test. `coverage_contract` is `FULL`;
  a request for `SUBSET` (or anything else) is refused with `UNSUPPORTED_COVERAGE_CONTRACT`, as the contracts validator
  does.
- **Bounded parallelism.** The plan has at most `max_parallel` lanes, so at most that many tests run at once.
- **Conflict keys.** Tests whose claims meet on an exclusive resource are ordered, in one lane or through a gate chain.
  They are never concurrent.
- **Pool capacities.** The lane count is capped so that the largest claimants running together fit every declared
  capacity (memory, CPU, browser slots, ...).
- **Budgets within caps.** Every item's budget and recovery ceiling are at most the declared run caps, and at most the
  spec's own budget when the specs are supplied. Model use is off unless a cap grants it.

Estimated, not promised: start and finish times and the makespan. They come from the declared estimates only and are used
to choose between candidate lane counts. The explanation reports their basis.

The plan hash is the contracts' `rcv1` hash of the plan's canonical form. The same inputs give the same plan and the same
hash across processes, hash seeds and locales; the planner version is part of the plan's identity. Items, gates, lanes and
claims are emitted in the contracts' canonical orders.

The scheduling cost grows with the number of tests and the lane limit (each candidate lane count is scheduled once, and the
search stops when extra lanes stay unused); it is meant for suites of tens to a few hundred tests.

## Resource claims

A claim is `(resource, amount, exclusive)`. Nodes (a test, a fixture, an environment, a setup step) declare claims; a test's
effective claims are its own plus those of the non-test nodes it needs. An exclusive claim is a conflict key. A claim that is
neither exclusive nor against a declared pool capacity is refused, because nothing bounds it. A node marked `exclusive` is a
lock on the resource `<kind>:<id>`.

## The `ui_action` policy phase

`ui_action` decides whether one UI step may be performed. It is fail-closed:

- no matching rule is a deny;
- an unknown side-effect class (absent, `UNKNOWN`, or not a known class) is a deny, and so is a mutating step whose
  idempotency is `UNKNOWN`;
- a provider error or a timeout is a deny, and so is a failure to build the decision record;
- only `allow`, `deny` and `require_approval` are meaningful: a `redact` or `flag` rule counts as a deny;
- `WRITES_EXTERNAL`, `NOTIFIES_HUMAN` and `SPENDS_MONEY` are denied unless the reviewed spec lists the class in
  `allowed_side_effects` and the target is a local sink;
- the strictest matching rule wins (deny, then require_approval, then allow); a rule that cannot be evaluated is ignored
  when it allows and applies when it denies.

Every decision, allow or deny, emits a `PolicyDecision` from the plane, and `UiActionPolicy` builds the canonical
`QAPolicyDecision` for it. The record's `content_hash` is the reference a receipt cites (`policy_decision_ref`); an action
without such a reference must be refused by whatever performs actions. The default QA action policy is a document whose hash
is `default_action_policy_ref()`; a plan can carry it as `action_policy_ref`, which makes the policy part of the plan's
identity.

## The mandatory QA job

The `qa` job in `.github/workflows/tests.yml` is a required check. It fetches `runtime-contracts` at the pinned revision from
its public repository without a token (`contents: read`), verifies that the checked-out HEAD equals the pin, runs
`scripts/qa_pins.py check`, and runs `scripts/qa_ci_gate.py`. The gate fails, and never skips, when the dependency is missing
or unfetchable, when the fetched pin differs from the one declared for the `qa` extra, when zero tests under `tests/qa` are
collected, or when any test is skipped (the allow-list of skips is empty). The general unit-test job installs no git extras;
it leaves `tests/qa` out only when the package is absent and the run did not ask for those tests, and it says so in its
summary.

## Dependency pins: from commits to release tags

Until release tags exist, the QA code is pinned by commit, and each commit pin is temporary (`scripts/qa_pins.toml`,
`temp_pin = true`). Where this repository owns the text, a `TEMP-PIN` marker comment flags the declaration. The declarations
are the `qa` extra, the workflow's `RUNTIME_CONTRACTS_PIN`, the `context-state` extra, the `agent-harness` dependency in
`pyproject.toml`, and the `agent-harness` archive arguments in `Dockerfile` and
`deploy/proxmox-demo/Dockerfile.contextruntime`. The `agent-harness` pin is the one already in use; this change does not
move it.

[`scripts/qa_pins.py`](../scripts/qa_pins.py) checks that every declaration agrees with the manifest and prepares the move.

```bash
# every declaration agrees with the manifest; offline
python scripts/qa_pins.py check

# before a release: fails while any pin is a temporary commit pin or any marker remains
python scripts/qa_pins.py check --release
```

When both tags exist, the move is a separate, reviewed change:

```bash
CONTRACTS_TAG="${CONTRACTS_TAG:?the runtime-contracts tag that carries qa/v1}"
HARNESS_TAG="${HARNESS_TAG:?the agent-harness tag}"

# 1. confirm that both tags exist on the remotes (network)
python scripts/qa_pins.py verify-tags --contracts-tag "$CONTRACTS_TAG" --harness-tag "$HARNESS_TAG"

# 2. the sha256 of the harness tag's source archive, as the Dockerfiles verify it
HARNESS_SHA256="$(curl -fsSL "https://github.com/redevops-io/agent-harness/archive/refs/tags/${HARNESS_TAG}.tar.gz" | sha256sum | cut -d' ' -f1)"

# 3. dry run: prints a diff, changes nothing
python scripts/qa_pins.py rewrite --contracts-tag "$CONTRACTS_TAG" --harness-tag "$HARNESS_TAG" --harness-sha256 "$HARNESS_SHA256"

# 4. apply: rewrites every declaration together and removes the markers
python scripts/qa_pins.py rewrite --contracts-tag "$CONTRACTS_TAG" --harness-tag "$HARNESS_TAG" --harness-sha256 "$HARNESS_SHA256" --write --confirm-tags-exist

# 5. the pins now agree and none is temporary
python scripts/qa_pins.py check --release
```

The pins only ever move together, so the `qa` and `context-state` extras resolve to one contracts tag. After applying, update
the expectations in `tests/test_container.py` (the Dockerfile argument becomes `AGENT_HARNESS_REF`, a tag), refresh the pin
comments, build the image locally (the unit-test job does not run Docker), write the release notes and bump the version.
The script creates no tag, no release and no commit. The tests exercise the whole procedure against a copy of the files
(`tests/qa/test_qa_pins.py`).

## Tests

`tests/qa/` holds the QA tests: closure, selection (including property-style tests that run each plan with random real
durations and check conflicts, parallelism and capacities), the contracts validators and hash reproducibility, determinism
across hash seeds and locales, the golden plan, explanation, the `ui_action` phase, the import boundary, the pin tooling and
the CI gate. Mutation tests take a guard out of a throw-away copy of the package (cycle detection, the conflict-key check, the
FULL-coverage guarantee, the fail-closed default of `ui_action`, and others) and require the tests to fail.
`tests/test_policy_ui_action.py` covers the plane's `ui_action` phase and needs no QA schema package.
