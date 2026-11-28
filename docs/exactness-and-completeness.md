# Exactness and completeness are hard requirements

Status: candidate, in 0.5.x line. Everything below was measured on an internal development set; see **Not established**.

## What was wrong (reproduced on v0.5.2, `108c0d3`)

| id | defect | observed effect |
|---|---|---|
| D-1 | `HeuristicEstimator` tested the temporal / multi-hop *bucket* before the *analytical representation*, although `RuleCandidateGenerator` had already restricted the candidates to the representation. | A question that needs an exact aggregate ("total ... per year", "how many ... per type") and mentions a date was costed as a temporal question, so `hybrid` (top-k document retrieval) outranked `sql`. |
| D-2 | There was no way to say "the answer must be computed, not estimated". | Nothing stopped an exact aggregate from being answered out of the top-k of a retrieval. |
| D-3 | The retriever received `plan.intent.normalized`: lower-cased, stop-words dropped, de-duplicated and **sorted**. | Text-to-SQL received a scrambled token set, not the question. |
| D-4 | The reasoners presented `Question: {normalized}` to the model. | The answering model was shown the scrambled key as "the question". |

Routing probe, 17 questions, shipped v0.5.2: every question whose representation is analytical but whose bucket is temporal or multi-hop was routed to `hybrid`.

## What changes

* `Constraints.require_exact` / `require_complete` (default `False`). When set, **feasibility is checked before cost**: only plans whose retrieval method is served by a provider that declares the `exact` capability are eligible. If there is none, `plan()` raises `NoEligiblePlan` and `run()` returns a typed `Limitation` (`no_eligible_plan`) instead of a prose guess.
* An exact query that returns no rows is `empty_exact_result`; one cut at a row cap with `require_complete` is `incomplete_evidence`. Exact aggregation never silently falls back to top-k.
* Providers declare capabilities (`PluginInfo.capabilities`); `AnalyticalRetriever` declares `exact`. `HopRouterRetriever` reports the capabilities of the provider bound to each method.
* The estimator evaluates the analytical representation first.
* For exactness-constrained requests (or `Config.natural_question=True`) the retriever and the reasoners get the user's own question. The default for every other request is unchanged.
* The plan-cache key includes the two constraints only when they are set, so existing keys are byte-identical.

## Compatibility

* Defaults change nothing **except** estimator precedence (D-1), which is a behaviour fix on the default path: a request that is analytical by representation is no longer costed by its temporal / multi-hop bucket. The Go port (`context-runtime-go` @ `47cd956`) does not have this defect, but only because its planner never routes to `sql`: its keyword head never yields the analytical representation and `MethodForRepresentation` maps it to `hybrid`, so on all 29 probe questions it chose `hybrid`. It has no exactness constraint either. That is a separate parity gap, tracked in the Go port, not covered by this change.
* Test suite on the base commit: 1108 passed, 0 failed, 34 skipped. With this change: 1152 passed (1108 + 44 new), 0 failed, 34 skipped. Run with a clean environment (`env -i`).
* `tests/test_exactness_feasibility.py` (44 tests) is cross-domain (finance, procurement, inventory, temporal) and names no customer or domain in the runtime. Each test fails when its change is reverted.

## Measured effect (internal development set, 17 questions, three tabular sources, strict evaluator)

| configuration | passes per run |
|---|---|
| shipped v0.5.2 | 4 / 17 |
| this change, `require_exact=True`, shipped text-to-SQL | 11 / 17 (three runs) |

The remaining failures with the shipped text-to-SQL are errors in the SQL the model wrote (wrong filter, wrong aggregate), not routing: the executed query is in each hit's `meta["sql"]`. Providers that run typed operations under the `exact` capability reach 17 / 17 on the same set; that provider lives in the application, not here. See `docs/evidence/cr_exactness_dev_set.yaml`.

## Typed operations, narration verification and the fast path (follow-up, opt-in)

* `TypedOperationRetriever` (`adapters/store_typed.py`): the application supplies operations (`match`, `run`, `render`, `facts`, `provides`) and a measure vocabulary; the runtime holds no domain. It declares `exact` and `typed`.
* `verification/narration.py`: `verify_narration` / `serve_verified` check required facts (numbers with units, identifiers, dates, text) on token boundaries (`42` is not found in `00042` or `142`; `INV-00042` is not the number 42) and flag any figure in the answer that is not in the structured result or the question. A failing narration is replaced by the structured rendering, recorded in `RunResult.narration`. An explicit abstention is not required to repeat the facts but may not invent figures. Where the narration adds nothing, the structured result is what is served.
* Fast path (`Config.fast_path`, default off): requires `require_exact`, an analytical classification, a resolution whose measures are all provided by the matched operations, and no optimizer policy (the policy is evaluated per plan; the shortcut must not go around it). Otherwise the normal path (with its model) runs, so an ambiguous request still gets the fallback.

## Not established

* No result here generalises: the set is a development set, the holdout is a separate protocol.
* Cost-based optimisation showed no measurable value on this workload: once exactness is required every eligible method is exact and the remaining choice is arbitrary. Its value is untested where several meaningfully different feasible plans exist.
