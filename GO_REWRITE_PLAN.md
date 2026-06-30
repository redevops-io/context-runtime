# Context Runtime — Go rewrite plan

Port the Python `context_runtime/` implementation to idiomatic Go under `go/`, driven by
**sidekick** (the `openai` branch, **GPT-5.5 Pro** driver). Each part is delegated as a
sidekick subtask with `go build/vet/test` as the acceptance bar. Parts are grouped into
**dependency tiers**: within a tier they're independent and run in parallel (sidekick's
internal `--concurrency`, or a second instance); tiers run in order because each needs the
previous ones to compile.

## Module & toolchain

- `go/` — module `github.com/redevops-io/context-runtime`, Go 1.26.
- One Go module, package-per-subsystem mirroring the Python layout.

## OSS Go frameworks (use where they fit)

| Concern | Python | Go OSS |
|---|---|---|
| LLM driver | LiteLLM / urllib | **github.com/sashabaranov/go-openai** (OpenAI + compatible: Kimi, etc.) |
| CLI | argparse | **github.com/spf13/cobra** |
| Control-plane HTTP | FastAPI | **net/http + github.com/go-chi/chi/v5** |
| Config / registry | pyyaml | **gopkg.in/yaml.v3** |
| Trace export | OpenLLMetry | **go.opentelemetry.io/otel** (+ stdout/JSONL) |
| Tests / asserts | pytest | stdlib `testing` + **github.com/stretchr/testify** |
| JSON | stdlib | stdlib `encoding/json` |
| Bandit / optimizer | hand-rolled | hand-rolled (small); **gonum.org/v1/gonum** only if needed |

**No-Go-equivalent deps** (port as thin HTTP/CLI binding or stub, note in code):
torch-backed redevops-rag → HTTP binding to the Python service or stub; HippoRAG → stub
graph retriever; Dagster → a small in-process executor; langfuse/otel are optional exporters.

## Package map (Python → Go) and tiers

**Tier 0 — foundation** *(running now)*
- `crtypes` ← `types.py` — all data structs with JSON tags + string-const enums.

**Tier 1 — depend only on `crtypes` (independent → parallel)**
- `jsonio` ← `jsonio.py` — round-trip + unknown-field preserve + spec-version guard.
- `plugins` ← `plugins/base.go` — the seam **interfaces** (Model, Reasoner, Retriever, Store, Scheduler, Cost*, Tool, TraceExporter, planner trio).
- `constraints` ← `constraints/hard.py` — feasibility.
- `execution` ← `execution/graph.py` — Execution-Graph IR + validate.
- `compression` ← `compression/structural.py` — clip + assemble.
- `verification` ← `verification/verifier.py` — citation verifier.
- `observability` ← `observability/{traces,exporters}.py` — trace builder + JSONL/OTel exporters.
- `tools` ← `tools/{base,retriever}.py` — ToolSpec/Result/Registry + ApprovalPolicy + ToolRetriever.
- `planner` ← `planner/{rules,intent,candidates}.py` — rule tables, intent analyzer, candidate gen.
- `costmodel` ← `costmodel/{score,estimators,statistics}.py` — PlanScore + heuristic estimator + stats.

**Tier 2 — depend on tier 1**
- `optimizer` ← `optimizer/knapsack.py` (costmodel + constraints).
- `scheduler` ← `scheduler/schedule.py` (execution).
- `reasoner` ← `reasoner/single_shot.py` (plugins).
- `adapters` ← `adapters/{model_stub, store_inmemory}.py` + **`model_openai.go`** (go-openai; replaces LiteLLM, routes gpt-5*pro → Responses) + `store_redevops.go` (HTTP binding) + `store_hipporag.go`/`store_router.go` (sim graph + hop router).
- `bandit` ← `integrations/bandit.py` — contextual ε-greedy + persistence.

**Tier 3 — depend on tier 2**
- `runtime` ← `runtime/{config,runtime}.py` — wires the lifecycle (plan→build→execute→verify→explain→simulate).
- `integrations` ← `integrations/{modules,sidekick,redevops_rag,edge_sentinel,vibexgen}.py` — the tenant fleet (bandit + runtime).

**Tier 4 — depend on tier 3**
- `controlplane` ← `control_plane/{registry,fleet,app}.py` — chi router serving the SAME API (`/health /status /modules /m/<name> /dispatch /approvals /agent/{run,outcome,policy} /vibex/*`), fleet = ModuleTenants, persistence under `$HOME`.
- `cmd/context-runtime` ← `runtime/cli.py` — cobra CLI (`run/explain/simulate/serve`).

## Acceptance bar (every part)

```
cd go && go build ./... && go vet ./... && go test ./...
```
Ported tests preserve the Python tests' intent (JSON round-trip, planner routing,
bandit convergence, tenant learning, control-plane endpoints). The module must stay
green after each tier merges.

## Delegation procedure

1. **One sidekick run per tier.** Task lists the tier's packages; sidekick fans out one
   subtask per package (`--concurrency 3–4`, `--max-subtasks` = tier size). Each subtask
   reads the named Python file(s) and writes the Go package + its test.
2. **Sequential across tiers** — verify `go build/test` green on the merged result before
   the next tier (a later tier won't compile until its deps landed).
3. **Multiple instances** only when a tier exceeds the subtask cap, or to run a second
   provider branch (e.g. kimi) on a disjoint package subset — kept disjoint to avoid
   merge races on the shared repo.
4. Env for every run: `PATH` includes the Go bin, `OPENAI_MODEL=gpt-5.5-pro`,
   `GOTOOLCHAIN=local`, `--no-vscode`.

## Status

- [x] Tier 0 `crtypes` — DONE (module + 26 types + round-trip test; go build/vet/test green).
- [ ] Tier 1 (10 packages) — pending tier 0 green.
- [ ] Tier 2 (5 packages) — pending tier 1.
- [ ] Tier 3 (runtime + integrations) — pending tier 2.
- [ ] Tier 4 (control plane + CLI) — pending tier 3.
