# Release 0.5.0: fail-closed planning and tool authorization

Read **Upgrading from 0.4.1** before upgrading: several changes refuse what 0.4.1 allowed. The full list is in
[CHANGELOG.md](../CHANGELOG.md).

## Policy and planning (security)

- **A policy-rejected plan is no longer executed.** When no candidate was both within the cost and latency limits and
  permitted, both optimizers fell back to *all* candidates, so a plan the `PolicyProvider` had rejected was served, marked
  feasible. A candidate over a hard limit was never even shown to the policy. The fallback is now limited to what the
  policy permits (marked infeasible). If it permits nothing, planning raises `PolicyDenied`.
- **A cached plan is no longer replayed against the policy.** `ContextRuntime.plan()` selected a plan and then returned
  the cached one. A revoked plan kept running, and identities sharing a cache shared its permission. A cached plan is now
  replayed only if the policy permits it for this goal now.

Affected: any deployment that passes a `PolicyProvider` to an optimizer. Without one, plan selection is as before.

## Upgrading from 0.4.1

1. **Handle `context_runtime.plugins.base.PolicyDenied`** where you call `plan`, `run`, `explain` or `simulate` with a
   policy installed. It carries `.rejected`, the candidates and reasons.
2. **A tool authorizer must return a decision.** A bare `None` used to mean "allow"; it now means "nobody decided" and is
   **denied**. Return `agent_harness.tools.Authorization.allow(...)` / `.deny(...)`. If other code reads your authorizer's
   `reason | None` answer, keep it and attach the decision-returning form as its `engine_authorizer` attribute: a tool
   registry calls that one.
3. **A tenant does not start without an authorization plane**: pass `authorizer=`, install one with
   `set_default_authorizer(...)`, or declare local trust with a written reason (`CR_LOCAL_AUTHORITY="<why>"`; `1` is
   refused).
4. **Declare each tool's risk class** (`mount_mcp(..., actions=..., default_action="write")`, `AgentConsole.tool(...,
   action=...)`). It is no longer inferred from what an MCP server says about itself.
5. **Corpus references that name a Windows device** (`con`, `nul`, `com1`, ...) are refused on every platform, and policy
   scopes with such names are stored under a different file name; nothing is migrated automatically.
6. Outside containers, state and caches default to the platform's directories; a non-empty `.context-runtime` in the
   working directory is still used, with a warning. See [native-portability.md](native-portability.md).

## Learning

- Every observation of a persisted bandit policy is recorded with who or what produced it; there is a kill switch
  (`CR_LEARNING_FROZEN=1`), and per-context rollback and reset. A policy file that cannot be read is set aside, not
  overwritten. See [LEARNERS.md](LEARNERS.md), which also says what is not built.
- A reward that is not a finite number is refused.

## Verification

The test suite passes on the released commit. The enterprise policy engine was also run through this runtime with canary
retrievers and models, and its tests of the cached-plan cases fail on a runtime without the fix.
