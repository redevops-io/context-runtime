# Release 0.5.2

An addition on [0.5.1](release-0.5.1.md). No settings or existing interfaces change.

- **Self-Improvement Memory selection** (`context_runtime.learning.sim_select`). When an `agent-harness.sim/1` record does
  not fit a token budget, it returns a view that keeps every problem visible, every attempt and measured field of each
  problem it shows, and the regression evidence that bears on the decision. The complete record is returned when it fits,
  and `BudgetTooSmall` is raised when the evidence that must be kept does not fit.

**Not established:** that an optimizer given a selected view does as well as one given the complete record. No such
comparison has been run.

## Verification

The test suite passes on the released commit.
