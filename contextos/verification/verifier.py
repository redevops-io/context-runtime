"""Verification — citation/grounding check (SPEC §4.7, §5.9).

v0.1 ships a deterministic citation verifier: every ``[n]`` the answer cites must
correspond to a real assembled context block. The heavier RAGAS/Instructor verifiers
plug in behind the same contract. Verification is part of execution: a high-risk
result isn't complete until it passes.
"""
from __future__ import annotations

import re

from ..types import BuiltContext, ModelResult, Plan, Verdict

_CITE = re.compile(r"\[(\d+)\]")


class CitationVerifier:
    def verify(self, result: ModelResult, plan: Plan, ctx: BuiltContext) -> Verdict:
        cited = {int(m) for m in _CITE.findall(result.text)}
        n_blocks = len(ctx.hits)
        findings: list[str] = []

        if not cited:
            findings.append("answer cites no sources")
            return Verdict(passed=False, confidence=0.2, findings=tuple(findings))

        dangling = sorted(c for c in cited if c < 1 or c > n_blocks)
        if dangling:
            findings.append(f"citations refer to non-existent blocks: {dangling}")

        passed = not dangling
        confidence = 0.9 if passed else 0.3
        return Verdict(passed=passed, confidence=confidence, findings=tuple(findings))
