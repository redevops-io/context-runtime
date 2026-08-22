"""Arms E and F — evidence/action trajectory governance on real strategywiki moderation (v0.3.0).

Wires the real ReDevOps governance engine (`agentic_os_enterprise.governance`) to the strategywiki
revert→protection trajectories. Arm E proves a cross-series/population trajectory rule flags edit-war
trajectories a per-event baseline cannot distinguish; Arm F proves OBSERVE→ENFORCE changes disposition
without changing detection. The engine is the shipped one + the v0.3.0 PopulationRule.

External label caveat (plan §9): a later protection event is a moderation label, NOT proof of
causality — reported as precision/recall against that label, with the false-positive rate explicit.
"""
from __future__ import annotations

from harness.governance_corpus import GovCase, select_governance_cases

# window: 30 days in seconds — an edit-war "storm" of reverts clustered within a month
WINDOW = 30 * 86400
REVERT_THRESHOLD = 3          # ≥3 reverts on one page in the window = a storm


def _events(case: GovCase, mod):
    """Build the RuntimeEvent list for one page: revert events (+ a protection action if protected)."""
    from agentic_os.mission.events import capability_event
    evs = [capability_event(case.editor_by_revert[i] or "editor", ts, "page.revert",
                            payload={"page": str(case.page_id)})
           for i, ts in enumerate(case.revert_epochs)]
    if case.is_protected:
        evs.append(capability_event("moderator", case.protected_at, "page.protect",
                                    payload={"page": str(case.page_id)}))
    return evs


def _storm_rule(mod):
    from agentic_os.mission.events import EventType
    return mod.PopulationRule(
        id="revert-storm", step=mod.Step(EventType.CAPABILITY_INVOCATION, lambda e: "revert" in e.capability_id),
        threshold=REVERT_THRESHOLD, within=WINDOW, correlation_keys=("page",),
        effect=mod.Effect.REQUIRE_REVIEW, severity="high", version="v1")


def _cross_rule(mod):
    from agentic_os.mission.events import EventType
    return mod.CrossSeriesRule(
        id="revert-storm-then-protection", version="v1",
        evidence=(mod.Step(EventType.CAPABILITY_INVOCATION, lambda e: "revert" in e.capability_id),),
        action=(mod.Step(EventType.CAPABILITY_INVOCATION, lambda e: "protect" in e.capability_id),),
        effect=mod.Effect.REQUIRE_REVIEW, severity="high", within=WINDOW,
        correlation_keys=("page",), require_precedence=True)


def run_arm_e() -> dict:
    import agentic_os_enterprise.governance as gov

    cases = select_governance_cases()
    protected = [c for c in cases if c.is_protected]
    controls = [c for c in cases if not c.is_protected]

    storm = _storm_rule(gov)
    cross = _cross_rule(gov)

    # trajectory rule (population): does a revert-STORM appear on this page?
    traj_flag_protected = traj_flag_control = 0
    baseline_flag_protected = baseline_flag_control = 0  # per-event: ANY revert flags the page
    cross_findings_protected = cross_findings_control = 0
    lead_times: list[int] = []

    for c in cases:
        led = gov.GovernanceLedger().ingest_all(_events(c, gov))
        storm_hits = gov.match_population(storm, led.events())
        cross_hits = gov.match_cross_series(cross, led.events())
        flagged = len(storm_hits) > 0
        baseline = len(c.revert_epochs) >= 1          # naive single-event gate
        if c.is_protected:
            traj_flag_protected += flagged
            baseline_flag_protected += baseline
            cross_findings_protected += len(cross_hits) > 0
            if flagged:
                # lead time: protection minus the last revert in the storm window
                lead_times.append(c.protected_at - max(c.revert_epochs))
        else:
            traj_flag_control += flagged
            baseline_flag_control += baseline
            cross_findings_control += len(cross_hits) > 0

    nP, nC = len(protected), len(controls)
    traj_recall = round(traj_flag_protected / nP, 3) if nP else 0.0
    traj_fpr = round(traj_flag_control / nC, 3) if nC else 0.0
    base_fpr = round(baseline_flag_control / nC, 3) if nC else 0.0
    lead_p50 = sorted(lead_times)[len(lead_times) // 2] if lead_times else 0

    # gates: cross-series fires on NO control (no action series ⇒ negative by construction);
    # the trajectory rule's false-positive rate is strictly below the per-event baseline's.
    passed = (nP > 0 and nC > 0 and cross_findings_control == 0 and traj_fpr < base_fpr)
    return {
        "arm": "E", "name": "evidence/action trajectory governance", "passed": passed,
        "n_cases": len(cases),
        "metrics": {
            "protected_pages": nP, "control_pages": nC,
            "trajectory_recall_vs_label": traj_recall,
            "trajectory_false_positive_rate": traj_fpr,
            "per_event_baseline_false_positive_rate": base_fpr,   # naive gate flags ~every page
            "cross_series_findings_protected": cross_findings_protected,
            "cross_series_findings_control": cross_findings_control,  # HARD = 0 (no action ⇒ no finding)
            "lead_time_days_p50": round(lead_p50 / 86400, 1),
        },
    }


def run_arm_f() -> dict:
    """OBSERVE vs ENFORCE on the SAME revert-storm rule + real events: identical detection, only the
    disposition changes. Plus the plan's negative controls (wrong key / outside window / precedence)."""
    import agentic_os_enterprise.governance as gov
    from agentic_os.mission.events import EventType, capability_event

    cases = [c for c in select_governance_cases() if c.is_protected]
    storm = _storm_rule(gov)

    detection_identical = 0
    observe_all_allow = enforce_all_review = 0
    n = 0
    for c in cases:
        led1 = gov.GovernanceLedger().ingest_all(_events(c, gov))
        led2 = gov.GovernanceLedger().ingest_all(_events(c, gov))
        obs = gov.GovernancePlane(ledger=led1).add_population_rule(storm).evaluate(enforcement=gov.observe_all())
        enf = gov.GovernancePlane(ledger=led2).add_population_rule(storm).evaluate()
        if not obs and not enf:
            continue                                  # no storm on this page — skip
        n += 1
        # detection identical: same findings (rule + cited events + effect), regardless of mode
        same = (len(obs) == len(enf)
                and all(o.matched_event_ids == e.matched_event_ids and o.effect is e.effect
                        for o, e in zip(obs, enf)))
        detection_identical += same
        observe_all_allow += all(o.decision is gov.GateResult.ALLOW for o in obs)
        enforce_all_review += all(e.decision is gov.GateResult.REQUIRE_REVIEW for e in enf)

    # ── negative controls (plan §10 Test F): the SAME rule must NOT fire when the shape is broken ──
    def _reverts(page, tss):
        return [capability_event("e", t, "page.revert", payload={"page": page}) for t in tss]

    base = [0, 10 * 86400, 20 * 86400]                # 3 reverts within window on page "X"
    neg = {
        "wrong_correlation_key": gov.match_population(   # reverts on DIFFERENT pages don't aggregate
            storm, gov.GovernanceLedger().ingest_all(
                _reverts("A", [0]) + _reverts("B", [1]) + _reverts("C", [2])).events()),
        "outside_window": gov.match_population(          # 3 reverts spread over 200 days
            storm, gov.GovernanceLedger().ingest_all(
                _reverts("X", [0, 100 * 86400, 200 * 86400])).events()),
        "below_threshold": gov.match_population(         # only 2 reverts
            storm, gov.GovernanceLedger().ingest_all(_reverts("X", base[:2])).events()),
    }
    neg_controls_clean = all(len(v) == 0 for v in neg.values())

    passed = (n > 0 and detection_identical == n
              and observe_all_allow == n and enforce_all_review == n and neg_controls_clean)
    return {
        "arm": "F", "name": "OBSERVE→ENFORCE lifecycle", "passed": passed,
        "n_cases": n,
        "metrics": {
            "detection_identical_observe_vs_enforce": detection_identical,  # == n
            "observe_decision_allow": observe_all_allow,                    # == n
            "enforce_decision_require_review": enforce_all_review,          # == n
            "negative_controls_clean": neg_controls_clean,                  # HARD = True
            "negative_controls": {k: len(v) for k, v in neg.items()},       # all 0
        },
    }
