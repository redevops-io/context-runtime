"""Every component that changes its behaviour from observations, and what it is allowed to change.

THE RULE: a learner chooses among alternatives somebody declared, or keeps a statistic that informs such a choice. No
learner may change policy, approval, authorization, egress or verification code or configuration. Anything stronger than a
choice among declared alternatives (a new prompt, a new tool default, a new threshold that gates an action) is a change
like any other and goes through the same review and approval - it is not something a learner does.

``tests/test_learner_registry.py`` walks the package: a class that grows a learning method has to be listed here, as a
learner (with what it changes, where it keeps it, how it is stopped and reset) or as not one (with why). The list of things
a learner may change is closed; adding one is a reviewed change to this file.

An honest entry says "none" where there is no kill switch or reset.
"""
from __future__ import annotations

from dataclasses import dataclass

LEARNING_METHODS = ("update", "observe", "learn", "record_outcome", "record", "observe_outcome", "fit", "record_feedback", "record_judgment",
                    "record_reward", "record_result", "calibrate", "train", "promote")
MAY_CHANGE = ("choice among declared arms", "cost estimate used in planning", "statistic that informs a choice",
              "review flag (only adds review)", "choice among plan shapes named by outcome events")


@dataclass(frozen=True)
class Learner:
    key: str            # "<module under context_runtime>:<Class>"
    learns: str
    changes: str        # one of MAY_CHANGE
    persisted: str
    kill_switch: str
    reset: str
    provenance: str


_BANDIT_STOP = "CR_LEARNING_FROZEN=1, or the bandit's .frozen"
_BANDIT_RESET = "bandit.reset(ctx, by=...) / rollback(...)"
_BANDIT_WHO = "ledger beside the persisted policy"


def _tenant(key: str, learns: str) -> Learner:
    """A tenant that learns ONLY through the shared bandit: it inherits the bandit's switch, reset and ledger. Its policy
    survives a restart (and has a ledger) only where its bandit was given a persist_path."""
    return Learner(key, learns, "choice among declared arms", "the shared bandit's, where a persist_path is set",
                   _BANDIT_STOP, _BANDIT_RESET, _BANDIT_WHO)


LEARNERS: tuple[Learner, ...] = (
    Learner("integrations/bandit.py:EpsilonGreedyBandit", "mean reward per (context, arm)", "choice among declared arms",
            "JSON file at persist_path", _BANDIT_STOP, _BANDIT_RESET, _BANDIT_WHO),
    Learner("optimizer/online.py:BanditOptimizer", "plan choice, through the bandit", "choice among declared arms",
            "through the bandit", _BANDIT_STOP, _BANDIT_RESET, _BANDIT_WHO),
    _tenant("control_plane/fleet.py:Fleet", "per-module bundle choice from outcomes"),
    _tenant("integrations/librechat.py:LibreChatTenant", "retrieval strategy per request kind, from feedback and judged scores"),
    _tenant("integrations/vibexgen.py:VibexgenPlanner", "generation chain per scene, from render scores"),
    # The one learner whose set of arms is NOT fixed in code: an outcome event naming an unknown plan shape adds it. Whoever
    # can publish to its bus can therefore add an arm, so the bus is part of what it trusts.
    Learner("learning/aggregator.py:LearningAggregator", "reward per plan shape from the outcome bus",
            "choice among plan shapes named by outcome events", "the shared bandit's, where a persist_path is set",
            _BANDIT_STOP, _BANDIT_RESET, _BANDIT_WHO),
    _tenant("integrations/agentic_billing.py:AgenticBillingTenant", "billing bundle per need"),
    _tenant("integrations/agentic_books.py:AgenticBooksTenant", "bookkeeping bundle per need"),
    _tenant("integrations/agentic_compliance.py:AgenticComplianceTenant", "compliance bundle per need"),
    _tenant("integrations/agentic_support.py:AgenticSupportTenant", "support bundle per need"),
    _tenant("integrations/chat_memory.py:ChatMemoryTenant", "recall mode"),
    _tenant("integrations/control_tower.py:ControlTowerTenant", "analysis arm per goal"),
    _tenant("integrations/edge_sentinel.py:SOCTriageTenant", "triage context bundle (not what gets blocked)"),
    _tenant("integrations/growth_engine.py:GrowthEngineTenant", "attribution arm"),
    _tenant("integrations/market_radar.py:MarketRadarTenant", "research arm per goal"),
    _tenant("integrations/modules.py:ModuleTenant", "module bundle"),
    _tenant("integrations/outreach_engine.py:OutreachEngineTenant", "research bundle per account signal (not whether to send)"),
    _tenant("integrations/redevops_rag.py:ContextRuntimeRetrieverTuner", "retrieval plan"),
    _tenant("integrations/revenue_intelligence.py:RevenueIntelligenceTenant", "provider bundle per need (NO_OUTREACH is not learned)"),
    _tenant("integrations/runtime_scan.py:RuntimeScanTenant", "scan bundle"),
    _tenant("integrations/sidekick.py:ContextRuntimeSkillStore", "skill choice per request"),
    _tenant("integrations/social_autopilot.py:SocialAutopilotTenant", "content arm (not whether to publish)"),
    _tenant("integrations/zoning_intelligence.py:ZoningIntelligenceTenant", "which providers to consult (never the determination)"),
    Learner("integrations/zoning_intelligence.py:PopulationGovernor", "recent downgrade rate per series", "review flag (only adds review)",
            "in memory", "none", "a new instance", "none"),
    Learner("optimizer/pattern_memory.py:PatternMemory", "reward per plan pattern", "choice among declared arms",
            "in memory", "none", "a new instance", "none"),
    Learner("optimizer/pattern_memory.py:_Stat", "PatternMemory's running statistic", "statistic that informs a choice",
            "in memory", "none", "a new instance", "none"),
    Learner("costmodel/estimators.py:HeuristicEstimator", "estimate-versus-actual corrections", "cost estimate used in planning",
            "through StatisticsStore (and CostProfile)", "none", "delete its statistics file", "none"),
    Learner("costmodel/statistics.py:StatisticsStore", "calibration statistics per estimator", "cost estimate used in planning",
            "JSON file at its path", "none", "delete its file", "none"),
    Learner("costmodel/statistics.py:_Accum", "StatisticsStore's accumulator", "statistic that informs a choice",
            "through StatisticsStore", "none", "delete the store's file", "none"),
    Learner("costmodel/profile.py:CostProfile", "measured stage latency", "cost estimate used in planning",
            "JSON file at its path", "none", "delete its file", "none"),
    Learner("quality.py:QualityLedger", "quality and cost per (context, choice)", "statistic that informs a choice",
            "JSON file, where given a path", "none", "delete its file", "none"),
)

#: classes that have a method with one of the names above and learn nothing from it
NOT_LEARNERS: dict[str, str] = {
    "plugins/base.py:CostEstimator": "the interface; implementations are listed on their own",
    "integrations/job_leads.py:OutreachLedger": "a log of what was sent; nothing reads it to change behaviour toward an arm",
    "integrations/zoning_intelligence.py:RegridProvider": "observe() reads a data source",
    "integrations/zoning_intelligence.py:AttomProvider": "observe() reads a data source",
    "integrations/zoning_intelligence.py:MunicipalGisProvider": "observe() reads a data source",
    "integrations/zoning_intelligence.py:OrdinanceProvider": "observe() reads a data source",
}

__all__ = ["Learner", "LEARNERS", "NOT_LEARNERS", "LEARNING_METHODS", "MAY_CHANGE"]
