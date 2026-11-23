"""Fleet orchestrator — Context Runtime edition.

Same surface the control-plane app expects (registry, status, up, down, dispatch,
context, jobs), but the brain is Context Runtime's ``ModuleTenant`` fleet instead of
the old docker-compose-shelling Fleet. Standing a module up registers its tenant and
flips it to ``deployed`` — so ``/status`` reflects real tenants, not marker dirs.

Every module shares ONE ContextRuntime, so the cost-model statistics and learning
compound across the whole fleet.
"""
from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass

from pathlib import Path

from ..costmodel.estimators import HeuristicEstimator
from ..integrations.modules import CATALOG, ModuleSpec, ModuleTenant
from ..platform_services import ensure_private_dir, state_dir
from ..runtime.runtime import ContextRuntime
from ..tools.authz import (
    DIAGNOSTIC,
    PrivilegedExecutionUnavailable,
    execution_mode,
    require_plane,
)
from .context import Approval, Context
from .registry import Module, Registry

# registry module name → integrations.modules CATALOG key
_ALIAS = {
    "edge-sentinel": "edge_sentinel", "agentic-support": "support", "agentic-billing": "billing",
    "agentic-books": "books", "agentic-compliance": "compliance", "control-tower": "control_tower",
    "market-radar": "market_radar", "growth-engine": "growth_engine", "social-autopilot": "social",
    "agentic-crm": "crm", "lifecycle": "lifecycle", "agentic-privacy": "privacy",
    "outreach-engine": "outreach",
}


#: Recorded once per process so /health can say what this fleet was allowed to become. It is a start-up fact,
#: which is exactly why it is no longer a warning: see ``require_plane``.
_MODE: tuple[str, str] | None = None


def mode() -> tuple[str, str]:
    """``("execute", "")`` or ``("diagnostic", reason)`` for this fleet — decided once, at construction.

    The control plane is a privileged execution service: it stands up tenants whose tools block IPs, file
    tickets and bill customers. It therefore refuses to come up at all with nothing to ask, and the only
    instance of it that may boot without a plane is one where execution has been taken out of the build —
    no tenants, no dispatch, nothing for a missing grant to be about.
    """
    return _MODE if _MODE is not None else execution_mode(context="module fleet")


def spec_for(m: Module) -> ModuleSpec:
    """A CATALOG tenant spec if we have one; otherwise a generic spec from the registry."""
    key = _ALIAS.get(m.name)
    if key and key in CATALOG:
        return CATALOG[key]
    if m.name in CATALOG:
        return CATALOG[m.name]
    # Not a CATALOG row: the action names come from a module the registry discovered at run time, so
    # this file cannot vouch for what they do. They are registered unclassified (denied) rather than
    # assumed safe — see integrations/modules.py::_action_tool.
    return ModuleSpec(
        name=m.name.replace("-", "_"), core="", pain=m.pain,
        sources=("primary", "secondary", "context"), metric="task_success",
        actions=tuple(a.replace("-", "_") for a in m.approval_required) or tuple(m.agents[:1]),
        curated_actions=False,
    )


@dataclass
class ModuleStatus:
    name: str
    deployed: bool
    agents: tuple[str, ...]
    detail: str = ""


class Fleet:
    def __init__(self, registry: Registry, router=None, context: Context | None = None,
                 workdir=None, runtime: ContextRuntime | None = None, home: str | None = None,
                 privileged: bool | None = None):
        global _MODE
        self.registry = registry
        # Decided once, here, before anything can run: a privileged fleet refuses construction without a plane
        # (``require_plane``), a diagnostic one comes up and is handed nothing to execute with.
        _MODE = execution_mode(context="module fleet")
        self.privileged = (_MODE[0] != DIAGNOSTIC) if privileged is None else privileged
        if self.privileged:
            require_plane(context="module fleet")
        self.router = router                      # kept for app compatibility; unused
        self.home = Path(home) if home else state_dir()   # persistence root (the /data volume in a container)
        self.context = context or Context(ensure_private_dir(self.home))
        # ONE shared runtime; its cost-model statistics persist so calibration survives restarts
        if runtime is None:
            estimator = HeuristicEstimator(stats_path=str(self.home / "costmodel_stats.json"))
            runtime = ContextRuntime.default([], estimator=estimator)
        self.runtime = runtime
        self.tenants: dict[str, ModuleTenant] = {}
        self._up: set[str] = set()
        self.jobs: dict[str, dict] = {}

    def _names(self) -> list[str]:
        return [m.name for m in self.registry]

    # --- lifecycle ---
    def up(self, *names: str) -> list[ModuleStatus]:
        if not self.privileged:
            raise PrivilegedExecutionUnavailable(
                "this fleet is diagnostic: it does not stand modules up. It exists to show a fleet that has no "
                "execution path — no tenants, no tool registry, nothing a missing grant could be about — and "
                "standing one up would be the moment it silently became an execution service again. Run the "
                "execution instance for that.")
        out = []
        for n in (names or self._names()):
            m = self.registry.get(n)
            # the fleet gates approvals (via Context); the tenant's own tools default-deny.
            # each tenant's learned policy persists to the /data volume → survives restarts.
            self.tenants[n] = ModuleTenant(
                spec_for(m), runtime=self.runtime, approver=lambda a: False,
                persist_path=str(self.home / "bandits" / f"{n}.json"))
            self._up.add(n)
            out.append(ModuleStatus(n, True, m.agents, "context-runtime tenant"))
        return out

    def down(self, *names: str) -> list[ModuleStatus]:
        out = []
        for n in (names or list(self._up)):
            self._up.discard(n)
            self.tenants.pop(n, None)
            out.append(ModuleStatus(n, False, self.registry.get(n).agents, "stopped"))
        return out

    def status(self) -> list[ModuleStatus]:
        return [ModuleStatus(m.name, m.name in self._up, m.agents,
                             "context-runtime tenant" if m.name in self._up else "")
                for m in self.registry]

    # --- dispatch (now a Context Runtime plan, not a raw model call) ---
    def dispatch(self, module_name: str, agent: str, action: str, prompt: str,
                 capability: str = "reason", background: bool = False) -> Approval | dict:
        if not self.privileged:
            raise PrivilegedExecutionUnavailable(
                f"dispatch({module_name}/{action}) is refused: this fleet is diagnostic, so it has no tenants "
                f"and no tool path behind them. A diagnostic instance answers questions about a fleet; it does "
                f"not act as one.")
        m = self.registry.get(module_name)
        if agent not in m.agents:
            raise ValueError(f"module {module_name} has no agent {agent!r} (agents: {m.agents})")
        if m.needs_approval(action):
            return self.context.request_approval(
                module=module_name, action=action,
                summary=f"{agent} wants to {action}", payload={"prompt": prompt})
        if module_name not in self.tenants:
            self.up(module_name)
        if background:
            return self._dispatch_background(module_name, action, prompt)
        return self._plan(module_name, agent, action, prompt)

    def _plan(self, module_name: str, agent: str, action: str, prompt: str) -> dict:
        tenant = self.tenants[module_name]
        r = tenant.handle(prompt)
        return {
            "kind": "result", "module": module_name, "agent": agent, "action": action,
            "intent": r.kind, "sources": list(r.bundle.sources),
            "available_sources": list(tenant.spec.sources),
            "recommended_action": r.recommended_action, "context": r.context,
            "plan_id": r.plan.id, "policy": tenant.policy(),
        }

    def _dispatch_background(self, module_name: str, action: str, prompt: str) -> dict:
        job_id = f"job-{len(self.jobs) + 1:05d}"
        self.jobs[job_id] = {"status": "running", "module": module_name, "action": action, "result": None}

        def _worker() -> None:
            try:
                self.jobs[job_id].update(status="done", result=self._plan(module_name, "", action, prompt))
            except Exception as e:  # noqa: BLE001
                self.jobs[job_id].update(status="error", result=str(e))

        threading.Thread(target=_worker, name=f"bg-{job_id}", daemon=True).start()
        return {"job_id": job_id, "status": "running", "module": module_name, "action": action}

    def job(self, job_id: str) -> dict | None:
        return self.jobs.get(job_id)

    def record_outcome(self, module_name: str, question: str, success: bool) -> float:
        """Close the loop: feed a module's task outcome back so its tenant learns."""
        t = self.tenants.get(module_name)
        return t.record_outcome(question, success) if t else 0.0
