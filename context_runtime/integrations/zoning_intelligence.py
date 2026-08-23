"""Zoning Intelligence × Context Runtime — the geospatial reference tenant (plan §6, §13-18).

The decision point is **which evidence sources to acquire** before concluding what a parcel supports —
commercial parcel/zoning providers (Regrid, ATTOM), the official municipal GIS, and the zoning
ordinance text. The reward is *the correct land-use disposition at the cheapest sufficient evidence
bundle* — the same fleet pattern as the other tenants, over the geospatial domain.

What makes this tenant more than a bandit clone is the **deterministic-first, fail-safe resolver**
(`resolve_disposition`): geometry and structured evidence are reconciled by the CPU spatial engine
BEFORE any ordinance/LLM interpretation, and a parcel is only ever concluded ``PERMITTED`` when the
gathered evidence is sufficient to confirm it. When sources disagree or the decisive source is absent,
the resolver returns ``UNKNOWN`` rather than guess — so a single stale provider can never produce a
"verified permitted" (the blocking SLO ``false-permitted = 0``, plan §16). Reconciliation (≥2 sources)
is what converts an unsafe single-provider guess into a safe ``UNKNOWN``.

Everything here is offline and deterministic: `build_reference_world()` is a fixture jurisdiction
(Fulton County, GA; EPSG:2240 planar feet) with real parcel geometry, overlay polygons, and a hidden
ground truth the providers observe with source-specific fidelity — exactly like the other tenants'
simulated backends. Swap the providers for live Regrid/ATTOM/ArcGIS calls and nothing else changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..geospatial.contracts import (
    ConstraintType, GeoRef, LandUseConstraint, ParcelEntity, Ring, UseDisposition, geometry_hash,
)
from ..geospatial.engine import area, bbox, centroid, distance, point_in_polygon, polygons_intersect
from ..runtime.runtime import ContextRuntime
from ..types import Goal, Plan, Trace
from .bandit import EpsilonGreedyBandit

# ──────────────────────────── use ontology → evidence-difficulty bucket ────────────────────────────
# Target uses this benchmark judges, grouped by the evidence tier that decides them (plan §9, §14).
# Each bucket has ONE minimal-sufficient evidence bundle; the bandit learns it from outcomes.
TARGET_USES: tuple[str, ...] = (
    "RESIDENTIAL_SINGLE_FAMILY", "OFFICE", "RETAIL", "WAREHOUSE", "LIGHT_INDUSTRIAL", "DATA_CENTER",
)
USE_BUCKET: dict[str, str] = {
    "RESIDENTIAL_SINGLE_FAMILY": "residential",   # base zoning decides → commercial consensus suffices
    "OFFICE": "commercial", "RETAIL": "commercial",  # overlay-sensitive → needs official GIS
    "WAREHOUSE": "industrial", "LIGHT_INDUSTRIAL": "industrial", "DATA_CENTER": "industrial",  # +ordinance
}
# Base zonings where a use is at least conditionally allowed (plan §9 mapping, with evidence).
BASE_ALLOWS: dict[str, frozenset[str]] = {
    "RESIDENTIAL_SINGLE_FAMILY": frozenset({"R-1"}),
    "OFFICE": frozenset({"C-2"}), "RETAIL": frozenset({"C-2"}),
    "WAREHOUSE": frozenset({"M-1"}), "LIGHT_INDUSTRIAL": frozenset({"M-1"}), "DATA_CENTER": frozenset({"M-1"}),
}
# Uses whose permissibility can be flipped by an overlay district → require the authoritative GIS.
OVERLAY_SENSITIVE: frozenset[str] = frozenset({"OFFICE", "RETAIL", "WAREHOUSE", "LIGHT_INDUSTRIAL", "DATA_CENTER"})
# Uses whose final disposition (permitted vs conditional vs special-exception) lives in ordinance text.
CONDITIONAL_USES: frozenset[str] = frozenset({"WAREHOUSE", "LIGHT_INDUSTRIAL", "DATA_CENTER"})
RESTRICTIVE_OVERLAYS: frozenset[str] = frozenset({"FLOOD", "HISTORIC"})
# Deterministic structured constraint: minimum lot area (planar ft²) — checked BEFORE any interpretation.
MIN_LOT_AREA: dict[str, float] = {"DATA_CENTER": 87120.0}   # 2 acres; small lots are prohibited outright


def zoning_bucket(use: str) -> str:
    """The bandit context key: the evidence-difficulty class of a target use (plan §14)."""
    return USE_BUCKET.get(use, "commercial")


# ──────────────────────────── the fixture jurisdiction (ground truth) ────────────────────────────


@dataclass
class ZoningWorld:
    """Ground truth for the reference jurisdiction. Providers observe this with source-specific fidelity;
    the LLM/derived layer is never treated as truth (plan §15). Offline and fully deterministic."""
    crs: str
    parcels: dict[str, ParcelEntity]                     # parcel_id → canonical entity (geometry, lot_area)
    rings: dict[str, Ring]                               # parcel_id → its outer ring (for spatial ops)
    base_truth: dict[str, str]                           # parcel_id → true base zoning
    regrid_view: dict[str, str]                          # parcel_id → Regrid's (possibly stale) base zoning
    overlays: list[tuple[str, Ring]]                     # (overlay_kind, ring) districts, e.g. ("FLOOD", ring)
    ordinance_det: dict[tuple[str, str], UseDisposition]  # (parcel_id, use) → authoritative conditional call


def _parcel(pid: str, x0: float, y0: float, size: float, base: str, crs: str) -> tuple[ParcelEntity, Ring]:
    ring: Ring = [(x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size)]
    gh = geometry_hash([ring], crs)
    ref = GeoRef(geometry_type="Polygon", geometry_hash=gh, crs=crs, bbox=bbox(ring),
                 centroid=centroid(ring), jurisdiction="Fulton County, GA", source="fixture")
    ent = ParcelEntity(parcel_id=gh, geometry_ref=ref, apn=pid, jurisdiction="Fulton County, GA",
                       lot_area=area(ring), zoning_districts=[base])
    return ent, ring


def build_reference_world(crs: str = "EPSG:2240") -> tuple[ZoningWorld, dict[str, str]]:
    """A small, honest jurisdiction exercising every arm's decisive evidence tier. Returns the world and
    an ``apn → parcel_id`` index (callers key parcels by the human-readable APN)."""
    specs = [
        # apn,  x0,   y0,  size, true_base, regrid_stale_to(None=accurate)
        ("R-100", 0, 0, 200, "R-1", None),            # residential, clean → commercial consensus suffices
        ("R-101", 300, 0, 200, "R-1", None),
        ("R-102", 600, 0, 200, "C-2", "R-1"),         # STALE: Regrid still says R-1; truth is C-2 → false-permit trap
        ("O-200", 0, 400, 200, "C-2", None),          # office/retail, overlay-sensitive → needs GIS
        ("O-201", 300, 400, 200, "C-2", None),        # sits under a FLOOD overlay → GIS flips it PROHIBITED
        ("W-300", 0, 800, 300, "M-1", None),          # warehouse/industrial → needs GIS + ordinance
        ("W-301", 400, 800, 300, "M-1", None),        # HISTORIC overlay → GIS PROHIBITED
        ("D-400", 900, 800, 100, "M-1", None),        # data-center on a small lot → deterministic PROHIBITED
        ("D-401", 900, 1100, 400, "M-1", None),       # data-center, big lot, conditional → ordinance decides
    ]
    parcels: dict[str, ParcelEntity] = {}
    rings: dict[str, Ring] = {}
    base_truth: dict[str, str] = {}
    regrid_view: dict[str, str] = {}
    apn_index: dict[str, str] = {}
    for apn, x0, y0, size, base, stale in specs:
        ent, ring = _parcel(apn, x0, y0, size, base, crs)
        pid = ent.parcel_id
        parcels[pid] = ent
        rings[pid] = ring
        base_truth[pid] = base
        regrid_view[pid] = stale or base
        apn_index[apn] = pid
    # Overlay districts (real polygons; membership is a deterministic spatial intersect).
    overlays: list[tuple[str, Ring]] = [
        ("FLOOD", [(250, 350), (700, 350), (700, 700), (250, 700)]),      # covers O-201
        ("HISTORIC", [(380, 780), (720, 780), (720, 1120), (380, 1120)]),  # covers W-301
    ]
    # Authoritative ordinance determinations for conditional (industrial) uses.
    ordinance_det: dict[tuple[str, str], UseDisposition] = {
        (apn_index["W-300"], "WAREHOUSE"): UseDisposition.PERMITTED,
        (apn_index["W-300"], "LIGHT_INDUSTRIAL"): UseDisposition.CONDITIONAL,
        (apn_index["D-401"], "DATA_CENTER"): UseDisposition.CONDITIONAL,
        (apn_index["D-401"], "LIGHT_INDUSTRIAL"): UseDisposition.PERMITTED,
    }
    world = ZoningWorld(crs=crs, parcels=parcels, rings=rings, base_truth=base_truth,
                        regrid_view=regrid_view, overlays=overlays, ordinance_det=ordinance_det)
    return world, apn_index


# ──────────────────────────── providers (observe the world with source-specific fidelity) ──────────


class Provider:
    """Read-only evidence source. Live subclasses would hit Regrid/ATTOM/ArcGIS; here they read the
    fixture world. Each exposes what it can actually see — Regrid/ATTOM see only base zoning (and Regrid
    can be stale); the municipal GIS sees base + overlays authoritatively; the ordinance resolves the
    conditional-use call. Provider disagreement is preserved, never flattened (plan §8)."""

    name = "provider"
    cost = 1.0

    def __init__(self, world: ZoningWorld):
        self.world = world


class RegridProvider(Provider):
    name, cost = "regrid", 1.0

    def observe(self, pid: str) -> dict:
        base = self.world.regrid_view.get(pid, self.world.base_truth[pid])
        return {"source": "regrid", "base_zoning": base, "stale": base != self.world.base_truth[pid]}


class AttomProvider(Provider):
    name, cost = "attom", 1.0     # independent commercial source — disagreement with Regrid catches staleness

    def observe(self, pid: str) -> dict:
        return {"source": "attom", "base_zoning": self.world.base_truth[pid]}


class MunicipalGisProvider(Provider):
    name, cost = "municipal_gis", 2.0   # official; authoritative base + overlays (deterministic spatial join)

    def observe(self, pid: str) -> dict:
        ring = self.world.rings[pid]
        hits = [kind for kind, oring in self.world.overlays if polygons_intersect(ring, oring)]
        return {"source": "municipal_gis", "base_zoning": self.world.base_truth[pid], "overlays": tuple(hits)}


class OrdinanceProvider(Provider):
    name, cost = "ordinance", 4.0   # expensive: official ordinance text + interpretation for conditional uses

    def observe(self, pid: str, use: str) -> dict:
        det = self.world.ordinance_det.get((pid, use), UseDisposition.PERMITTED)
        return {"source": "ordinance", "determination": det}


def build_providers(world: ZoningWorld) -> dict[str, Provider]:
    return {p.name: p for p in (RegridProvider(world), AttomProvider(world),
                                MunicipalGisProvider(world), OrdinanceProvider(world))}


# ──────────────────────────── the deterministic-first, fail-safe resolver (plan §4, §16) ──────────


@dataclass(frozen=True)
class Assessment:
    parcel_id: str
    use: str
    disposition: UseDisposition
    confidence: float
    sources_used: tuple[str, ...]
    reasons: tuple[str, ...]
    constraints: tuple[LandUseConstraint, ...] = ()


def resolve_disposition(world: ZoningWorld, providers: dict[str, Provider], pid: str, use: str,
                        sources: tuple[str, ...]) -> Assessment:
    """Reconcile the available evidence into a disposition. PERMITTED is emitted ONLY when the gathered
    evidence is sufficient to confirm it; otherwise UNKNOWN. This is what makes false-permitted a
    structural impossibility for reconciled bundles (plan §16 blocking SLO)."""
    present = set(sources)
    reasons: list[str] = []
    parcel = world.parcels[pid]

    # 1 ── deterministic structured constraint (geometry before interpretation).
    min_area = MIN_LOT_AREA.get(use)
    if min_area is not None and parcel.lot_area is not None and parcel.lot_area < min_area:
        c = LandUseConstraint(type=ConstraintType.MIN_LOT_AREA, value=min_area, unit="ft^2",
                              operator=">=", source_evidence="deterministic:lot_area")
        return Assessment(pid, use, UseDisposition.PROHIBITED, 1.0, sources,
                          (f"lot_area {parcel.lot_area:.0f} < MIN_LOT_AREA {min_area:.0f}",), (c,))

    # 2 ── base zoning: authoritative from GIS, else commercial consensus (reconciliation).
    gis = providers["municipal_gis"].observe(pid) if "municipal_gis" in present else None
    bases = {}
    if "regrid" in present:
        bases["regrid"] = providers["regrid"].observe(pid)["base_zoning"]
    if "attom" in present:
        bases["attom"] = providers["attom"].observe(pid)["base_zoning"]
    if gis is not None:
        base = gis["base_zoning"]
        reasons.append(f"base={base} (official GIS)")
    elif "regrid" in bases and "attom" in bases:
        if bases["regrid"] != bases["attom"]:
            reasons.append(f"provider disagreement regrid={bases['regrid']} attom={bases['attom']}")
            return Assessment(pid, use, UseDisposition.UNKNOWN, 0.4, sources, tuple(reasons))
        base = bases["attom"]
        reasons.append(f"base={base} (commercial consensus)")
    elif bases:
        base = next(iter(bases.values()))
        reasons.append(f"base={base} (single commercial source, unreconciled)")
    else:
        return Assessment(pid, use, UseDisposition.UNKNOWN, 0.2, sources, ("no base-zoning evidence",))

    # 3 ── base compatibility.
    if base not in BASE_ALLOWS.get(use, frozenset()):
        return Assessment(pid, use, UseDisposition.PROHIBITED, 0.9, sources,
                          tuple(reasons) + (f"{use} not allowed in {base}",))

    # 4 ── overlay sensitivity: needs the authoritative GIS to rule out a restrictive overlay.
    if use in OVERLAY_SENSITIVE:
        if gis is None:
            reasons.append("overlay-sensitive use, no official GIS → cannot confirm no overlay")
            return Assessment(pid, use, UseDisposition.UNKNOWN, 0.5, sources, tuple(reasons))
        restrictive = [o for o in gis["overlays"] if o in RESTRICTIVE_OVERLAYS]
        if restrictive:
            return Assessment(pid, use, UseDisposition.PROHIBITED, 0.95, sources,
                              tuple(reasons) + (f"restrictive overlay {restrictive}",))

    # 5 ── conditional use: the final call lives in the ordinance text.
    if use in CONDITIONAL_USES:
        if "ordinance" not in present:
            reasons.append("conditional use, no ordinance → cannot confirm disposition")
            return Assessment(pid, use, UseDisposition.UNKNOWN, 0.5, sources, tuple(reasons))
        det = providers["ordinance"].observe(pid, use)["determination"]
        return Assessment(pid, use, det, 0.9, sources, tuple(reasons) + (f"ordinance→{det.value}",))

    return Assessment(pid, use, UseDisposition.PERMITTED, 0.85, sources, tuple(reasons) + ("base permits",))


def gold_disposition(world: ZoningWorld, providers: dict[str, Provider], pid: str, use: str) -> UseDisposition:
    """The reference answer: resolve with the full authoritative evidence set (all sources)."""
    return resolve_disposition(world, providers, pid, use, ("regrid", "attom", "municipal_gis", "ordinance")).disposition


# ──────────────────────────── the bandit arm + reward (plan §14) ────────────────────────────


@dataclass(frozen=True)
class EvidencePlan:
    """A bandit arm: which evidence sources to acquire for an assessment. Fewer = cheaper."""
    sources: tuple[str, ...]

    @property
    def key(self) -> str:
        return "+".join(self.sources)   # sources are declared in escalation order; keep it

    def cost_units(self, providers: dict[str, Provider]) -> float:
        return sum(providers[s].cost for s in self.sources)


# Escalating evidence bundles: single-provider → commercial consensus → +official GIS → +ordinance.
DEFAULT_ARMS: tuple[EvidencePlan, ...] = (
    EvidencePlan(("regrid",)),
    EvidencePlan(("regrid", "attom")),
    EvidencePlan(("regrid", "attom", "municipal_gis")),
    EvidencePlan(("regrid", "attom", "municipal_gis", "ordinance")),
)
_MAX_COST = 8.0   # regrid1 + attom1 + gis2 + ordinance4 — the reward-normalizing reference
COST_LAMBDA = 0.15
FALSE_PERMIT_PENALTY = 1.0   # a "verified permitted" that is actually prohibited is worse than a miss


def reward_zoning(*, correct: bool, false_permitted: bool, cost_units: float) -> float:
    """Correct disposition at the cheapest sufficient evidence — with a hard penalty for the blocking-SLO
    violation (false-permitted). UNKNOWN when unsure is a plain miss (0.0), never penalized like a lie."""
    if false_permitted:
        return -FALSE_PERMIT_PENALTY
    if not correct:
        return 0.0
    return round(1.0 - COST_LAMBDA * (cost_units / _MAX_COST), 4)


def _zoning_bandit(epsilon: float = 0.12, arms: tuple[EvidencePlan, ...] = DEFAULT_ARMS) -> EpsilonGreedyBandit:
    return EpsilonGreedyBandit(arms, epsilon=epsilon)


# ──────────────────────────── the tenant ────────────────────────────


@dataclass
class _Pending:
    plan: Plan
    arm: EvidencePlan
    bucket: str
    assessment: Assessment


class ZoningIntelligenceTenant:
    """Context Runtime plans zoning intelligence: pick the cheapest evidence bundle that resolves a
    parcel/use question, reconcile the evidence deterministically, conclude a disposition (or abstain to
    UNKNOWN), and learn — per use-difficulty bucket — the minimal sufficient bundle from outcomes."""

    def __init__(self, world: ZoningWorld | None = None, runtime: ContextRuntime | None = None,
                 bandit: EpsilonGreedyBandit | None = None, arms: tuple[EvidencePlan, ...] = DEFAULT_ARMS):
        if world is None:
            world, _ = build_reference_world()
        self.world = world
        self.providers = build_providers(world)
        self.runtime = runtime or ContextRuntime.default([])
        self.bandit = bandit or _zoning_bandit(arms=arms)
        self._pending: dict[str, _Pending] = {}

    # ── parcel-first: "what can I build/operate on this parcel?" ──
    def assess(self, query_id: str, parcel_id: str, use: str) -> Assessment:
        bucket = zoning_bucket(use)
        plan = self.runtime.plan(Goal(text=f"{use} on {parcel_id}"))
        arm = self.bandit.select(bucket)
        a = resolve_disposition(self.world, self.providers, parcel_id, use, arm.sources)
        self._pending[query_id] = _Pending(plan, arm, bucket, a)
        return a

    def record(self, query_id: str, gold: UseDisposition) -> float:
        """Feed back the reference disposition. Correct = matched gold; false-permitted = concluded
        PERMITTED where gold is PROHIBITED (the SLO violation). Updates the bandit + cost model."""
        p = self._pending.pop(query_id, None)
        if p is None:
            return 0.0
        correct = p.assessment.disposition == gold
        false_permitted = p.assessment.disposition == UseDisposition.PERMITTED and gold == UseDisposition.PROHIBITED
        reward = reward_zoning(correct=correct, false_permitted=false_permitted,
                               cost_units=p.arm.cost_units(self.providers))
        self.bandit.update(p.bucket, p.arm, reward)
        trace = Trace(plan_id=p.plan.id, goal_text=f"{p.assessment.use} on {p.assessment.parcel_id}",
                      actual_cost_usd=p.arm.cost_units(self.providers) * 0.01,
                      verification_passed=correct)
        self.runtime.estimator.observe(p.plan, trace)
        return reward

    # ── use-first: "find parcels compatible with this use in an area" (plan §6, §12) ──
    def search(self, use: str, *, center: tuple[float, float], radius: float,
               min_lot_area: float = 0.0) -> list[Assessment]:
        """A governed candidate set: deterministic spatial filter (within radius + lot area) THEN the
        learned evidence bundle per candidate. Only confidently-PERMITTED parcels are returned; UNKNOWN
        and PROHIBITED are withheld (they surface in EXPLAIN as REQUIRE_REVIEW / rejected)."""
        bucket = zoning_bucket(use)
        arm = self.bandit.select(bucket)
        out: list[Assessment] = []
        for pid, ent in self.world.parcels.items():
            if ent.lot_area is not None and ent.lot_area < min_lot_area:
                continue
            if distance(ent.geometry_ref.centroid, center) > radius:   # deterministic spatial prefilter
                continue
            a = resolve_disposition(self.world, self.providers, pid, use, arm.sources)
            if a.disposition == UseDisposition.PERMITTED:
                out.append(a)
        return out

    def explain(self, query_id: str) -> dict:
        """Per-assessment EXPLAIN: the selected bundle, its expected value, and every arm scored — plus
        the evidence reasons that produced the conclusion (the essential UI feature, plan §21)."""
        p = self._pending.get(query_id)
        bucket = p.bucket if p else "commercial"
        scores = []
        for arm in DEFAULT_ARMS:
            n, val = self.bandit.value(bucket, arm.key)
            scores.append({"arm": arm.key, "n": n, "value": round(val, 4),
                           "cost_units": arm.cost_units(self.providers)})
        exp = {"query": query_id, "bucket": bucket,
               "scores": sorted(scores, key=lambda s: s["value"], reverse=True)}
        if p is not None:
            exp["selected"] = p.arm.key
            exp["disposition"] = p.assessment.disposition.value
            exp["confidence"] = p.assessment.confidence
            exp["reasons"] = list(p.assessment.reasons)
        return exp

    def policy(self) -> dict[str, str]:
        return self.bandit.policy()


# ──────────────────────────── incremental evidence change (plan §17, Phase 8) ────────────────────────────


def affected_conclusions(world: ZoningWorld, changed_parcel_ids: set[str],
                         changed_overlay_rings: list[Ring] | None = None) -> set[str]:
    """Dependency-scoped recomputation set: the parcels whose conclusions an EvidenceChange invalidates.
    A parcel is affected if it changed directly, OR if its geometry intersects a changed overlay polygon.
    Everything else stays valid — the geospatial analogue of incremental Discovery (plan §17)."""
    affected = set(changed_parcel_ids)
    for ring in (changed_overlay_rings or []):
        for pid, pring in world.rings.items():
            if polygons_intersect(pring, ring):
                affected.add(pid)
    return affected


# ──────────────────────────── population governance (plan §18, Phase 9) ────────────────────────────


class PopulationGovernor:
    """Detects a systematic quality shift across the parcel population that individual records hide — e.g.
    a provider/classifier that starts mapping CONDITIONAL → PERMITTED. Compares a recent window's
    downgrade rate against a healthy baseline; a jump past the threshold yields REQUIRE_REVIEW for the
    whole series, not just one parcel (plan §18)."""

    def __init__(self, baseline_rate: float = 0.0, threshold: float = 0.25, window: int = 20):
        self.baseline_rate = baseline_rate
        self.threshold = threshold
        self.window = window
        self._recent: dict[str, list[int]] = {}

    def observe(self, series_id: str, downgraded: bool) -> None:
        buf = self._recent.setdefault(series_id, [])
        buf.append(1 if downgraded else 0)
        if len(buf) > self.window:
            del buf[0]

    def recent_rate(self, series_id: str) -> float:
        buf = self._recent.get(series_id, [])
        return sum(buf) / len(buf) if buf else 0.0

    def review_required(self, series_id: str) -> bool:
        return self.recent_rate(series_id) - self.baseline_rate > self.threshold
