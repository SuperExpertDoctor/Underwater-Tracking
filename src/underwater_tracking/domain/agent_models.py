# src/underwater_tracking/domain/agent_models.py
"""Strict cross-layer contracts for the carrier assistant.

Implements spec sections 6.5-6.9 (``IntentHypothesis``,
``PredictedTrackRef``, ``StrategyProposal``, ``TrackingPlan``,
``ExpertDirective``, ``DecisionRecord``) plus the intermediate carrier
contracts ``StrategySet``, ``PlanCommand``, ``PlanDiff``,
``ValidationIssue`` and ``ValidationReport``.

All models are strict (``extra="forbid"``) but intentionally NOT frozen:
plan-level immutability is enforced at commit boundaries by revision and
snapshot-version checks in the repositories, not by frozen models (the
plan's stale-plan tests mutate ``TrackingPlan.base_snapshot_revision``
directly). Final member IDs and waypoints live only in ``TrackingPlan``
(and the derived ``PlanCommand``); ``StrategyProposal`` is the LLM-facing
concept contract and must never carry them.
"""

from __future__ import annotations

from collections.abc import Iterator
from math import isfinite
from typing import Literal

from pydantic import Field, field_validator, model_validator

from underwater_tracking.domain.execution_models import IMMModelForecast
from underwater_tracking.domain.models import StrictModel
from underwater_tracking.domain.regional_models import RegionTask, TargetRegionPlan

IntentLabel = Literal["transit", "patrol", "loiter", "evade", "approach", "withdraw", "unknown"]
IntentMotiveLabel = Literal[
    "persistent_straight_transit",
    "hard_turn_evasion",
    "sprint_escape",
    "weaving_evasion",
    "speed_deception",
]
Concept = Literal["quality_first", "balanced", "resource_saving", "hold_current"]
SuggestionCategory = Literal[
    "tracking_quality",
    "segmented_handoff",
    "resource_rotation",
    "commander_preference",
]
PlanStatus = Literal[
    "draft", "validating", "active", "superseded", "completed", "rejected", "degraded"
]
PredictionRegime = Literal[
    "known_submarine",
    "public_prior",
    "short_history",
    "bspline",
    "imm",
    "boundary_recovery",
]
TrajectoryDiffStatus = Literal[
    "comparable",
    "first_prediction",
    "no_new_evidence",
    "insufficient_overlap",
    "predictor_regime_reset",
    "target_mismatch",
    "invalid_prediction",
]
TrajectoryDiffGateTransition = Literal[
    "none",
    "accumulating",
    "suspected",
    "verifying",
    "confirmed",
    "reset",
]


class IntentMotive(StrictModel):
    label: IntentMotiveLabel
    probability: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=400)


class IntentHypothesis(StrictModel):
    label: IntentLabel
    confidence: float = Field(ge=0, le=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    alternatives: dict[IntentLabel, float] = Field(default_factory=dict)
    ranked_motives: tuple[IntentMotive, ...] = Field(default_factory=tuple, max_length=5)
    planning_effects: tuple[str, ...] = ()
    model_id: str
    prompt_version: str


class PredictedTrackRef(StrictModel):
    """Reference to a B-spline track prediction (spec 6.6).

    ``times_s``/``points_xy``/``corridor_radius_m`` mirror the deterministic
    ``TrackPrediction`` arrays; ``spline_knots`` and the control points are
    the fitted spline parameters; ``clipping_records`` list velocity and
    turn-rate clip events; ``fallback_used``/``fallback_reason`` record the
    IMM-extrapolation fallback and its cause.
    """

    prediction_id: str
    target_id: str
    sim_time_s: int = Field(ge=0)
    horizon_s: float = Field(gt=0)
    sample_step_s: float = Field(gt=0)
    times_s: tuple[float, ...] = ()
    points_xy: tuple[tuple[float, float], ...] = ()
    corridor_radius_m: tuple[float, ...] = ()
    point_confidence: tuple[float, ...] = ()
    # IMM owns the uncertainty band; the historical cubic B-spline owns the
    # dashed centerline shown by the operator view.
    imm_times_s: tuple[float, ...] = ()
    imm_centerline_xy: tuple[tuple[float, float], ...] = ()
    imm_corridor_radius_m: tuple[float, ...] = ()
    bspline_times_s: tuple[float, ...] = ()
    bspline_centerline_xy: tuple[tuple[float, float], ...] = ()
    spline_degree: int = Field(default=3, ge=1, le=5)
    spline_knots: tuple[float, ...] = ()
    spline_control_x: tuple[float, ...] = ()
    spline_control_y: tuple[float, ...] = ()
    source_belief_history_ids: tuple[str, ...] = ()
    clipping_records: tuple[str, ...] = ()
    fallback_used: bool = False
    fallback_reason: str | None = None
    prediction_regime: PredictionRegime = "short_history"
    imm_model_probabilities: dict[str, float] = Field(default_factory=dict)
    imm_model_states: tuple[IMMModelForecast, ...] = ()
    imm_covariance_xy: tuple[tuple[float, float, float, float], ...] = ()
    imm_clipping_records: tuple[str, ...] = ()

    @field_validator("imm_model_probabilities")
    @classmethod
    def imm_probabilities_are_valid(cls, value: dict[str, float]) -> dict[str, float]:
        if not value:
            return value
        if any(not isfinite(probability) or probability < 0.0 for probability in value.values()):
            raise ValueError("IMM model probabilities must be finite and non-negative")
        if sum(value.values()) <= 0.0:
            raise ValueError("IMM model probabilities must have positive mass")
        return value


class TrajectoryDiffResult(StrictModel):
    """Auditable comparison of two time-aligned public target forecasts."""

    diff_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    previous_prediction_id: str | None = None
    current_prediction_id: str = Field(min_length=1)
    previous_sim_time_s: int | None = Field(default=None, ge=0)
    current_sim_time_s: int = Field(ge=0)
    status: TrajectoryDiffStatus
    reason: str | None = None
    overlap_start_s: float | None = Field(default=None, allow_inf_nan=False)
    overlap_end_s: float | None = Field(default=None, allow_inf_nan=False)
    overlap_duration_s: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    comparison_step_s: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    sample_count: int = Field(default=0, ge=0)
    absolute_rms_m: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    normalized_rms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    p90_distance_m: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    max_distance_m: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    max_distance_time_s: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    js_distance: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    previous_leading_model: str | None = None
    current_leading_model: str | None = None
    leading_model_changed: bool = False
    previous_evidence_ids: tuple[str, ...] = ()
    current_evidence_ids: tuple[str, ...] = ()
    normalized_threshold: float = Field(gt=0, allow_inf_nan=False)
    absolute_floor_m: float = Field(gt=0, allow_inf_nan=False)
    reset_normalized_threshold: float = Field(ge=0, allow_inf_nan=False)
    reset_absolute_floor_m: float = Field(ge=0, allow_inf_nan=False)
    threshold_schema_version: str = Field(min_length=1)
    confirmation_cycles: int = Field(ge=1)
    exceeded: bool = False
    consecutive_count: int = Field(default=0, ge=0)
    latched: bool = False
    gate_transition: TrajectoryDiffGateTransition = "none"


class IntentVerificationCallRef(StrictModel):
    """Checkpoint-safe hashes for one successful semantic intent call."""

    operation: Literal["intent"] = "intent"
    model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    request_hash: str = Field(min_length=1)
    response_hash: str = Field(min_length=1)
    sim_time_s: int = Field(ge=0)
    scenario_id: str = Field(min_length=1)


class TrajectoryDiffGateState(StrictModel):
    """Checkpointed detector state for one target."""

    target_id: str = Field(min_length=1)
    consecutive_count: int = Field(default=0, ge=0)
    latched: bool = False
    verification_pending: bool = False
    suspicion_event_id: str | None = None
    suspicion_diff_id: str | None = None
    latest_diff_id: str | None = None
    last_intent_verification_sim_time_s: int | None = Field(default=None, ge=0)
    last_intent_verification_diff_id: str | None = None
    intent_baseline_label: str | None = None
    intent_verification_label: str | None = None
    intent_verification_calls: tuple[IntentVerificationCallRef, ...] = ()


class Segment(StrictModel):
    """One relay-tracking time slice of a predicted track (spec 6.7 amendment, R3).

    ``intercept_xy`` is the track point where the assigned group
    initializes its standoff; ``start_s``/``end_s`` are absolute
    simulation times inside the prediction horizon.
    """

    index: int = Field(ge=0)
    start_s: int = Field(ge=0)
    end_s: int = Field(ge=0)
    group_id: str
    intercept_xy: tuple[float, float]


class SegmentPlan(StrictModel):
    """Ordered track segments across the tracking groups (R3)."""

    segments: tuple[Segment, ...] = ()


class StrategyProposal(StrictModel):
    concept: Concept
    target_priorities: dict[str, float]
    required_quality: dict[str, float]
    reinforcement_policy: dict[str, str]
    releasable_soft_constraints: tuple[str, ...]
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    rationale: str
    segment_plan: SegmentPlan | None = None

    @field_validator("target_priorities")
    @classmethod
    def target_priorities_are_finite(cls, value: dict[str, float]) -> dict[str, float]:
        for target, priority in value.items():
            if not isfinite(priority) or priority < 0.0:
                raise ValueError(f"target priority for {target!r} must be finite and non-negative")
        return value

    @field_validator("required_quality")
    @classmethod
    def required_quality_is_finite_and_bounded(cls, value: dict[str, float]) -> dict[str, float]:
        for target, quality in value.items():
            if not isfinite(quality) or not 0.0 <= quality <= 1.0:
                raise ValueError(f"required quality for {target!r} must be finite and in [0, 1]")
        return value

    @field_validator("reinforcement_policy", mode="before")
    @classmethod
    def coerce_policy_values_to_str(cls, value: object) -> object:
        """Coerce scalar policy VALUES to str; keys and strictness intact.

        Providers occasionally emit an int where the contract wants a str
        (e.g. ``max_additional_groups=1``). Only the VALUES are normalized
        — keys are policy names and stay as-is — so the declared
        ``dict[str, str]`` type and the strict schema are unchanged; any
        other input shape still fails validation as before.
        """
        if isinstance(value, dict):
            return {
                key: str(child) if isinstance(child, (bool, int, float)) else child
                for key, child in value.items()
            }
        return value


class StrategySet(StrictModel):
    """Ordered candidate concepts for one planning cycle.

    Iterating a ``StrategySet`` yields its ``proposals`` so downstream
    nodes can select concepts without unpacking the container.
    """

    trigger_event_ids: tuple[str, ...] = ()
    proposals: tuple[StrategyProposal, ...] = ()

    # Intentional semantic override: iterate the proposals, not the
    # BaseModel (name, value) field pairs.
    def __iter__(self) -> Iterator[StrategyProposal]:  # type: ignore[override]
        return iter(self.proposals)


class PlanAdjustmentSuggestion(StrictModel):
    """One LLM-generated, operator-facing plan adjustment suggestion."""

    suggestion_id: str = Field(min_length=1, max_length=80)
    category: SuggestionCategory
    title: str = Field(min_length=1, max_length=120)
    rationale: str = Field(min_length=1, max_length=500)
    proposed_feedback: str = Field(min_length=1, max_length=500)
    target_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)


class PlanAdjustmentSuggestionSet(StrictModel):
    """Exactly four current-observation suggestions for the command dialog."""

    suggestions: tuple[PlanAdjustmentSuggestion, ...] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def contains_one_suggestion_per_category(self) -> PlanAdjustmentSuggestionSet:
        expected = {
            "tracking_quality",
            "segmented_handoff",
            "resource_rotation",
            "commander_preference",
        }
        categories = [item.category for item in self.suggestions]
        if len(set(categories)) != len(categories) or set(categories) != expected:
            raise ValueError("suggestions must contain exactly one item for each required category")
        if len({item.suggestion_id for item in self.suggestions}) != len(self.suggestions):
            raise ValueError("suggestion_id values must be unique")
        return self


class Waypoint(StrictModel):
    x: float
    y: float
    arrive_at_s: int = Field(default=0, ge=0)


class PlanDiff(StrictModel):
    """Member and waypoint differences of a plan versus its predecessor."""

    from_plan_id: str | None = None
    from_revision: int | None = None
    to_plan_id: str
    to_revision: int = Field(default=1, ge=1)
    members_added: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    members_removed: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    waypoints_changed: tuple[str, ...] = ()
    summary: str = ""


class RegionalPlanMetrics(StrictModel):
    """Planning proxies derived from regional tasks, never sensor truth."""

    regional_quality_by_region: dict[str, float] = Field(default_factory=dict)
    coverage_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    relay_links_by_region: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    degraded_regions: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    uncovered_region_ids: tuple[str, ...] = ()
    metrics_are_planning_proxies: bool = True


class TrackingPlan(StrictModel):
    """A committed or candidate plan with final members and waypoints.

    Per spec 6.8 this is the only contract that carries the final group
    members, roles, per-UUV waypoints and rotation/release/return/emergency
    actions. Plans are strict but mutable: staleness is rejected at commit
    time by comparing ``base_snapshot_revision`` with the stored snapshot
    revision, never by freezing the model.
    """

    plan_id: str
    scenario_id: str
    revision: int = Field(ge=1)
    base_snapshot_revision: int = Field(ge=0)
    status: PlanStatus = "draft"
    valid_from_s: int = Field(default=0, ge=0)
    valid_until_s: int = Field(default=0, ge=0)
    concept: Concept = "hold_current"
    target_priorities: dict[str, float] = Field(default_factory=dict)
    required_quality: dict[str, float] = Field(default_factory=dict)
    member_ids_by_target: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    roles_by_member: dict[str, str] = Field(default_factory=dict)
    intent_refs: dict[str, str] = Field(default_factory=dict)
    prediction_refs: dict[str, str] = Field(default_factory=dict)
    waypoints_by_member: dict[str, tuple[Waypoint, ...]] = Field(default_factory=dict)
    rotation_conditions: dict[str, str] = Field(default_factory=dict)
    rotation_uuv_ids: tuple[str, ...] = ()
    release_actions: dict[str, str] = Field(default_factory=dict)
    return_actions: dict[str, str] = Field(default_factory=dict)
    emergency_actions: dict[str, str] = Field(default_factory=dict)
    active_uuv_ids: tuple[str, ...] = ()
    standby_uuv_ids: tuple[str, ...] = ()
    returning_uuv_ids: tuple[str, ...] = ()
    failed_uuv_ids: tuple[str, ...] = ()
    predicted_quality: dict[str, float] = Field(default_factory=dict)
    predicted_fim: dict[str, float] = Field(default_factory=dict)
    predicted_active_count: int = Field(default=0, ge=0)
    predicted_energy: float = Field(default=0.0, ge=0)
    predicted_risk: float = Field(default=0.0, ge=0, le=1)
    diff: PlanDiff | None = None
    trigger_event_ids: tuple[str, ...] = ()
    solver_run_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    regional_plans: dict[str, TargetRegionPlan] = Field(default_factory=dict)
    # Target-scoped provenance for regional strategy decisions. Hashes let
    # replay link a regional revision to an LLM exchange without storing bodies.
    regional_llm_hashes: dict[str, tuple[str, str]] = Field(default_factory=dict)
    region_tasks: dict[str, RegionTask] = Field(default_factory=dict)
    regional_metrics: RegionalPlanMetrics | None = None
    segment_plan: SegmentPlan | None = None


class PlanCommand(StrictModel):
    """Versioned per-group execution command (spec 5.2)."""

    command_id: str
    plan_id: str
    plan_revision: int = Field(ge=1)
    scenario_id: str
    group_id: str
    region_id: str | None = None
    target_id: str
    sim_time_s: int = Field(ge=0)
    member_ids: tuple[str, ...] = ()
    waypoints_by_member: dict[str, tuple[Waypoint, ...]] = Field(default_factory=dict)
    actions: dict[str, str] = Field(default_factory=dict)
    sensor_mode: Literal["active", "passive"] = "passive"


class VerificationCommand(StrictModel):
    """Engine-facing active-sonar verification protocol command (spec 17.3).

    ``sensor_mode`` drives the engine's ping simulation: ``ping`` turns
    active sonar on for ``uuv_ids``, ``return_to_passive`` turns it off,
    ``dispatch`` promotes a verified submarine contact into the tracking
    loop, and ``drop`` discards a classified decoy. Commands are emitted
    by the deterministic verification node and applied by the agent loop
    after plan commands, so the protocol's sensor-mode writes win.
    """

    command_id: str
    target_id: str
    sensor_mode: Literal["ping", "return_to_passive", "dispatch", "drop"]
    uuv_ids: tuple[str, ...] = ()
    sim_time_s: int = 0


class ValidationIssue(StrictModel):
    code: str
    field: str = ""
    message: str = ""
    observed: str | None = None
    expected: str | None = None


class ValidationReport(StrictModel):
    """Outcome of one schema/semantic validation cycle with repairs."""

    valid: bool
    issues: tuple[ValidationIssue, ...] = ()
    repair_attempts: int = Field(default=0, ge=0)
    degraded: bool = False


class SolverMetrics(StrictModel):
    solve_status: str = "solved"
    seed: int | None = None
    wall_time_ms: float = Field(default=0.0, ge=0)
    hard_violations: int = Field(default=0, ge=0)
    active_count: int = Field(default=0, ge=0)
    economic_cost: float = Field(default=0.0, ge=0)


class ExpertDirective(StrictModel):
    directive_id: str
    raw_text: str
    target_scope: tuple[str, ...]
    locked_members: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    # The RUF012 tags on the two defaults below are intentional: pydantic
    # deep-copies mutable defaults per instance, so the pattern is safe.
    target_priorities: dict[str, float] = {}  # noqa: RUF012
    minimum_quality: dict[str, float] = {}  # noqa: RUF012
    disabled_uuv_ids: tuple[str, ...] = ()
    return_uuv_ids: tuple[str, ...] = ()
    directive_type: Literal["constraint", "assignment", "feedback"] = "constraint"
    assignment_target_id: str | None = None
    assignment_uuv_ids: tuple[str, ...] = ()
    tracking_mode: Literal["dedicated", "regional"] | None = None
    dedicated_uuv_ids: tuple[str, ...] = ()
    feedback_region_ids: tuple[str, ...] = ()
    feedback_text: str | None = None
    confidence: float = Field(ge=0, le=1)
    conflicts: tuple[str, ...] = ()
    status: Literal["preview", "applied", "rejected", "needs_clarification"] = "preview"

    @model_validator(mode="after")
    def ambiguity_requires_clarification(self) -> ExpertDirective:
        if self.confidence < 0.70 and self.status == "applied":
            raise ValueError("low-confidence directives cannot be applied")
        if self.directive_type == "feedback" and (
            self.locked_members
            or self.target_priorities
            or self.minimum_quality
            or self.disabled_uuv_ids
            or self.return_uuv_ids
            or self.assignment_target_id is not None
            or self.assignment_uuv_ids
            or self.tracking_mode is not None
            or self.dedicated_uuv_ids
        ):
            raise ValueError("feedback directives cannot carry planning constraints")
        if self.tracking_mode != "dedicated" and self.dedicated_uuv_ids:
            raise ValueError("dedicated UUV ids require dedicated tracking mode")
        return self


class DecisionRecord(StrictModel):
    """One planning decision, fully traceable (spec 6.9 and section 16).

    ``candidates`` holds the strategy concepts; ``candidate_plan_ids``
    reference the corresponding numeric plans; ``rejected_candidates`` maps
    candidate id to rejection reason; ``verification_records`` captures each
    schema/semantic validation and repair round; ``expert_inputs`` keeps the
    expert directives that shaped the decision.
    """

    decision_id: str
    scenario_id: str
    sim_time_s: int = Field(ge=0)
    trigger_event_ids: tuple[str, ...] = ()
    snapshot_revision: int = Field(default=0, ge=0)
    snapshot_hash: str = ""
    input_evidence_ids: tuple[str, ...] = ()
    model_version: str = ""
    prompt_version: str = ""
    schema_version: str = ""
    config_version: str = ""
    code_version: str = ""
    candidates: tuple[StrategyProposal, ...] = ()
    candidate_plan_ids: tuple[str, ...] = ()
    solver_metrics: SolverMetrics | None = None
    rejected_candidates: dict[str, str] = Field(default_factory=dict)
    verification_records: tuple[ValidationReport, ...] = ()
    final_plan_id: str | None = None
    final_plan_diff: PlanDiff | None = None
    expert_inputs: tuple[ExpertDirective, ...] = ()
    plan_adjustment_suggestions: tuple[PlanAdjustmentSuggestion, ...] = ()


def derive_legacy_views(
    regional_plans: dict[str, TargetRegionPlan],
    region_tasks: dict[str, RegionTask] | None = None,
) -> dict[str, object]:
    """Derive target/group compatibility fields from regional tasks.

    Regional tasks remain authoritative. A target is retained even when all
    of its regions are degraded, and each member receives deterministic
    center waypoints ordered by active-window start and region ID.
    """
    members_by_target: dict[str, list[str]] = {
        target_id: [] for target_id in sorted(regional_plans)
    }
    roles_by_member: dict[str, str] = {}
    waypoints_by_member: dict[str, list[Waypoint]] = {}
    authoritative_tasks = region_tasks or {}
    active_uuv_ids: set[str] = set()
    degraded_regions: dict[str, tuple[str, ...]] = {}
    uncovered_region_ids: list[str] = []
    regional_quality_by_region: dict[str, float] = {}
    relay_links_by_region: dict[str, tuple[str, ...]] = {}
    total_regions = 0
    covered_regions = 0

    for target_id, plan in sorted(regional_plans.items()):
        cells = {cell.region_id: cell for cell in plan.cells}
        tasks = sorted(
            (authoritative_tasks.get(task.region_id, task) for task in plan.tasks),
            key=lambda task: (task.active_window.start_s, task.region_id),
        )
        for task in tasks:
            total_regions += 1
            regional_quality_by_region[task.region_id] = task.required_quality
            relay_links_by_region[task.region_id] = task.communication_links
            if task.assignment_status != "uncovered":
                covered_regions += 1
            if task.assignment_status in {"degraded", "uncovered"}:
                degraded_regions[task.region_id] = tuple(sorted(task.degraded_reasons)) or (
                    task.assignment_status,
                )
            if task.assignment_status == "uncovered":
                uncovered_region_ids.append(task.region_id)
            cell = cells.get(task.region_id)
            if cell is None:
                continue
            members = tuple(sorted(set(task.assigned_uuv_ids)))
            members_by_target.setdefault(target_id, []).extend(members)
            for index, member_id in enumerate(members):
                role = task.uuv_roles[index] if index < len(task.uuv_roles) else "passive_tracker"
                roles_by_member.setdefault(member_id, role)
                active_uuv_ids.add(member_id)
                waypoints_by_member.setdefault(member_id, []).append(
                    Waypoint(
                        x=cell.center_xy[0],
                        y=cell.center_xy[1],
                        arrive_at_s=task.active_window.start_s,
                    )
                )

    member_ids_by_target = {
        target_id: tuple(sorted(set(member_ids)))
        for target_id, member_ids in sorted(members_by_target.items())
    }
    return {
        "member_ids_by_target": member_ids_by_target,
        "roles_by_member": dict(sorted(roles_by_member.items())),
        "waypoints_by_member": {
            member_id: tuple(waypoints)
            for member_id, waypoints in sorted(waypoints_by_member.items())
        },
        "active_uuv_ids": tuple(sorted(active_uuv_ids)),
        # Preserve the legacy top-level view while the richer regional
        # metrics object remains authoritative for new callers.
        "degraded_regions": dict(sorted(degraded_regions.items())),
        "regional_metrics": RegionalPlanMetrics(
            regional_quality_by_region=dict(sorted(regional_quality_by_region.items())),
            coverage_rate=(covered_regions / total_regions) if total_regions else 0.0,
            relay_links_by_region=dict(sorted(relay_links_by_region.items())),
            degraded_regions=dict(sorted(degraded_regions.items())),
            uncovered_region_ids=tuple(sorted(uncovered_region_ids)),
        ),
    }
