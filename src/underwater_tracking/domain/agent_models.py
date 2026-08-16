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
from typing import Literal

from pydantic import Field, field_validator, model_validator

from underwater_tracking.domain.models import StrictModel

IntentLabel = Literal["transit", "patrol", "loiter", "evade", "approach", "withdraw", "unknown"]
Concept = Literal["quality_first", "balanced", "resource_saving", "hold_current"]
PlanStatus = Literal["draft", "validating", "active", "superseded", "completed", "rejected", "degraded"]


class IntentHypothesis(StrictModel):
    label: IntentLabel
    confidence: float = Field(ge=0, le=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    alternatives: dict[IntentLabel, float] = Field(default_factory=dict)
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
    spline_degree: int = Field(default=3, ge=1, le=5)
    spline_knots: tuple[float, ...] = ()
    spline_control_x: tuple[float, ...] = ()
    spline_control_y: tuple[float, ...] = ()
    source_belief_history_ids: tuple[str, ...] = ()
    clipping_records: tuple[str, ...] = ()
    fallback_used: bool = False
    fallback_reason: str | None = None


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
    segment_plan: SegmentPlan | None = None


class PlanCommand(StrictModel):
    """Versioned per-group execution command (spec 5.2)."""

    command_id: str
    plan_id: str
    plan_revision: int = Field(ge=1)
    scenario_id: str
    group_id: str
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
    directive_type: Literal["constraint", "assignment"] = "constraint"
    assignment_target_id: str | None = None
    assignment_uuv_ids: tuple[str, ...] = ()
    confidence: float = Field(ge=0, le=1)
    conflicts: tuple[str, ...] = ()
    status: Literal["preview", "applied", "rejected", "needs_clarification"] = "preview"

    @model_validator(mode="after")
    def ambiguity_requires_clarification(self) -> ExpertDirective:
        if self.confidence < 0.70 and self.status == "applied":
            raise ValueError("low-confidence directives cannot be applied")
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
