"""Truth-safe contracts for the rule-based event world-model showcase.

The showcase deliberately predicts *consequences* only.  Its inputs are
limited to estimator-visible IMM belief state, a B-spline trajectory
prediction, operational UUV state, association trends, and public map
bounds.  No simulator truth or low-level control command is represented in
the contract, and every output explicitly declares that it has no control
authority.
"""

from __future__ import annotations

from enum import StrEnum
from itertools import pairwise
from math import isfinite
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from underwater_tracking.domain.models import EventLevel, StrictModel


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
PositiveFloat = Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
PointXY = tuple[FiniteFloat, FiniteFloat]
MapBoundsXY = tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]


class FrozenStrictModel(StrictModel):
    """Strict immutable base used by every public world-model contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class HorizonName(StrEnum):
    H1 = "H1"
    H2 = "H2"
    H3 = "H3"
    H4 = "H4"


class EventType(StrEnum):
    TARGET_TURN_LEFT = "target_turn_left"
    TARGET_TURN_RIGHT = "target_turn_right"
    HIGH_SPEED_ESCAPE = "high_speed_escape"
    AREA_EXIT_RISK = "area_exit_risk"
    GEOMETRY_DEGRADATION = "geometry_degradation"
    TRACK_LOSS_RISK = "track_loss_risk"
    DECOY_OR_NEW_CONTACT_AMBIGUITY = "decoy_or_new_contact_ambiguity"
    UUV_COVERAGE_GAP = "uuv_coverage_gap"
    TARGET_ABNORMAL_STOP = "target_abnormal_stop"


class DataStatus(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    EXPIRED = "expired"
    UNAVAILABLE = "unavailable"


class ForecastProvenance(FrozenStrictModel):
    """Identity of the public evidence, separate from generation/frame time."""
    source_track_revision: int | None = Field(default=None, ge=1)
    prediction_revision: int | None = Field(default=None, ge=1)
    last_observed_at_s: NonNegativeFloat | None = None
    generated_at_s: NonNegativeFloat | None = None
    valid_until_s: NonNegativeFloat | None = None
    source_prediction_id: str | None = None
    source_plan_revision: int | None = Field(default=None, ge=1)
    owner_group_id: str | None = None
    region_id: str | None = None
    region_geometry_revision: int | None = Field(default=None, ge=1)
    source_group_id: str | None = None
    control_authority: Literal[False] = False


class HorizonSpec(FrozenStrictModel):
    name: HorizonName
    start_offset_s: NonNegativeFloat
    end_offset_s: PositiveFloat

    @model_validator(mode="after")
    def interval_is_ordered(self) -> HorizonSpec:
        if self.end_offset_s <= self.start_offset_s:
            raise ValueError("horizon end_offset_s must be after start_offset_s")
        return self


class RuleThresholds(FrozenStrictModel):
    turn_model_probability_min: Probability = 0.45
    turn_heading_change_rad: PositiveFloat = 0.25
    turn_heading_change_strong_rad: PositiveFloat = 0.70
    sprint_speed_threshold_mps: PositiveFloat = 10.0
    sprint_reference_speed_mps: PositiveFloat = 14.0
    stop_speed_threshold_mps: NonNegativeFloat = 0.20
    stop_confirmation_s: PositiveFloat = 60.0
    min_tracking_uuvs: int = Field(default=2, ge=1)
    low_energy_fraction: Probability = 0.10
    max_uuv_state_age_s: PositiveFloat = 30.0
    geometry_warning_od: Probability = 0.35
    geometry_critical_od: Probability = 0.15
    fim_rank_tolerance: PositiveFloat = 1.0e-12
    association_confidence_drop: Probability = 0.20
    association_entropy_rise: NonNegativeFloat = 0.20
    track_loss_corridor_m: PositiveFloat = 800.0
    track_loss_quality_threshold: Probability = 0.40
    event_min_confidence: Probability = 0.55

    @model_validator(mode="after")
    def related_thresholds_are_ordered(self) -> RuleThresholds:
        if self.turn_heading_change_strong_rad <= self.turn_heading_change_rad:
            raise ValueError("strong turn threshold must exceed the trigger threshold")
        if self.sprint_reference_speed_mps <= self.sprint_speed_threshold_mps:
            raise ValueError("sprint reference speed must exceed the trigger threshold")
        if self.geometry_warning_od <= self.geometry_critical_od:
            raise ValueError("geometry warning threshold must exceed the critical threshold")
        return self


class RuleWorldModelConfig(FrozenStrictModel):
    schema_version: str = "world-model-rule-config-v1"
    enabled: bool = True
    horizons: tuple[HorizonSpec, ...] = (
        HorizonSpec(name=HorizonName.H1, start_offset_s=0.0, end_offset_s=120.0),
        HorizonSpec(name=HorizonName.H2, start_offset_s=120.0, end_offset_s=300.0),
        HorizonSpec(name=HorizonName.H3, start_offset_s=300.0, end_offset_s=900.0),
        HorizonSpec(name=HorizonName.H4, start_offset_s=900.0, end_offset_s=1800.0),
    )
    thresholds: RuleThresholds = Field(default_factory=RuleThresholds)

    @field_validator("horizons")
    @classmethod
    def horizons_are_complete_and_non_overlapping(
        cls, value: tuple[HorizonSpec, ...]
    ) -> tuple[HorizonSpec, ...]:
        if tuple(item.name for item in value) != tuple(HorizonName):
            raise ValueError("horizons must contain H1, H2, H3, H4 in order")
        for previous, current in pairwise(value):
            if current.start_offset_s < previous.end_offset_s:
                raise ValueError("horizon intervals must not overlap")
        return value


class ImmBeliefInput(FrozenStrictModel):
    position_xy: PointXY
    velocity_xy_mps: PointXY
    turn_rate_rad_s: FiniteFloat = 0.0
    covariance_trace_m2: NonNegativeFloat
    model_probabilities: dict[str, Probability]

    @field_validator("model_probabilities")
    @classmethod
    def model_probabilities_form_distribution(
        cls, value: dict[str, float]
    ) -> dict[str, float]:
        if not value:
            raise ValueError("IMM model_probabilities must not be empty")
        total = sum(value.values())
        if not isfinite(total) or abs(total - 1.0) > 1.0e-6:
            raise ValueError("IMM model_probabilities must sum to 1")
        return dict(sorted(value.items()))


class TrajectoryForecastInput(FrozenStrictModel):
    prediction_id: str = Field(min_length=1)
    times_s: tuple[FiniteFloat, ...] = Field(min_length=1)
    points_xy: tuple[PointXY, ...] = Field(min_length=1)
    corridor_radius_m: tuple[NonNegativeFloat, ...] = Field(min_length=1)
    fallback_used: bool = False
    fallback_reason: str | None = None
    prediction_regime: Literal["imm", "bspline", "short_history", "boundary_recovery"] = "bspline"

    @model_validator(mode="after")
    def arrays_are_aligned_and_time_is_increasing(self) -> TrajectoryForecastInput:
        count = len(self.times_s)
        if len(self.points_xy) != count or len(self.corridor_radius_m) != count:
            raise ValueError("trajectory times, points, and corridor arrays must align")
        if any(right <= left for left, right in zip(self.times_s, self.times_s[1:])):
            raise ValueError("trajectory times_s must be strictly increasing")
        if self.fallback_reason is not None and not self.fallback_used:
            raise ValueError("fallback_reason requires fallback_used=True")
        return self


class UuvForecastInput(FrozenStrictModel):
    uuv_id: str = Field(min_length=1)
    position_xy: PointXY
    velocity_xy_mps: PointXY = (0.0, 0.0)
    passive_range_m: PositiveFloat
    bearing_variance_rad2: PositiveFloat
    energy_fraction: Probability = 1.0
    healthy: bool = True
    communication_ok: bool = True
    state_age_s: NonNegativeFloat = 0.0
    state_time_s: NonNegativeFloat | None = None
    planned_times_s: tuple[FiniteFloat, ...] = ()
    planned_points_xy: tuple[PointXY, ...] = ()

    @model_validator(mode="after")
    def planned_track_is_aligned(self) -> UuvForecastInput:
        if len(self.planned_times_s) != len(self.planned_points_xy):
            raise ValueError("planned UUV times and points must align")
        if any(
            right <= left
            for left, right in zip(self.planned_times_s, self.planned_times_s[1:])
        ):
            raise ValueError("planned UUV times must be strictly increasing")
        return self


class TrackingContextInput(FrozenStrictModel):
    quality_ewma: Probability
    current_contact_count: int = Field(default=1, ge=0)
    previous_contact_count: int | None = Field(default=None, ge=0)
    association_confidence: Probability | None = None
    previous_association_confidence: Probability | None = None
    association_entropy: NonNegativeFloat | None = None
    previous_association_entropy: NonNegativeFloat | None = None
    observability_hypotheses: dict[str, Probability] = Field(default_factory=dict)

    @field_validator("observability_hypotheses")
    @classmethod
    def hypotheses_are_sorted(cls, value: dict[str, float]) -> dict[str, float]:
        if any(not name for name in value):
            raise ValueError("observability hypothesis names must not be empty")
        return dict(sorted(value.items()))


class RuleWorldModelInput(ForecastProvenance):
    scenario_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    as_of_s: NonNegativeFloat
    belief: ImmBeliefInput
    trajectory: TrajectoryForecastInput
    uuvs: tuple[UuvForecastInput, ...] = ()
    tracking: TrackingContextInput
    map_bounds_xy: MapBoundsXY | None = None
    source_observation_ids: tuple[str, ...] = ()
    source_observability_event_ids: tuple[str, ...] = ()
    source_plan_revision: int | None = Field(default=None, ge=1)
    source_status: Literal["current", "degraded", "expired", "unavailable"] = "unavailable"
    source_reason_codes: tuple[str, ...] = ()
    task_region_bounds_xy: MapBoundsXY | None = None

    @model_validator(mode="after")
    def operational_inputs_are_consistent(self) -> RuleWorldModelInput:
        if self.trajectory.times_s[0] <= self.as_of_s:
            raise ValueError("trajectory samples must be strictly later than as_of_s")
        ids = [uuv.uuv_id for uuv in self.uuvs]
        if len(ids) != len(set(ids)):
            raise ValueError("uuv_ids must be unique")
        if self.map_bounds_xy is not None:
            min_x, max_x, min_y, max_y = self.map_bounds_xy
            if max_x <= min_x or max_y <= min_y:
                raise ValueError("map_bounds_xy must have positive area")
        return self


EvidenceSource = Literal[
    "imm",
    "bspline",
    "tracking_context",
    "uuv_projection",
    "map_bounds",
    "observability",
    "short_history",
    "boundary_recovery",
    "task_region",
]


class RuleEvidence(FrozenStrictModel):
    key: str = Field(min_length=1)
    source: EvidenceSource
    value: FiniteFloat
    threshold: FiniteFloat | None = None
    unit: str = "1"
    description: str = Field(min_length=1)


class PredictedEvent(ForecastProvenance):
    event_id: str = Field(min_length=1)
    event_type: EventType
    target_id: str = Field(min_length=1)
    horizon: HorizonName
    predicted_time_s: NonNegativeFloat
    time_to_event_s: NonNegativeFloat
    predicted_position_xy: PointXY
    confidence: Probability
    level: EventLevel
    rule_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    evidence: tuple[RuleEvidence, ...] = Field(min_length=1)


class HorizonCoverage(FrozenStrictModel):
    name: HorizonName
    start_offset_s: NonNegativeFloat
    end_offset_s: PositiveFloat
    sample_count: int = Field(ge=0)
    covered: bool


class WorldModelForecast(ForecastProvenance):
    schema_version: str = "world-model-event-forecast-v1"
    model_kind: Literal["rule_demo"] = "rule_demo"
    model_version: str = "rule-event-v1"
    control_authority: Literal[False] = False
    scenario_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    as_of_s: NonNegativeFloat
    source_prediction_id: str = Field(min_length=1)
    source_observation_ids: tuple[str, ...] = ()
    source_observability_event_ids: tuple[str, ...] = ()
    source_plan_revision: int | None = Field(default=None, ge=1)
    data_status: DataStatus
    trajectory_fallback_used: bool
    imm_model_probabilities: dict[str, Probability]
    horizons: tuple[HorizonCoverage, ...]
    events: tuple[PredictedEvent, ...]
    warnings: tuple[str, ...] = ()
