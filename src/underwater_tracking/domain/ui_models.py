# src/underwater_tracking/domain/ui_models.py
"""Versioned UI frame contracts with strict truth isolation (spec 17.3).

``OperationalFrame`` and its view models carry only estimator-visible
state: positions, covariances, bearings, groups, plans, events, ledger
rows and metrics.  They must never contain target truth — that gate is
enforced by ``tests/api/test_frame_contracts.py`` at the schema level and
by ``tests/api/test_truth_isolation.py`` at the route level (task 10).

Target truth lives exclusively in ``EvaluationFrame``, a standalone model
(no inheritance from ``OperationalFrame``) that wraps ``TargetTruth``
dataclasses together with the identifiers pairing it to the operational
run it was collected under.  It is served only by separately enabled
evaluation routes.
"""
from __future__ import annotations

from math import pi
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from underwater_tracking.domain.agent_models import Concept, IntentLabel, PlanStatus
from underwater_tracking.domain.models import (
    CarrierStatus,
    DeploymentState,
    EventLevel,
    StrictModel,
    UUVStatus,
)
from underwater_tracking.domain.relationships import (
    normalize_legacy_carrier_relationships,
    normalize_legacy_uuv_deployment_state,
)
from underwater_tracking.domain.truth import TargetTruth


class Point2D(StrictModel):
    x: float
    y: float


class MapBounds(StrictModel):
    """Axis-aligned region the tactical map clips geometry to."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @model_validator(mode="after")
    def axes_ordered(self) -> MapBounds:
        if self.min_x > self.max_x or self.min_y > self.max_y:
            raise ValueError("map bounds require min <= max on both axes")
        return self


class CovarianceEllipse(StrictModel):
    """Covariance rendered as an ellipse: axes in meters and rotation."""

    semimajor_m: float = Field(gt=0)
    semiminor_m: float = Field(gt=0)
    rotation_rad: float

    @field_validator("rotation_rad")
    @classmethod
    def wrap_angle(cls, value: float) -> float:
        return (value + pi) % (2 * pi) - pi

    @model_validator(mode="after")
    def axes_ordered(self) -> CovarianceEllipse:
        if self.semiminor_m > self.semimajor_m:
            raise ValueError("semiminor axis must not exceed semimajor axis")
        return self


class UUVView(StrictModel):
    uuv_id: str
    status: UUVStatus
    deployment_state: DeploymentState = DeploymentState.DEPLOYED
    position: Point2D
    heading_rad: float
    speed_mps: float = Field(ge=0)
    energy_fraction: float = Field(ge=0, le=1)
    group_id: str | None = None
    current_waypoint: Point2D | None = None
    breadcrumb: tuple[Point2D, ...] = ()
    sensor_mode: Literal["active", "passive"] = "passive"
    reserved: bool = False

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_deployment_state(cls, value: Any) -> Any:
        return normalize_legacy_uuv_deployment_state(value)

    @model_validator(mode="after")
    def status_matches_deployment_state(self) -> UUVView:
        if self.status is UUVStatus.RETURNING and self.deployment_state is not DeploymentState.RETURNING:
            raise ValueError("returning status requires returning deployment_state")
        if self.status is UUVStatus.FAILED and self.deployment_state is not DeploymentState.FAILED:
            raise ValueError("failed status requires failed deployment_state")
        return self


class CarrierView(StrictModel):
    carrier_id: str
    position: Point2D
    heading_rad: float
    speed_mps: float = Field(ge=0)
    status: CarrierStatus = CarrierStatus.TRANSIT
    onboard_uuv_ids: tuple[str, ...] = ()
    deployed_uuv_ids: tuple[str, ...] = ()
    returning_uuv_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def relationship_lists_are_disjoint(self) -> CarrierView:
        raw_lists = (
            self.onboard_uuv_ids,
            self.deployed_uuv_ids,
            self.returning_uuv_ids,
        )
        if any(len(ids) != len(set(ids)) for ids in raw_lists):
            raise ValueError("carrier relationship lists must not contain duplicate IDs")
        lists = tuple(set(ids) for ids in raw_lists)
        if any(left & right for index, left in enumerate(lists) for right in lists[index + 1 :]):
            raise ValueError("carrier relationship lists must be disjoint")
        return self


class IntentView(StrictModel):
    label: IntentLabel
    confidence: float = Field(ge=0, le=1)
    alternatives: dict[IntentLabel, float] = Field(default_factory=dict)


class PredictionCorridorView(StrictModel):
    """Predicted track centerline with a radius envelope per sample."""

    horizon_s: float = Field(gt=0)
    sample_step_s: float = Field(gt=0)
    centerline_xy: tuple[Point2D, ...] = ()
    radius_m: tuple[float, ...] = ()


class EstimateQualityView(StrictModel):
    """Estimator-visible quality proxies; never true error."""

    quality_score: float = Field(ge=0, le=1)
    estimated_rmse_m: float = Field(ge=0)
    fim_min_eigenvalue: float = Field(ge=0)
    fim_condition: float = Field(ge=0)


class TargetEstimateView(StrictModel):
    target_id: str
    mean: Point2D
    covariance_ellipse: CovarianceEllipse
    intent: IntentView
    prediction: PredictionCorridorView | None = None
    quality: EstimateQualityView
    classification: Literal["submarine", "decoy", "unknown"] = "unknown"
    last_ping_s: int | None = None


class BearingRayView(StrictModel):
    observation_id: str
    uuv_id: str
    target_id: str
    origin: Point2D
    azimuth_rad: float
    variance_rad2: float = Field(gt=0)
    confidence: float = Field(ge=0, le=1)

    @field_validator("azimuth_rad")
    @classmethod
    def wrap_angle(cls, value: float) -> float:
        return (value + pi) % (2 * pi) - pi


class GroupQualityView(StrictModel):
    instant: float = Field(ge=0, le=1)
    window_mean: float = Field(ge=0, le=1)
    ewma: float = Field(ge=0, le=1)
    components: dict[str, float] = Field(default_factory=dict)
    hard_guard_reasons: tuple[str, ...] = ()


class GroupView(StrictModel):
    group_id: str
    target_id: str
    member_ids: tuple[str, ...] = ()
    quality: GroupQualityView


class EventView(StrictModel):
    event_id: str
    sim_time_s: int = Field(ge=0)
    event_type: str
    level: EventLevel
    entity_id: str | None = None
    message: str = ""


class PlanView(StrictModel):
    """One plan as rendered to the operator (current or candidate).

    ``version`` is the version the frame's ``plan_version`` must agree
    with whenever this plan is the active one.
    """

    plan_id: str
    version: int = Field(ge=1)
    status: PlanStatus
    concept: Concept = "hold_current"
    reason: str = ""
    affected_targets: tuple[str, ...] = ()
    group_changes: tuple[str, ...] = ()
    valid_from_s: int = Field(default=0, ge=0)
    valid_until_s: int | None = None
    segment_plan: tuple[str, ...] = ()


class LedgerView(StrictModel):
    """One traceable decision row for the decision ledger."""

    decision_id: str
    sim_time_s: int = Field(ge=0)
    outcome: Literal["committed", "degraded", "rejected"] = "committed"
    trigger_event_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    final_plan_id: str | None = None
    final_plan_version: int | None = None


class MetricView(StrictModel):
    metric_id: str
    label: str = ""
    value: float
    unit: str = ""
    threshold: float | None = None
    window_s: int = Field(default=0, ge=0)
    series: tuple[float, ...] = ()


class OperationalFrame(StrictModel):
    """Versioned operational snapshot broadcast over the live wire.

    ``plan_version`` is the committed plan version the frame renders; it
    must agree with the version of the active plan carried in ``plans``.
    """

    schema_version: str = "1.0"
    frame_id: int = Field(ge=0)
    sim_time_s: int = Field(ge=0)
    plan_version: int = Field(ge=0)
    map_bounds: MapBounds
    carrier: CarrierView | None = None
    uuvs: tuple[UUVView, ...] = ()
    target_estimates: tuple[TargetEstimateView, ...] = ()
    bearing_rays: tuple[BearingRayView, ...] = ()
    groups: tuple[GroupView, ...] = ()
    events: tuple[EventView, ...] = ()
    plans: tuple[PlanView, ...] = ()
    ledger: tuple[LedgerView, ...] = ()
    metrics: tuple[MetricView, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_carrier_relationships(cls, value: Any) -> Any:
        return normalize_legacy_carrier_relationships(value)

    @model_validator(mode="after")
    def plan_version_matches_active_plan(self) -> OperationalFrame:
        for plan in self.plans:
            if plan.status == "active" and plan.version != self.plan_version:
                raise ValueError(
                    f"frame plan_version {self.plan_version} does not match active "
                    f"plan {plan.plan_id!r} version {plan.version}"
                )
        return self

    @model_validator(mode="after")
    def carrier_relationships_match_uuvs(self) -> OperationalFrame:
        if self.carrier is None:
            return self
        uuvs_by_id = {uuv.uuv_id: uuv for uuv in self.uuvs}
        relationships = {
            DeploymentState.ONBOARD: self.carrier.onboard_uuv_ids,
            DeploymentState.DEPLOYED: self.carrier.deployed_uuv_ids,
            DeploymentState.RETURNING: self.carrier.returning_uuv_ids,
        }
        if any(len(ids) != len(set(ids)) for ids in relationships.values()):
            raise ValueError("carrier relationship lists must not contain duplicate IDs")
        relationship_sets = tuple(set(ids) for ids in relationships.values())
        if any(
            left & right
            for index, left in enumerate(relationship_sets)
            for right in relationship_sets[index + 1 :]
        ):
            raise ValueError("carrier relationship lists must be disjoint")
        listed_ids = {uuv_id for ids in relationships.values() for uuv_id in ids}
        for expected_state, ids in relationships.items():
            for uuv_id in ids:
                uuv = uuvs_by_id.get(uuv_id)
                if uuv is None:
                    raise ValueError(f"carrier lists unknown UUV {uuv_id!r}")
                if (
                    uuv.status is UUVStatus.RETURNING
                    and uuv.deployment_state is not DeploymentState.RETURNING
                ) or (
                    uuv.status is UUVStatus.FAILED
                    and uuv.deployment_state is not DeploymentState.FAILED
                ):
                    raise ValueError(f"uuv {uuv_id!r} status contradicts deployment_state")
                if uuv.status is UUVStatus.FAILED or uuv.deployment_state is DeploymentState.FAILED:
                    raise ValueError(f"carrier lists must omit failed UUV {uuv_id!r}")
                if uuv.deployment_state is not expected_state:
                    raise ValueError(
                        f"carrier list {expected_state.value!r} contains {uuv_id!r} "
                        f"with deployment_state {uuv.deployment_state.value!r}"
                    )
        for uuv in self.uuvs:
            if uuv.status is UUVStatus.FAILED or uuv.deployment_state is DeploymentState.FAILED:
                if uuv.uuv_id in listed_ids:
                    raise ValueError(f"carrier lists must omit failed UUV {uuv.uuv_id!r}")
                continue
            if uuv.uuv_id not in listed_ids:
                raise ValueError(f"carrier lists omit non-failed UUV {uuv.uuv_id!r}")
        return self


class EvaluationFrame(StrictModel):
    """Truth-only frame paired to the operational run it evaluates.

    Standalone by design: it never inherits from ``OperationalFrame`` so
    the operational contract cannot structurally admit truth fields.
    ``scenario_id``, ``run_id`` and ``plan_version`` pair this frame to
    the operational run it was collected under.
    """

    schema_version: str = "1.0"
    frame_id: int = Field(ge=0)
    sim_time_s: int = Field(ge=0)
    scenario_id: str
    run_id: str
    plan_version: int = Field(ge=0)
    targets: tuple[TargetTruth, ...] = ()
