# src/underwater_tracking/domain/models.py
from __future__ import annotations
from enum import StrEnum
from math import pi
from typing import Annotated, Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from underwater_tracking.domain.relationships import (
    expected_carrier_status,
    normalize_legacy_carrier_relationships,
    normalize_legacy_uuv_deployment_state,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EventLevel(StrEnum):
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    INFORMATIONAL = "informational"


class UUVStatus(StrEnum):
    AVAILABLE = "available"
    TRACKING = "tracking"
    RETURNING = "returning"
    FAILED = "failed"


class CarrierStatus(StrEnum):
    STANDBY = "standby"
    TRANSIT = "transit"
    DEPLOYING = "deploying"
    RECOVERING = "recovering"


class DeploymentState(StrEnum):
    ONBOARD = "onboard"
    DEPLOYED = "deployed"
    RETURNING = "returning"
    FAILED = "failed"


class ContactClassification(StrEnum):
    UNVERIFIED = "unverified"
    SUBMARINE = "submarine"
    DECOY = "decoy"


class IntelligenceSource(StrEnum):
    TECHNICAL_RECONNAISSANCE = "technical_reconnaissance"
    SIGINT = "sigint"
    ELINT = "elint"
    HUMINT = "humint"
    SONAR = "sonar"


_UnitInterval = Annotated[float, Field(ge=0, le=1)]


class SurveillanceCapability(StrictModel):
    """The sensing and maneuver limits available to one UUV."""

    passive_range_m: float = Field(default=4000.0, gt=0)
    active_range_m: float = Field(default=3000.0, gt=0)
    bearing_variance_rad2: float = Field(default=1e-2, gt=0)
    active_sonar_available: bool = True
    max_speed_mps: float = Field(default=4.0, gt=0)
    max_turn_rate_rad_s: float = Field(default=pi / 60.0, gt=0)


class OperationalScheme(StrictModel):
    """A time-bounded, traceable set of deterministic tracking constraints."""

    scheme_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    target_priorities: dict[str, float] = Field(default_factory=dict)
    minimum_quality: dict[str, _UnitInterval] = Field(default_factory=dict)
    valid_from_s: int = Field(ge=0)
    valid_until_s: int = Field(ge=0)
    constraints: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validity_interval_is_positive(self) -> OperationalScheme:
        if self.valid_until_s <= self.valid_from_s:
            raise ValueError("valid_until_s must be after valid_from_s")
        return self


class IntelligenceReport(StrictModel):
    """A source-attributed operational assessment with a finite lifetime."""

    report_id: str = Field(min_length=1)
    source: IntelligenceSource
    target_id: str = Field(min_length=1)
    confidence: _UnitInterval
    issued_at_s: int = Field(ge=0)
    valid_until_s: int = Field(ge=0)
    assessment: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def expiry_is_after_issue_time(self) -> IntelligenceReport:
        if self.valid_until_s <= self.issued_at_s:
            raise ValueError("valid_until_s must be after issued_at_s")
        return self


class Contact(StrictModel):
    """One operational sonar contact (spec 11.1 amendment, R5).

    The classification is the operational measurement produced by active
    pings; it is not truth (decoy truth stays truth-side only). Targets
    being tracked enter classified SUBMARINE (already dispatched); decoys
    enter UNVERIFIED and are pinged by the active-verification protocol.
    """

    contact_id: str
    sim_time_s: int = Field(ge=0)
    bearing_rays: tuple[BearingObservation, ...] = ()
    classification: ContactClassification = ContactClassification.UNVERIFIED
    classification_evidence: tuple[str, ...] = ()
    estimated_position_xy: tuple[float, float] | None = None


class BearingObservation(StrictModel):
    observation_id: str
    scenario_id: str
    sim_time_s: int = Field(ge=0)
    uuv_id: str
    target_id: str
    azimuth_rad: float
    variance_rad2: float = Field(gt=0)
    detection_confidence: float = Field(ge=0, le=1)

    @field_validator("azimuth_rad")
    @classmethod
    def wrap_angle(cls, value: float) -> float:
        return (value + pi) % (2 * pi) - pi


class UUVState(StrictModel):
    uuv_id: str
    position_xy: tuple[float, float]
    heading_rad: float
    speed_mps: float = Field(ge=0)
    energy_fraction: float = Field(ge=0, le=1)
    status: UUVStatus
    deployment_state: DeploymentState = DeploymentState.DEPLOYED
    group_id: str | None = None
    sensor_mode: Literal["passive", "active"] = "passive"
    capability: SurveillanceCapability = Field(default_factory=SurveillanceCapability)
    reserved: bool = False

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_deployment_state(cls, value: Any) -> Any:
        return normalize_legacy_uuv_deployment_state(value)

    @model_validator(mode="after")
    def status_matches_deployment_state(self) -> UUVState:
        if self.status is UUVStatus.TRACKING and self.deployment_state is DeploymentState.ONBOARD:
            raise ValueError("tracking status cannot be onboard")
        if self.status is UUVStatus.TRACKING and self.deployment_state is DeploymentState.FAILED:
            raise ValueError("tracking status cannot be failed")
        if self.status is UUVStatus.RETURNING and self.deployment_state is not DeploymentState.RETURNING:
            raise ValueError("returning status requires returning deployment_state")
        if self.status is UUVStatus.FAILED and self.deployment_state is not DeploymentState.FAILED:
            raise ValueError("failed status requires failed deployment_state")
        if self.deployment_state is DeploymentState.RETURNING and self.status is not UUVStatus.RETURNING:
            raise ValueError("returning deployment_state requires returning status")
        if self.deployment_state is DeploymentState.FAILED and self.status is not UUVStatus.FAILED:
            raise ValueError("failed deployment_state requires failed status")
        return self


class CarrierState(StrictModel):
    carrier_id: str
    position_xy: tuple[float, float]
    heading_rad: float
    speed_mps: float = Field(ge=0)
    status: CarrierStatus = CarrierStatus.TRANSIT
    onboard_uuv_ids: tuple[str, ...] = ()
    deployed_uuv_ids: tuple[str, ...] = ()
    returning_uuv_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def relationship_lists_are_disjoint(self) -> CarrierState:
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
        expected = expected_carrier_status(
            self.speed_mps, self.onboard_uuv_ids, self.deployed_uuv_ids, self.returning_uuv_ids
        )
        if str(self.status) != expected:
            if self.returning_uuv_ids:
                raise ValueError("returning UUVs require recovering status")
            if self.status is CarrierStatus.RECOVERING:
                raise ValueError("recovering status requires returning UUVs")
            if self.status is CarrierStatus.DEPLOYING:
                raise ValueError("deploying status requires onboard and deployed UUVs")
            if self.status is CarrierStatus.STANDBY:
                raise ValueError("standby status requires zero speed")
            if self.status is CarrierStatus.TRANSIT:
                raise ValueError("transit status requires movement")
        return self


class TargetBelief(StrictModel):
    target_id: str
    sim_time_s: int
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    model_probabilities: dict[str, float]
    source_observation_ids: tuple[str, ...] = ()
    fim_min_eigenvalue: float = 0.0
    fim_condition: float = float("inf")


class GroupQuality(StrictModel):
    instant: float = Field(ge=0, le=1)
    window_mean: float = Field(ge=0, le=1)
    ewma: float = Field(ge=0, le=1)
    components: dict[str, float]
    hard_guard_reasons: tuple[str, ...] = ()


class GroupReport(StrictModel):
    group_id: str
    target_id: str
    sim_time_s: int
    member_ids: tuple[str, ...]
    belief: TargetBelief
    quality: GroupQuality
    plan_revision: int
    event_types: tuple[str, ...] = ()


class RuntimeEvent(StrictModel):
    event_id: str
    scenario_id: str
    sim_time_s: int
    event_type: str
    entity_id: str | None = None
    level: EventLevel
    payload: dict[str, Any] = {}


class SituationSnapshot(StrictModel):
    scenario_id: str
    snapshot_revision: int
    sim_time_s: int
    uuvs: tuple[UUVState, ...]
    carrier: CarrierState | None = None
    group_reports: tuple[GroupReport, ...]
    pending_events: tuple[RuntimeEvent, ...]
    contacts: tuple[Contact, ...] = ()
    active_plan_id: str | None = None
    active_plan_revision: int | None = None
    operational_scheme: OperationalScheme | None = None
    intelligence_reports: tuple[IntelligenceReport, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_carrier_relationships(cls, value: Any) -> Any:
        return normalize_legacy_carrier_relationships(value)

    @model_validator(mode="after")
    def carrier_relationships_match_uuvs(self) -> SituationSnapshot:
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
