# src/underwater_tracking/domain/models.py
from __future__ import annotations
from enum import StrEnum
from math import pi
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class ContactClassification(StrEnum):
    UNVERIFIED = "unverified"
    SUBMARINE = "submarine"
    DECOY = "decoy"


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
    group_id: str | None = None
    sensor_mode: Literal["passive", "active"] = "passive"
    reserved: bool = False


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
    group_reports: tuple[GroupReport, ...]
    pending_events: tuple[RuntimeEvent, ...]
    contacts: tuple[Contact, ...] = ()
    active_plan_id: str | None = None
    active_plan_revision: int | None = None
