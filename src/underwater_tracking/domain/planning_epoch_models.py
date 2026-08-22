"""Durable contracts for a single central planning attempt.

The epoch freezes the inputs a provider may read.  Physics revisions can move
while the provider is running; semantic revalidation, not revision equality,
decides whether the resulting executable plan can be committed.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from underwater_tracking.domain.mission_models import ExecutableMissionPlan
from underwater_tracking.domain.models import SituationSnapshot, StrictModel
from underwater_tracking.runtime.mission_controller import MissionSnapshot


class PlanningEpochStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMMITTED = "committed"
    INVALIDATED = "invalidated"
    REJECTED = "rejected"
    FAILED = "failed"


def _unique_ids(value: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if any(not item for item in value):
        raise ValueError(f"{field_name} must not contain empty IDs")
    if len(value) != len(set(value)):
        raise ValueError(f"{field_name} must contain unique IDs")
    return value


class PlanningEpoch(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epoch_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    base_physics_revision: int = Field(ge=0)
    base_sim_time_s: int = Field(ge=0)
    observation_batch_id: str = Field(min_length=1)
    critical_event_ids: tuple[str, ...] = ()
    public_target_prior_ids: tuple[str, ...] = ()
    public_target_estimate_ids: tuple[str, ...] = ()
    resource_manifest_hash: str = Field(min_length=1)
    active_plan_version: int = Field(ge=0)
    expert_request_version: int | None = Field(default=None, ge=0)
    status: PlanningEpochStatus = PlanningEpochStatus.QUEUED

    @model_validator(mode="after")
    def validate_ids(self) -> PlanningEpoch:
        _unique_ids(self.critical_event_ids, "critical_event_ids")
        _unique_ids(self.public_target_prior_ids, "public_target_prior_ids")
        _unique_ids(self.public_target_estimate_ids, "public_target_estimate_ids")
        return self


class EpochCommitResult(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epoch_id: str = Field(min_length=1)
    status: Literal["committed", "invalidated", "rejected", "failed"]
    plan_id: str | None = None
    plan_version: int | None = Field(default=None, ge=1)
    validation_report_id: str | None = None
    executable_plan: ExecutableMissionPlan | None = None
    invalidated_reason: str | None = None
    failure_category: Literal["timeout", "provider", "schema", "configuration", "internal"] | None = None
    failure_message: str | None = Field(default=None, max_length=2000)
    consumed_event_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_terminal_contract(self) -> EpochCommitResult:
        _unique_ids(self.consumed_event_ids, "consumed_event_ids")
        if self.status == "committed":
            if self.plan_id is None:
                raise ValueError("committed result requires plan_id")
            if self.plan_version is None:
                raise ValueError("committed result requires plan_version")
            if self.validation_report_id is None:
                raise ValueError("committed result requires validation_report_id")
            if self.executable_plan is None:
                raise ValueError("committed result requires executable_plan")
            if self.executable_plan.revision != self.plan_version:
                raise ValueError("committed executable_plan revision must match plan_version")
        elif self.status == "invalidated":
            if self.validation_report_id is None:
                raise ValueError("invalidated result requires validation_report_id")
            if not self.invalidated_reason:
                raise ValueError("invalidated result requires invalidated_reason")
            if any(value is not None for value in (self.plan_id, self.plan_version, self.executable_plan)):
                raise ValueError("invalidated result cannot expose a selected plan")
        elif self.status == "rejected":
            if self.validation_report_id is None:
                raise ValueError("rejected result requires validation_report_id")
        else:
            if self.failure_category is None:
                raise ValueError("failed result requires failure_category")
            if not self.failure_message:
                raise ValueError("failed result requires failure_message")
        return self


class PlanningEpochCapture(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    epoch: PlanningEpoch
    situation: SituationSnapshot
    mission: MissionSnapshot

