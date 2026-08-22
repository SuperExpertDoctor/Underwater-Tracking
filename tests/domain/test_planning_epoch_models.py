from __future__ import annotations

import pytest

from underwater_tracking.domain.mission_models import ExecutableMissionPlan
from underwater_tracking.domain.models import (
    EventLevel,
    RuntimeEvent,
    SituationSnapshot,
)
from underwater_tracking.domain.planning_epoch_models import (
    EpochCommitResult,
    PlanningEpoch,
    PlanningEpochCapture,
    PlanningEpochStatus,
)
from underwater_tracking.runtime.mission_controller import MissionSnapshot


def make_epoch(epoch_id: str = "epoch:S1:1") -> PlanningEpoch:
    return PlanningEpoch(
        epoch_id=epoch_id,
        scenario_id="S1",
        base_physics_revision=1,
        base_sim_time_s=30,
        observation_batch_id="observation:S1:1",
        critical_event_ids=("event-1",),
        public_target_prior_ids=("prior-1",),
        public_target_estimate_ids=(),
        resource_manifest_hash="manifest-1",
        active_plan_version=0,
    )


def test_invalidated_epoch_requires_reason() -> None:
    with pytest.raises(ValueError, match="invalidated_reason"):
        EpochCommitResult(
            epoch_id="epoch:S1:1",
            status="invalidated",
            validation_report_id="validation:S1:1",
        )


def test_committed_epoch_requires_plan_identity() -> None:
    with pytest.raises(ValueError, match="plan_id"):
        EpochCommitResult(
            epoch_id="epoch:S1:1",
            status="committed",
            validation_report_id="validation:S1:1",
        )


def test_planning_epoch_rejects_duplicate_ids_and_is_frozen() -> None:
    with pytest.raises(ValueError, match="critical_event_ids"):
        PlanningEpoch.model_validate(
            {**make_epoch().model_dump(), "critical_event_ids": ("event-1", "event-1")}
        )
    epoch = make_epoch()
    with pytest.raises(Exception):
        epoch.status = PlanningEpochStatus.RUNNING  # type: ignore[misc]


def test_capture_contains_typed_immutable_inputs() -> None:
    situation = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=1,
        sim_time_s=30,
        uuvs=(),
        group_reports=(),
        pending_events=(
            RuntimeEvent(
                event_id="event-1",
                scenario_id="S1",
                sim_time_s=30,
                event_type="initialization",
                level=EventLevel.STRATEGIC,
            ),
        ),
    )
    mission = MissionSnapshot(scenario_id="S1", sim_time_s=30, plan_revision=0)
    capture = PlanningEpochCapture(epoch=make_epoch(), situation=situation, mission=mission)
    assert capture.epoch.epoch_id == "epoch:S1:1"


def test_failed_result_requires_bounded_category() -> None:
    result = EpochCommitResult(
        epoch_id="epoch:S1:1",
        status="failed",
        failure_category="provider",
        failure_message="provider timeout",
    )
    assert result.failure_category == "provider"


def test_committed_result_requires_matching_executable_revision() -> None:
    with pytest.raises(ValueError, match="plan_version"):
        EpochCommitResult(
            epoch_id="epoch:S1:1",
            status="committed",
            plan_id="plan:S1:1",
            plan_version=2,
            validation_report_id="validation:S1:1",
            executable_plan=ExecutableMissionPlan(revision=1),
        )
