from __future__ import annotations

from pathlib import Path

import pytest

from underwater_tracking.domain.planning_epoch_models import (
    EpochCommitResult,
    PlanningEpoch,
    PlanningEpochCapture,
)
from underwater_tracking.domain.mission_models import ExecutableMissionPlan
from underwater_tracking.planning.mission_revalidation import MissionRevalidationReport
from underwater_tracking.runtime.mission_controller import MissionSnapshot
from underwater_tracking.domain.models import SituationSnapshot
from underwater_tracking.persistence.planning_epochs import PlanningEpochRepository


def make_epoch(epoch_id: str = "epoch:S1:1") -> PlanningEpoch:
    return PlanningEpoch(
        epoch_id=epoch_id,
        scenario_id="S1",
        base_physics_revision=1,
        base_sim_time_s=30,
        observation_batch_id="observation:S1:1",
        critical_event_ids=("event-1",),
        public_target_prior_ids=(),
        public_target_estimate_ids=(),
        resource_manifest_hash="manifest-1",
        active_plan_version=0,
    )


def make_capture(epoch: PlanningEpoch) -> PlanningEpochCapture:
    return PlanningEpochCapture(
        epoch=epoch,
        situation=SituationSnapshot(
            scenario_id="S1",
            snapshot_revision=1,
            sim_time_s=30,
            uuvs=(),
            group_reports=(),
            pending_events=(),
        ),
        mission=MissionSnapshot(scenario_id="S1", sim_time_s=30, plan_revision=0),
    )


def make_failed_result(epoch_id: str, error: str = "provider timeout") -> EpochCommitResult:
    return EpochCommitResult(
        epoch_id=epoch_id,
        status="failed",
        failure_category="provider",
        failure_message=error,
    )


def test_epoch_terminal_result_is_idempotent_but_not_replaceable(tmp_path: Path) -> None:
    repo = PlanningEpochRepository(tmp_path / "agent.db")
    epoch = make_epoch()
    result = make_failed_result(epoch.epoch_id)
    repo.create(make_capture(epoch))
    repo.mark_running(epoch.epoch_id)
    repo.finish(result)
    repo.finish(result)
    with pytest.raises(ValueError, match="already finished"):
        repo.finish(make_failed_result(epoch.epoch_id, error="different failure"))
    repo.close()


def test_epoch_capture_round_trips_without_live_state(tmp_path: Path) -> None:
    repo = PlanningEpochRepository(tmp_path / "agent.db")
    capture = make_capture(make_epoch())
    repo.create(capture)
    loaded = repo.get_capture(capture.epoch.epoch_id)
    assert loaded == capture
    assert repo.get(capture.epoch.epoch_id) == capture.epoch
    repo.close()


def test_latest_returns_terminal_result(tmp_path: Path) -> None:
    repo = PlanningEpochRepository(tmp_path / "agent.db")
    capture = make_capture(make_epoch())
    repo.create(capture)
    result = make_failed_result(capture.epoch.epoch_id)
    repo.finish(result)
    latest = repo.latest("S1")
    assert latest == (capture.epoch.model_copy(update={"status": "failed"}), result)
    repo.close()


def test_revalidation_report_and_result_round_trip_atomically(tmp_path: Path) -> None:
    repo = PlanningEpochRepository(tmp_path / "agent.db")
    capture = make_capture(make_epoch())
    repo.create(capture)
    report = MissionRevalidationReport(
        report_id="validation:S1:1",
        epoch_id=capture.epoch.epoch_id,
        current_physics_revision=2,
        current_plan_version=0,
        valid=True,
        rebased_plan=ExecutableMissionPlan(revision=1),
    )
    result = EpochCommitResult(
        epoch_id=capture.epoch.epoch_id,
        status="committed",
        plan_id="plan:S1:1",
        plan_version=1,
        validation_report_id=report.report_id,
        executable_plan=ExecutableMissionPlan(revision=1),
    )
    repo.finish_with_revalidation(report, result)
    assert repo.get_revalidation(report.report_id) == report
    assert repo.latest("S1") == (capture.epoch.model_copy(update={"status": "committed"}), result)
    repo.close()
