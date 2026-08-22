from __future__ import annotations

from pathlib import Path

from underwater_tracking.domain.agent_models import TrackingPlan
from underwater_tracking.domain.mission_models import ExecutableMissionPlan
from underwater_tracking.domain.planning_epoch_models import (
    EpochCommitResult,
    PlanningEpoch,
    PlanningEpochCapture,
)
from underwater_tracking.domain.models import SituationSnapshot
from underwater_tracking.persistence.plans import PlanRepository
from underwater_tracking.persistence.planning_epochs import PlanningEpochRepository
from underwater_tracking.persistence.uuv_plan_commits import UUVPlanCommitRepository
from underwater_tracking.runtime.mission_controller import MissionSnapshot
from underwater_tracking.planning.mission_revalidation import MissionRevalidationReport


def _epoch() -> PlanningEpoch:
    return PlanningEpoch(
        epoch_id="epoch:S1:1:a1",
        scenario_id="S1",
        base_physics_revision=1,
        base_sim_time_s=1,
        observation_batch_id="obs:S1:1",
        resource_manifest_hash="manifest",
        active_plan_version=0,
    )


def _capture(epoch: PlanningEpoch) -> PlanningEpochCapture:
    return PlanningEpochCapture(
        epoch=epoch,
        situation=SituationSnapshot(
            scenario_id="S1",
            snapshot_revision=1,
            sim_time_s=1,
            uuvs=(),
            group_reports=(),
            pending_events=(),
        ),
        mission=MissionSnapshot(scenario_id="S1", sim_time_s=1, plan_revision=0),
    )


def _audit() -> TrackingPlan:
    return TrackingPlan(plan_id="plan:S1:1", scenario_id="S1", revision=1, base_snapshot_revision=1)


def test_prepared_commit_is_invisible_until_finish(tmp_path: Path) -> None:
    path = tmp_path / "plans.db"
    plans = PlanRepository(path)
    epoch_repo = PlanningEpochRepository(path)
    epoch = _epoch()
    epoch_repo.create(_capture(epoch))
    plans.set_snapshot_revision("S1", 1)
    report = MissionRevalidationReport(
        report_id="validation:S1:1",
        epoch_id=epoch.epoch_id,
        current_physics_revision=1,
        current_plan_version=0,
        valid=True,
        rebased_plan=ExecutableMissionPlan(revision=1),
    )
    prepared = UUVPlanCommitRepository(plans).prepare(
        epoch=epoch,
        report=report,
        audit_projection=_audit(),
        executable_plan=ExecutableMissionPlan(revision=1),
        expected_active_plan_revision=0,
    )
    assert plans.get_active("S1") is None
    prepared.rollback()
    assert plans.get_active("S1") is None
    plans.close()
    epoch_repo.close()


def test_prepared_commit_finishes_with_typed_result(tmp_path: Path) -> None:
    path = tmp_path / "plans.db"
    plans = PlanRepository(path)
    epoch_repo = PlanningEpochRepository(path)
    epoch = _epoch()
    epoch_repo.create(_capture(epoch))
    plans.set_snapshot_revision("S1", 1)
    report = MissionRevalidationReport(
        report_id="validation:S1:1",
        epoch_id=epoch.epoch_id,
        current_physics_revision=1,
        current_plan_version=0,
        valid=True,
        rebased_plan=ExecutableMissionPlan(revision=1),
    )
    prepared = UUVPlanCommitRepository(plans).prepare(
        epoch=epoch,
        report=report,
        audit_projection=_audit(),
        executable_plan=ExecutableMissionPlan(revision=1),
        expected_active_plan_revision=0,
    )
    prepared.finish(
        EpochCommitResult(
            epoch_id=epoch.epoch_id,
            status="committed",
            plan_id="plan:S1:1",
            plan_version=1,
            validation_report_id=report.report_id,
            executable_plan=ExecutableMissionPlan(revision=1),
        )
    )
    assert plans.get_active("S1") is not None
    plans.close()
    epoch_repo.close()
