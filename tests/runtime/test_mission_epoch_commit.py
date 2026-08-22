from __future__ import annotations

from pathlib import Path

from underwater_tracking.domain.agent_models import TrackingPlan
from underwater_tracking.domain.mission_models import ExecutableMissionPlan
from underwater_tracking.domain.models import SituationSnapshot
from underwater_tracking.domain.planning_epoch_models import PlanningEpoch, PlanningEpochCapture
from underwater_tracking.persistence.planning_epochs import PlanningEpochRepository
from underwater_tracking.persistence.plans import PlanRepository
from underwater_tracking.runtime.mission_controller import MissionController
from underwater_tracking.runtime.mission_epoch_commit import MissionEpochCommitPort
from underwater_tracking.runtime.scenario_transition import ScenarioTransitionCoordinator


def _epoch() -> PlanningEpoch:
    return PlanningEpoch(
        epoch_id="epoch:S1:1:a1",
        scenario_id="S1",
        base_physics_revision=1,
        base_sim_time_s=1,
        observation_batch_id="observation:S1:1",
        resource_manifest_hash="manifest",
        active_plan_version=0,
    )


def _capture(epoch: PlanningEpoch) -> PlanningEpochCapture:
    situation = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=1,
        sim_time_s=1,
        uuvs=(),
        group_reports=(),
        pending_events=(),
    )
    from underwater_tracking.runtime.mission_controller import MissionSnapshot

    return PlanningEpochCapture(
        epoch=epoch,
        situation=situation,
        mission=MissionSnapshot(scenario_id="S1", sim_time_s=1, plan_revision=0),
    )


def _audit() -> TrackingPlan:
    return TrackingPlan(
        plan_id="plan:S1:1",
        scenario_id="S1",
        revision=1,
        base_snapshot_revision=1,
    )


def _port(path: Path, controller: MissionController) -> tuple[MissionEpochCommitPort, PlanningEpochRepository, PlanRepository]:
    plans = PlanRepository(path)
    epochs = PlanningEpochRepository(plans.connection)
    epoch = _epoch()
    epochs.create(_capture(epoch))
    port = MissionEpochCommitPort(
        plans=plans,
        epochs=epochs,
        mission_controller=controller,
        transition_coordinator=ScenarioTransitionCoordinator("S1"),
        situation_provider=lambda: _capture(epoch).situation,
    )
    return port, epochs, plans


def test_mission_epoch_commit_applies_controller_and_persists_one_result(tmp_path: Path) -> None:
    controller = MissionController(scenario_id="S1")
    port, epochs, plans = _port(tmp_path / "commit.db", controller)

    result = port.commit(
        epoch=_epoch(),
        audit_projection=_audit(),
        executable_plan=ExecutableMissionPlan(revision=1),
    )

    assert result.status == "committed"
    assert controller.snapshot().plan_revision == 1
    assert plans.get_active("S1") is not None
    latest = epochs.latest("S1")
    assert latest is not None and latest[1] is not None
    assert latest[1].status == "committed"
    assert plans.connection.execute(
        "SELECT COUNT(*) AS count FROM plan_commands"
    ).fetchone()["count"] == 0
    plans.close()
    epochs.close()


def test_mission_epoch_commit_restores_controller_when_apply_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    controller = MissionController(scenario_id="S1")
    port, epochs, plans = _port(tmp_path / "rollback.db", controller)
    original = controller.apply_revalidated_plan

    def fail_after_apply(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected apply failure")

    monkeypatch.setattr(controller, "apply_revalidated_plan", fail_after_apply)

    result = port.commit(
        epoch=_epoch(),
        audit_projection=_audit(),
        executable_plan=ExecutableMissionPlan(revision=1),
    )

    assert result.status == "failed"
    assert controller.snapshot().plan_revision == 0
    assert plans.get_active("S1") is None
    latest = epochs.latest("S1")
    assert latest is not None and latest[1] is not None
    assert latest[1].status == "failed"
    plans.close()
    epochs.close()


class _SQLFailingPrepared:
    def finish(self, result) -> None:
        del result
        raise RuntimeError("injected SQL commit failure")

    def rollback(self) -> None:
        return None


class _SQLFailingRepository:
    def prepare(self, **kwargs):
        del kwargs
        return _SQLFailingPrepared()


def test_mission_epoch_commit_restores_controller_when_sql_finish_fails(
    tmp_path: Path,
) -> None:
    controller = MissionController(scenario_id="S1")
    port, epochs, plans = _port(tmp_path / "sql-rollback.db", controller)
    port._commit_repository = _SQLFailingRepository()  # type: ignore[assignment]

    result = port.commit(
        epoch=_epoch(),
        audit_projection=_audit(),
        executable_plan=ExecutableMissionPlan(revision=1),
    )

    assert result.status == "failed"
    assert controller.snapshot().plan_revision == 0
    assert plans.get_active("S1") is None
    latest = epochs.latest("S1")
    assert latest is not None and latest[1] is not None
    assert latest[1].status == "failed"
    plans.close()
    epochs.close()
