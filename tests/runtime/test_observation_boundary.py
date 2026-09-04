from __future__ import annotations

import pytest

from tests.runtime.test_mission_controller import _runtime_execution_snapshot
from underwater_tracking.domain.models import SituationSnapshot
from underwater_tracking.runtime.mission_controller import MissionController
from underwater_tracking.runtime.observation_boundary import (
    ObservationBoundaryCommitter,
    PhysicalObservationBatch,
)
from underwater_tracking.runtime.scenario_transition import ScenarioTransitionCoordinator


def _situation(controller: MissionController) -> SituationSnapshot:
    return SituationSnapshot(
        scenario_id=controller.scenario_id,
        snapshot_revision=controller.snapshot().sim_time_s,
        sim_time_s=controller.snapshot().sim_time_s,
        uuvs=(),
        group_reports=(),
        pending_events=(),
    )


def test_observation_boundary_commits_physics_and_mission_as_one_bundle() -> None:
    controller = MissionController(scenario_id="S1")
    transitions = ScenarioTransitionCoordinator("S1")

    committer = ObservationBoundaryCommitter(
        transitions,
        mission_controller=controller,
        apply_delta=lambda delta: controller.advance(
            delta.sim_time_s, delta.as_observations()
        ),
        situation_provider=lambda: _situation(controller),
        mission_snapshot_provider=controller.snapshot,
    )

    bundle = committer.commit(
        PhysicalObservationBatch(
            physics_revision=1,
            sim_time_s=30,
            observations={"external_events": ()},
        )
    )

    assert bundle.physics_revision == 1
    assert bundle.mission_revision == 0
    assert bundle.situation.sim_time_s == 30
    assert bundle.mission is not None and bundle.mission.sim_time_s == 30
    assert transitions.active_kind is None


def test_observation_boundary_restores_controller_and_publishes_no_partial_bundle() -> None:
    controller = MissionController(scenario_id="S1")
    transitions = ScenarioTransitionCoordinator("S1")
    fail = True

    def apply(delta: PhysicalObservationBatch) -> None:
        controller.advance(delta.sim_time_s, delta.as_observations())

    def reconcile() -> None:
        if fail:
            raise RuntimeError("reconciliation failed")

    committer = ObservationBoundaryCommitter(
        transitions,
        mission_controller=controller,
        apply_delta=apply,
        reconcile=reconcile,
        situation_provider=lambda: _situation(controller),
        mission_snapshot_provider=controller.snapshot,
    )

    with pytest.raises(RuntimeError, match="reconciliation failed"):
        committer.commit(PhysicalObservationBatch(physics_revision=1, sim_time_s=30))

    assert controller.snapshot().sim_time_s == 0
    assert transitions.active_kind is None

    fail = False
    bundle = committer.commit(PhysicalObservationBatch(physics_revision=1, sim_time_s=30))
    assert bundle.situation.sim_time_s == 30


def test_observation_batch_rejects_duplicate_deployment_ids() -> None:
    with pytest.raises(ValueError, match="deployed_uuv_ids must be unique"):
        PhysicalObservationBatch(
            physics_revision=1,
            sim_time_s=30,
            deployed_uuv_ids=("uuv_01", "uuv_01"),
        )


def test_observation_boundary_rolls_back_runtime_group_state() -> None:
    controller = MissionController(scenario_id="S1")
    execution = _runtime_execution_snapshot()
    controller.advance(int(execution.valid_from_s), {})
    assert controller.apply_execution_snapshot(execution) is True
    before = controller.snapshot()
    transitions = ScenarioTransitionCoordinator("S1")

    def apply(delta: PhysicalObservationBatch) -> None:
        controller.observe(
            {
                "region_entry_probabilities": {
                    "target_00:task:01": 0.9,
                }
            }
        )

    def fail_reconcile() -> None:
        raise RuntimeError("reconcile failed")

    committer = ObservationBoundaryCommitter(
        transitions,
        mission_controller=controller,
        apply_delta=apply,
        reconcile=fail_reconcile,
        situation_provider=lambda: _situation(controller),
        mission_snapshot_provider=controller.snapshot,
    )

    with pytest.raises(RuntimeError, match="reconcile failed"):
        committer.commit(PhysicalObservationBatch(physics_revision=1, sim_time_s=121))

    assert controller.snapshot() == before
