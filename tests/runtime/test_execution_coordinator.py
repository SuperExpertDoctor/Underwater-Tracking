from __future__ import annotations

from pathlib import Path
from threading import Event

import pytest

from tests.domain.test_execution_models import _snapshot as _domain_snapshot
from underwater_tracking.persistence.plans import PlanRepository
from underwater_tracking.runtime.execution_coordinator import ExecutionCoordinator
from underwater_tracking.runtime.mission_controller import MissionController


def _snapshot(**updates: object):
    revision = int(updates.get("execution_revision", 1))
    base = _domain_snapshot()
    updates.setdefault(
        "regions",
        tuple(
            region.model_copy(update={"execution_revision": revision})
            for region in base.regions
        ),
    )
    updates.setdefault(
        "task_groups",
        tuple(
            group.model_copy(update={"execution_revision": revision})
            for group in base.task_groups
        ),
    )
    updates["execution_revision"] = revision
    return base.model_copy(deep=True, update=updates)


def _candidate(base: object, **updates: object):
    revision = int(updates.get("execution_revision", base.execution_revision))
    updates.setdefault(
        "regions",
        tuple(
            region.model_copy(update={"execution_revision": revision})
            for region in base.regions
        ),
    )
    updates.setdefault(
        "task_groups",
        tuple(
            group.model_copy(update={"execution_revision": revision})
            for group in base.task_groups
        ),
    )
    updates["execution_revision"] = revision
    return base.model_copy(deep=True, update=updates)


def test_startup_revision_one_is_immediately_readable_and_executable() -> None:
    initial = _snapshot(execution_revision=1, base_execution_revision=None)
    coordinator = ExecutionCoordinator(snapshot=initial)

    assert coordinator.current == initial
    assert coordinator.active_mission_plan() == initial
    assert coordinator.is_executable


def test_rolling_check_is_due_every_450_simulation_seconds() -> None:
    coordinator = ExecutionCoordinator(snapshot=_snapshot())

    assert coordinator.rolling_check_due(120)
    coordinator.mark_rolling_check(120)
    assert not coordinator.rolling_check_due(569)
    assert coordinator.rolling_check_due(570)


def test_prediction_leaving_the_active_chain_requests_immediate_replan() -> None:
    coordinator = ExecutionCoordinator(snapshot=_snapshot())

    assert coordinator.prediction_leaves_chain(
        ("target_00:task:02", "target_00:task:03")
    )
    assert not coordinator.prediction_leaves_chain(
        ("target_00:task:01", "target_00:task:02")
    )


def test_compare_and_set_rejects_a_stale_execution_base() -> None:
    coordinator = ExecutionCoordinator(snapshot=_snapshot(execution_revision=1))
    first = _candidate(
        coordinator.current,
        execution_revision=2,
        base_execution_revision=1,
    )
    assert coordinator.commit(first).status == "committed"

    stale = _candidate(
        first,
        execution_revision=3,
        base_execution_revision=1,
    )
    result = coordinator.commit(stale)

    assert result.status == "stale"
    assert not result.accepted
    assert result.preserved_execution_revision == 2
    assert coordinator.current.execution_revision == 2


def test_physics_only_drift_allows_a_controlled_rebase_when_evidence_is_valid() -> None:
    coordinator = ExecutionCoordinator(snapshot=_snapshot(execution_revision=1))
    candidate = _candidate(
        coordinator.current,
        execution_revision=2,
        base_execution_revision=0,
        source_snapshot_revision=13,
    )

    result = coordinator.commit(
        candidate,
        allow_rebase=True,
        evidence_valid=True,
    )

    assert result.status == "committed"
    assert result.was_rebased
    assert result.snapshot is not None
    assert result.snapshot.base_execution_revision == 1
    assert result.snapshot.execution_revision == 2


@pytest.mark.parametrize(
    ("field", "expected_reason"),
    [
        ("expert_request_version", "manual_revision_changed"),
    ],
)
def test_rebase_rejects_manual_revision_changes(
    field: str, expected_reason: str
) -> None:
    coordinator = ExecutionCoordinator(snapshot=_snapshot(execution_revision=1))
    candidate = _candidate(
        coordinator.current,
        execution_revision=2,
        base_execution_revision=0,
        **{field: 1},
    )

    result = coordinator.commit(candidate, allow_rebase=True, evidence_valid=True)

    assert result.status == "stale"
    assert expected_reason in result.reason


def test_rebase_rejects_target_and_resource_revision_changes() -> None:
    coordinator = ExecutionCoordinator(snapshot=_snapshot(execution_revision=1))
    current = coordinator.current
    assert current is not None
    changed_track = current.target_track.model_copy(update={"track_revision": 8})
    changed_prediction = current.prediction.model_copy(
        update={"source_track_revision": 8}
    )
    changed_target = _candidate(
        current,
        execution_revision=2,
        base_execution_revision=0,
        target_track=changed_track,
        prediction=changed_prediction,
    )
    target_result = coordinator.commit(
        changed_target,
        allow_rebase=True,
        evidence_valid=True,
    )
    assert target_result.status == "stale"
    assert "target_revision_changed" in target_result.reason

    changed_group = current.task_groups[0].model_copy(
        update={
            "member_uuv_ids": ("uuv_12", "uuv_01"),
            "active_verifier_uuv_id": "uuv_12",
        }
    )
    changed_resource = _candidate(
        current,
        execution_revision=2,
        base_execution_revision=0,
        task_groups=(changed_group, *current.task_groups[1:]),
    )
    resource_result = coordinator.commit(
        changed_resource,
        allow_rebase=True,
        evidence_valid=True,
    )
    assert resource_result.status == "stale"
    assert "resource_revision_changed" in resource_result.reason


def test_failed_apply_preserves_the_current_snapshot_and_records_degradation() -> None:
    coordinator = ExecutionCoordinator(snapshot=_snapshot(execution_revision=1))
    candidate = _candidate(
        coordinator.current,
        execution_revision=2,
        base_execution_revision=1,
    )

    result = coordinator.commit(candidate, apply=lambda _snapshot: False)

    assert result.status == "preserved"
    assert result.active_plan_preserved
    assert result.preserved_execution_revision == 1
    assert coordinator.current.execution_revision == 1


def test_mission_controller_accepts_the_execution_snapshot_as_uuv_only_plan() -> None:
    controller = MissionController(scenario_id="S1")
    snapshot = _snapshot(execution_revision=1)

    assert controller.apply_execution_snapshot(snapshot)
    mission = controller.snapshot()

    assert mission.plan_revision == 1
    assert len(mission.regions) == 4
    assert mission.carrier_missions == {}
    assert mission.uuv_modes["uuv_00"].value == "ACTIVE_SCAN"
    assert mission.uuv_modes["uuv_01"].value == "PASSIVE_TRACK"


def test_active_reader_loads_the_highest_validated_revision(tmp_path: Path) -> None:
    repository = PlanRepository(tmp_path / "agent.db")
    first = _snapshot(execution_revision=1)
    coordinator = ExecutionCoordinator(
        snapshot=first,
        plans=repository,
        scenario_id=first.scenario_id,
    )
    second = _candidate(
        first,
        execution_revision=2,
        base_execution_revision=1,
    )
    assert coordinator.commit(second).status == "committed"
    repository.close()

    reopened_repository = PlanRepository(tmp_path / "agent.db")
    reopened = ExecutionCoordinator(
        plans=reopened_repository,
        scenario_id="S1",
    )

    assert reopened.active_mission_plan().execution_revision == 2
    reopened_repository.close()


class _BlockingOptimizer:
    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()

    def __call__(self, baseline):
        self.started.set()
        assert self.release.wait(timeout=5)
        return _candidate(
            baseline,
            execution_revision=baseline.execution_revision + 1,
            base_execution_revision=baseline.execution_revision,
            plan_source="llm_optimized",
        )


def test_baseline_is_committed_and_published_before_optimizer_returns() -> None:
    optimizer = _BlockingOptimizer()
    coordinator = ExecutionCoordinator(scenario_id="S1")
    baseline = _snapshot(
        execution_revision=1,
        base_execution_revision=None,
        plan_source="deterministic",
    )
    published: list[int] = []

    try:
        result = coordinator.commit_baseline_then_optimize(
            baseline,
            optimizer=optimizer,
            publish=lambda snapshot: published.append(snapshot.execution_revision),
        )

        active = coordinator.active_mission_plan()
        assert result.status == "committed"
        assert active is not None
        assert active.plan_source == "deterministic"
        assert len(active.regions) == 4
        assert len(active.task_groups) == 4
        assert published == [1]
        assert optimizer.started.wait(timeout=1)
    finally:
        optimizer.release.set()


def test_stale_semantic_optimization_cannot_replace_newer_baseline() -> None:
    first = _snapshot(execution_revision=1, base_execution_revision=None)
    coordinator = ExecutionCoordinator(snapshot=first)
    second = _candidate(
        first,
        execution_revision=2,
        base_execution_revision=1,
        plan_source="deterministic",
    )
    assert coordinator.commit(second).committed
    stale_candidate = _candidate(
        first,
        execution_revision=2,
        base_execution_revision=1,
        plan_source="llm_optimized",
    )

    result = coordinator.commit_semantic_optimization(
        stale_candidate,
        base_execution_revision=1,
    )

    assert result.status == "rejected"
    assert result.reason == "stale_execution_base"
    assert coordinator.execution_revision == 2


def test_semantic_optimization_cannot_change_physical_execution_fields() -> None:
    baseline = _snapshot(execution_revision=1, base_execution_revision=None)
    coordinator = ExecutionCoordinator(snapshot=baseline)
    changed_region = baseline.regions[0].model_copy(
        update={"geometry": ((0.0, 0.0), (3.0, 0.0), (0.0, 3.0))}
    )
    candidate = _candidate(
        baseline,
        execution_revision=2,
        base_execution_revision=1,
        plan_source="llm_optimized",
        regions=(changed_region, *baseline.regions[1:]),
    )

    result = coordinator.commit_semantic_optimization(
        candidate,
        base_execution_revision=1,
    )

    assert result.status == "rejected"
    assert result.reason == "semantic_optimization_changed_physical_fields"
    assert coordinator.current == baseline


def test_deterministic_planning_failure_marks_execution_failed() -> None:
    baseline = _snapshot(execution_revision=1, base_execution_revision=None)
    coordinator = ExecutionCoordinator(snapshot=baseline)

    coordinator.mark_failed("baseline_build_failed")
    health = coordinator.execution_health(sim_time_s=100, hard_stale_s=900)

    assert health.status == "failed"
    assert health.reason_codes == ("baseline_build_failed",)
    assert health.executable is False
    assert coordinator.current == baseline
