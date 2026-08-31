from __future__ import annotations

from underwater_tracking.domain.models import SituationSnapshot
from underwater_tracking.domain.planning_epoch_models import EpochCommitResult
from underwater_tracking.planning.mission_revalidation import MissionRevalidationReport
from underwater_tracking.runtime.mission_controller import MissionSnapshot
from underwater_tracking.runtime.planning_epoch import (
    EpochTrigger,
    PlanningEpochCoordinator,
)
from underwater_tracking.persistence.planning_epochs import PlanningEpochRepository


def situation(revision: int, sim_time_s: int | None = None) -> SituationSnapshot:
    return SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=revision,
        sim_time_s=sim_time_s if sim_time_s is not None else revision,
        uuvs=(),
        group_reports=(),
        pending_events=(),
    )


def mission() -> MissionSnapshot:
    return MissionSnapshot(scenario_id="S1", sim_time_s=0, plan_revision=0)


def failed(epoch_id: str, category: str = "provider") -> EpochCommitResult:
    return EpochCommitResult(
        epoch_id=epoch_id,
        status="failed",
        failure_category=category,  # type: ignore[arg-type]
        failure_message="provider timeout",
    )


def test_observe_only_replaces_latest_situation(tmp_path) -> None:
    coordinator = PlanningEpochCoordinator(scenario_id="S1", database_path=tmp_path / "agent.db")
    coordinator.observe(situation(revision=1))
    coordinator.observe(situation(revision=200))
    assert coordinator.latest_situation().snapshot_revision == 200
    assert coordinator.next_epoch(mission()) is None
    coordinator.close()


def test_request_deduplicates_event_ids(tmp_path) -> None:
    coordinator = PlanningEpochCoordinator(scenario_id="S1", database_path=tmp_path / "agent.db")
    coordinator.observe(situation(revision=1))
    trigger = EpochTrigger("event-1", "initialization", 30, 100)
    coordinator.request((trigger, trigger))
    capture = coordinator.next_epoch(mission())
    assert capture is not None
    assert capture.epoch.critical_event_ids == ("event-1",)
    assert coordinator.next_epoch(mission()) is None
    coordinator.close()


def test_running_epoch_keeps_new_mailbox_event_for_next_epoch(tmp_path) -> None:
    coordinator = PlanningEpochCoordinator(scenario_id="S1", database_path=tmp_path / "agent.db")
    coordinator.observe(situation(revision=1))
    coordinator.request((EpochTrigger("event-1", "initialization", 1, 100),))
    first = coordinator.next_epoch(mission())
    assert first is not None
    coordinator.mark_running(first.epoch.epoch_id)
    coordinator.request((EpochTrigger("event-2", "target_added", 2, 90),))
    assert coordinator.next_epoch(mission()) is None
    coordinator.finish(
        EpochCommitResult(
            epoch_id=first.epoch.epoch_id,
            status="failed",
            failure_category="schema",
            failure_message="invalid candidate",
            consumed_event_ids=("event-1",),
        )
    )
    # A schema failure dead-letters event-1, while the newer event survives.
    assert coordinator.health().dead_letter_event_ids == ("event-1",)
    assert coordinator.health().dead_letter_reasons["event-1"] == "invalid candidate"
    second = coordinator.next_epoch(mission())
    assert second is not None
    assert second.epoch.critical_event_ids == ("event-2",)
    coordinator.close()


def test_provider_retry_uses_5_15_45_seconds_then_dead_letters(tmp_path) -> None:
    now = [0]
    coordinator = PlanningEpochCoordinator(
        scenario_id="S1", database_path=tmp_path / "agent.db", utc_now_ms=lambda: now[0]
    )
    coordinator.observe(situation(revision=1))
    trigger = EpochTrigger("event-1", "initialization", 1, 100)
    coordinator.request((trigger,))
    for expected_delay in (5_000, 15_000, 45_000):
        current = coordinator.next_epoch(mission())
        assert current is not None
        coordinator.mark_running(current.epoch.epoch_id)
        coordinator.finish(failed(current.epoch.epoch_id))
        assert coordinator.next_epoch(mission()) is None
        now[0] += expected_delay
    assert coordinator.next_epoch(mission()) is None
    assert coordinator.health().dead_letter_event_ids == ("event-1",)
    coordinator.close()


def test_revalidation_invalidation_requeues_dynamic_state_trigger(tmp_path) -> None:
    now = [0]
    coordinator = PlanningEpochCoordinator(
        scenario_id="S1", database_path=tmp_path / "agent.db", utc_now_ms=lambda: now[0]
    )
    coordinator.observe(situation(revision=1))
    trigger = EpochTrigger("event-replan", "target_estimate_updated", 1, 100)
    coordinator.request((trigger,))
    capture = coordinator.next_epoch(mission())
    assert capture is not None
    coordinator.mark_running(capture.epoch.epoch_id)

    invalidated = EpochCommitResult(
        epoch_id=capture.epoch.epoch_id,
        status="invalidated",
        validation_report_id="validation:dynamic-state",
        invalidated_reason="deployment_changed; prior_changed",
    )
    report = MissionRevalidationReport(
        report_id="validation:dynamic-state",
        epoch_id=capture.epoch.epoch_id,
        current_physics_revision=2,
        current_plan_version=0,
        valid=False,
    )
    repository = PlanningEpochRepository(tmp_path / "agent.db")
    repository.finish_with_revalidation(report, invalidated)
    repository.close()
    coordinator.finish(invalidated)

    assert coordinator.next_epoch(mission()) is None
    now[0] = 5_000
    retried = coordinator.next_epoch(mission())
    assert retried is not None
    assert retried.epoch.critical_event_ids == ("event-replan",)
    coordinator.close()


def test_internal_failure_dead_letters_without_automatic_retry_and_supports_expert_retry(
    tmp_path,
) -> None:
    coordinator = PlanningEpochCoordinator(scenario_id="S1", database_path=tmp_path / "agent.db")
    coordinator.observe(situation(revision=1))
    trigger = EpochTrigger("event-internal", "strategic_review", 1, 100)
    coordinator.request((trigger,))
    capture = coordinator.next_epoch(mission())
    assert capture is not None
    coordinator.mark_running(capture.epoch.epoch_id)
    coordinator.finish(failed(capture.epoch.epoch_id, category="internal"))

    assert coordinator.next_epoch(mission()) is None
    health = coordinator.health()
    assert health.dead_letter_event_ids == ("event-internal",)
    assert health.dead_letter_reasons["event-internal"] == "provider timeout"

    coordinator.retry_dead_letter_event("event-internal")
    retried = coordinator.next_epoch(mission())
    assert retried is not None
    assert retried.epoch.critical_event_ids == ("event-internal",)
    coordinator.close()


def test_finish_reconciles_terminal_result_written_by_commit_port(tmp_path) -> None:
    coordinator = PlanningEpochCoordinator(scenario_id="S1", database_path=tmp_path / "agent.db")
    coordinator.observe(situation(revision=1))
    coordinator.request((EpochTrigger("event-1", "initialization", 1, 100),))
    capture = coordinator.next_epoch(mission())
    assert capture is not None
    coordinator.mark_running(capture.epoch.epoch_id)

    # The graph commit port can persist the terminal result before the outer
    # background loop performs its bookkeeping pass.
    persisted = failed(capture.epoch.epoch_id)
    from underwater_tracking.persistence.planning_epochs import PlanningEpochRepository

    repository = PlanningEpochRepository(tmp_path / "agent.db")
    repository.finish(persisted)
    repository.close()

    coordinator.finish(persisted.model_copy(update={"failure_message": "outer-loop error"}))
    assert coordinator.health().last_result_status == "failed"
    assert coordinator.health().last_error == "provider timeout"

    # A second outer-loop callback after coordinator bookkeeping is harmless.
    coordinator.finish(persisted)
    assert coordinator.health().epoch_id is None
    coordinator.close()
