from __future__ import annotations

from types import SimpleNamespace

from underwater_tracking.cli import _AgentLoop
from underwater_tracking.domain.models import EventLevel, RuntimeEvent, SituationSnapshot
from underwater_tracking.domain.mission_models import ExecutableMissionPlan
from tests.runtime.test_execution_coordinator import _candidate, _snapshot
from underwater_tracking.runtime.execution_coordinator import ExecutionCoordinator


class _RecordingEngine:
    def __init__(self) -> None:
        self.applied_revisions: list[int] = []
        self.reservations: list[object] = []

    def apply_verified_mission_plan(self, plan: ExecutableMissionPlan) -> bool:
        self.applied_revisions.append(plan.revision)
        return True

    def set_reservations(self, reservations: object) -> None:
        self.reservations.append(reservations)


class _EventDrivenRuntime:
    def __init__(self, initial: ExecutableMissionPlan, replanned: ExecutableMissionPlan) -> None:
        self._active = initial
        self._replanned = replanned
        self.submitted: list[RuntimeEvent] = []

    def active_mission_plan(self) -> ExecutableMissionPlan:
        return self._active

    def active_plan(self) -> None:
        return None

    def reservations(self) -> dict[str, tuple[str, ...]]:
        return {}

    def drain_sensor_controls(self) -> tuple[object, ...]:
        return ()

    def submit_events(self, events: tuple[RuntimeEvent, ...]) -> None:
        self.submitted.extend(events)

    def tick(self) -> dict[str, str]:
        if any(event.event_type == "uuv_range_exhausted" for event in self.submitted):
            self._active = self._replanned
        return {"commit_status": "committed"}


def test_uuv_resource_event_drives_higher_executable_revision_to_engine() -> None:
    initial = ExecutableMissionPlan(revision=1)
    replanned = ExecutableMissionPlan(revision=2)
    engine = _RecordingEngine()
    runtime = _EventDrivenRuntime(initial, replanned)
    loop = _AgentLoop.__new__(_AgentLoop)
    loop._config = SimpleNamespace(scenario=SimpleNamespace(uuv_only=True))
    loop._engine = engine
    loop._runtime = runtime
    loop._clients = {}
    loop.situation = None
    loop.paused = False
    loop.reconnectable = True
    loop._next_llm_retry_at = 0.0
    loop._last_mission_revision = 0
    loop._initialization_submitted = True
    loop._feedback_events = lambda situation: ()
    loop._local_brain_decisions = lambda situation: ((), ())
    loop.carrier_error_count = 0
    loop._llm_failure_count = 0
    loop.llm_pause_reason = None

    event = RuntimeEvent(
        event_id="S1:uuv_range_exhausted:U1:30",
        scenario_id="S1",
        sim_time_s=30,
        event_type="uuv_range_exhausted",
        entity_id="U1",
        level=EventLevel.STRATEGIC,
        payload={"remaining_range_m": 0.0},
    )
    situation = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=1,
        sim_time_s=30,
        uuvs=(),
        group_reports=(),
        pending_events=(event,),
    )

    loop._run_synchronous_carrier_cycle(situation)

    assert runtime.submitted == [event]
    assert engine.applied_revisions == [1, 2]
    assert loop._last_mission_revision == 2


def test_execution_coordinator_keeps_runtime_projection_at_observation_boundary() -> None:
    baseline = _snapshot(execution_revision=1)
    coordinator = ExecutionCoordinator(snapshot=baseline)
    current = coordinator.current
    assert current is not None
    runtime = _candidate(current, execution_revision=1)
    runtime = runtime.model_copy(
        deep=True,
        update={
            "task_groups": tuple(
                group.model_copy(update={"status": "active"})
                for group in runtime.task_groups
            )
        },
    )

    assert coordinator.update_runtime_projection(
        runtime,
        expected_execution_revision=1,
    ) is True
    assert coordinator.current == runtime
