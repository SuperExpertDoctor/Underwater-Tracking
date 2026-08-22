from collections import deque
from types import SimpleNamespace
from threading import Condition, Event, RLock, Thread
import pytest

from underwater_tracking.cli import _AgentLoop, _BackgroundCarrierCycle
from underwater_tracking.domain.mission_models import ExecutableMissionPlan
from underwater_tracking.domain.models import EventLevel, RuntimeEvent, SituationSnapshot
from underwater_tracking.domain.planning_epoch_models import EpochCommitResult, PlanningEpoch
from underwater_tracking.runtime.mission_controller import MissionSnapshot


def _situation(revision: int, event_id: str | None = None) -> SituationSnapshot:
    events = ()
    if event_id is not None:
        events = (
            RuntimeEvent(
                event_id=event_id,
                scenario_id="S1",
                sim_time_s=revision * 30,
                event_type="test_event",
                entity_id="S1",
                level=EventLevel.INFORMATIONAL,
            ),
        )
    return SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=revision,
        sim_time_s=revision * 30,
        uuvs=(),
        group_reports=(),
        pending_events=events,
    )


def test_background_cycle_keeps_latest_mailbox_while_cycle_runs() -> None:
    loop = _AgentLoop.__new__(_AgentLoop)
    loop._carrier_cycle_lock = RLock()
    loop._background_cycle = object()
    loop._background_mailbox = None
    loop.situation = _situation(1)

    loop._start_background_cycle(_situation(2))

    assert loop.situation.snapshot_revision == 2
    assert loop._background_mailbox.snapshot_revision == 2


class _SummaryWriter:
    def __init__(self, *, accepting: bool = True) -> None:
        self.accepting = accepting
        self.events: list[RuntimeEvent] = []

    def submit(self, event: RuntimeEvent) -> bool:
        if not self.accepting:
            return False
        self.events.append(event)
        return True


def _summary_loop(*, writer: _SummaryWriter) -> _AgentLoop:
    loop = _AgentLoop.__new__(_AgentLoop)
    loop._config = SimpleNamespace(timing=SimpleNamespace(progress_report_s=600))
    loop._engine = SimpleNamespace(
        mission_snapshot=lambda: MissionSnapshot(
            scenario_id="S1",
            sim_time_s=0,
            plan_revision=0,
        )
    )
    loop._periodic_summary_writer = writer
    loop._periodic_summary_source_ids = set()
    loop._last_built_periodic_summary = None
    loop._pending_periodic_summaries = deque()
    loop._periodic_summary_next_boundary_s = 600
    loop._periodic_summary_backlog_overflow = 0
    loop._periodic_summary_degradation_events = []
    loop._background_carrier = True
    loop._start_background_cycle = lambda _situation: None
    return loop


def test_periodic_summary_is_scheduled_once_per_boundary_and_collects_sources() -> None:
    writer = _SummaryWriter()
    loop = _summary_loop(writer=writer)

    for sim_time_s in range(30, 601, 30):
        loop.on_situation(_situation(sim_time_s // 30, f"event-{sim_time_s}"))
    loop.on_situation(_situation(20, "duplicate-boundary-event"))
    loop.on_situation(_situation(40, "event-1200"))

    assert [event.event_id for event in writer.events] == [
        "periodic_situation_summary:S1:600",
        "periodic_situation_summary:S1:1200",
    ]
    assert writer.events[0].payload["source_event_ids"] == [
        *sorted(f"event-{sim_time_s}" for sim_time_s in range(30, 601, 30)),
    ]


def test_periodic_summary_backlog_keeps_oldest_boundaries_until_writer_recovers() -> None:
    writer = _SummaryWriter(accepting=False)
    loop = _summary_loop(writer=writer)

    loop.on_situation(_situation(20, "event-600"))
    loop.on_situation(_situation(40, "event-1200"))
    assert [event.event_id for _, event in loop._pending_periodic_summaries] == [
        "periodic_situation_summary:S1:600",
        "periodic_situation_summary:S1:1200",
    ]

    writer.accepting = True
    loop.on_situation(_situation(60, "event-1800"))

    assert [event.event_id for event in writer.events] == [
        "periodic_situation_summary:S1:600",
        "periodic_situation_summary:S1:1200",
        "periodic_situation_summary:S1:1800",
    ]
    assert loop._pending_periodic_summaries == deque()


def test_agent_loop_close_keeps_resources_open_when_memory_worker_is_still_running() -> None:
    calls: list[str] = []

    class BlockingWorker:
        def __init__(self) -> None:
            self.stops = [False, True]

        def stop(self, *, timeout: float) -> bool:
            del timeout
            return self.stops.pop(0)

    class Closable:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            calls.append(self.name)

    loop = _AgentLoop.__new__(_AgentLoop)
    loop._closed = False
    loop._closing = False
    loop._carrier_cycle_lock = RLock()
    loop._background_mailbox = None
    loop._background_thread = None
    loop._memory_worker = BlockingWorker()
    loop._memory_short_term = Closable("short-term")
    loop._memory_long_term = Closable("long-term")
    loop._knowledge_client = Closable("knowledge")
    loop._memory_embedding_provider = Closable("embedding")
    loop._clients = {"master": Closable("llm")}
    loop._publisher = None
    loop._runtime = None
    loop.plans = Closable("plans")
    loop.events = Closable("events")
    loop.ledger = Closable("ledger")

    assert loop.close() is False
    assert calls == []
    assert loop._closed is False

    assert loop.close() is True
    assert calls[:4] == ["embedding", "llm", "short-term", "long-term"]
    assert loop._closed is True


def test_agent_loop_concurrent_close_closes_shared_resources_once() -> None:
    calls: list[str] = []
    stop_started = Event()
    release_stop = Event()

    class Worker:
        stop_calls = 0

        def stop(self, *, timeout: float) -> bool:
            del timeout
            self.stop_calls += 1
            stop_started.set()
            release_stop.wait(1.0)
            return True

    class Closable:
        def close(self) -> None:
            calls.append("shared")

    worker = Worker()
    shared = Closable()
    loop = _AgentLoop.__new__(_AgentLoop)
    loop._closed = False
    loop._closing = False
    loop._carrier_cycle_lock = RLock()
    loop._close_condition = Condition(RLock())
    loop._background_mailbox = None
    loop._background_thread = None
    loop._memory_worker = worker
    loop._memory_short_term = shared
    loop._memory_long_term = None
    loop._knowledge_client = None
    loop._memory_embedding_provider = None
    loop._clients = {"master": shared}
    loop._publisher = None
    loop._runtime = None
    loop.plans = None
    loop.events = None
    loop.ledger = None
    results: list[bool] = []
    threads = [Thread(target=lambda: results.append(loop.close())) for _ in range(2)]

    threads[0].start()
    assert stop_started.wait(1.0)
    threads[1].start()
    release_stop.set()
    for thread in threads:
        thread.join(1.0)

    assert results == [True, True]
    assert worker.stop_calls == 1
    assert calls == ["shared"]


def test_agent_loop_close_retries_resources_after_a_cleanup_exception() -> None:
    calls: list[str] = []

    class Flaky:
        def __init__(self) -> None:
            self.attempts = 0

        def close(self) -> None:
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("close failed")
            calls.append("llm")

    class Closable:
        def __init__(self, name: str) -> None:
            self.name = name

        def close(self) -> None:
            calls.append(self.name)

    loop = _AgentLoop.__new__(_AgentLoop)
    loop._closed = False
    loop._closing = False
    loop._carrier_cycle_lock = RLock()
    loop._background_mailbox = None
    loop._background_thread = None
    loop._memory_worker = None
    loop._memory_short_term = Closable("short-term")
    loop._memory_long_term = Closable("long-term")
    loop._knowledge_client = None
    loop._memory_embedding_provider = None
    loop._clients = {"master": Flaky()}
    loop._publisher = None
    loop._runtime = None
    loop.plans = Closable("plans")
    loop.events = Closable("events")
    loop.ledger = Closable("ledger")

    with pytest.raises(RuntimeError, match="close failed"):
        loop.close()
    assert loop._closed is False

    assert loop.close() is True
    assert calls.count("llm") == 1


def test_background_mailbox_merges_pending_events_from_skipped_situations() -> None:
    loop = _AgentLoop.__new__(_AgentLoop)
    loop._carrier_cycle_lock = RLock()
    loop._background_cycle = object()
    loop._background_mailbox = _situation(2, "event-2")
    loop.situation = _situation(2, "event-2")

    loop._start_background_cycle(_situation(3, "event-3"))

    assert loop._background_mailbox.snapshot_revision == 3
    assert [event.event_id for event in loop._background_mailbox.pending_events] == [
        "event-2",
        "event-3",
    ]


def test_background_mailbox_does_not_requeue_active_revision() -> None:
    loop = _AgentLoop.__new__(_AgentLoop)
    loop._carrier_cycle_lock = RLock()
    loop._background_cycle = _BackgroundCarrierCycle(
        situation=_situation(2, "event-2"),
        adversary_contexts=(),
        slave_contexts=(),
    )
    loop._background_mailbox = None
    loop.situation = _situation(2, "event-2")

    loop._start_background_cycle(_situation(1, "event-1"))

    assert loop._background_mailbox is None


def test_stale_background_result_is_discarded_and_latest_cycle_started() -> None:
    loop = _AgentLoop.__new__(_AgentLoop)
    loop._background_carrier = True
    loop._carrier_cycle_lock = RLock()
    loop._background_thread = object()
    loop._background_mailbox = None
    loop.situation = _situation(2)
    loop._background_cycle = _BackgroundCarrierCycle(
        situation=_situation(1),
        adversary_contexts=(),
        slave_contexts=(),
        result={"commit_status": "committed"},
        done=True,
    )
    loop._runtime = SimpleNamespace(
        active_plan=lambda: None,
        reservations=lambda: {},
    )
    loop._engine = SimpleNamespace(set_reservations=lambda _value: None)
    loop._config = SimpleNamespace(
        scenario=SimpleNamespace(uuv_only=False),
        environment=None,
    )
    loop._apply_new_commands = lambda: (_ for _ in ()).throw(
        AssertionError("stale cycle was applied")
    )
    loop._apply_verification_commands = lambda _result: (_ for _ in ()).throw(
        AssertionError("stale verification was applied")
    )
    loop.mark_llm_recovered = lambda: (_ for _ in ()).throw(
        AssertionError("stale cycle was acknowledged")
    )
    started: list[SituationSnapshot] = []
    loop._start_background_cycle = lambda situation: started.append(situation)

    loop.apply_background_cycle()

    assert [item.snapshot_revision for item in started] == [2]


def test_completed_epoch_is_applied_after_physics_revision_drift() -> None:
    epoch = PlanningEpoch(
        epoch_id="epoch:S1:1:a1",
        scenario_id="S1",
        base_physics_revision=1,
        base_sim_time_s=30,
        observation_batch_id="observation:S1:1",
        resource_manifest_hash="manifest",
        active_plan_version=0,
    )
    plan = ExecutableMissionPlan(revision=1)
    commit_result = EpochCommitResult(
        epoch_id=epoch.epoch_id,
        status="committed",
        plan_id="plan:S1:1",
        plan_version=1,
        validation_report_id="validation:S1:1",
        executable_plan=plan,
    )
    applied: list[ExecutableMissionPlan] = []
    finished: list[EpochCommitResult] = []
    started: list[SituationSnapshot] = []
    loop = _AgentLoop.__new__(_AgentLoop)
    loop._background_carrier = True
    loop._carrier_cycle_lock = RLock()
    loop._background_thread = object()
    loop._background_mailbox = None
    loop.situation = _situation(200)
    loop._background_cycle = _BackgroundCarrierCycle(
        situation=_situation(1),
        adversary_contexts=(),
        slave_contexts=(),
        epoch=epoch,
        result={"commit_status": "committed", "epoch_commit_result": commit_result},
        done=True,
    )
    loop._runtime = SimpleNamespace(
        active_plan=lambda: None,
        reservations=lambda: (),
    )
    loop._engine = SimpleNamespace(
        apply_verified_mission_plan=lambda candidate: applied.append(candidate) or True,
        set_reservations=lambda _value: None,
    )
    loop._config = SimpleNamespace(
        scenario=SimpleNamespace(uuv_only=True),
        environment=None,
    )
    loop._last_mission_revision = 0
    loop._epoch_coordinator = SimpleNamespace(finish=finished.append)
    loop._apply_new_commands = lambda: None
    loop._apply_verification_commands = lambda _result: None
    loop.mark_llm_recovered = lambda: None
    loop._start_background_cycle = started.append

    loop.apply_background_cycle()

    assert applied == [plan]
    assert finished == [commit_result]
    assert [item.snapshot_revision for item in started] == [200]
