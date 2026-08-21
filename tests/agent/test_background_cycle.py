from types import SimpleNamespace
from threading import Condition, Event, RLock, Thread
import pytest

from underwater_tracking.cli import _AgentLoop, _BackgroundCarrierCycle
from underwater_tracking.domain.models import EventLevel, RuntimeEvent, SituationSnapshot


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
