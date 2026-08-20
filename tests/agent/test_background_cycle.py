from types import SimpleNamespace
from threading import RLock

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
