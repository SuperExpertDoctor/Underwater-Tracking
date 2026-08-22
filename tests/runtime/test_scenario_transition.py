from __future__ import annotations

from threading import Event, Thread

from underwater_tracking.runtime.scenario_transition import ScenarioTransitionCoordinator


def test_plan_and_observation_transitions_share_one_serial_boundary() -> None:
    coordinator = ScenarioTransitionCoordinator("S1")
    first_entered = Event()
    release_first = Event()
    second_entered = Event()
    order: list[str] = []

    def worker(kind: str) -> None:
        with coordinator.transition(kind):
            order.append(f"{kind}:start")
            if kind == "plan":
                first_entered.set()
                release_first.wait(timeout=2)
            else:
                second_entered.set()
            order.append(f"{kind}:end")

    first = Thread(target=worker, args=("plan",))
    second = Thread(target=worker, args=("observation",))
    first.start()
    second.start()
    assert first_entered.wait(timeout=2)
    assert not second_entered.is_set()
    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)
    assert len(order) == 4
    assert order[0].endswith(":start")
    assert order[1].endswith(":end")
    assert order[2].endswith(":start")
    assert order[3].endswith(":end")
