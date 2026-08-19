"""Runtime-level latching for repeated regional replan signals."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from underwater_tracking.agent.graphs.central import CarrierDependencies
from underwater_tracking.agent.runtime import CarrierRuntime
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.plans import PlanRepository
from underwater_tracking.simulation.clock import SimulationClock


class _SituationHolder:
    def __init__(self, situation: SimpleNamespace) -> None:
        self.situation = situation

    def __call__(self, _: str) -> SimpleNamespace:
        return self.situation


class _Graph:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.fail = False

    def get_state(self, _: dict[str, object]) -> SimpleNamespace:
        return SimpleNamespace(
            values={"known_target_ids": ("T1",), "lost_target_ids": ()}
        )

    def invoke(self, state: dict[str, object], *, config: dict[str, object]) -> dict[str, object]:
        del config
        self.calls.append(state)
        if self.fail:
            raise RuntimeError("regional graph failed")
        return {"route": "strategic"}


def _situation(*, sim_time_s: int, covariance_trace: float) -> SimpleNamespace:
    half_trace = covariance_trace / 2
    return SimpleNamespace(
        scenario_id="S1",
        sim_time_s=sim_time_s,
        group_reports=(
            SimpleNamespace(
                target_id="T1",
                belief=SimpleNamespace(
                    covariance=((half_trace, 0.0), (0.0, half_trace)),
                    source_observation_ids=(),
                ),
            ),
        ),
        uuvs=(),
        platform_snapshot=None,
    )


@pytest.fixture
def runtime_bundle(tmp_path):
    database_path = tmp_path / "runtime.db"
    plans = PlanRepository(database_path)
    events = EventRepository(database_path)
    ledger = DecisionLedger(database_path)
    holder = _SituationHolder(_situation(sim_time_s=10, covariance_trace=200.0))
    runtime = CarrierRuntime(
        CarrierDependencies(
            plans=plans,
            events=events,
            ledger=ledger,
            llm=object(),
            predictor=lambda situation, target_id: None,
            situation_provider=holder,
            clock=SimulationClock(step_s=30),
            covariance_cap_m2=100.0,
        ),
        scenario_id="S1",
        database_path=database_path,
    )
    graph = _Graph()
    runtime._graph = graph
    try:
        yield runtime, holder, graph
    finally:
        runtime.close()
        plans.close()
        events.close()
        ledger.close()


def test_continuous_regional_degradation_latches_until_recovery(runtime_bundle) -> None:
    runtime, holder, graph = runtime_bundle

    runtime._run_cycle()
    first_events = graph.calls[-1]["pending_events"]
    assert len(first_events) == 1
    assert first_events[0].event_type == "covariance_threshold_exceeded"
    assert first_events[0].entity_id == "T1"

    holder.situation = _situation(sim_time_s=20, covariance_trace=200.0)
    runtime._run_cycle()
    assert graph.calls[-1]["pending_events"] == ()

    holder.situation = _situation(sim_time_s=30, covariance_trace=50.0)
    runtime._run_cycle()
    assert graph.calls[-1]["pending_events"] == ()

    holder.situation = _situation(sim_time_s=40, covariance_trace=200.0)
    runtime._run_cycle()
    retriggered = graph.calls[-1]["pending_events"]
    assert len(retriggered) == 1
    assert retriggered[0].event_type == "covariance_threshold_exceeded"
    assert retriggered[0].sim_time_s == 40


def test_regional_replan_latch_rolls_back_when_graph_raises(runtime_bundle) -> None:
    runtime, _, graph = runtime_bundle
    graph.fail = True

    with pytest.raises(RuntimeError, match="regional graph failed"):
        runtime._run_cycle()

    assert runtime._regional_replan_latches == set()
    assert len(runtime._pending) == 1

    graph.fail = False
    runtime._run_cycle()
    assert len(graph.calls[-1]["pending_events"]) == 1
    assert runtime._regional_replan_latches == {
        ("covariance_threshold_exceeded", "T1")
    }
