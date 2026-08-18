"""Regional strategy routing tests for the carrier graph."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from underwater_tracking.agent.graphs.central import (
    EventMonitorNode,
    RegionalGenerationWiringNode,
    RegionalStrategyWiringNode,
)
from underwater_tracking.agent.llm import LLMError
from underwater_tracking.agent.nodes.event_monitor import EventMonitor
from underwater_tracking.agent.runtime import CarrierRuntime
from underwater_tracking.agent.graphs.central import CarrierDependencies
from underwater_tracking.domain.models import EventLevel, RuntimeEvent
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.persistence.ledger import DecisionLedger
from underwater_tracking.persistence.plans import PlanRepository
from underwater_tracking.simulation.clock import SimulationClock


def _event(event_type: str) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=f"S1:{event_type}",
        scenario_id="S1",
        sim_time_s=900,
        event_type=event_type,
        entity_id="R01",
        level=EventLevel.INFORMATIONAL,
        payload={},
    )


@pytest.mark.parametrize(
    "event_type",
    (
        "regional_feedback_received",
        "relay_radius_exceeded",
        "endurance_threshold_crossed",
        "communication_link_lost",
        "covariance_threshold_exceeded",
        "target_reacquired",
        "intent_change_confirmed",
        "target_lost",
        "operational_scheme_updated",
        "intelligence_report_received",
        "directive_applied",
    ),
)
def test_regional_replan_signals_route_strategically(event_type: str) -> None:
    node = EventMonitorNode(
        EventMonitor(scenario_id="S1"),
        lambda _: SimpleNamespace(group_reports=()),
    )

    result = node({"snapshot_ref": "S1:live", "pending_events": (_event(event_type),)})

    assert result["route"] is EventLevel.STRATEGIC
    assert result["coalesced_events"][0].level is EventLevel.STRATEGIC


def test_unknown_event_is_deferred_to_the_graph_error_route() -> None:
    node = EventMonitorNode(
        EventMonitor(scenario_id="S1"),
        lambda _: SimpleNamespace(group_reports=()),
    )

    result = node({"snapshot_ref": "S1:live", "pending_events": (_event("unknown"),)})

    assert result == {"node_error": "event_monitor failed: unknown event type: 'unknown'"}


class _DeterministicFailure:
    def __call__(self, state):
        del state
        raise ValueError("invalid regional geometry")


class _ProviderFailure:
    def __call__(self, state):
        del state
        raise LLMError("provider unavailable")


def test_regional_generation_defers_deterministic_failures_to_handle_error() -> None:
    result = RegionalGenerationWiringNode(_DeterministicFailure())({})

    assert result == {"node_error": "regional_generation failed: invalid regional geometry"}


def test_regional_strategy_preserves_llm_error_for_runtime_pause() -> None:
    with pytest.raises(LLMError, match="provider unavailable"):
        RegionalStrategyWiringNode(_ProviderFailure())({})


def test_regional_strategy_defers_deterministic_failures_to_handle_error() -> None:
    result = RegionalStrategyWiringNode(_DeterministicFailure())({})

    assert result == {"node_error": "regional_strategy failed: invalid regional geometry"}


class _FailingGraph:
    def __init__(self) -> None:
        self.state = None

    def invoke(self, state, *, config):
        self.state = state
        del config
        raise LLMError("regional provider unavailable")


def test_runtime_pauses_and_retains_regional_replan_after_llm_error(tmp_path) -> None:
    database_path = tmp_path / "runtime.db"
    plans = PlanRepository(database_path)
    events = EventRepository(database_path)
    ledger = DecisionLedger(database_path)
    clock = SimulationClock(step_s=30)
    runtime = CarrierRuntime(
        CarrierDependencies(
            plans=plans,
            events=events,
            ledger=ledger,
            llm=object(),
            predictor=lambda situation, target_id: None,
            situation_provider=lambda ref: None,
            clock=clock,
        ),
        scenario_id="S1",
        database_path=database_path,
    )
    failing_graph = _FailingGraph()
    edges = {
        (edge.source, edge.target)
        for edge in runtime._graph.get_graph().edges
    }
    assert {
        ("trajectory_prediction", "regional_generation"),
        ("regional_generation", "regional_strategy"),
        ("regional_strategy", "strategy_generation"),
        ("strategy_generation", "verify_strategy"),
        ("verify_strategy", "resource_optimizer"),
        ("resource_optimizer", "verify_plan"),
        ("verify_plan", "commit_plan"),
    } <= edges
    runtime._graph = failing_graph
    runtime.submit_regional_replan(
        reason="relay_radius",
        entity_id="R01",
        sim_time_s=0,
    )
    try:
        with pytest.raises(LLMError, match="regional provider unavailable"):
            runtime.tick()

        assert runtime.llm_paused is True
        assert runtime.llm_pause_reason == "regional provider unavailable"
        assert clock.sim_time_s == 0
        assert failing_graph.state["pending_events"][0].event_type == "relay_radius_exceeded"
    finally:
        runtime.close()
        plans.close()
        events.close()
        ledger.close()
