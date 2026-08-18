"""Regional strategy routing tests for the carrier graph."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from underwater_tracking.agent.graphs.central import (
    CarrierDependencies,
    EventMonitorNode,
    RegionalGenerationWiringNode,
    RegionalStrategyWiringNode,
    RegionalStrategyToStrategySetNode,
    assess_regional_replan_events,
)
from underwater_tracking.agent.llm import LLMError
from underwater_tracking.agent.nodes.event_monitor import EventMonitor
from underwater_tracking.agent.runtime import CarrierRuntime
from underwater_tracking.domain.models import EventLevel, RuntimeEvent
from underwater_tracking.domain.regional_models import (
    CommunicationRequirement,
    GridSpec,
    RegionCell,
    RegionTask,
    RegionalPolicy,
    RegionalStrategySet,
    SonarPolicy,
    TargetRegionPlan,
    TimeWindow,
)
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


def _regional_plan_and_policy() -> tuple[TargetRegionPlan, RegionalStrategySet]:
    cell = RegionCell(
        region_id="T1:cell:0:0",
        target_id="T1",
        grid_x=0,
        grid_y=0,
        min_x=0.0,
        max_x=100.0,
        min_y=0.0,
        max_y=100.0,
        center_xy=(50.0, 50.0),
        cell_size_m=100.0,
        first_entry_s=100,
        last_exit_s=180,
        visit_windows=(TimeWindow(start_s=100, end_s=180),),
        evidence_ids=("belief:T1", "intent:T1"),
    )
    task = RegionTask(
        region_id=cell.region_id,
        target_id="T1",
        active_window=TimeWindow(start_s=100, end_s=180),
        required_quality=0.8,
        required_uuv_count=1,
        required_usv_count=1,
        uuv_roles=("passive_tracker",),
        usv_role="surface_relay",
        sonar_policy=SonarPolicy(passive_required=True),
        communication=CommunicationRequirement(),
        evidence_ids=cell.evidence_ids,
    )
    plan = TargetRegionPlan(
        target_id="T1",
        grid_spec=GridSpec(),
        cell_size_m=100.0,
        cells=(cell,),
        tasks=(task,),
        prediction_id="prediction:T1:1",
        intent_label="patrol",
        intent_confidence=0.8,
        evidence_ids=cell.evidence_ids,
    )
    policy = RegionalPolicy(
        region_id=cell.region_id,
        coverage_mode="required",
        priority=1.0,
        required_quality=0.8,
        required_uuv_count=1,
        required_usv_count=1,
        uuv_roles=("passive_tracker",),
        usv_role="surface_relay",
        sonar_policy=SonarPolicy(passive_required=True),
        communication=CommunicationRequirement(),
        tracking_mode="uuv_primary_usv_relay",
        assigned_uuv_ids=("U1",),
        assigned_usv_ids=("S1",),
        rationale="keep the selected UUV on the region with a surface relay",
        evidence_ids=cell.evidence_ids,
    )
    return plan, RegionalStrategySet(policies=(policy,))


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


def test_regional_strategy_is_the_authoritative_legacy_strategy_input() -> None:
    plan, policy_set = _regional_plan_and_policy()
    result = RegionalStrategyToStrategySetNode()(
        {
            "scenario_id": "S1",
            "regional_plans": {"T1": plan},
            "regional_policies": {"T1": policy_set},
            "coalesced_events": (_event("regional_feedback_received"),),
        }
    )

    strategy = result["strategy_set"]
    assert len(strategy.proposals) == 1
    assert strategy.proposals[0].target_priorities == {"T1": 1.0}
    assert strategy.proposals[0].required_quality == {"T1": 0.8}
    assert result["regional_policies"]["T1"] == policy_set
    assert policy_set.policies[0].assigned_uuv_ids == ("U1",)
    assert policy_set.policies[0].assigned_usv_ids == ("S1",)
    assert policy_set.policies[0].tracking_mode == "uuv_primary_usv_relay"


def test_state_assessment_emits_evidence_backed_replan_events() -> None:
    plan, _ = _regional_plan_and_policy()
    task = plan.tasks[0].model_copy(
        update={
            "assigned_uuv_ids": ("U1",),
            "assignment_status": "active",
            "communication_links": ("carrier->S1", "S1->U1"),
        }
    )
    active_plan = SimpleNamespace(region_tasks={task.region_id: task})
    situation = SimpleNamespace(
        scenario_id="S1",
        sim_time_s=900,
        group_reports=(
            SimpleNamespace(
                target_id="T1",
                event_types=("intent_change_detected",),
                belief=SimpleNamespace(covariance=((200.0, 0.0), (0.0, 200.0))),
            ),
        ),
        uuvs=(SimpleNamespace(uuv_id="U1", energy_fraction=0.1),),
        platform_snapshot=SimpleNamespace(communication_links=()),
    )

    events = assess_regional_replan_events(
        situation,
        active_plan=active_plan,
        known_target_ids=("T0", "T1"),
        covariance_cap_m2=100.0,
        endurance_threshold=0.2,
    )

    event_types = {event.event_type for event in events}
    assert {
        "covariance_threshold_exceeded",
        "endurance_threshold_crossed",
        "communication_link_lost",
    } <= event_types
    assert "target_lost" not in event_types
    assert "intent_change_confirmed" not in event_types
    assert all(event.payload.get("evidence") for event in events)


def test_state_assessment_does_not_treat_a_missing_report_as_target_loss() -> None:
    situation = SimpleNamespace(
        scenario_id="S1",
        sim_time_s=900,
        group_reports=(),
        uuvs=(),
        platform_snapshot=None,
    )

    events = assess_regional_replan_events(
        situation,
        active_plan=None,
        known_target_ids=("T1",),
        covariance_cap_m2=100.0,
    )

    assert events == ()


def test_state_assessment_limits_endurance_and_relay_checks_to_active_assignments() -> None:
    plan, _ = _regional_plan_and_policy()
    task = plan.tasks[0].model_copy(
        update={
            "assigned_uuv_ids": ("U1",),
            "assigned_usv_ids": ("S1",),
            "assignment_status": "active",
        }
    )
    active_plan = SimpleNamespace(region_tasks={task.region_id: task})
    situation = SimpleNamespace(
        scenario_id="S1",
        sim_time_s=900,
        group_reports=(),
        uuvs=(
            SimpleNamespace(uuv_id="U1", energy_fraction=0.1),
            SimpleNamespace(uuv_id="U2", energy_fraction=0.1),
        ),
        platform_snapshot=SimpleNamespace(
            carrier=SimpleNamespace(position_xy=(0.0, 0.0), support_radius_m=500.0),
            roster=SimpleNamespace(
                usvs=(SimpleNamespace(platform_id="S1", position_xy=(600.0, 0.0)),)
            ),
            communication_links=(),
        ),
    )

    events = assess_regional_replan_events(
        situation,
        active_plan=active_plan,
        known_target_ids=(),
        covariance_cap_m2=100.0,
        endurance_threshold=0.2,
    )

    assert {(event.event_type, event.entity_id) for event in events} == {
        ("endurance_threshold_crossed", "U1"),
        ("relay_radius_exceeded", "S1"),
    }


@pytest.mark.parametrize("assignment_status", ("planned", "degraded", "uncovered"))
def test_state_assessment_ignores_members_of_non_active_regional_tasks(
    assignment_status: str,
) -> None:
    plan, _ = _regional_plan_and_policy()
    task = plan.tasks[0].model_copy(
        update={
            "assigned_uuv_ids": ("U1",),
            "assigned_usv_ids": ("S1",),
            "assignment_status": assignment_status,
        }
    )
    active_plan = SimpleNamespace(region_tasks={task.region_id: task})
    situation = SimpleNamespace(
        scenario_id="S1",
        sim_time_s=900,
        group_reports=(),
        uuvs=(SimpleNamespace(uuv_id="U1", energy_fraction=0.1),),
        platform_snapshot=SimpleNamespace(
            carrier=SimpleNamespace(position_xy=(0.0, 0.0), support_radius_m=500.0),
            roster=SimpleNamespace(
                usvs=(SimpleNamespace(platform_id="S1", position_xy=(600.0, 0.0)),)
            ),
            communication_links=(),
        ),
    )

    events = assess_regional_replan_events(
        situation,
        active_plan=active_plan,
        known_target_ids=(),
        covariance_cap_m2=100.0,
        endurance_threshold=0.2,
    )

    assert events == ()


def test_state_assessment_reacquires_a_previously_lost_target() -> None:
    situation = SimpleNamespace(
        scenario_id="S1",
        sim_time_s=930,
        group_reports=(
            SimpleNamespace(
                target_id="T1",
                event_types=(),
                belief=SimpleNamespace(covariance=((1.0, 0.0), (0.0, 1.0))),
            ),
        ),
        uuvs=(),
        platform_snapshot=None,
    )

    events = assess_regional_replan_events(
        situation,
        active_plan=None,
        known_target_ids=("T1",),
        lost_target_ids=("T1",),
        covariance_cap_m2=100.0,
    )

    assert [event.event_type for event in events] == ["target_reacquired"]
    assert events[0].payload["evidence"]


def test_event_monitor_records_real_group_loss_by_target_then_reacquires() -> None:
    lost_situation = SimpleNamespace(
        scenario_id="S1",
        sim_time_s=900,
        uuvs=(),
        group_reports=(
            SimpleNamespace(
                group_id="G-T1",
                target_id="T1",
                quality=SimpleNamespace(ewma=1.0, hard_guard_reasons=()),
                belief=SimpleNamespace(covariance=((200.0, 0.0), (0.0, 200.0))),
            ),
        ),
        platform_snapshot=None,
    )
    node = EventMonitorNode(
        EventMonitor(target_lost_gap_s=300, covariance_cap_m2=100.0),
        lambda _: lost_situation,
        last_bearing_time=lambda target_id: 500 if target_id == "T1" else None,
    )

    checkpoint_state = node({"snapshot_ref": "S1:live", "pending_events": ()})

    assert checkpoint_state["coalesced_events"][0].event_type == "target_lost"
    assert checkpoint_state["coalesced_events"][0].entity_id == "G-T1"
    assert checkpoint_state["lost_target_ids"] == ("T1",)

    reacquired = assess_regional_replan_events(
        lost_situation,
        active_plan=None,
        known_target_ids=checkpoint_state["known_target_ids"],
        lost_target_ids=checkpoint_state["lost_target_ids"],
        covariance_cap_m2=1_000.0,
    )

    assert [(event.event_type, event.entity_id) for event in reacquired] == [
        ("target_reacquired", "T1")
    ]


def test_event_monitor_prefers_target_id_payload_and_accepts_target_entity_id() -> None:
    node = EventMonitorNode(
        EventMonitor(scenario_id="S1"),
        lambda _: SimpleNamespace(group_reports=()),
    )
    group_event = RuntimeEvent(
        event_id="S1:target_lost:G-T1",
        scenario_id="S1",
        sim_time_s=900,
        event_type="target_lost",
        entity_id="G-T1",
        level=EventLevel.STRATEGIC,
        payload={"target_id": "T1"},
    )
    target_event = RuntimeEvent(
        event_id="S1:target_lost:T2",
        scenario_id="S1",
        sim_time_s=900,
        event_type="target_lost",
        entity_id="T2",
        level=EventLevel.STRATEGIC,
        payload={},
    )

    result = node(
        {
            "snapshot_ref": "S1:live",
            "pending_events": (group_event, target_event),
        }
    )

    assert result["lost_target_ids"] == ("T1", "T2")


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
        ("regional_strategy", "regional_strategy_adapter"),
        ("regional_strategy_adapter", "verify_strategy"),
        ("verify_strategy", "resource_optimizer"),
        ("resource_optimizer", "verify_plan"),
        ("verify_plan", "commit_plan"),
    } <= edges
    assert "strategy_generation" not in runtime._graph.get_graph().nodes
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
