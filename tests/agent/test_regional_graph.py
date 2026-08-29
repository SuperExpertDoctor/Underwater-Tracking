"""Regional strategy routing tests for the carrier graph."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from underwater_tracking.agent.graphs.central import (
    CarrierDependencies,
    EventMonitorNode,
    ResourceOptimizerNode,
    RegionalGenerationWiringNode,
    RegionalStrategyWiringNode,
    RegionalStrategyToStrategySetNode,
    assess_regional_replan_events,
    _build_live_regional_generation,
    _route_after_prediction,
    _route_question_branch,
)
from underwater_tracking.agent.llm import LLMError
from underwater_tracking.agent.nodes.event_monitor import EventMonitor
from underwater_tracking.agent.nodes.optimize import (
    PlanningConfig,
    _refresh_uuv_only_regional_plans,
)
from underwater_tracking.agent.nodes.snapshot import PlanningSnapshot
from underwater_tracking.agent.runtime import CarrierRuntime
from underwater_tracking.domain.agent_models import (
    IntentHypothesis,
    PredictedTrackRef,
    StrategyProposal,
    StrategySet,
)
from underwater_tracking.domain.models import EventLevel, RuntimeEvent
from underwater_tracking.domain.prediction_models import AcceptedPrediction, PredictionHealth
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
from underwater_tracking.planning.dynamic_regions import DynamicRegionChain
from underwater_tracking.planning.region_baseline import build_four_region_baseline
from underwater_tracking.planning.regions import generate_target_region_plan
from underwater_tracking.simulation.clock import SimulationClock


def _event(
    event_type: str,
    *,
    payload: dict[str, object] | None = None,
    entity_id: str = "R01",
) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=f"S1:{event_type}",
        scenario_id="S1",
        sim_time_s=900,
        event_type=event_type,
        entity_id=entity_id,
        level=EventLevel.INFORMATIONAL,
        payload=payload or {},
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
        uuv_roles=("passive_tracker",),
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
        uuv_roles=("passive_tracker",),
        sonar_policy=SonarPolicy(passive_required=True),
        communication=CommunicationRequirement(),
        tracking_mode="heuristic_uuv",
        assigned_uuv_ids=("U1",),
        rationale="keep the selected UUV on the region",
        evidence_ids=cell.evidence_ids,
    )
    return plan, RegionalStrategySet(policies=(policy,))


@pytest.mark.parametrize(
    "event_type",
    (
        "regional_feedback_received",
        "communication_link_lost",
        "endurance_threshold_crossed",
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

    payload = (
        {"plan_impact": True}
        if event_type
        not in {"target_reacquired", "target_lost", "operational_scheme_updated", "directive_applied"}
        else {}
    )
    result = node({"snapshot_ref": "S1:live", "pending_events": (_event(event_type, payload=payload),)})

    assert result["route"] is EventLevel.STRATEGIC
    assert result["coalesced_events"][0].level is EventLevel.STRATEGIC


def test_normal_carrier_dispatch_is_not_a_strategic_replan() -> None:
    node = EventMonitorNode(
        EventMonitor(scenario_id="S1"),
        lambda _: SimpleNamespace(group_reports=()),
        active_plan_provider=lambda _: SimpleNamespace(region_tasks={}),
    )

    result = node({"snapshot_ref": "S1:live", "pending_events": (_event("carrier_dispatch_completed"),)})

    assert result["route"] is EventLevel.INFORMATIONAL
    assert result["coalesced_events"][0].level is EventLevel.INFORMATIONAL


def test_public_target_estimate_with_plan_impact_routes_tactically() -> None:
    node = EventMonitorNode(
        EventMonitor(scenario_id="S1"),
        lambda _: SimpleNamespace(group_reports=()),
        active_plan_provider=lambda _: SimpleNamespace(region_tasks={}),
    )

    result = node(
        {
            "snapshot_ref": "S1:live",
            "pending_events": (
                _event(
                    "target_estimate_updated",
                    entity_id="T1",
                    payload={
                        "observation_ids": ("obs:T1:900",),
                        "source": "fused_public_estimate",
                        "plan_impact": True,
                    },
                ),
            ),
        }
    )

    assert result["route"] is EventLevel.TACTICAL
    assert result["coalesced_events"][0].level is EventLevel.TACTICAL
    assert result["coalesced_events"][0].payload["plan_impact"] is True


def test_carrier_plan_degraded_is_a_registered_tactical_event() -> None:
    node = EventMonitorNode(
        EventMonitor(scenario_id="S1"),
        lambda _: SimpleNamespace(group_reports=()),
        active_plan_provider=lambda _: SimpleNamespace(region_tasks={}),
    )

    result = node(
        {
            "snapshot_ref": "S1:live",
            "pending_events": (
                _event(
                    "carrier_plan_degraded",
                    entity_id="carrier_01",
                    payload={
                        "candidate_id": "handoff:T1:1",
                        "carrier_id": "carrier_01",
                        "plan_revision": 3,
                        "plan_impact": True,
                        "reason": "deployment_window_infeasible",
                    },
                ),
            ),
        }
    )

    assert "node_error" not in result
    assert result["route"] is EventLevel.TACTICAL
    assert result["coalesced_events"][0].event_type == "carrier_plan_degraded"
    assert result["coalesced_events"][0].level is EventLevel.TACTICAL


def test_quality_critical_strategic_replan_skips_unrelated_intent_analysis() -> None:
    event = _event(
        "group_quality_critical",
        payload={"target_id": "T1", "hard_guard_reasons": ["fim_degenerate"]},
        entity_id="G-T1",
    ).model_copy(update={"level": EventLevel.STRATEGIC})

    assert _route_question_branch(
        {"route": EventLevel.STRATEGIC, "coalesced_events": (event,)}
    ) == "strategic_prediction"


def test_periodic_strategic_review_refreshes_plan_without_forcing_intent_llm() -> None:
    event = _event("strategic_review").model_copy(
        update={"level": EventLevel.STRATEGIC}
    )

    assert _route_question_branch(
        {"route": EventLevel.STRATEGIC, "coalesced_events": (event,)}
    ) == "strategic_prediction"


def test_quality_event_routes_strategically_only_when_active_quality_is_affected() -> None:
    node = EventMonitorNode(
        EventMonitor(scenario_id="S1"),
        lambda _: SimpleNamespace(
            sim_time_s=900,
            group_reports=(SimpleNamespace(target_id="T1", group_id="G-T1", quality=SimpleNamespace(
                ewma=0.40, hard_guard_reasons=()
            ), belief=SimpleNamespace(covariance=((1.0, 0.0), (0.0, 1.0))),),)
        ),
        active_plan_provider=lambda _: SimpleNamespace(
            region_tasks={"R1": SimpleNamespace(
                region_id="R1", target_id="T1", assignment_status="active",
                assigned_uuv_ids=("U1",), communication_links=(),
                required_quality=0.70,
            )},
            required_quality={"T1": 0.70},
            member_ids_by_target={"T1": ("U1",)},
        ),
    )

    result = node({
        "snapshot_ref": "S1:live",
        "pending_events": (_event("region_coverage_degraded", payload={"region_id": "R1"}),),
    })

    assert result["route"] is EventLevel.STRATEGIC
    assert result["coalesced_events"][0].level is EventLevel.STRATEGIC


def test_observed_quality_event_is_informational_without_an_active_plan() -> None:
    node = EventMonitorNode(
        EventMonitor(scenario_id="S1", critical_hold_s=0),
        lambda _: SimpleNamespace(
            scenario_id="S1",
            sim_time_s=900,
            group_reports=(
                SimpleNamespace(
                    target_id="T1",
                    group_id="G-T1",
                    quality=SimpleNamespace(ewma=0.30, hard_guard_reasons=()),
                    belief=SimpleNamespace(covariance=((1.0, 0.0), (0.0, 1.0))),
                ),
            ),
        ),
        active_plan_provider=lambda _: None,
    )

    result = node({"snapshot_ref": "S1:live", "pending_events": ()})

    assert result["route"] is EventLevel.INFORMATIONAL
    event = result["coalesced_events"][0]
    assert event.event_type == "group_quality_critical"
    assert event.level is EventLevel.INFORMATIONAL


def test_observed_quality_event_is_key_only_when_active_plan_quality_is_breached() -> None:
    node = EventMonitorNode(
        EventMonitor(scenario_id="S1", critical_hold_s=0),
        lambda _: SimpleNamespace(
            scenario_id="S1",
            sim_time_s=900,
            group_reports=(
                SimpleNamespace(
                    target_id="T1",
                    group_id="G-T1",
                    quality=SimpleNamespace(ewma=0.30, hard_guard_reasons=()),
                    belief=SimpleNamespace(covariance=((1.0, 0.0), (0.0, 1.0))),
                ),
            ),
        ),
        active_plan_provider=lambda _: SimpleNamespace(
            region_tasks={
                "R1": SimpleNamespace(
                    region_id="R1",
                    target_id="T1",
                    assignment_status="active",
                    assigned_uuv_ids=("U1",),
                    communication_links=(),
                    required_quality=0.70,
                )
            },
            required_quality={"T1": 0.70},
            member_ids_by_target={"T1": ("U1",)},
        ),
    )

    result = node({"snapshot_ref": "S1:live", "pending_events": ()})

    assert result["route"] is EventLevel.STRATEGIC
    event = result["coalesced_events"][0]
    assert event.level is EventLevel.STRATEGIC
    assert event.payload["plan_impact"] is True


def test_quality_sources_are_coalesced_into_one_plan_trigger() -> None:
    node = EventMonitorNode(
        EventMonitor(scenario_id="S1", critical_hold_s=0),
        lambda _: SimpleNamespace(
            scenario_id="S1",
            sim_time_s=900,
            group_reports=(
                SimpleNamespace(
                    target_id="T1",
                    group_id="G-T1",
                    quality=SimpleNamespace(ewma=0.30, hard_guard_reasons=()),
                    belief=SimpleNamespace(covariance=((1.0, 0.0), (0.0, 1.0))),
                ),
            ),
        ),
        active_plan_provider=lambda _: SimpleNamespace(
            region_tasks={
                "R1": SimpleNamespace(
                    region_id="R1",
                    target_id="T1",
                    assignment_status="active",
                    assigned_uuv_ids=("U1",),
                    communication_links=(),
                    required_quality=0.70,
                )
            },
            required_quality={"T1": 0.70},
            member_ids_by_target={"T1": ("U1",)},
        ),
    )

    result = node(
        {
            "snapshot_ref": "S1:live",
            "pending_events": (
                _event(
                    "regional_feedback_received",
                    payload={"target_id": "T1", "region_id": "R1"},
                ),
            ),
        }
    )

    assert result["route"] is EventLevel.STRATEGIC
    assert len(result["coalesced_events"]) == 1
    assert set(result["coalesced_events"][0].payload["coalesced_event_types"]) == {
        "group_quality_critical",
        "regional_feedback_received",
    }


def test_unknown_event_is_deferred_to_the_graph_error_route() -> None:
    node = EventMonitorNode(
        EventMonitor(scenario_id="S1"),
        lambda _: SimpleNamespace(group_reports=()),
    )

    result = node({"snapshot_ref": "S1:live", "pending_events": (_event("unknown"),)})

    assert result == {"node_error": "event_monitor failed: unknown event type: 'unknown'"}


def test_missing_fresh_prediction_uses_existing_regional_plan_tactically() -> None:
    assert _route_after_prediction(
        {
            "route": EventLevel.STRATEGIC,
            "predictions": {},
            "regional_plans": {"T1": object()},
        }
    ) == "tactical"


def test_goal_uuv_only_continuation_stays_tactical_after_public_replan() -> None:
    assert _route_after_prediction(
        {
            "route": EventLevel.STRATEGIC,
            "uuv_only": True,
            "predictions": {"T1": object()},
            "regional_plans": {"T1": object()},
            "executable_mission_plan": object(),
        }
    ) == "tactical"


def test_goal_uuv_only_public_estimate_update_reenters_regional_strategy() -> None:
    event = _event(
        "target_estimate_updated",
        entity_id="T1",
        payload={"observation_ids": ("obs:T1:900",)},
    ).model_copy(update={"level": EventLevel.TACTICAL})

    assert _route_after_prediction(
        {
            "route": EventLevel.STRATEGIC,
            "uuv_only": True,
            "predictions": {"T1": object()},
            "regional_plans": {"T1": object()},
            "executable_mission_plan": object(),
            "coalesced_events": (event,),
        }
    ) == "strategic"


def test_public_maneuver_refreshes_uuv_region_geometry_without_llm() -> None:
    plan, _ = _regional_plan_and_policy()
    prediction = PredictedTrackRef(
        prediction_id="prediction:T1:refresh",
        target_id="T1",
        sim_time_s=100,
        horizon_s=200.0,
        sample_step_s=50.0,
        times_s=(150.0, 200.0, 250.0),
        points_xy=((50.0, 50.0), (150.0, 50.0), (250.0, 50.0)),
        corridor_radius_m=(10.0, 12.0, 14.0),
        source_belief_history_ids=("belief:T1",),
    )
    intent = IntentHypothesis(
        label="transit",
        confidence=0.8,
        evidence_ids=("belief:T1",),
        model_id="test",
        prompt_version="test-v1",
    )
    snapshot = SimpleNamespace(
        sim_time_s=100,
        situation=SimpleNamespace(map_bounds_xy=(-1000.0, 1000.0, -1000.0, 1000.0)),
    )

    refreshed = _refresh_uuv_only_regional_plans(
        snapshot,
        {
            "coalesced_events": (
                _event(
                    "target_maneuver_observed",
                    entity_id="T1",
                    payload={"observation_ids": ("obs:T1:1",)},
                ),
            ),
            "predictions": {"T1": prediction},
            "intent_hypotheses": {"T1": intent},
        },
        {"T1": plan},
        PlanningConfig(bounds=(-1000.0, 1000.0, -1000.0, 1000.0)),
    )

    assert refreshed["T1"].prediction_id == prediction.prediction_id
    assert refreshed["T1"].cells
    assert refreshed["T1"].cells != plan.cells


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
    assert policy_set.policies[0].tracking_mode == "heuristic_uuv"


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


def test_state_assessment_limits_endurance_checks_to_active_assignments() -> None:
    plan, _ = _regional_plan_and_policy()
    task = plan.tasks[0].model_copy(
        update={
            "assigned_uuv_ids": ("U1",),
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
            roster=SimpleNamespace(uuvs=()),
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
    }


def test_state_assessment_uses_live_execution_groups_for_uuv_only_resources() -> None:
    plan, _ = _regional_plan_and_policy()
    task = plan.tasks[0].model_copy(
        update={"assigned_uuv_ids": ("U1",), "assignment_status": "planned"}
    )
    active_plan = SimpleNamespace(region_tasks={task.region_id: task})
    situation = SimpleNamespace(
        scenario_id="S1",
        sim_time_s=900,
        group_reports=(),
        execution_groups=(
            SimpleNamespace(mode="active_scan", member_ids=("U1",)),
        ),
        uuvs=(SimpleNamespace(uuv_id="U1", energy_fraction=0.1),),
        platform_snapshot=None,
    )

    events = assess_regional_replan_events(
        situation,
        active_plan=active_plan,
        known_target_ids=(),
        covariance_cap_m2=100.0,
        endurance_threshold=0.2,
    )

    assert [(event.event_type, event.entity_id) for event in events] == [
        ("endurance_threshold_crossed", "U1")
    ]


@pytest.mark.parametrize("assignment_status", ("planned", "degraded", "uncovered"))
def test_state_assessment_ignores_members_of_non_active_regional_tasks(
    assignment_status: str,
) -> None:
    plan, _ = _regional_plan_and_policy()
    task = plan.tasks[0].model_copy(
        update={
            "assigned_uuv_ids": ("U1",),
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
            roster=SimpleNamespace(uuvs=()),
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


def test_resource_optimizer_keeps_fresh_strategic_regional_evidence() -> None:
    """A new regional LLM result must not be filtered as stale."""

    snapshot = PlanningSnapshot(
        situation=SimpleNamespace(
            group_reports=(
                SimpleNamespace(
                    target_id="T1",
                    belief=SimpleNamespace(source_observation_ids=()),
                ),
            ),
        ),
        active_plan=None,
        applied_directives=(),
    )

    class RecordingOptimizer:
        def __init__(self) -> None:
            self.state = None

        def __call__(self, state):
            self.state = state
            return {"selected_plan_ref": "S1:candidate:1"}

    inner = RecordingOptimizer()
    node = ResourceOptimizerNode(
        inner,
        lambda ref: snapshot,
    )
    proposal = StrategyProposal(
        concept="balanced",
        target_priorities={"T1": 1.0},
        required_quality={"T1": 0.7},
        reinforcement_policy={"T1": "hold"},
        releasable_soft_constraints=(),
        evidence_ids=("prediction:S1:T1:900",),
        rationale="continue the fresh regional policy",
    )

    result = node(
        {
            "snapshot_ref": "S1:snapshot:3",
            "route": EventLevel.STRATEGIC,
            "regional_candidates": {"T1": ("T1:cell:0:0",)},
            "regional_policies": {"T1": ("policy",)},
            "strategy_set": StrategySet(proposals=(proposal,)),
        }
    )

    assert result == {"selected_plan_ref": "S1:candidate:1"}
    assert inner.state["strategy_set"].proposals == (proposal,)


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
        ("trajectory_prediction", "prediction_intent_analysis"),
        ("prediction_intent_analysis", "regional_generation"),
        ("prediction_intent_analysis", "resource_optimizer"),
        ("regional_generation", "regional_strategy"),
        ("regional_strategy", "regional_strategy_adapter"),
        ("regional_strategy_adapter", "verify_strategy"),
        ("verify_strategy", "resource_optimizer"),
        ("resource_optimizer", "verify_plan"),
        ("verify_plan", "commit_plan"),
    } <= edges
    assert ("prediction_intent_analysis", "trajectory_prediction") not in edges
    assert "strategy_generation" not in runtime._graph.get_graph().nodes
    runtime._graph = failing_graph
    runtime.submit_regional_replan(
        reason="communication_link",
        entity_id="R01",
        sim_time_s=0,
    )
    try:
        with pytest.raises(LLMError, match="regional provider unavailable"):
            runtime.tick()

        assert runtime.llm_paused is True
        assert runtime.llm_pause_reason == "regional provider unavailable"
        assert clock.sim_time_s == 0
        assert failing_graph.state["pending_events"][0].event_type == "communication_link_lost"
        assert any(event.event_type == "llm_degraded" for event in runtime._pending)
        runtime._graph = None
        with pytest.raises(LLMError, match="regional provider unavailable"):
            runtime.resume()
    finally:
        runtime.close()
        plans.close()
        events.close()
        ledger.close()


def test_production_region_wiring_reprojects_unavailable_prediction_without_geometry_llm() -> None:
    prediction = PredictedTrackRef(
        prediction_id="prediction:T1:prior",
        target_id="T1",
        sim_time_s=1_000,
        horizon_s=1_800.0,
        sample_step_s=100.0,
        times_s=tuple(1_000.0 + index * 100.0 for index in range(19)),
        points_xy=tuple((1_000.0 + index * 200.0, 2_000.0) for index in range(19)),
        corridor_radius_m=(100.0,) * 19,
        source_belief_history_ids=("belief:T1",),
        prediction_regime="imm",
    )
    intent = IntentHypothesis(
        label="transit",
        confidence=0.8,
        evidence_ids=("belief:T1",),
        model_id="test",
        prompt_version="test-v1",
    )
    accepted = AcceptedPrediction(
        prediction=prediction,
        health=PredictionHealth(
            status="valid",
            regime="imm",
            source_track_age_s=0.0,
            clipped_point_fraction=0.0,
            maximum_radius_m=100.0,
            raw_prediction_id=prediction.prediction_id,
        ),
    )
    baseline = build_four_region_baseline(
        accepted,
        target_id="T1",
        execution_revision=4,
        origin_sim_time_s=1_000.0,
        map_bounds_xy=(0.0, 8_000.0, 0.0, 6_000.0),
    )
    prior_chain = DynamicRegionChain(
        target_id="T1",
        prediction_id=prediction.prediction_id,
        execution_revision=4,
        geometry_revision=baseline.regions[0].geometry_revision,
        regions=baseline.regions,
    )
    prior_plan = generate_target_region_plan(
        prediction,
        intent,
        (0.0, 8_000.0, 0.0, 6_000.0),
        GridSpec(),
    )
    unavailable = AcceptedPrediction(
        prediction=None,
        health=PredictionHealth(
            status="unavailable",
            regime="boundary_recovery",
            reason_codes=("boundary_candidate_rejected", "all_candidates_rejected"),
            source_track_age_s=45.0,
            clipped_point_fraction=1.0,
            maximum_radius_m=0.0,
        ),
    )

    class NoCoordinateLLM:
        def invoke_structured(self, operation, *_args, **_kwargs):
            if operation == "task_regions":
                raise AssertionError("production live wiring called coordinate LLM")
            raise AssertionError(f"unexpected LLM operation: {operation}")

    snapshot = SimpleNamespace(scenario_id="S1", sim_time_s=1_100, active_plan=None)
    dependencies = SimpleNamespace(
        optimizer=SimpleNamespace(
            bounds=(0.0, 8_000.0, 0.0, 6_000.0),
            quality_warning=0.0,
        ),
        grid_spec=GridSpec(),
        llm=NoCoordinateLLM(),
        model_id="test",
        execution_strategy_node=None,
    )
    node = _build_live_regional_generation(dependencies, lambda _: snapshot)

    state = {
        "snapshot_ref": "S1:snapshot:1",
        "intent_hypotheses": {"T1": intent},
        "predictions": {},
        "accepted_predictions": {"T1": unavailable},
        "dynamic_region_chains": {"T1": prior_chain},
        "regional_plans": {"T1": prior_plan},
        "execution_revision": 5,
    }
    assert _route_after_prediction(state) == "strategic"

    result = node(state)

    chain = result["dynamic_region_chains"]["T1"]
    assert result["region_generation_modes"] == {"T1": "reprojected_previous"}
    assert result["region_generation_reason_codes"]["T1"] == (
        "boundary_candidate_rejected",
        "all_candidates_rejected",
        "reprojected_previous_regions",
    )
    assert tuple(region.geometry for region in chain.regions) == tuple(
        region.geometry for region in prior_chain.regions
    )
    assert chain.geometry_revision == prior_chain.geometry_revision
