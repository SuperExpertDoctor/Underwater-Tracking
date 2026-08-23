"""Tiered event classification and coalescing tests (spec 8.2).

Covers the brief's verbatim hysteresis test (warning requires the hold
duration and then deduplicates inside the cooldown window) plus the
escalation cases: immediate critical quality, two consecutive gated intent
analyses, tactical repair infeasibility escalating to strategic, and a
failed member routing strategic when the group drops below the minimum
size.
"""

from threading import RLock
from types import SimpleNamespace

import pytest

from underwater_tracking.agent.nodes.event_monitor import EventMonitor
from underwater_tracking.agent.state import CarrierState
from underwater_tracking.domain.agent_models import (
    ExpertDirective,
    IntentHypothesis,
    PredictedTrackRef,
    StrategyProposal,
    StrategySet,
)
from underwater_tracking.domain.event_registry import (
    EVENT_REGISTRY,
    PUBLIC_AUDIENCES,
    event_definition,
    validate_event_payload,
)
from underwater_tracking.domain.models import (
    DeploymentState,
    EventLevel,
    RuntimeEvent,
    SituationSnapshot,
    UUVState,
    UUVStatus,
)


def test_quality_warning_requires_two_minutes_and_deduplicates():
    monitor = EventMonitor(warning_threshold=0.65, warning_hold_s=120, cooldown_s=300)
    assert monitor.observe_quality("G-T1", 30, 0.60) == ()
    assert monitor.observe_quality("G-T1", 120, 0.60) == ()
    events = monitor.observe_quality("G-T1", 150, 0.60)
    assert [event.event_type for event in events] == ["group_quality_warning"]
    assert monitor.observe_quality("G-T1", 180, 0.59) == ()


def test_quality_warning_does_not_refire_until_recovery():
    monitor = EventMonitor(warning_threshold=0.65, warning_hold_s=120, cooldown_s=300)
    assert monitor.observe_quality("G-T1", 30, 0.60) == ()
    warning = monitor.observe_quality("G-T1", 150, 0.60)
    assert [event.event_type for event in warning] == ["group_quality_warning"]
    # Still below threshold but inside the cooldown window -> merged.
    assert monitor.observe_quality("G-T1", 300, 0.60) == ()
    # A persistent episode stays coalesced even beyond the old cooldown.
    refired = monitor.observe_quality("G-T1", 450, 0.60)
    assert refired == ()
    monitor.observe_quality("G-T1", 500, 0.80)
    monitor.observe_quality("G-T1", 650, 0.60)
    recovered = monitor.observe_quality("G-T1", 770, 0.60)
    assert [event.event_type for event in recovered] == ["group_quality_warning"]


def test_monitor_episode_state_can_be_rolled_back_for_a_failed_graph_cycle():
    monitor = EventMonitor(critical_hold_s=0)
    checkpoint = monitor.checkpoint()

    first = monitor.observe_quality("G-T1", 30, 0.30)
    assert [event.event_type for event in first] == ["group_quality_critical"]

    monitor.restore(checkpoint)
    retry = monitor.observe_quality("G-T1", 30, 0.30)
    assert [event.event_type for event in retry] == ["group_quality_critical"]


def test_quality_recovery_resets_the_warning_streak():
    monitor = EventMonitor(warning_threshold=0.65, warning_hold_s=120, cooldown_s=300)
    assert monitor.observe_quality("G-T1", 30, 0.60) == ()
    assert monitor.observe_quality("G-T1", 120, 0.60) == ()
    # Recovery above the warning threshold resets the streak.
    assert monitor.observe_quality("G-T1", 130, 0.70) == ()
    assert monitor.observe_quality("G-T1", 150, 0.60) == ()
    events = monitor.observe_quality("G-T1", 270, 0.60)
    assert [event.event_type for event in events] == ["group_quality_warning"]


def test_critical_quality_escalates_immediately_and_breaks_cooldown():
    monitor = EventMonitor(
        warning_threshold=0.65, warning_hold_s=120, cooldown_s=300, critical_threshold=0.40
    )
    assert monitor.observe_quality("G-T1", 30, 0.60) == ()
    assert monitor.observe_quality("G-T1", 120, 0.60) == ()
    warning = monitor.observe_quality("G-T1", 150, 0.60)
    assert [event.event_type for event in warning] == ["group_quality_warning"]
    critical = monitor.observe_quality("G-T1", 180, 0.35)
    assert len(critical) == 1
    assert critical[0].event_type == "group_quality_critical"
    assert critical[0].level == EventLevel.STRATEGIC
    # A repeated critical inside the cooldown window is coalesced, but the
    # latest payload is retained.
    assert monitor.observe_quality("G-T1", 200, 0.30) == ()
    assert monitor.coalesced_payload("G-T1", "group_quality_critical") == {
        "quality": 0.30,
        "threshold": 0.40,
    }


def test_intent_change_confirmed_after_two_consecutive_gated_analyses():
    monitor = EventMonitor()
    assert (
        monitor.observe_intent_analysis(
            "T1", 100, leading_label="evade", confidence=0.80, runner_up_confidence=0.55
        )
        == ()
    )
    events = monitor.observe_intent_analysis(
        "T1", 200, leading_label="evade", confidence=0.85, runner_up_confidence=0.50
    )
    assert [event.event_type for event in events] == ["target_intent_changed"]
    assert events[0].level == EventLevel.STRATEGIC
    assert events[0].entity_id == "T1"


def test_intent_gates_reset_on_failing_analysis_or_label_change():
    monitor = EventMonitor()
    monitor.observe_intent_analysis(
        "T1", 100, leading_label="evade", confidence=0.80, runner_up_confidence=0.55
    )
    # Margin 0.80 - 0.70 = 0.10 < 0.15 -> gate fails, streak resets.
    assert (
        monitor.observe_intent_analysis(
            "T1", 200, leading_label="evade", confidence=0.80, runner_up_confidence=0.70
        )
        == ()
    )
    # Only one consecutive pass so far -> not yet confirmed.
    assert (
        monitor.observe_intent_analysis(
            "T1", 300, leading_label="evade", confidence=0.80, runner_up_confidence=0.55
        )
        == ()
    )
    # A different leading label restarts the streak as well.
    assert (
        monitor.observe_intent_analysis(
            "T1", 400, leading_label="approach", confidence=0.80, runner_up_confidence=0.55
        )
        == ()
    )


def test_repair_infeasibility_escalates_to_strategic():
    monitor = EventMonitor()
    applied = monitor.observe_repair("U3", 300, feasible=True, target_id="T1")
    assert [event.event_type for event in applied] == ["repair_applied"]
    assert applied[0].level == EventLevel.INFORMATIONAL
    infeasible = monitor.observe_repair("U3", 320, feasible=False, target_id="T1")
    assert [event.event_type for event in infeasible] == ["repair_infeasible"]
    assert infeasible[0].level == EventLevel.STRATEGIC


def test_member_failed_routes_strategic_when_group_drops_below_minimum():
    monitor = EventMonitor(group_min_size=2)
    replaceable = monitor.observe_member_failed("U1", 400, target_id="T1", remaining_members=2)
    assert replaceable[0].level == EventLevel.TACTICAL
    escalated = monitor.observe_member_failed("U2", 410, target_id="T1", remaining_members=1)
    assert escalated[0].level == EventLevel.STRATEGIC
    assert escalated[0].payload["remaining_members"] == 1
    assert escalated[0].payload["group_min_size"] == 2


def test_classify_routes_default_tiers_and_rejects_unknown_types():
    monitor = EventMonitor()
    for event_type in (
        "initialization",
        "target_added",
        "target_removed",
        "target_lost",
        "major_failure",
        "repair_infeasible",
        "directive_applied",
    ):
        assert monitor.classify(event_type) == EventLevel.STRATEGIC
    for event_type in ("group_quality_warning", "geometry_degradation", "battery_rotation"):
        assert monitor.classify(event_type) == EventLevel.TACTICAL
    assert monitor.classify("target_maneuver") == EventLevel.TACTICAL
    for event_type in (
        "progress_report",
        "question",
        "state_changed",
        "repair_applied",
        "intent_change_confirmed",
        "imm_confidence_shifted",
        "imm_motion_mode_changed",
    ):
        assert monitor.classify(event_type) == EventLevel.INFORMATIONAL
    assert monitor.classify("target_intent_change_suspected") == EventLevel.TACTICAL
    assert monitor.classify("target_intent_changed") == EventLevel.STRATEGIC
    assert monitor.classify("member_failed", payload={"remaining_members": 2}) == EventLevel.TACTICAL
    assert monitor.classify("member_failed", payload={"remaining_members": 1}) == EventLevel.STRATEGIC
    with pytest.raises(ValueError):
        monitor.classify("unknown_event")


def test_classify_routes_forwarded_engine_and_feedback_events() -> None:
    monitor = EventMonitor()
    strategic = ("strategic_review", "operational_scheme_updated")
    informational = (
        "uuv_recovery_requested",
        "uuv_deployed",
        "uuv_recovered",
        "group_report_published",
        "intelligence_report_received",
    )
    for event_type in strategic:
        assert monitor.classify(event_type) is EventLevel.STRATEGIC
    for event_type in informational:
        assert monitor.classify(event_type) is EventLevel.INFORMATIONAL
    assert monitor.classify("quality_guard:fim_degenerate") is EventLevel.TACTICAL


def test_classify_routes_uuv_mission_events_to_strategic_replan() -> None:
    monitor = EventMonitor()
    candidate_events = (
        "imm_confidence_shifted",
        "imm_motion_mode_changed",
        "target_entered_region",
        "target_exit_predicted",
        "handoff_completed",
        "region_coverage_degraded",
        "carrier_dispatch_completed",
        "carrier_recovery_completed",
        "llm_degraded",
    )
    for event_type in candidate_events:
        assert monitor.classify(event_type) is EventLevel.INFORMATIONAL
    for event_type in (
        "uuv_range_exhausted",
        "uuv_energy_depleted",
        "uuv_failed",
        "uuv_capability_lost",
        "carrier_task_window_missed",
    ):
        assert monitor.classify(event_type) is EventLevel.STRATEGIC
    assert monitor.classify("carrier_recovery_health_check_pending") is EventLevel.INFORMATIONAL


def test_prediction_and_intent_event_semantics_are_distinct() -> None:
    assert (
        event_definition("target_intent_change_suspected").default_level
        is EventLevel.TACTICAL
    )
    assert event_definition("target_intent_changed").default_level is EventLevel.STRATEGIC
    assert event_definition("imm_motion_mode_changed").default_level is EventLevel.INFORMATIONAL
    assert EVENT_REGISTRY["target_intent_changed"].audiences == PUBLIC_AUDIENCES


def test_prediction_suspicion_requires_auditable_diff_payload() -> None:
    with pytest.raises(ValueError, match="requires payload keys"):
        validate_event_payload("target_intent_change_suspected", {})

    validate_event_payload(
        "target_intent_change_suspected",
        {
            "diff_id": "D1",
            "previous_prediction_id": "P1",
            "current_prediction_id": "P2",
            "observation_ids": ("O1", "O2"),
            "absolute_rms_m": 300.0,
            "normalized_rms": 3.0,
            "absolute_floor_m": 250.0,
            "normalized_threshold": 2.45,
            "consecutive_count": 2,
            "source": "trajectory_diff",
        },
    )


def test_runtime_batch_submission_preserves_event_ids_and_deduplicates() -> None:
    """The forwarding adapter must not replace source event IDs."""
    from underwater_tracking.agent.runtime import CarrierRuntime

    active_ping = RuntimeEvent(
        event_id="active-ping-30",
        scenario_id="S1",
        sim_time_s=30,
        event_type="active_ping",
        entity_id="U1",
        level=EventLevel.INFORMATIONAL,
    )
    lifecycle = RuntimeEvent(
        event_id="recovery-30",
        scenario_id="S1",
        sim_time_s=30,
        event_type="uuv_recovery_requested",
        entity_id="U1",
        level=EventLevel.INFORMATIONAL,
    )
    runtime = CarrierRuntime.__new__(CarrierRuntime)
    runtime._pending = []
    runtime._processed_event_ids = set()
    runtime._lock = RLock()

    runtime.submit_events((active_ping, lifecycle, active_ping))

    assert runtime._pending == [active_ping, lifecycle]


def test_runtime_batch_submission_deduplicates_after_successful_cycle() -> None:
    """A source event must not re-enter after the graph has consumed it."""
    from underwater_tracking.agent.runtime import CarrierRuntime

    event = RuntimeEvent(
        event_id="ping-30",
        scenario_id="S1",
        sim_time_s=30,
        event_type="active_ping",
        entity_id="U1",
        level=EventLevel.INFORMATIONAL,
    )

    class RecordingGraph:
        def invoke(self, state: dict[str, object], *, config: dict[str, object]) -> dict[str, object]:
            assert state["pending_events"] == (event,)
            del config
            return {}

    runtime = CarrierRuntime.__new__(CarrierRuntime)
    runtime._scenario_id = "S1"
    runtime._config = {}
    runtime._graph = RecordingGraph()
    runtime._pending = [event]
    runtime._processed_event_ids = set()
    runtime._lock = RLock()

    runtime._run_cycle()
    runtime.submit_events((event,))

    assert runtime._pending == []


def test_agent_loop_forwards_source_events_and_emits_review_and_rotation() -> None:
    """Feedback events enter the runtime before its observation-cycle tick."""
    from underwater_tracking.cli import _AgentLoop

    class RecordingRuntime:
        def __init__(self) -> None:
            self.events: list[RuntimeEvent] = []

        def reservations(self) -> dict[str, tuple[str, ...]]:
            return {}

        def submit_events(self, events: tuple[RuntimeEvent, ...]) -> None:
            self.events.extend(events)

        def tick(self) -> dict[str, object]:
            return {"commit_status": "unchanged"}

    class RecordingEngine:
        def set_reservations(self, reservations: object) -> None:
            assert reservations == {}

    source_events = (
        RuntimeEvent(
            event_id="ping-900",
            scenario_id="underwater-default",
            sim_time_s=900,
            event_type="active_ping",
            entity_id="uuv_00",
            level=EventLevel.INFORMATIONAL,
        ),
        RuntimeEvent(
            event_id="recover-900",
            scenario_id="underwater-default",
            sim_time_s=900,
            event_type="uuv_recovery_requested",
            entity_id="uuv_01",
            level=EventLevel.INFORMATIONAL,
        ),
        RuntimeEvent(
            event_id="guard-900",
            scenario_id="underwater-default",
            sim_time_s=900,
            event_type="quality_guard:fim_degenerate",
            entity_id="G-target_00",
            level=EventLevel.TACTICAL,
        ),
    )

    def situation(sim_time_s: int, events: tuple[RuntimeEvent, ...]) -> SituationSnapshot:
        return SituationSnapshot(
            scenario_id="underwater-default",
            snapshot_revision=sim_time_s // 30,
            sim_time_s=sim_time_s,
            uuvs=(
                UUVState(
                    uuv_id="uuv_00",
                    position_xy=(0.0, 0.0),
                    heading_rad=0.0,
                    speed_mps=1.0,
                    energy_fraction=0.2,
                    status=UUVStatus.AVAILABLE,
                    deployment_state=DeploymentState.DEPLOYED,
                    group_id="target_00",
                ),
                UUVState(
                    uuv_id="uuv_01",
                    position_xy=(0.0, 0.0),
                    heading_rad=0.0,
                    speed_mps=0.0,
                    energy_fraction=0.1,
                    status=UUVStatus.AVAILABLE,
                    deployment_state=DeploymentState.ONBOARD,
                ),
            ),
            group_reports=(),
            pending_events=events,
        )

    runtime = RecordingRuntime()
    loop = _AgentLoop.__new__(_AgentLoop)
    loop._config = SimpleNamespace(
        timing=SimpleNamespace(strategic_review_s=900),
        agent=SimpleNamespace(event_cooldown_s=300),
    )
    loop._runtime = runtime
    loop._engine = RecordingEngine()
    loop.scenario_id = "underwater-default"
    loop._initialization_submitted = True
    loop._last_plan_id = None
    loop._last_strategic_review_s = 0
    loop._last_battery_rotation_s = {}
    loop._publisher = None
    loop.carrier_error_count = 0

    loop.on_situation(situation(900, source_events))

    assert [event.event_id for event in runtime.events[:3]] == [
        "ping-900",
        "recover-900",
        "guard-900",
    ]
    assert [event.event_type for event in runtime.events[3:]] == [
        "strategic_review",
        "battery_rotation",
    ]
    assert runtime.events[-1].payload == {
        "energy_fraction": 0.2,
        "rotation_threshold": 0.3,
        "target_id": "target_00",
    }

    loop.on_situation(situation(930, ()))
    assert [event.event_type for event in runtime.events] == [
        "active_ping",
        "uuv_recovery_requested",
        "quality_guard:fim_degenerate",
        "strategic_review",
        "battery_rotation",
    ]


def test_agent_loop_review_uses_elapsed_time_between_snapshots() -> None:
    """A 50 s review interval fires at the 60 s observation snapshot."""
    from underwater_tracking.cli import _AgentLoop

    loop = _AgentLoop.__new__(_AgentLoop)
    loop.scenario_id = "underwater-default"
    loop._config = SimpleNamespace(
        timing=SimpleNamespace(strategic_review_s=50), agent=None
    )
    loop._last_strategic_review_s = 0
    loop._last_battery_rotation_s = {}

    def snapshot(sim_time_s: int) -> SituationSnapshot:
        return SituationSnapshot(
            scenario_id="underwater-default",
            snapshot_revision=sim_time_s // 30,
            sim_time_s=sim_time_s,
            uuvs=(),
            group_reports=(),
            pending_events=(),
        )

    assert loop._feedback_events(snapshot(30)) == ()
    events = loop._feedback_events(snapshot(60))
    assert [(event.event_type, event.sim_time_s) for event in events] == [
        ("strategic_review", 60)
    ]


def test_agent_loop_excludes_unassigned_low_energy_uuv_from_rotation() -> None:
    from underwater_tracking.cli import _AgentLoop

    loop = _AgentLoop.__new__(_AgentLoop)
    loop.scenario_id = "underwater-default"
    loop._config = SimpleNamespace(
        timing=SimpleNamespace(strategic_review_s=900), agent=None
    )
    loop._last_strategic_review_s = 0
    loop._last_battery_rotation_s = {}
    snapshot = SituationSnapshot(
        scenario_id="underwater-default",
        snapshot_revision=1,
        sim_time_s=30,
        uuvs=(
            UUVState(
                uuv_id="assigned",
                position_xy=(0.0, 0.0),
                heading_rad=0.0,
                speed_mps=1.0,
                energy_fraction=0.2,
                status=UUVStatus.AVAILABLE,
                deployment_state=DeploymentState.DEPLOYED,
                group_id="target_00",
            ),
            UUVState(
                uuv_id="unassigned",
                position_xy=(0.0, 0.0),
                heading_rad=0.0,
                speed_mps=1.0,
                energy_fraction=0.2,
                status=UUVStatus.AVAILABLE,
                deployment_state=DeploymentState.DEPLOYED,
            ),
        ),
        group_reports=(),
        pending_events=(),
    )

    events = loop._feedback_events(snapshot)

    assert [event.entity_id for event in events] == ["assigned"]


def test_carrier_state_holds_references_not_raw_histories():
    directive = ExpertDirective(
        directive_id="D1", raw_text="prioritize T1", target_scope=("T1",), confidence=0.9
    )
    event = RuntimeEvent(
        event_id="G-T1:group_quality_warning:150",
        scenario_id="S1",
        sim_time_s=150,
        event_type="group_quality_warning",
        entity_id="G-T1",
        level=EventLevel.TACTICAL,
        payload={"quality": 0.6},
    )
    proposal = StrategyProposal(
        concept="balanced",
        target_priorities={"T1": 1.0},
        required_quality={"T1": 0.7},
        reinforcement_policy={"T1": "reinforce"},
        releasable_soft_constraints=(),
        evidence_ids=("B:T1:900",),
        rationale="test",
    )
    state: CarrierState = {
        "scenario_id": "S1",
        "snapshot_revision": 7,
        "snapshot_ref": "snapshot:S1:7",
        "pending_events": (event,),
        "coalesced_events": (event,),
        "route": EventLevel.STRATEGIC,
        "intent_hypotheses": {
            "T1": IntentHypothesis(
                label="evade", confidence=0.8, evidence_ids=("B:T1:900",), model_id="m1",
                prompt_version="p1",
            )
        },
        "predictions": {
            "T1": PredictedTrackRef(
                prediction_id="P:T1:1", target_id="T1", sim_time_s=900, horizon_s=1800.0,
                sample_step_s=30.0,
            )
        },
        "strategy_set": StrategySet(proposals=(proposal,)),
        "validation_attempts": 2,
        "candidate_plan_refs": ("plan:S1:4",),
        "selected_plan_ref": "plan:S1:4",
        "latest_directive": directive,
        "latest_question": "Why is T1 downgraded?",
        "history_summaries": ("summary:S1:2",),
        "errors": (),
        "output_messages": ("carrier ready",),
    }
    assert state["scenario_id"] == "S1"
    assert state["snapshot_revision"] == 7
    assert state["snapshot_ref"] == "snapshot:S1:7"
    assert state["route"] == EventLevel.STRATEGIC
    assert state["candidate_plan_refs"] == ("plan:S1:4",)
    assert state["selected_plan_ref"] == "plan:S1:4"
    assert state["latest_directive"] is directive
    assert state["pending_events"][0].event_type == "group_quality_warning"
    assert state["intent_hypotheses"]["T1"].label == "evade"
    assert state["strategy_set"] is not None
    assert state["predictions"]["T1"].prediction_id == "P:T1:1"
