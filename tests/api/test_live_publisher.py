from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from underwater_tracking.api.frame_logger import FrameLogger
from underwater_tracking.api.hub import OperationalHub
from underwater_tracking.api.live import OperationalFramePublisher
from underwater_tracking.api.replay import ReplayService
from underwater_tracking.domain.agent_models import PlanAdjustmentSuggestion
from underwater_tracking.domain.models import (
    BearingObservation,
    CarrierState,
    Contact,
    SituationSnapshot,
    UUVState,
    UUVStatus,
)
from underwater_tracking.domain.models import EventLevel, RuntimeEvent
from underwater_tracking.runtime.mission_controller import MissionSnapshot


class Runtime:
    llm_paused = False

    def active_plan(self):
        return None

    def get_state(self):
        return {"intent_hypotheses": {}, "predictions": {}}


class SuggestionRuntime(Runtime):
    def get_state(self):
        return {
            "intent_hypotheses": {},
            "predictions": {},
            "plan_adjustment_suggestions": (
                PlanAdjustmentSuggestion(
                    suggestion_id="suggestion-1",
                    category="tracking_quality",
                    title="Improve track stability",
                    rationale="The current observation quality is weakening.",
                    proposed_feedback="Prioritize stable passive tracking for T1.",
                    target_ids=("T1",),
                    evidence_ids=("event-1",),
                    confidence=0.8,
                ),
            ),
        }


class Ledger:
    def list_decisions(self, scenario_id: str, limit: int = 100):
        return []

    def list_directives(self, scenario_id: str, status: str | None = None):
        return []


class Events:
    def list_events(self, **kwargs):
        return []


class HistoricalEvents(Events):
    def list_events(self, **kwargs):
        return [
            SimpleNamespace(
                event_id="old-manual-mode",
                scenario_id="S1",
                sim_time_s=0,
                event_type="manual_sensor_mode",
                severity="tactical",
                target_id="U1",
                payload={},
            )
        ]


class StalePlanningRuntime(Runtime):
    def get_state(self):
        return {
            "intent_hypotheses": {},
            "predictions": {},
            "snapshot_revision": 1,
            "snapshot_sim_time_s": 30,
        }


def test_publisher_bridges_runtime_state_to_hub_and_operational_replay(tmp_path: Path) -> None:
    hub = OperationalHub()
    log_path = tmp_path / "operational-frames.jsonl"
    logger = FrameLogger(log_path)
    publisher = OperationalFramePublisher(
        runtime=Runtime(), ledger=Ledger(), events=Events(), hub=hub, logger=logger
    )
    snapshot = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=2,
        sim_time_s=60,
        uuvs=(UUVState(
            uuv_id="U1", position_xy=(1.0, 2.0), heading_rad=0.0,
            speed_mps=2.0, energy_fraction=0.9, status=UUVStatus.RETURNING,
            deployment_state="returning",
        ),),
        carrier=CarrierState(
            carrier_id="carrier-01",
            position_xy=(-3000.0, -2995.0),
            heading_rad=1.57,
            speed_mps=1.0,
            status="recovering",
            returning_uuv_ids=("U1",),
        ),
        group_reports=(), pending_events=(),
    )

    frame = publisher.publish(snapshot)

    assert hub.snapshot() == frame
    assert frame.frame_id == 2
    assert frame.uuvs[0].uuv_id == "U1"
    assert frame.carrier is not None
    assert (frame.carrier.position.x, frame.carrier.position.y) == snapshot.carrier.position_xy
    logged_frame = ReplayService(log_path).range()[0]
    assert logged_frame.carrier == frame.carrier
    assert ReplayService(log_path).range() == [frame]
    publisher.close()


def test_publisher_exposes_llm_suggestions_in_replayable_frames(tmp_path: Path) -> None:
    log_path = tmp_path / "suggestions.jsonl"
    publisher = OperationalFramePublisher(
        runtime=SuggestionRuntime(),
        ledger=Ledger(),
        events=Events(),
        hub=OperationalHub(),
        logger=FrameLogger(log_path),
    )
    snapshot = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=1,
        sim_time_s=30,
        uuvs=(),
        group_reports=(),
        pending_events=(),
    )

    frame = publisher.publish(snapshot)

    assert frame.plan_adjustment_suggestions[0].proposed_feedback == (
        "Prioritize stable passive tracking for T1."
    )
    assert ReplayService(log_path).range()[0].plan_adjustment_suggestions == (
        frame.plan_adjustment_suggestions
    )
    publisher.close()


def test_publisher_skips_usv_ray_without_breaking_hub_or_jsonl(tmp_path: Path) -> None:
    hub = OperationalHub()
    log_path = tmp_path / "operational-frames.jsonl"
    publisher = OperationalFramePublisher(
        runtime=Runtime(), ledger=Ledger(), events=Events(), hub=hub, logger=FrameLogger(log_path)
    )
    snapshot = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=3,
        sim_time_s=90,
        uuvs=(),
        group_reports=(),
        pending_events=(),
        contacts=(
            Contact(
                contact_id="target-1",
                sim_time_s=90,
                bearing_rays=(
                    BearingObservation(
                        observation_id="passive:usv_00:target-1:90",
                        scenario_id="S1",
                        sim_time_s=90,
                        uuv_id="usv_00",
                        target_id="target-1",
                        azimuth_rad=0.25,
                        variance_rad2=0.02,
                        detection_confidence=0.8,
                    ),
                ),
            ),
        ),
    )

    frame = publisher.publish(snapshot)

    assert frame.bearing_rays == ()
    assert hub.snapshot() == frame
    assert ReplayService(log_path).range() == [frame]
    publisher.close()


def test_publisher_exposes_runtime_llm_pause_in_all_brain_views(tmp_path: Path) -> None:
    runtime = Runtime()
    runtime.llm_paused = True
    publisher = OperationalFramePublisher(
        runtime=runtime,
        ledger=Ledger(),
        events=Events(),
        hub=OperationalHub(),
        logger=FrameLogger(tmp_path / "paused.jsonl"),
    )
    snapshot = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=4,
        sim_time_s=120,
        uuvs=(),
        group_reports=(),
        pending_events=(),
    )

    frame = publisher.publish(snapshot)

    assert {brain.role for brain in frame.brains} == {"master", "slave", "adversary"}
    assert all(brain.status == "paused" for brain in frame.brains)
    assert all(brain.message == "等待 LLM 重连" for brain in frame.brains)
    publisher.close()


def test_publisher_advances_frame_id_for_repeated_observation_snapshot(tmp_path: Path) -> None:
    publisher = OperationalFramePublisher(
        runtime=Runtime(),
        ledger=Ledger(),
        events=Events(),
        hub=OperationalHub(),
        logger=FrameLogger(tmp_path / "frames.jsonl"),
    )
    snapshot = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=4,
        sim_time_s=120,
        uuvs=(),
        group_reports=(),
        pending_events=(),
    )

    first = publisher.publish(snapshot)
    second = publisher.publish(snapshot)

    assert [first.frame_id, second.frame_id] == [4, 5]
    publisher.close()


def test_publisher_marks_planning_data_age_against_physical_snapshot(tmp_path: Path) -> None:
    publisher = OperationalFramePublisher(
        runtime=StalePlanningRuntime(),
        ledger=Ledger(),
        events=Events(),
        hub=OperationalHub(),
        logger=FrameLogger(tmp_path / "stale.jsonl"),
    )
    snapshot = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=2,
        sim_time_s=60,
        uuvs=(),
        group_reports=(),
        pending_events=(),
    )

    frame = publisher.publish(snapshot)

    assert frame.planning_snapshot_revision == 1
    assert frame.planning_sim_time_s == 30
    assert frame.planning_data_age_s == 30
    assert frame.planning_data_status == "stale"
    publisher.close()


def test_publisher_limits_mission_event_tail(tmp_path: Path) -> None:
    mission = MissionSnapshot(
        scenario_id="S1",
        sim_time_s=60,
        plan_revision=1,
        events=tuple(
            RuntimeEvent(
                event_id=f"mission-{index}",
                scenario_id="S1",
                sim_time_s=index * 10,
                event_type="uuv_rotation",
                level=EventLevel.STRATEGIC,
            )
            for index in range(3)
        ),
    )
    publisher = OperationalFramePublisher(
        runtime=Runtime(),
        ledger=Ledger(),
        events=Events(),
        hub=OperationalHub(),
        logger=FrameLogger(tmp_path / "mission-tail.jsonl"),
        mission_snapshot_provider=lambda: mission,
        mission_event_history_limit=2,
    )
    snapshot = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=2,
        sim_time_s=60,
        uuvs=(),
        group_reports=(),
        pending_events=(),
    )

    frame = publisher.publish(snapshot)

    assert [event.event_id for event in frame.mission_events] == [
        "mission-1",
        "mission-2",
    ]
    publisher.close()


def test_publisher_projects_operator_thinking_and_stage_flags_to_all_frame_paths(
    tmp_path: Path,
) -> None:
    event = RuntimeEvent(
        event_id="manual-mode-1",
        scenario_id="S1",
        sim_time_s=60,
        event_type="manual_sensor_mode",
        level=EventLevel.TACTICAL,
        entity_id="U1",
        payload={"message": "operator selected passive sonar"},
    )
    publisher = OperationalFramePublisher(
        runtime=Runtime(),
        ledger=Ledger(),
        events=Events(),
        hub=OperationalHub(),
        logger=FrameLogger(tmp_path / "operator-context.jsonl"),
    )
    snapshot = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=6,
        sim_time_s=60,
        uuvs=(UUVState(
            uuv_id="U1",
            position_xy=(1.0, 2.0),
            heading_rad=0.0,
            speed_mps=2.0,
            energy_fraction=0.9,
            status=UUVStatus.TRACKING,
            deployment_state="deployed",
        ),),
        group_reports=(),
        pending_events=(event,),
    )

    frame = publisher.publish(snapshot)
    replayed = ReplayService(tmp_path / "operator-context.jsonl").range()[0]

    assert frame.operational_stage_flags == (
        "task_execution",
        "event_trigger",
        "human_feedback",
    )
    assert frame.llm_thinking
    assert frame.llm_thinking_trigger == "manual_sensor_mode"
    assert replayed.operational_stage_flags == frame.operational_stage_flags
    assert replayed.llm_thinking == frame.llm_thinking
    assert replayed.llm_thinking_trigger == frame.llm_thinking_trigger
    publisher.close()


def test_publisher_does_not_reuse_historical_event_as_current_thinking_trigger(
    tmp_path: Path,
) -> None:
    publisher = OperationalFramePublisher(
        runtime=Runtime(),
        ledger=Ledger(),
        events=HistoricalEvents(),
        hub=OperationalHub(),
        logger=FrameLogger(tmp_path / "historical-event.jsonl"),
    )
    snapshot = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=8,
        sim_time_s=60,
        uuvs=(),
        group_reports=(),
        pending_events=(),
    )

    frame = publisher.publish(snapshot)

    assert frame.llm_thinking_trigger == "等待首轮态势输入"
    assert frame.operational_stage_flags == ()
    publisher.close()
