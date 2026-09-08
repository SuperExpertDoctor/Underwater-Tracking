from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace

from underwater_tracking.api.frame_logger import FrameLogger
from underwater_tracking.api.hub import OperationalHub
from underwater_tracking.api.live import (
    FramePersistencePolicy,
    OperationalFramePublisher,
    compact_operational_frame,
)
from underwater_tracking.api.replay import ReplayService
from underwater_tracking.domain.agent_models import (
    PlanAdjustmentSuggestion,
    PredictedTrackRef,
    TrajectoryDiffGateState,
    TrajectoryDiffResult,
)
from underwater_tracking.domain.models import (
    BearingObservation,
    CarrierState,
    Contact,
    GroupQuality,
    GroupReport,
    SituationSnapshot,
    TargetBelief,
    UUVState,
    UUVStatus,
)
from underwater_tracking.domain.models import EventAudience, EventLevel, RuntimeEvent
from underwater_tracking.domain.prediction_models import AcceptedPrediction, PredictionHealth
from underwater_tracking.domain.ui_models import BrainActivityRecord, PlanningHealthView
from underwater_tracking.domain.ui_models import RegionTimelineView
from underwater_tracking.runtime.mission_controller import MissionSnapshot
from underwater_tracking.domain.mission_models import RegionMissionState
from underwater_tracking.domain.regional_models import RegionalMissionCandidate, TimeWindow
from underwater_tracking.world_model.demo import build_demo_input
from underwater_tracking.world_model.rules import predict_future_events


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


class PredictionDiffRuntime(Runtime):
    def get_state(self):
        prediction = PredictedTrackRef(
            prediction_id="P2",
            target_id="T1",
            sim_time_s=30,
            source_track_revision=1,
            prediction_revision=1,
            last_observed_at_s=30.,
            generated_at_s=30.,
            valid_until_s=930.,
            horizon_s=60.0,
            sample_step_s=30.0,
            times_s=(60.0, 90.0),
            points_xy=((60.0, 0.0), (90.0, 0.0)),
            corridor_radius_m=(10.0, 10.0),
        )
        return {
            "intent_hypotheses": {},
            "predictions": {"T1": prediction},
            "accepted_predictions": {
                "T1": AcceptedPrediction(
                    prediction=prediction,
                    health=PredictionHealth(
                        status="degraded",
                        regime="short_history",
                        source_track_age_s=0.0,
                        clipped_point_fraction=0.0,
                        maximum_radius_m=10.0,
                        raw_prediction_id=prediction.prediction_id,
                    ),
                )
            },
            "prediction_diffs": {
                "T1": TrajectoryDiffResult(
                    diff_id="D1",
                    target_id="T1",
                    previous_prediction_id="P1",
                    current_prediction_id="P2",
                    previous_sim_time_s=0,
                    current_sim_time_s=30,
                    status="comparable",
                    absolute_rms_m=300.0,
                    normalized_rms=3.0,
                    normalized_threshold=2.45,
                    absolute_floor_m=250.0,
                    reset_normalized_threshold=1.75,
                    reset_absolute_floor_m=150.0,
                    threshold_schema_version="trajectory-diff-v1",
                    confirmation_cycles=2,
                    exceeded=True,
                    consecutive_count=2,
                    latched=True,
                    gate_transition="verifying",
                )
            },
            "prediction_diff_gates": {
                "T1": TrajectoryDiffGateState(
                    target_id="T1",
                    consecutive_count=2,
                    latched=True,
                    verification_pending=True,
                    suspicion_event_id="E1",
                    latest_diff_id="D1",
                )
            },
        }


def test_live_publisher_does_not_publish_raw_prediction_as_valid() -> None:
    class RawPredictionRuntime(Runtime):
        def get_state(self):
            state = PredictionDiffRuntime().get_state()
            raw_prediction = state["predictions"]["T1"].model_copy(
                update={
                    "prediction_regime": "imm",
                    "imm_model_probabilities": {"CV": 1.0},
                }
            )
            return {
                "intent_hypotheses": {},
                "predictions": {"T1": raw_prediction},
            }

    snapshot = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=1,
        sim_time_s=30,
        uuvs=(),
        group_reports=(
            GroupReport(
                group_id="G1",
                target_id="T1",
                sim_time_s=30,
                member_ids=(),
                belief=TargetBelief(
                    target_id="T1",
                    sim_time_s=30,
                    mean=(0.0, 0.0),
                    covariance=((100.0, 0.0), (0.0, 100.0)),
                    model_probabilities={"cv": 1.0},
                    fim_condition=1.0,
                ),
                quality=GroupQuality(
                    instant=0.9,
                    window_mean=0.9,
                    ewma=0.9,
                    components={"fim": 0.9},
                ),
                plan_revision=1,
            ),
        ),
        pending_events=(),
    )
    publisher = OperationalFramePublisher(
        runtime=RawPredictionRuntime(),
        ledger=Ledger(),
        events=Events(),
        hub=OperationalHub(),
    )

    frame = publisher.publish(snapshot)

    prediction = frame.target_estimates[0].prediction
    assert prediction is None


class WorldModelRuntime(Runtime):
    def get_state(self):
        from tests.world_model.test_rule_world_model import _snapshot_from_demo
        demo = build_demo_input("left_turn").model_copy(update={"scenario_id": "S1"})
        _, prediction, _ = _snapshot_from_demo(demo)
        forecast = predict_future_events(demo)
        return {
            "intent_hypotheses": {},
            "predictions": {},
            "world_model_forecasts": {forecast.target_id: forecast},
            "accepted_predictions": {forecast.target_id: AcceptedPrediction(prediction=prediction,
                health=PredictionHealth(status="valid", regime="bspline", source_track_age_s=0.,
                    clipped_point_fraction=0., maximum_radius_m=100., raw_prediction_id=prediction.prediction_id))},
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


class PrivateAuditEvents(Events):
    def list_events(self, **kwargs):
        return [
            SimpleNamespace(
                event_id="target_mission_decision:target_00:decision-1",
                scenario_id="S1",
                sim_time_s=30,
                event_type="target_mission_decision",
                severity="informational",
                target_id="target_00",
                audiences=frozenset(
                    {
                        EventAudience.ADVERSARY_PRIVATE,
                        EventAudience.OPERATOR_AUDIT,
                        EventAudience.MEMORY_SOURCE,
                    }
                ),
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


class ActivityLedger(Ledger):
    def latest_role_activity(self, scenario_id: str):
        return {
            "master": BrainActivityRecord(
                brain_id="carrier-master",
                role="master",
                status="succeeded",
                operation="regional_strategy",
                sim_time_s=30,
                message="regional strategy returned",
            )
        }


def test_persistence_policy_only_treats_new_events_as_boundaries() -> None:
    policy = FramePersistencePolicy(sample_interval_s=30)
    first = SimpleNamespace(
        plan_version=1,
        events=(SimpleNamespace(event_id="event-1"),),
        mission_events=(),
        run_phase="running",
        sim_time_s=0,
    )
    unchanged = SimpleNamespace(
        plan_version=1,
        events=first.events,
        mission_events=(),
        run_phase="running",
        sim_time_s=5,
    )
    sampled = SimpleNamespace(
        plan_version=1,
        events=first.events,
        mission_events=(),
        run_phase="running",
        sim_time_s=30,
    )
    new_event = SimpleNamespace(
        plan_version=1,
        events=(
            *first.events,
            SimpleNamespace(event_id="event-2"),
        ),
        mission_events=(),
        run_phase="running",
        sim_time_s=35,
    )

    assert policy.should_persist(first)
    assert not policy.should_persist(unchanged)
    assert policy.should_persist(sampled)
    assert policy.should_persist(new_event)


def test_compact_operational_frame_bounds_replay_only() -> None:
    publisher = OperationalFramePublisher(
        runtime=Runtime(),
        ledger=Ledger(),
        events=Events(),
        hub=OperationalHub(),
    )
    snapshot = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=1,
        sim_time_s=30,
        uuvs=(
            UUVState(
                uuv_id="U1",
                position_xy=(1.0, 2.0),
                heading_rad=0.0,
                speed_mps=2.0,
                energy_fraction=0.9,
                status=UUVStatus.TRACKING,
                deployment_state="deployed",
            ),
        ),
        group_reports=(),
        pending_events=(),
    )
    frame = publisher.publish(snapshot)
    timeline_row = RegionTimelineView(
        region_id="region-1",
        target_id="target-1",
        center={"x": 0.0, "y": 0.0},
        bounds={"min_x": -1.0, "min_y": -1.0, "max_x": 1.0, "max_y": 1.0},
        start_offset_s=0.0,
        end_offset_s=10.0,
        status="active",
        priority=1.0,
        occupancy_likelihood=0.8,
    )
    expanded = frame.model_copy(
        update={
            "uuvs": (
                frame.uuvs[0].model_copy(
                    update={
                        "breadcrumb": tuple(frame.uuvs[0].breadcrumb) * 40,
                        "connected_peer_ids": tuple(f"uuv-{i}" for i in range(12)),
                    }
                ),
            ),
            "region_timeline": (timeline_row,) * 20,
            "events": (SimpleNamespace(event_id="event-1"),),
            "ledger": (SimpleNamespace(decision_id="decision-1"),),
        }
    )

    compact = compact_operational_frame(expanded)

    assert compact.plan_version == expanded.plan_version
    assert compact.uuvs[0].breadcrumb == expanded.uuvs[0].breadcrumb[-24:]
    assert len(compact.uuvs[0].connected_peer_ids) == 4
    assert len(compact.region_timeline) == 16
    assert compact.events == expanded.events
    assert compact.ledger == expanded.ledger
    publisher.close()


def test_compact_operational_frame_bounds_historical_replay_arrays() -> None:
    publisher = OperationalFramePublisher(
        runtime=Runtime(),
        ledger=Ledger(),
        events=Events(),
        hub=OperationalHub(),
    )
    snapshot = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=1,
        sim_time_s=30,
        uuvs=(),
        group_reports=(),
        pending_events=(),
    )
    frame = publisher.publish(snapshot).model_copy(
        update={
            "events": tuple(SimpleNamespace(event_id=f"event-{i}") for i in range(100)),
            "mission_events": tuple(
                SimpleNamespace(event_id=f"mission-{i}") for i in range(100)
            ),
            "ledger": tuple(SimpleNamespace(decision_id=f"decision-{i}") for i in range(100)),
            "operator_audit_event_ids": tuple(f"audit-{i}" for i in range(200)),
            "plan_timeline": tuple(SimpleNamespace(plan_id=f"plan-{i}") for i in range(100)),
        }
    )

    compact = compact_operational_frame(frame)

    assert [item.event_id for item in compact.events] == [
        f"event-{i}" for i in range(36, 100)
    ]
    assert [item.event_id for item in compact.mission_events] == [
        f"mission-{i}" for i in range(84, 100)
    ]
    assert [item.decision_id for item in compact.ledger] == [
        f"decision-{i}" for i in range(68, 100)
    ]
    assert compact.operator_audit_event_ids == tuple(
        f"audit-{i}" for i in range(200)
    )
    assert [item.plan_id for item in compact.plan_timeline] == [
        f"plan-{i}" for i in range(68, 100)
    ]
    publisher.close()


def test_publisher_uses_public_projection_for_hub_and_replay(tmp_path: Path) -> None:
    publisher = OperationalFramePublisher(
        runtime=Runtime(),
        ledger=Ledger(),
        events=Events(),
        hub=OperationalHub(),
        logger=FrameLogger(tmp_path / "projected.jsonl"),
        persistence_projection=lambda frame: frame.model_copy(
            update={"operator_audit_event_ids": ("public-audit",)}
        ),
    )
    snapshot = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=1,
        sim_time_s=30,
        uuvs=(),
        group_reports=(),
        pending_events=(),
    )

    full_frame = publisher.publish(snapshot)

    assert full_frame.operator_audit_event_ids == ()
    assert publisher._hub.snapshot().operator_audit_event_ids == ("public-audit",)
    assert ReplayService(tmp_path / "projected.jsonl").last().operator_audit_event_ids == (
        "public-audit",
    )
    publisher.close()


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


def test_publisher_reuses_one_immutable_payload_for_latest_websocket_and_jsonl(
    tmp_path: Path,
) -> None:
    hub = OperationalHub()
    log_path = tmp_path / "atomic.jsonl"
    publisher = OperationalFramePublisher(
        runtime=Runtime(),
        ledger=Ledger(),
        events=Events(),
        hub=hub,
        logger=FrameLogger(log_path),
    )
    snapshot = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=2,
        sim_time_s=60,
        uuvs=(),
        group_reports=(),
        pending_events=(),
    )

    publisher.publish(snapshot)

    latest_payload = hub.serialized_snapshot()

    async def websocket_payload() -> bytes:
        stream = hub.stream_serialized()
        try:
            return await anext(stream)
        finally:
            await stream.aclose()

    streamed_payload = asyncio.run(websocket_payload())
    jsonl_payload = log_path.read_bytes().splitlines()[-1]
    assert latest_payload is streamed_payload
    assert hashlib.sha256(latest_payload).digest() == hashlib.sha256(
        jsonl_payload
    ).digest()
    publisher.close()


def test_publisher_projects_checkpointed_prediction_diff_to_replay(tmp_path: Path) -> None:
    log_path = tmp_path / "prediction-diff.jsonl"
    publisher = OperationalFramePublisher(
        runtime=PredictionDiffRuntime(),
        ledger=Ledger(),
        events=Events(),
        hub=OperationalHub(),
        logger=FrameLogger(log_path),
    )
    report = GroupReport(
        group_id="G1",
        target_id="T1",
        sim_time_s=30,
        member_ids=(),
        belief=TargetBelief(
            target_id="T1",
            sim_time_s=30,
            mean=(30.0, 0.0),
            track_revision=1, last_observed_at_s=30, valid_until_s=930, source_observation_ids=("O1",),
            covariance=((100.0, 0.0), (0.0, 100.0)),
            model_probabilities={"cv": 1.0},
        ),
        quality=GroupQuality(
            instant=0.8,
            window_mean=0.8,
            ewma=0.8,
            components={},
        ),
        plan_revision=0,
    )
    snapshot = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=1,
        sim_time_s=30,
        uuvs=(),
        group_reports=(report,),
        pending_events=(),
    )

    frame = publisher.publish(snapshot)
    replayed = ReplayService(log_path).range()[0]

    assert frame.target_estimates[0].prediction.diff.state == "verifying"
    assert frame.target_estimates[0].prediction.health.status == "degraded"
    assert frame.target_estimates[0].prediction.centerline_xy[0].x == 60.0
    assert replayed.target_estimates[0].prediction.diff == (
        frame.target_estimates[0].prediction.diff
    )
    publisher.close()


def test_publisher_projects_world_model_forecast_to_replay(tmp_path: Path) -> None:
    log_path = tmp_path / "world-model.jsonl"
    publisher = OperationalFramePublisher(
        runtime=WorldModelRuntime(),
        ledger=Ledger(),
        events=Events(),
        hub=OperationalHub(),
        logger=FrameLogger(log_path),
    )
    report = GroupReport(
        group_id="G1",
        target_id="submarine_01",
        sim_time_s=300,
        member_ids=(),
        belief=TargetBelief(
            target_id="submarine_01",
            sim_time_s=300,
            mean=(0.0, 0.0),
            covariance=((100.0, 0.0), (0.0, 100.0)),
            model_probabilities={"left_turn": 0.78, "cv": 0.17, "right_turn": 0.05},
            track_revision=1, last_observed_at_s=300, valid_until_s=1200, source_observation_ids=("demo-observation-01",),
        ),
        quality=GroupQuality(
            instant=0.8,
            window_mean=0.8,
            ewma=0.8,
            components={},
        ),
        plan_revision=0,
    )
    snapshot = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=1,
        sim_time_s=300,
        uuvs=(),
        group_reports=(report,),
        pending_events=(),
    )

    frame = publisher.publish(snapshot)
    replayed = ReplayService(log_path).range()[0]

    world_model = frame.target_estimates[0].world_model
    assert world_model is not None
    assert world_model.control_authority is False
    assert world_model.events[0].event_type == "target_turn_left"
    assert replayed.target_estimates[0].world_model == world_model
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


def test_publisher_exposes_operator_audit_ids_without_private_payload(
    tmp_path: Path,
) -> None:
    publisher = OperationalFramePublisher(
        runtime=Runtime(),
        ledger=Ledger(),
        events=PrivateAuditEvents(),
        hub=OperationalHub(),
        logger=FrameLogger(tmp_path / "audit-ids.jsonl"),
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

    assert frame.operator_audit_event_ids == (
        "target_mission_decision:target_00:decision-1",
    )
    assert all(
        "intent" not in str(value).lower()
        for value in frame.model_dump(mode="json").values()
    )
    publisher.close()


def test_epoch_terminal_status_overrides_successful_subcall_activity(tmp_path: Path) -> None:
    publisher = OperationalFramePublisher(
        runtime=Runtime(),
        ledger=ActivityLedger(),
        events=Events(),
        hub=OperationalHub(),
        logger=FrameLogger(tmp_path / "rejected-epoch.jsonl"),
        planning_health_provider=lambda: PlanningHealthView(
            status="rejected",
            epoch_id="epoch-1",
            last_result_status="rejected",
            last_error="semantic correction rejected",
        ),
    )
    snapshot = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=1,
        sim_time_s=0,
        uuvs=(),
        group_reports=(),
        pending_events=(),
    )

    frame = publisher.publish(snapshot)

    master = next(brain for brain in frame.brains if brain.role == "master")
    assert master.status == "failed"
    assert master.operation == "planning_epoch"
    assert frame.plan_version == 0
    assert frame.planning is not None
    assert frame.planning.status == "rejected"
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


def test_publisher_projects_deterministic_candidate_geometry_when_graph_state_is_empty(
    tmp_path: Path,
) -> None:
    candidate = RegionalMissionCandidate(
        candidate_id="target_00:task:01",
        cell_ids=("target_00:r1:cell:0:0",),
        time_window=TimeWindow(start_s=30, end_s=330),
        perimeter_points=(
            (-1000.0, -1000.0),
            (-1000.0, 1000.0),
            (1000.0, 1000.0),
            (1000.0, -1000.0),
        ),
        required_uuv_count=2,
    )
    mission = MissionSnapshot(
        scenario_id="S1",
        sim_time_s=30,
        plan_revision=1,
        regions=(
            RegionMissionState(
                region_id=candidate.candidate_id,
                target_id="target_00",
                plan_revision=1,
            ),
        ),
    )
    publisher = OperationalFramePublisher(
        runtime=Runtime(),
        ledger=Ledger(),
        events=Events(),
        hub=OperationalHub(),
        logger=FrameLogger(tmp_path / "deterministic-regions.jsonl"),
        mission_snapshot_provider=lambda: mission,
        candidate_regions_provider=lambda: {candidate.candidate_id: candidate},
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

    assert tuple(
        (point.x, point.y) for point in frame.regional_missions[0].geometry
    ) == candidate.perimeter_points
    assert frame.regional_missions[0].entry_s == 30
    assert frame.regional_missions[0].exit_s == 330
    publisher.close()


def test_publisher_uses_executable_region_polygon_without_candidate_cache(
    tmp_path: Path,
) -> None:
    polygon = (
        (-2000.0, -1000.0),
        (-2000.0, 1000.0),
        (1000.0, 1000.0),
        (1000.0, -1000.0),
    )
    mission = MissionSnapshot(
        scenario_id="S1",
        sim_time_s=60,
        plan_revision=3,
        regions=(
            RegionMissionState(
                region_id="target_00:task:01",
                target_id="target_00",
                region_polygon=polygon,
                plan_revision=3,
            ),
        ),
    )
    publisher = OperationalFramePublisher(
        runtime=Runtime(),
        ledger=Ledger(),
        events=Events(),
        hub=OperationalHub(),
        logger=FrameLogger(tmp_path / "mission-polygon.jsonl"),
        mission_snapshot_provider=lambda: mission,
    )
    frame = publisher.publish(
        SituationSnapshot(
            scenario_id="S1",
            snapshot_revision=2,
            sim_time_s=60,
            uuvs=(),
            group_reports=(),
            pending_events=(),
        )
    )

    assert tuple((point.x, point.y) for point in frame.regional_missions[0].geometry) == polygon
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
    assert frame.llm_thinking_trigger == "expert_feedback"
    assert frame.llm_thinking_epoch_id
    assert frame.llm_thinking_source_event_ids == ("manual-mode-1",)
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

    assert frame.llm_thinking_trigger == "initialization"
    assert frame.operational_stage_flags == ()
    publisher.close()
