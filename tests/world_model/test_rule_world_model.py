from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from underwater_tracking.domain.agent_models import PredictedTrackRef, TrackingPlan, Waypoint
from underwater_tracking.domain.models import (
    DeploymentState,
    GroupQuality,
    GroupReport,
    SituationSnapshot,
    SurveillanceCapability,
    TargetBelief,
    RuntimeEvent,
    EventLevel,
    UUVState,
    UUVStatus,
)
from underwater_tracking.world_model.adapter import (
    build_world_model_input,
    build_world_model_forecasts,
    planned_uuv_tracks_from_plan,
    predict_snapshot_events,
)
from underwater_tracking.world_model.config import (
    DEFAULT_WORLD_MODEL_CONFIG,
    load_world_model_config,
)
from underwater_tracking.world_model.demo import SCENARIOS, build_demo_input
from underwater_tracking.world_model.models import DataStatus, EventType
from underwater_tracking.world_model.rules import predict_future_events


EXPECTED_EVENT = {
    "normal": None,
    "left_turn": EventType.TARGET_TURN_LEFT,
    "sprint": EventType.HIGH_SPEED_ESCAPE,
    "area_exit": EventType.AREA_EXIT_RISK,
    "decoy": EventType.DECOY_OR_NEW_CONTACT_AMBIGUITY,
    "geometry_bad": EventType.GEOMETRY_DEGRADATION,
    "coverage_gap": EventType.UUV_COVERAGE_GAP,
    "track_loss": EventType.TRACK_LOSS_RISK,
    "stop": EventType.TARGET_ABNORMAL_STOP,
}


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_demo_scenarios_emit_the_expected_rule_event(scenario: str) -> None:
    inputs = build_demo_input(scenario)  # type: ignore[arg-type]
    forecast = predict_future_events(inputs)
    event_types = {event.event_type for event in forecast.events}
    expected = EXPECTED_EVENT[scenario]
    if expected is None:
        assert event_types == set()
    else:
        assert expected in event_types
    assert forecast.control_authority is False
    assert all(event.confidence >= 0.55 for event in forecast.events)


def test_rule_output_is_deterministic_and_frontend_ready() -> None:
    inputs = build_demo_input("left_turn")
    first = predict_future_events(inputs)
    second = predict_future_events(inputs)
    assert first == second
    assert first.model_dump_json() == second.model_dump_json()
    assert first.events[0].horizon.value == "H1"
    assert first.events[0].predicted_position_xy in inputs.trajectory.points_xy
    assert first.events[0].evidence
    assert tuple(horizon.name.value for horizon in first.horizons) == (
        "H1",
        "H2",
        "H3",
        "H4",
    )


def test_area_exit_is_a_mid_horizon_forecast() -> None:
    forecast = predict_future_events(build_demo_input("area_exit"))
    event = next(item for item in forecast.events if item.event_type is EventType.AREA_EXIT_RISK)
    assert event.horizon.value == "H3"
    assert 300.0 <= event.time_to_event_s < 900.0


def test_turn_requires_imm_support_as_well_as_a_curved_spline() -> None:
    inputs = build_demo_input("left_turn")
    low_turn_belief = inputs.belief.model_copy(
        update={
            "model_probabilities": {
                "cv": 0.90,
                "left_turn": 0.05,
                "right_turn": 0.05,
            }
        }
    )
    forecast = predict_future_events(inputs.model_copy(update={"belief": low_turn_belief}))
    assert EventType.TARGET_TURN_LEFT not in {
        event.event_type for event in forecast.events
    }


def test_mirrored_curve_and_right_turn_imm_probability_predict_right_turn() -> None:
    inputs = build_demo_input("left_turn")
    mirrored = inputs.trajectory.model_copy(
        update={
            "points_xy": tuple((point[0], -point[1]) for point in inputs.trajectory.points_xy)
        }
    )
    right_belief = inputs.belief.model_copy(
        update={
            "turn_rate_rad_s": -0.0105,
            "model_probabilities": {
                "cv": 0.17,
                "left_turn": 0.05,
                "right_turn": 0.78,
            },
        }
    )
    forecast = predict_future_events(
        inputs.model_copy(update={"belief": right_belief, "trajectory": mirrored})
    )
    assert EventType.TARGET_TURN_RIGHT in {
        event.event_type for event in forecast.events
    }


def test_short_history_fallback_is_visible_as_degraded_data() -> None:
    inputs = build_demo_input("normal")
    fallback = inputs.trajectory.model_copy(
        update={"fallback_used": True, "fallback_reason": "short history"}
    )
    forecast = predict_future_events(inputs.model_copy(update={"trajectory": fallback}))
    assert forecast.data_status is DataStatus.DEGRADED
    assert forecast.trajectory_fallback_used is True
    assert "short history" in forecast.warnings


def test_truth_or_unknown_payloads_are_rejected_at_the_input_boundary() -> None:
    payload = build_demo_input("normal").model_dump(mode="python")
    payload["target_position_truth"] = (1.0, 2.0)
    with pytest.raises(ValidationError):
        type(build_demo_input("normal")).model_validate(payload)


def test_yaml_config_matches_the_checked_in_defaults() -> None:
    path = Path("configs/world_model_rules.yaml")
    loaded = load_world_model_config(path)
    assert loaded == DEFAULT_WORLD_MODEL_CONFIG


def test_adapter_uses_live_operational_contracts() -> None:
    demo = build_demo_input("left_turn")
    snapshot, prediction, planned = _snapshot_from_demo(demo)
    inputs = build_world_model_input(snapshot, prediction, planned_uuv_tracks=planned)
    assert inputs.target_id == prediction.target_id
    assert inputs.belief.position_xy == (0.0, 0.0)
    assert inputs.belief.velocity_xy_mps == (8.0, 0.0)
    assert sum(inputs.belief.model_probabilities.values()) == pytest.approx(1.0)
    assert inputs.trajectory.prediction_id == prediction.prediction_id
    assert len(inputs.uuvs) == 3
    forecast = predict_snapshot_events(
        snapshot,
        prediction,
        planned_uuv_tracks=planned,
    )
    assert EventType.TARGET_TURN_LEFT in {
        event.event_type for event in forecast.events
    }
    assert "ground_truth" not in forecast.model_dump_json()


def test_adapter_prefers_imm_provenance_carried_by_the_prediction() -> None:
    demo = build_demo_input("left_turn")
    snapshot, prediction, planned = _snapshot_from_demo(demo)
    prediction = prediction.model_copy(
        update={"imm_model_probabilities": {"cv": 0.9, "left_turn": 0.1}}
    )

    inputs = build_world_model_input(snapshot, prediction, planned_uuv_tracks=planned)

    assert inputs.belief.model_probabilities == {"cv": 0.9, "left_turn": 0.1}


def test_committed_plan_waypoints_feed_future_uuv_projection() -> None:
    demo = build_demo_input("normal")
    snapshot, prediction, _planned = _snapshot_from_demo(demo)
    first_uuv_id = snapshot.uuvs[0].uuv_id
    plan = TrackingPlan(
        plan_id="plan-4",
        scenario_id=snapshot.scenario_id,
        revision=4,
        base_snapshot_revision=snapshot.snapshot_revision,
        status="active",
        waypoints_by_member={
            first_uuv_id: (
                Waypoint(x=100.0, y=200.0, arrive_at_s=snapshot.sim_time_s + 60),
                Waypoint(x=150.0, y=250.0, arrive_at_s=snapshot.sim_time_s + 120),
            )
        },
    )

    tracks = planned_uuv_tracks_from_plan(plan, as_of_s=snapshot.sim_time_s)
    inputs = build_world_model_input(snapshot, prediction, active_plan=plan)

    assert tracks[first_uuv_id] == (
        (float(snapshot.sim_time_s + 60), 100.0, 200.0),
        (float(snapshot.sim_time_s + 120), 150.0, 250.0),
    )
    projected = next(uuv for uuv in inputs.uuvs if uuv.uuv_id == first_uuv_id)
    assert projected.planned_points_xy == ((100.0, 200.0), (150.0, 250.0))
    assert inputs.source_plan_revision == 4


def test_observability_decoy_evidence_is_reused_without_claiming_ground_truth() -> None:
    demo = build_demo_input("normal")
    snapshot, prediction, planned = _snapshot_from_demo(demo)
    observability = RuntimeEvent(
        event_id="observability:report-1",
        scenario_id=snapshot.scenario_id,
        sim_time_s=snapshot.sim_time_s,
        event_type="observability_urgent",
        entity_id=prediction.target_id,
        level=EventLevel.TACTICAL,
        payload={
            "events": [
                {
                    "event_id": "event-decoy-1",
                    "track_id": prediction.target_id,
                    "hypothesis": "DECOY_OR_NEW_TARGET",
                    "confidence": 0.81,
                    "recovery": False,
                }
            ]
        },
    )
    snapshot = snapshot.model_copy(update={"pending_events": (observability,)})

    inputs = build_world_model_input(snapshot, prediction, planned_uuv_tracks=planned)
    forecast = predict_future_events(inputs)

    assert inputs.tracking.observability_hypotheses == {"DECOY_OR_NEW_TARGET": 0.81}
    assert inputs.source_observability_event_ids == ("event-decoy-1",)
    assert EventType.DECOY_OR_NEW_CONTACT_AMBIGUITY in {
        event.event_type for event in forecast.events
    }
    assert "ground_truth" not in forecast.model_dump_json()


def test_batch_adapter_returns_one_forecast_per_tracked_prediction() -> None:
    demo = build_demo_input("normal")
    snapshot, prediction, _planned = _snapshot_from_demo(demo)

    forecasts = build_world_model_forecasts(
        snapshot,
        {prediction.target_id: prediction},
    )

    assert tuple(forecasts) == (prediction.target_id,)
    assert forecasts[prediction.target_id].source_prediction_id == prediction.prediction_id


def _snapshot_from_demo(
    demo: object,
) -> tuple[
    SituationSnapshot,
    PredictedTrackRef,
    dict[str, tuple[tuple[float, float, float], ...]],
]:
    from underwater_tracking.world_model.models import RuleWorldModelInput

    assert isinstance(demo, RuleWorldModelInput)
    covariance = tuple(
        tuple(
            200.0
            if row == column and row < 2
            else 1.0
            if row == column
            else 0.0
            for column in range(5)
        )
        for row in range(5)
    )
    report = GroupReport(
        group_id="group_01",
        target_id=demo.target_id,
        sim_time_s=int(demo.as_of_s),
        member_ids=tuple(uuv.uuv_id for uuv in demo.uuvs),
        belief=TargetBelief(
            target_id=demo.target_id,
            sim_time_s=int(demo.as_of_s),
            mean=(0.0, 0.0, 8.0, 0.0, 0.0),
            covariance=covariance,
            model_probabilities={"cv": 1.7, "left_turn": 7.8, "right_turn": 0.5},
            source_observation_ids=("observation_01",),
        ),
        quality=GroupQuality(
            instant=0.85,
            window_mean=0.85,
            ewma=0.85,
            components={"fim": 0.85},
        ),
        plan_revision=1,
    )
    states = tuple(
        UUVState(
            uuv_id=uuv.uuv_id,
            position_xy=uuv.position_xy,
            heading_rad=0.0,
            speed_mps=0.0,
            energy_fraction=uuv.energy_fraction,
            status=UUVStatus.TRACKING,
            deployment_state=DeploymentState.DEPLOYED,
            group_id="group_01",
            capability=SurveillanceCapability(
                passive_range_m=uuv.passive_range_m,
                bearing_variance_rad2=uuv.bearing_variance_rad2,
            ),
        )
        for uuv in demo.uuvs
    )
    snapshot = SituationSnapshot(
        scenario_id=demo.scenario_id,
        snapshot_revision=10,
        sim_time_s=int(demo.as_of_s),
        uuvs=states,
        group_reports=(report,),
        pending_events=(),
        map_bounds_xy=demo.map_bounds_xy,
    )
    prediction = PredictedTrackRef(
        prediction_id=demo.trajectory.prediction_id,
        target_id=demo.target_id,
        sim_time_s=int(demo.as_of_s),
        horizon_s=1800.0,
        sample_step_s=30.0,
        times_s=demo.trajectory.times_s,
        points_xy=demo.trajectory.points_xy,
        corridor_radius_m=demo.trajectory.corridor_radius_m,
    )
    planned = {
        uuv.uuv_id: tuple(
            (time_s, point[0], point[1])
            for time_s, point in zip(uuv.planned_times_s, uuv.planned_points_xy)
        )
        for uuv in demo.uuvs
    }
    return snapshot, prediction, planned
