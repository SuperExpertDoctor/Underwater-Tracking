"""Fault injection for world-model source identity, expiry and read-only ownership."""

from types import SimpleNamespace
import pytest

from tests.world_model.test_rule_world_model import _snapshot_from_demo
from tests.runtime.test_execution_snapshot_factory import _inputs
from underwater_tracking.api.frame_builder import build_operational_frame
from underwater_tracking.config.models import TrackingPolicyConfig
from underwater_tracking.domain.models import (
    GroupReport,
    GroupQuality,
    TargetBelief,
    UUVState,
    DeploymentState,
)
from underwater_tracking.domain.execution_models import (
    OperationalExecutionSnapshot,
    GroupSensorMode,
    TaskGroupLifecycle,
)
from underwater_tracking.runtime.execution_snapshot_factory import build_execution_snapshot
from underwater_tracking.world_model.adapter import (
    build_world_model_forecasts,
    build_world_model_input,
    prediction_from_execution,
)
from underwater_tracking.world_model.demo import build_demo_input
from underwater_tracking.world_model.rules import predict_future_events


def test_missing_provenance_never_produces_success_or_events():
    value = build_demo_input("left_turn").model_copy(update={"source_track_revision": None})
    result = predict_future_events(value)
    assert result.data_status == "unavailable" and not result.events


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"source_track_revision": 999}, "revision_mismatch"),
        ({"valid_until_s": 0}, "expired"),
        ({"generated_at_s": 999999}, "generation_time_invalid"),
        ({"prediction_revision": None}, "provenance_missing"),
    ],
)
def test_raw_invalid_forecast_is_a_visible_unavailable_result(change, reason):
    snapshot, prediction, _ = _snapshot_from_demo(build_demo_input("left_turn"))
    result = build_world_model_forecasts(
        snapshot, {prediction.target_id: prediction.model_copy(update=change)}
    )[prediction.target_id]
    assert result.data_status in {"expired", "unavailable"} and not result.events
    assert reason in result.warnings[0]


def test_same_frame_rejects_a_different_prediction_id():
    snapshot, prediction, _planned = _snapshot_from_demo(build_demo_input("left_turn"))
    good = build_world_model_forecasts(snapshot, {prediction.target_id: prediction})
    assert good[prediction.target_id].events
    wrong = good[prediction.target_id].model_copy(update={"source_prediction_id": "wrong-id"})
    frame = build_operational_frame(
        snapshot,
        None,
        (),
        (),
        (),
        predictions={prediction.target_id: prediction},
        world_model_forecasts={prediction.target_id: wrong},
    )
    assert frame.target_estimates[0].world_model.data_status == "unavailable"
    assert not frame.target_estimates[0].world_model.events


def test_no_new_input_expires_instead_of_refreshing_event_validity():
    snapshot, prediction, _ = _snapshot_from_demo(build_demo_input("left_turn"))
    later = snapshot.model_copy(update={"sim_time_s": int(prediction.valid_until_s)})
    result = build_world_model_forecasts(later, {prediction.target_id: prediction})[
        prediction.target_id
    ]
    assert result.data_status == "expired" and not result.events


def test_final_publication_rejects_mixed_event_metadata():
    snapshot, prediction, _ = _snapshot_from_demo(build_demo_input("left_turn"))
    good = build_world_model_forecasts(snapshot, {prediction.target_id: prediction})[
        prediction.target_id
    ]
    wrong = good.model_copy(
        update={
            "events": tuple(
                event.model_copy(update={"source_track_revision": 999}) for event in good.events
            )
        }
    )
    frame = build_operational_frame(
        snapshot,
        None,
        (),
        (),
        (),
        predictions={prediction.target_id: prediction},
        world_model_forecasts={prediction.target_id: wrong},
    )
    assert frame.target_estimates[0].world_model.data_status == "unavailable"
    assert "world_model_event_provenance_mismatch" in frame.target_estimates[0].world_model.warnings
    assert not frame.target_estimates[0].world_model.events


def test_non_execution_prediction_exposes_provenance_and_expires():
    snapshot, prediction, _ = _snapshot_from_demo(build_demo_input("left_turn"))
    frame = build_operational_frame(
        snapshot, None, (), (), (), predictions={prediction.target_id: prediction}
    )
    view = frame.target_estimates[0].prediction
    assert view.source_track_revision == prediction.source_track_revision
    assert view.valid_until_s == prediction.valid_until_s
    expired = build_operational_frame(
        snapshot.model_copy(update={"sim_time_s": int(prediction.valid_until_s)}),
        None,
        (),
        (),
        (),
        predictions={prediction.target_id: prediction},
    )
    assert expired.target_estimates[0].prediction is None


def execution_fixture(owner_index=0):
    snapshot, track, accepted, baseline, intent, resources = _inputs()
    prediction = accepted.prediction.model_copy(
        update={
            "source_track_revision": 1,
            "prediction_revision": 7,
            "last_observed_at_s": 0.0,
            "generated_at_s": 0.0,
            "valid_until_s": 400.0,
        }
    )
    accepted = accepted.model_copy(update={"prediction": prediction})
    track = track.model_copy(
        update={
            "source_kind": "observed",
            "last_observed_at_s": 0.0,
            "valid_until_s": 400.0,
            "covariance_xy": (100.0, 0.0, 0.0, 100.0),
        }
    )
    execution = build_execution_snapshot(
        situation=snapshot,
        target_track=track,
        accepted_prediction=accepted,
        baseline=baseline,
        intent=intent,
        uuv_resources=resources,
        execution_revision=1,
        prediction_revision=7,
        tracking_policy=TrackingPolicyConfig(),
    )
    groups = tuple(
        group.model_copy(
            update={
                "ownership_status": "owner",
                "sensor_mode": GroupSensorMode.PASSIVE,
                "lifecycle": TaskGroupLifecycle.PASSIVE_TRACK,
            }
        )
        if i == owner_index
        else group
        for i, group in enumerate(execution.task_groups)
    )
    execution = OperationalExecutionSnapshot.model_validate(
        execution.model_copy(
            update={
                "task_groups": groups,
                "tracking_control": execution.tracking_control.model_copy(
                    update={"tracking_owner_group_id": groups[owner_index].group_instance_id}
                ),
            }
        ).model_dump()
    )
    report = GroupReport(
        group_id="observation_source_group",
        target_id="T1",
        sim_time_s=0,
        member_ids=groups[0].member_uuv_ids,
        plan_revision=1,
        belief=TargetBelief(
            target_id="T1",
            sim_time_s=0,
            mean=(0.0, 0.0, 1.0, 0.0, 0.0),
            covariance=((100.0, 0.0), (0.0, 100.0)),
            model_probabilities={"cv": 1.0},
            source_observation_ids=track.source_event_ids,
            track_revision=1,
            last_observed_at_s=0,
            valid_until_s=400,
        ),
        quality=GroupQuality(instant=0.8, window_mean=0.8, ewma=0.8, components={}),
    )
    uuvs = tuple(
        UUVState(
            uuv_id=resource.uuv_id,
            position_xy=(float(i * 100), 100.0),
            heading_rad=0.0,
            speed_mps=1.0,
            energy_fraction=1.0,
            status="track",
            deployment_state=DeploymentState.DEPLOYED,
        )
        for i, resource in enumerate(resources)
    )
    return snapshot.model_copy(update={"group_reports": (report,), "uuvs": uuvs}), execution


def test_owner_region_and_members_are_bound_to_execution_not_old_plan():
    snapshot, execution = execution_fixture(owner_index=1)
    old_plan = SimpleNamespace(waypoints_by_member={"old_member": ()}, revision=99)
    inputs = build_world_model_input(
        snapshot,
        prediction_from_execution(execution),
        execution_snapshot=execution,
        active_plan=old_plan,
    )
    owner = execution.task_groups[1]
    assert inputs.owner_group_id == owner.group_instance_id
    assert inputs.source_group_id == "observation_source_group"
    assert inputs.source_plan_revision == execution.execution_revision
    assert inputs.region_id == owner.region_id
    assert {u.uuv_id for u in inputs.uuvs} == set(owner.member_uuv_ids)
    assert inputs.task_region_bounds_xy != snapshot.map_bounds_xy
    assert all(not u.planned_times_s and u.state_time_s == snapshot.sim_time_s for u in inputs.uuvs)
    result = predict_future_events(inputs)
    assert result.events
    for event in result.events:
        assert event.owner_group_id == owner.group_instance_id
        assert event.source_track_revision == 1 and event.prediction_revision == 7
        assert event.valid_until_s == 400 and event.control_authority is False


def test_changing_owner_invalidates_old_cards_even_if_prediction_unchanged():
    snapshot, old = execution_fixture(0)
    _, new = execution_fixture(1)
    forecasts = build_world_model_forecasts(snapshot, {}, execution_snapshot=old)
    frame = build_operational_frame(
        snapshot, None, (), (), (), execution_snapshot=new, world_model_forecasts=forecasts
    )
    result = frame.target_estimates[0].world_model
    assert result.data_status == "unavailable" and not result.events
    assert "world_model_execution_context_mismatch" in result.warnings


def test_cached_prediction_age_uses_current_frame_clock():
    snapshot, execution = execution_fixture()
    snapshot = snapshot.model_copy(update={"sim_time_s": 200})
    frame = build_operational_frame(snapshot, None, (), (), (), execution_snapshot=execution)
    assert frame.target_estimates[0].prediction.health.source_track_age_s == 200
    assert frame.target_estimates[0].prediction.health.status == "degraded"
    expired = build_operational_frame(
        snapshot.model_copy(update={"sim_time_s": 400}),
        None,
        (),
        (),
        (),
        execution_snapshot=execution,
    )
    assert expired.target_estimates[0].prediction is None


def test_missing_report_has_no_fabricated_perfect_quality():
    snapshot, execution = execution_fixture()
    frame = build_operational_frame(
        snapshot.model_copy(update={"group_reports": ()}),
        None,
        (),
        (),
        (),
        execution_snapshot=execution,
    )
    assert frame.target_estimates[0].quality.quality_score is None
    assert frame.target_estimates[0].quality.estimated_rmse_m is None
