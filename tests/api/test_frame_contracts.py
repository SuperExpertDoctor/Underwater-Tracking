import json

import pytest
from pydantic import ValidationError

from underwater_tracking.domain import (
    BearingRayView,
    CarrierView,
    CovarianceEllipse,
    EstimateQualityView,
    EvaluationFrame,
    EventView,
    GroupQualityView,
    GroupView,
    IntentView,
    LedgerView,
    MapBounds,
    MetricView,
    PlanView,
    Point2D,
    PredictionCorridorView,
    RegionalMissionView,
    TargetEstimateView,
    UUVView,
)
from underwater_tracking.domain.truth import TargetTruth
from underwater_tracking.domain.ui_models import OperationalFrame


def test_operational_frame_schema_contains_no_truth_fields():
    forbidden = {"truth", "true_position", "target_truth", "ground_truth"}
    schema_text = str(OperationalFrame.model_json_schema()).lower()
    assert all(name not in schema_text for name in forbidden)


def test_regional_mission_view_derives_square_corners_from_json_geometry():
    view = RegionalMissionView.model_validate_json(
        json.dumps(
            {
                "region_id": "T1:region:1",
                "target_id": "T1",
                "geometry": [
                    {"x": 0.0, "y": 0.0},
                    {"x": 4.0, "y": 0.0},
                    {"x": 4.0, "y": 4.0},
                    {"x": 0.0, "y": 4.0},
                ],
                "entry_s": 10,
                "exit_s": 20,
                "lifecycle": "PLANNED",
                "coverage": 0.0,
                "tracking_quality": 0.0,
            }
        )
    )

    assert view.top_left_xy == Point2D(x=0.0, y=4.0)
    assert view.bottom_right_xy == Point2D(x=4.0, y=0.0)


def _full_frame(*, plan_version: int = 4) -> OperationalFrame:
    plan = PlanView(
        plan_id="plan-7",
        version=4,
        status="active",
        concept="balanced",
        reason="improve T1 FIM coverage",
        affected_targets=("T1",),
        group_changes=("G1 adds UUV-2",),
        valid_from_s=10,
        valid_until_s=60,
    )
    uuv = UUVView(
        uuv_id="UUV-1",
        status="tracking",
        position=Point2D(x=100.0, y=200.0),
        heading_rad=0.5,
        speed_mps=2.0,
        energy_fraction=0.8,
        group_id="G1",
        current_waypoint=Point2D(x=300.0, y=400.0),
        breadcrumb=(Point2D(x=90.0, y=190.0), Point2D(x=100.0, y=200.0)),
    )
    carrier = CarrierView(
        carrier_id="carrier-01",
        position=Point2D(x=-3000.0, y=-3000.0),
        heading_rad=0.25,
        speed_mps=1.5,
        status="transit",
        deployed_uuv_ids=("UUV-1",),
    )
    estimate = TargetEstimateView(
        target_id="T1",
        mean=Point2D(x=310.0, y=390.0),
        covariance_ellipse=CovarianceEllipse(semimajor_m=25.0, semiminor_m=8.0, rotation_rad=0.3),
        intent=IntentView(label="transit", confidence=0.85),
        prediction=PredictionCorridorView(
            prediction_id="prediction:T1:4",
            prediction_revision=4,
            origin_sim_time_s=20.0,
            health={
                "status": "valid",
                "regime": "imm",
                "reason_codes": (),
                "source_track_age_s": 0.0,
                "clipped_point_fraction": 0.0,
                "maximum_radius_m": 24.0,
                "raw_prediction_id": "prediction:T1:4",
            },
            horizon_s=30.0,
            sample_step_s=1.0,
            centerline_xy=(Point2D(x=310.0, y=390.0), Point2D(x=340.0, y=380.0)),
            radius_m=(20.0, 24.0),
            point_confidence=(0.9, 0.7),
        ),
        quality=EstimateQualityView(
            quality_score=0.9,
            estimated_rmse_m=12.5,
            fim_min_eigenvalue=0.4,
            fim_condition=3.0,
        ),
    )
    ray = BearingRayView(
        observation_id="obs-1",
        uuv_id="UUV-1",
        target_id="T1",
        origin=Point2D(x=100.0, y=200.0),
        azimuth_rad=0.7,
        variance_rad2=0.02,
        confidence=0.9,
    )
    group = GroupView(
        group_id="G1",
        target_id="T1",
        member_ids=("UUV-1", "UUV-2"),
        quality=GroupQualityView(instant=0.92, window_mean=0.9, ewma=0.91, components={"fim": 0.95}),
    )
    event = EventView(
        event_id="evt-1",
        sim_time_s=20,
        event_type="plan_commit",
        level="tactical",
        entity_id="plan-7",
        message="plan committed",
    )
    ledger_row = LedgerView(
        decision_id="dec-1",
        sim_time_s=20,
        outcome="committed",
        trigger_event_ids=("evt-1",),
        evidence_ids=("obs-1",),
        final_plan_id="plan-7",
        final_plan_version=4,
    )
    metric = MetricView(
        metric_id="active_uuv_hours",
        label="Active UUV-hours",
        value=3.5,
        unit="h",
        threshold=4.0,
        window_s=60,
        series=(2.0, 2.5, 3.0, 3.5),
    )
    return OperationalFrame(
        schema_version="1.0",
        frame_id=1,
        sim_time_s=20,
        plan_version=plan_version,
        map_bounds=MapBounds(min_x=0.0, min_y=0.0, max_x=1000.0, max_y=1000.0),
        carrier=carrier,
        uuvs=(uuv,),
        target_estimates=(estimate,),
        bearing_rays=(ray,),
        groups=(group,),
        events=(event,),
        plans=(plan,),
        ledger=(ledger_row,),
        metrics=(metric,),
    )


def test_operational_frame_valid_round_trip():
    frame = _full_frame()
    restored = frame.model_validate_json(frame.model_dump_json())
    assert restored == frame
    assert restored.frame_id == 1
    assert restored.sim_time_s == 20
    assert restored.plan_version == 4
    assert restored.uuvs[0].status == "track"
    assert restored.target_estimates[0].covariance_ellipse.semimajor_m == 25.0
    assert restored.events[0].level == "tactical"


def test_unknown_fields_are_rejected():
    payload = _full_frame().model_dump()
    with pytest.raises(ValidationError):
        OperationalFrame.model_validate({**payload, "phantom": True})
    with pytest.raises(ValidationError):
        UUVView(
            uuv_id="UUV-1",
            status="tracking",
            position=Point2D(x=0.0, y=0.0),
            heading_rad=0.0,
            speed_mps=1.0,
            energy_fraction=0.5,
            energy_percent=50.0,
        )


def test_plan_version_mismatch_is_rejected():
    with pytest.raises(ValidationError):
        _full_frame(plan_version=5)
    frame = _full_frame(plan_version=4)
    assert frame.plan_version == 4


def test_operational_frame_rejects_unknown_or_failed_carrier_members():
    payload = _full_frame().model_dump()
    payload["carrier"]["deployed_uuv_ids"] = ["UUV-99"]
    with pytest.raises(ValidationError, match="unknown UUV"):
        OperationalFrame.model_validate(payload)

    payload = _full_frame().model_dump()
    payload["uuvs"][0]["deployment_state"] = "failed"
    payload["uuvs"][0]["status"] = "failed"
    with pytest.raises(ValidationError, match="failed UUV"):
        OperationalFrame.model_validate(payload)


def test_legacy_frame_normalizes_missing_carrier_relationships():
    payload = _full_frame().model_dump()
    payload["uuvs"][0].pop("deployment_state")
    payload["carrier"].pop("onboard_uuv_ids")
    payload["carrier"].pop("deployed_uuv_ids")
    payload["carrier"].pop("returning_uuv_ids")
    payload["carrier"].pop("status")
    restored = OperationalFrame.model_validate(payload)
    assert restored.uuvs[0].deployment_state == "deployed"
    assert restored.carrier is not None
    assert restored.carrier.deployed_uuv_ids == ("UUV-1",)


def test_carrier_view_rejects_overlapping_relationships():
    with pytest.raises(ValidationError, match="carrier relationship lists must be disjoint"):
        CarrierView(
            carrier_id="carrier-01",
            position=Point2D(x=0.0, y=0.0),
            heading_rad=0.0,
            speed_mps=1.0,
            onboard_uuv_ids=("UUV-1",),
            deployed_uuv_ids=("UUV-1",),
        )


def test_carrier_view_rejects_status_contradicting_its_relationships() -> None:
    with pytest.raises(ValidationError, match="returning UUVs require recovering status"):
        CarrierView(
            carrier_id="carrier-01",
            position=Point2D(x=0.0, y=0.0),
            heading_rad=0.0,
            speed_mps=1.0,
            status="transit",
            returning_uuv_ids=("UUV-1",),
        )
    with pytest.raises(ValidationError, match="duplicate IDs"):
        CarrierView(
            carrier_id="carrier-01",
            position=Point2D(x=0.0, y=0.0),
            heading_rad=0.0,
            speed_mps=1.0,
            deployed_uuv_ids=("UUV-1", "UUV-1"),
        )


def test_operational_frame_normalizes_typed_old_carrier_and_rejects_duplicates():
    uuv = UUVView(
        uuv_id="UUV-1",
        status="tracking",
        position=Point2D(x=100.0, y=200.0),
        heading_rad=0.5,
        speed_mps=2.0,
        energy_fraction=0.8,
    )
    frame = OperationalFrame.model_validate(
        {**_full_frame().model_dump(), "uuvs": (uuv,), "carrier": CarrierView(
            carrier_id="carrier-01",
            position=Point2D(x=-3000.0, y=-3000.0),
            heading_rad=0.25,
            speed_mps=1.5,
        )}
    )
    assert frame.carrier is not None
    assert frame.carrier.deployed_uuv_ids == ("UUV-1",)

    duplicate_carrier = CarrierView.model_construct(
        carrier_id="carrier-01",
        position=Point2D(x=-3000.0, y=-3000.0),
        heading_rad=0.25,
        speed_mps=1.5,
        onboard_uuv_ids=(),
        deployed_uuv_ids=("UUV-1", "UUV-1"),
        returning_uuv_ids=(),
    )
    with pytest.raises(ValidationError, match="duplicate IDs"):
        OperationalFrame.model_validate(
            {**_full_frame().model_dump(), "carrier": duplicate_carrier}
        )


def test_legacy_returning_frame_normalizes_missing_deployment_state():
    payload = _full_frame().model_dump()
    payload["uuvs"][0]["status"] = "returning"
    payload["uuvs"][0].pop("deployment_state")
    payload["carrier"].pop("onboard_uuv_ids")
    payload["carrier"].pop("deployed_uuv_ids")
    payload["carrier"].pop("returning_uuv_ids")
    payload["carrier"].pop("status")
    frame = OperationalFrame.model_validate(payload)
    assert frame.uuvs[0].deployment_state == "returning"
    assert frame.carrier is not None
    assert frame.carrier.returning_uuv_ids == ("UUV-1",)


def test_operational_frame_json_contains_no_truth_fields():
    forbidden = {"truth", "true_position", "target_truth", "ground_truth"}
    serialized = str(_full_frame().model_dump_json()).lower()
    assert all(name not in serialized for name in forbidden)


def test_evaluation_frame_pairs_truth_with_run_metadata_and_stands_alone():
    truth = TargetTruth(
        target_id="T1",
        position_xy=(310.0, 390.0),
        velocity_xy=(1.5, -0.5),
        intent_label="transit",
    )
    frame = EvaluationFrame(
        schema_version="1.0",
        frame_id=1,
        sim_time_s=20,
        scenario_id="scenario-20260814",
        run_id="run-3",
        plan_version=4,
        targets=(truth,),
    )
    assert not issubclass(EvaluationFrame, OperationalFrame)
    restored = frame.model_validate_json(frame.model_dump_json())
    assert restored == frame
    assert restored.scenario_id == "scenario-20260814"
    assert restored.run_id == "run-3"
    assert restored.plan_version == 4
    assert restored.targets[0] == truth


def test_uuv_view_carries_sensor_mode_and_reservation_state():
    frame = _full_frame()
    assert frame.uuvs[0].sensor_mode == "passive"
    assert frame.uuvs[0].reserved is False
    active = UUVView(
        uuv_id="UUV-2",
        status="tracking",
        position=Point2D(x=500.0, y=600.0),
        heading_rad=0.2,
        speed_mps=2.0,
        energy_fraction=0.6,
        sensor_mode="active",
        reserved=True,
    )
    assert active.sensor_mode == "active"
    assert active.reserved is True


def test_target_estimate_carries_classification_and_ping_recency():
    estimate = TargetEstimateView(
        target_id="T1",
        mean=Point2D(x=310.0, y=390.0),
        covariance_ellipse=CovarianceEllipse(
            semimajor_m=25.0, semiminor_m=8.0, rotation_rad=0.3
        ),
        intent=IntentView(label="transit", confidence=0.85),
        quality=EstimateQualityView(
            quality_score=0.9,
            estimated_rmse_m=12.5,
            fim_min_eigenvalue=0.4,
            fim_condition=3.0,
        ),
        classification="submarine",
        last_ping_s=15,
    )
    assert estimate.classification == "submarine"
    assert estimate.last_ping_s == 15
    assert _full_frame().target_estimates[0].classification == "unknown"
    assert _full_frame().target_estimates[0].last_ping_s is None


def test_plan_view_carries_the_segmented_relay_plan():
    plan = PlanView(
        plan_id="plan-7",
        version=4,
        status="active",
        segment_plan=("relay:G-T1:0-300", "relay:G-T2:300-600"),
    )
    assert plan.segment_plan == ("relay:G-T1:0-300", "relay:G-T2:300-600")
    assert _full_frame().plans[0].segment_plan == ()
