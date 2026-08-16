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
    TargetEstimateView,
    UUVView,
)
from underwater_tracking.domain.truth import TargetTruth
from underwater_tracking.domain.ui_models import OperationalFrame


def test_operational_frame_schema_contains_no_truth_fields():
    forbidden = {"truth", "true_position", "target_truth", "ground_truth"}
    schema_text = str(OperationalFrame.model_json_schema()).lower()
    assert all(name not in schema_text for name in forbidden)


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
        status="recovering",
        onboard_uuv_ids=("uuv_03",),
        deployed_uuv_ids=("uuv_01",),
        returning_uuv_ids=("uuv_02",),
    )
    estimate = TargetEstimateView(
        target_id="T1",
        mean=Point2D(x=310.0, y=390.0),
        covariance_ellipse=CovarianceEllipse(semimajor_m=25.0, semiminor_m=8.0, rotation_rad=0.3),
        intent=IntentView(label="transit", confidence=0.85),
        prediction=PredictionCorridorView(
            horizon_s=30.0,
            sample_step_s=1.0,
            centerline_xy=(Point2D(x=310.0, y=390.0), Point2D(x=340.0, y=380.0)),
            radius_m=(20.0, 24.0),
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
    assert restored.uuvs[0].status == "tracking"
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
