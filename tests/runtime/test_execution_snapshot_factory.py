import pytest

from underwater_tracking.domain.agent_models import IntentHypothesis, PredictedTrackRef
from underwater_tracking.domain.execution_models import (
    GroupSensorMode,
    GlobalTargetTrackView,
    GlobalTrackSample,
    OperationalExecutionSnapshot,
    TaskGroupInstance,
    TaskGroupLifecycle,
)
from underwater_tracking.domain.models import SituationSnapshot
from underwater_tracking.domain.mission_models import UUVResourceState
from underwater_tracking.domain.prediction_models import (
    AcceptedPrediction,
    PredictionHealth,
)
from underwater_tracking.planning.region_baseline import build_four_region_baseline
from underwater_tracking.runtime.execution_snapshot_factory import build_execution_snapshot
from underwater_tracking.runtime.task_group_instances import AlwaysAvailableTaskGroupFactory


def _inputs(
    prediction_regime: str = "imm",
    *,
    health_status: str | None = None,
):
    points = tuple((float(index * 1_000), 0.0) for index in range(61))
    prediction = PredictedTrackRef(
        prediction_id="prediction:T1:1",
        target_id="T1",
        sim_time_s=0,
        horizon_s=1_800.0,
        sample_step_s=30.0,
        times_s=tuple(float((index + 1) * 30) for index in range(60)),
        points_xy=points[1:],
        corridor_radius_m=tuple(100.0 for _ in range(60)),
        prediction_regime=prediction_regime,
        imm_model_probabilities={"CV": 0.6, "CT_LEFT": 0.2, "CT_RIGHT": 0.2},
    )
    intent = IntentHypothesis(
        label="transit",
        confidence=0.9,
        evidence_ids=("intent:T1",),
        model_id="LongCat-2.0",
        prompt_version="intent-v3",
    )
    target_track = GlobalTargetTrackView(
        target_id="T1",
        track_revision=1,
        sim_time_s=0,
        position_xy=(0.0, 0.0),
        velocity_xy=(1.0, 0.0),
        heading_rad=0.0,
        acceleration_xy=(0.0, 0.0),
        turn_rate_rad_s=0.0,
        bounded_history=(GlobalTrackSample(sim_time_s=0, position_xy=(0.0, 0.0)),),
        source_event_ids=("track:T1:0",),
    )
    situation = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=1,
        sim_time_s=0,
        uuvs=(),
        group_reports=(),
        pending_events=(),
        map_bounds_xy=(-10_000.0, 70_000.0, -10_000.0, 10_000.0),
    )
    selected_health_status = health_status or (
        "valid" if prediction_regime == "imm" else "degraded"
    )
    accepted = AcceptedPrediction(
        prediction=prediction,
        health=PredictionHealth(
            status=selected_health_status,
            regime=prediction_regime,
            reason_codes=(
                ()
                if selected_health_status == "valid"
                else ("imm_point_out_of_bounds",)
            ),
            source_track_age_s=0.0,
            clipped_point_fraction=0.0,
            maximum_radius_m=100.0,
            raw_prediction_id=prediction.prediction_id,
        ),
    )
    baseline = build_four_region_baseline(
        accepted,
        target_id="T1",
        execution_revision=1,
        origin_sim_time_s=float(situation.sim_time_s),
        map_bounds_xy=situation.map_bounds_xy,
    )
    resources = [
        UUVResourceState(
            uuv_id=f"uuv_{index:02d}",
            mileage_m=0.0,
            energy_fraction=1.0,
            deployment_state="deployed",
        )
        for index in range(12)
    ]
    return situation, target_track, accepted, baseline, intent, resources


def _build_snapshot(prediction_regime: str = "imm") -> OperationalExecutionSnapshot:
    situation, target_track, accepted, baseline, intent, resources = _inputs(
        prediction_regime
    )

    return build_execution_snapshot(
        situation=situation,
        target_track=target_track,
        accepted_prediction=accepted,
        baseline=baseline,
        intent=intent,
        uuv_resources=resources,
        execution_revision=1,
        prediction_revision=7,
        plan_source="llm_optimized",
    )


def test_execution_snapshot_keeps_imm_band_separate_from_bspline_centerline() -> None:
    situation, target_track, accepted, baseline, intent, resources = _inputs()
    prediction = accepted.prediction
    assert prediction is not None
    imm_points = prediction.points_xy
    bspline_points = tuple(
        (float(index * 1_000), 100.0 + float(index))
        for index in range(len(imm_points))
    )
    separated = prediction.model_copy(
        update={
            "points_xy": bspline_points,
            "corridor_radius_m": tuple(900.0 for _ in imm_points),
            "imm_times_s": prediction.times_s,
            "imm_centerline_xy": imm_points,
            "imm_corridor_radius_m": tuple(100.0 for _ in imm_points),
            "bspline_times_s": prediction.times_s,
            "bspline_centerline_xy": bspline_points,
        }
    )
    snapshot = build_execution_snapshot(
        situation=situation,
        target_track=target_track,
        accepted_prediction=accepted.model_copy(update={"prediction": separated}),
        baseline=baseline,
        intent=intent,
        uuv_resources=resources,
        execution_revision=1,
        prediction_revision=7,
        plan_source="llm_optimized",
    )

    assert snapshot.prediction.centerline_xy == imm_points
    assert snapshot.prediction.corridor_radius_m == (100.0,) * len(imm_points)
    assert snapshot.prediction.bspline_centerline_xy == bspline_points



def test_execution_snapshot_accepts_llm_plan_source() -> None:
    snapshot = _build_snapshot()

    assert snapshot.plan_source == "llm_optimized"


@pytest.mark.parametrize(
    "regime", ("imm", "bspline", "short_history", "boundary_recovery")
)
def test_execution_snapshot_preserves_prediction_regime(regime: str) -> None:
    snapshot = _build_snapshot(regime)

    assert snapshot.prediction.prediction_regime == regime


def test_execution_snapshot_uses_accepted_baseline_and_fixed_freshness_window() -> None:
    situation, target_track, accepted, baseline, intent, resources = _inputs()

    snapshot = build_execution_snapshot(
        situation=situation,
        target_track=target_track,
        accepted_prediction=accepted,
        baseline=baseline,
        intent=intent,
        uuv_resources=resources,
        execution_revision=1,
        prediction_revision=7,
    )

    assert snapshot.valid_from_s == 0.0
    assert snapshot.valid_until_s == 450.0
    assert snapshot.prediction_revision == 7
    assert snapshot.prediction.prediction_revision == 7
    assert tuple(region.geometry for region in snapshot.regions) == tuple(
        region.geometry for region in baseline.regions
    )
    assert all(region.prediction_id == snapshot.prediction_id for region in snapshot.regions)
    members = tuple(
        member for group in snapshot.task_groups for member in group.member_uuv_ids
    )
    assert len(members) == 8
    assert len(set(members)) == 8
    assert len(snapshot.reserve_uuvs) == 4
    assert not set(members) & {reserve.uuv_id for reserve in snapshot.reserve_uuvs}
    assert "prediction_revision:7" in snapshot.evidence_ids
    assert all("prediction_revision:7" in region.evidence_ids for region in snapshot.regions)
    assert all("prediction_revision:7" in group.evidence_ids for group in snapshot.task_groups)


def test_uuv_execution_snapshot_creates_four_entering_three_member_groups() -> None:
    situation, target_track, accepted, baseline, intent, resources = _inputs()

    snapshot = build_execution_snapshot(
        situation=situation,
        target_track=target_track,
        accepted_prediction=accepted,
        baseline=baseline,
        intent=intent,
        uuv_resources=resources,
        execution_revision=1,
        tracking_policy="uuv_only",
        instance_factory=AlwaysAvailableTaskGroupFactory(scenario_id="S1"),
    )

    assert len(snapshot.task_groups) == 4
    assert all(isinstance(group, TaskGroupInstance) for group in snapshot.task_groups)
    assert all(len(group.member_uuv_ids) == 3 for group in snapshot.task_groups)
    assert all(group.lifecycle is TaskGroupLifecycle.ENTERING for group in snapshot.task_groups)
    assert all(group.sensor_mode is GroupSensorMode.ACTIVE for group in snapshot.task_groups)
    assert tuple(
        member
        for group in snapshot.task_groups
        for member in group.member_uuv_ids
    ) == tuple(f"uuv_{index:02d}" for index in range(12))
    assert snapshot.reserve_uuvs == ()
    assert snapshot.tracking_policy == "uuv_only"


def test_execution_snapshot_records_baseline_mode_and_prediction_health_reasons() -> None:
    snapshot = _build_snapshot("bspline")

    assert snapshot.degradation.status == "degraded"
    assert "region_generation_mode:degraded_prediction" in snapshot.degradation.reasons
    assert "imm_point_out_of_bounds" in snapshot.degradation.reasons
    assert "prediction_health:imm_point_out_of_bounds" in snapshot.evidence_ids


def test_execution_snapshot_rejects_unavailable_prediction() -> None:
    situation, target_track, accepted, baseline, intent, resources = _inputs()
    unavailable = AcceptedPrediction(
        prediction=None,
        health=accepted.health.model_copy(
            update={"status": "unavailable", "reason_codes": ("all_fallbacks_failed",)}
        ),
    )

    with pytest.raises(ValueError, match="unavailable prediction"):
        build_execution_snapshot(
            situation=situation,
            target_track=target_track,
            accepted_prediction=unavailable,
            baseline=baseline,
            intent=intent,
            uuv_resources=resources,
            execution_revision=1,
        )


def test_execution_snapshot_uses_prediction_sim_time_when_revision_not_provided() -> None:
    situation, target_track, accepted, baseline, intent, resources = _inputs()
    accepted = accepted.model_copy(
        update={
            "prediction": accepted.prediction.model_copy(update={"sim_time_s": 33})
        }
    )

    snapshot = build_execution_snapshot(
        situation=situation,
        target_track=target_track,
        accepted_prediction=accepted,
        baseline=baseline,
        intent=intent,
        uuv_resources=resources,
        execution_revision=1,
    )

    assert snapshot.prediction_revision == 33
    assert snapshot.prediction.prediction_revision == 33
