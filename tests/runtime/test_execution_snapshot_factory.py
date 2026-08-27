from underwater_tracking.domain.agent_models import IntentHypothesis, PredictedTrackRef
from underwater_tracking.domain.execution_models import GlobalTargetTrackView, GlobalTrackSample
from underwater_tracking.domain.models import SituationSnapshot
from underwater_tracking.domain.mission_models import UUVResourceState
from underwater_tracking.runtime.execution_snapshot_factory import build_execution_snapshot


def test_execution_snapshot_accepts_llm_plan_source() -> None:
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
        prediction_regime="imm",
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

    snapshot = build_execution_snapshot(
        situation=situation,
        target_track=target_track,
        prediction=prediction,
        intent=intent,
        uuv_resources=[
            UUVResourceState(
                uuv_id=f"uuv_{index:02d}",
                mileage_m=0.0,
                energy_fraction=1.0,
                deployment_state="deployed",
            )
            for index in range(12)
        ],
        execution_revision=1,
        plan_source="llm_optimized",
    )

    assert snapshot.plan_source == "llm_optimized"
