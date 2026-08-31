from underwater_tracking.domain.agent_models import TrajectoryDiffResult


def test_trajectory_diff_result_round_trips_with_auditable_thresholds() -> None:
    result = TrajectoryDiffResult(
        diff_id="S1:T1:prediction-diff:30:60",
        target_id="T1",
        previous_prediction_id="P1",
        current_prediction_id="P2",
        previous_sim_time_s=30,
        current_sim_time_s=60,
        status="comparable",
        reason=None,
        overlap_start_s=90.0,
        overlap_end_s=390.0,
        overlap_duration_s=300.0,
        comparison_step_s=30.0,
        sample_count=11,
        absolute_rms_m=300.0,
        normalized_rms=3.0,
        p90_distance_m=350.0,
        max_distance_m=400.0,
        max_distance_time_s=390.0,
        js_distance=0.25,
        previous_leading_model="cv",
        current_leading_model="left_turn",
        leading_model_changed=True,
        previous_evidence_ids=("O1",),
        current_evidence_ids=("O2",),
        normalized_threshold=2.45,
        absolute_floor_m=250.0,
        reset_normalized_threshold=1.75,
        reset_absolute_floor_m=150.0,
        threshold_schema_version="trajectory-diff-v1",
        confirmation_cycles=2,
        exceeded=True,
    )

    restored = TrajectoryDiffResult.model_validate_json(result.model_dump_json())

    assert restored == result
