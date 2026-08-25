from types import SimpleNamespace

from underwater_tracking.prediction.port import make_snapshot_predictor


def test_short_history_prediction_rebases_to_current_simulation_time() -> None:
    report = SimpleNamespace(
        target_id="target-01",
        belief=SimpleNamespace(
            mean=(0.0, 0.0, 2.0, 0.0),
            covariance=((100.0, 0.0), (0.0, 100.0)),
            source_observation_ids=("obs-01",),
            model_probabilities={"right_turn": 0.1, "cv": 0.8, "left_turn": 0.1},
        ),
    )
    snapshot = SimpleNamespace(
        scenario_id="scenario-01",
        snapshot_revision=9,
        sim_time_s=600,
        group_reports=(report,),
    )
    predictor = make_snapshot_predictor(
        belief_history=lambda _snapshot, _target_id: ((0, -200.0, 0.0), (100, 0.0, 0.0)),
        horizon_s=300.0,
        sample_step_s=30.0,
    )

    prediction = predictor(snapshot, "target-01")

    assert prediction.times_s[0] == 630.0
    assert prediction.times_s[-1] == 900.0
    assert prediction.points_xy[0][0] == 1060.0
    assert all(time_s > snapshot.sim_time_s for time_s in prediction.times_s)
    assert prediction.prediction_regime == "short_history"
    assert list(prediction.imm_model_probabilities) == ["cv", "left_turn", "right_turn"]
    assert prediction.imm_model_probabilities == {
        "cv": 0.8,
        "left_turn": 0.1,
        "right_turn": 0.1,
    }


def test_spline_prediction_carries_imm_metadata() -> None:
    report = SimpleNamespace(
        target_id="target-01",
        belief=SimpleNamespace(
            mean=(540.0, 0.0, 2.0, 0.0),
            covariance=((100.0, 0.0), (0.0, 100.0)),
            source_observation_ids=("obs-08", "obs-09"),
            model_probabilities={"left_turn": 0.2, "cv": 0.7, "right_turn": 0.1},
        ),
    )
    snapshot = SimpleNamespace(
        scenario_id="scenario-01",
        snapshot_revision=10,
        sim_time_s=300,
        group_reports=(report,),
    )
    predictor = make_snapshot_predictor(
        belief_history=lambda _snapshot, _target_id: tuple(
            (time_s, 2.0 * time_s, 0.0) for time_s in range(0, 271, 30)
        ),
        horizon_s=300.0,
        sample_step_s=30.0,
    )

    prediction = predictor(snapshot, "target-01")

    assert prediction.prediction_regime == "bspline"
    assert prediction.fallback_used is False
    assert prediction.source_belief_history_ids == ("obs-08", "obs-09")
    assert prediction.imm_model_probabilities == {
        "cv": 0.7,
        "left_turn": 0.2,
        "right_turn": 0.1,
    }


def test_global_trajectory_history_overrides_estimated_history_for_prediction() -> None:
    report = SimpleNamespace(
        target_id="target-01",
        belief=SimpleNamespace(
            mean=(0.0, 0.0, 0.0, 0.0),
            covariance=((100.0, 0.0), (0.0, 100.0)),
            source_observation_ids=("obs-01",),
            model_probabilities={},
        ),
    )
    snapshot = SimpleNamespace(
        scenario_id="scenario-01",
        snapshot_revision=1,
        sim_time_s=300,
        group_reports=(report,),
    )
    predictor = make_snapshot_predictor(
        belief_history=lambda _snapshot, _target_id: ((0, 0.0, 0.0),),
        global_trajectory_history=lambda _snapshot, _target_id: tuple(
            (time_s, 3.0 * time_s, 0.0) for time_s in range(0, 271, 30)
        ),
        horizon_s=300.0,
        sample_step_s=30.0,
    )

    prediction = predictor(snapshot, "target-01")

    assert prediction.prediction_regime == "bspline"
    assert prediction.points_xy[0][0] >= 900.0
