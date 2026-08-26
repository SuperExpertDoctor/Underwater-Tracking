from inspect import signature
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


def test_imm_prediction_uses_model_probabilities_in_future_centerline() -> None:
    report = SimpleNamespace(
        target_id="target-01",
        belief=SimpleNamespace(
            mean=(540.0, 0.0, 2.0, 0.0),
            covariance=((100.0, 0.0), (0.0, 100.0)),
            source_observation_ids=("obs-08", "obs-09"),
            model_probabilities={"left_turn": 0.8, "cv": 0.15, "right_turn": 0.05},
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

    assert prediction.prediction_regime == "imm"
    assert prediction.fallback_used is False
    assert prediction.points_xy[-1][1] > 0.0
    assert prediction.source_belief_history_ids == ("obs-08", "obs-09")
    assert prediction.imm_model_probabilities == {
        "cv": 0.15,
        "left_turn": 0.8,
        "right_turn": 0.05,
    }

    right_report = SimpleNamespace(
        target_id="target-01",
        belief=SimpleNamespace(
            mean=report.belief.mean,
            covariance=report.belief.covariance,
            source_observation_ids=report.belief.source_observation_ids,
            model_probabilities={"left_turn": 0.05, "cv": 0.15, "right_turn": 0.8},
        ),
    )
    right_prediction = predictor(
        SimpleNamespace(**{**snapshot.__dict__, "group_reports": (right_report,)}),
        "target-01",
    )
    assert right_prediction.points_xy[-1][1] < 0.0


def test_imm_prediction_exposes_branch_states_and_mixed_covariance() -> None:
    report = SimpleNamespace(
        target_id="target-01",
        belief=SimpleNamespace(
            mean=(0.0, 0.0, 2.0, 0.0, 0.0),
            covariance=((10.0, 0.0), (0.0, 10.0)),
            source_observation_ids=("obs-01",),
            model_probabilities={"cv": 0.6, "left_turn": 0.3, "right_turn": 0.1},
        ),
    )
    snapshot = SimpleNamespace(
        scenario_id="scenario-01",
        snapshot_revision=11,
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

    assert tuple(state.model_name for state in prediction.imm_model_states) == (
        "CV",
        "CT_LEFT",
        "CT_RIGHT",
    )
    assert len(prediction.imm_covariance_xy) == len(prediction.times_s)
    assert prediction.imm_clipping_records == ()
    assert prediction.imm_model_states[0].source_observation_ids == ("obs-01",)


def test_predictor_exposes_no_simulator_truth_history_port() -> None:
    assert "global_trajectory_history" not in signature(make_snapshot_predictor).parameters
