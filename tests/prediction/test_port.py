from inspect import signature
from types import SimpleNamespace

import pytest

from underwater_tracking.config.models import PredictionHealthConfig
from underwater_tracking.domain.agent_models import PredictedTrackRef
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
        map_bounds_xy=(-10_000.0, 10_000.0, -10_000.0, 10_000.0),
    )
    predictor = make_snapshot_predictor(
        belief_history=lambda _snapshot, _target_id: ((0, -200.0, 0.0), (100, 0.0, 0.0)),
        horizon_s=300.0,
        sample_step_s=30.0,
    )

    accepted = predictor(snapshot, "target-01")

    assert accepted.health.status == "degraded"
    assert accepted.health.source_track_age_s == 500.0
    prediction = accepted.prediction
    assert prediction is not None
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
        map_bounds_xy=(-1_000_000_000.0, 1_000_000_000.0, -1_000_000_000.0, 1_000_000_000.0),
    )
    predictor = make_snapshot_predictor(
        belief_history=lambda _snapshot, _target_id: tuple(
            (time_s, 2.0 * time_s, 0.0) for time_s in range(0, 271, 30)
        ),
        horizon_s=300.0,
        sample_step_s=30.0,
        max_turn_rate_rad_s=1.0,
        health_config=_permissive_imm_health_config(),
    )

    accepted = predictor(snapshot, "target-01")

    assert accepted.health.status == "valid", accepted.health
    prediction = accepted.prediction
    assert prediction is not None
    assert accepted.health.status == "valid", accepted.health
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
    right_accepted = predictor(
        SimpleNamespace(**{**snapshot.__dict__, "group_reports": (right_report,)}),
        "target-01",
    )
    right_prediction = right_accepted.prediction
    assert right_prediction is not None
    assert right_prediction.points_xy[-1][1] < 0.0


def test_valid_imm_prediction_also_carries_independent_bspline_centerline() -> None:
    imm = _candidate(
        "raw-imm-id",
        "imm",
        points_xy=((0.0, 0.0), (100.0, 0.0)),
    )
    bspline = _candidate(
        "raw-bspline-id",
        "bspline",
        points_xy=((0.0, 10.0), (100.0, 50.0)),
    )
    predictor = make_snapshot_predictor(
        belief_history=lambda _snapshot, _target_id: ((0, -200.0, 0.0), (100, 0.0, 0.0)),
        horizon_s=60.0,
        sample_step_s=30.0,
        health_config=_health_config(),
        imm_forecaster=lambda _context: imm,
        bspline_forecaster=lambda _context: bspline,
    )

    accepted = predictor(_snapshot_with_track_history(), "target-01")

    assert accepted.health.regime == "imm"
    assert accepted.prediction is not None
    assert accepted.prediction.imm_centerline_xy == imm.points_xy
    assert accepted.prediction.imm_corridor_radius_m == imm.corridor_radius_m
    assert accepted.prediction.bspline_centerline_xy == bspline.points_xy

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
        map_bounds_xy=(-1_000_000_000.0, 1_000_000_000.0, -1_000_000_000.0, 1_000_000_000.0),
    )
    predictor = make_snapshot_predictor(
        belief_history=lambda _snapshot, _target_id: tuple(
            (time_s, 2.0 * time_s, 0.0) for time_s in range(0, 271, 30)
        ),
        horizon_s=300.0,
        sample_step_s=30.0,
        max_turn_rate_rad_s=1.0,
        health_config=_permissive_imm_health_config(),
    )

    accepted = predictor(snapshot, "target-01")

    prediction = accepted.prediction
    assert prediction is not None
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


def test_predictor_does_not_accept_global_target_truth_history() -> None:
    assert "global_trajectory_history" not in signature(
        make_snapshot_predictor
    ).parameters


def _health_config() -> PredictionHealthConfig:
    return PredictionHealthConfig(
        max_corridor_radius_m=1_000.0,
        max_corridor_map_fraction=0.5,
        minimum_point_confidence=0.02,
    )


def _permissive_imm_health_config() -> PredictionHealthConfig:
    return PredictionHealthConfig(
        max_corridor_radius_m=1_000_000_000.0,
        max_corridor_map_fraction=1.0,
        minimum_point_confidence=0.0,
    )


def _candidate(
    prediction_id: str,
    regime: str,
    *,
    points_xy: tuple[tuple[float, float], ...] = ((0.0, 0.0), (100.0, 0.0)),
) -> PredictedTrackRef:
    return PredictedTrackRef(
        prediction_id=prediction_id,
        target_id="target-01",
        sim_time_s=100,
        horizon_s=60.0,
        sample_step_s=30.0,
        times_s=(130.0, 160.0),
        points_xy=points_xy,
        corridor_radius_m=(100.0, 200.0),
        prediction_regime=regime,
        imm_covariance_xy=(
            ((10.0, 0.0, 0.0, 10.0), (20.0, 0.0, 0.0, 20.0))
            if regime == "imm"
            else ()
        ),
    )


def _snapshot_with_track_history() -> SimpleNamespace:
    report = SimpleNamespace(
        target_id="target-01",
        belief=SimpleNamespace(
            mean=(0.0, 0.0, 2.0, 0.0),
            covariance=((100.0, 0.0), (0.0, 100.0)),
            source_observation_ids=("raw-observation-9",),
            model_probabilities={"cv": 1.0},
        ),
    )
    return SimpleNamespace(
        scenario_id="scenario-01",
        snapshot_revision=12,
        sim_time_s=100,
        group_reports=(report,),
        map_bounds_xy=(-1_000.0, 1_000.0, -1_000.0, 1_000.0),
    )


def test_predictor_falls_back_from_invalid_imm_to_bounded_bspline() -> None:
    invalid_imm = _candidate(
        "raw-imm-id",
        "imm",
        points_xy=((2_001.0, 0.0), (2_002.0, 0.0)),
    )
    valid_bspline = _candidate("raw-bspline-id", "bspline")

    predictor = make_snapshot_predictor(
        belief_history=lambda _snapshot, _target_id: ((0, -200.0, 0.0), (100, 0.0, 0.0)),
        horizon_s=60.0,
        sample_step_s=30.0,
        health_config=_health_config(),
        imm_forecaster=lambda _context: invalid_imm,
        bspline_forecaster=lambda _context: valid_bspline,
        short_history_forecaster=lambda _context: pytest.fail(
            "short history must not run after a valid B-spline candidate"
        ),
    )

    accepted = predictor(_snapshot_with_track_history(), "target-01")

    assert accepted.health.status == "degraded"
    assert accepted.health.regime == "bspline"
    assert accepted.health.reason_codes == ("imm_point_out_of_bounds",)
    assert accepted.health.raw_prediction_id == "raw-bspline-id"
    assert accepted.prediction is not None
    assert accepted.prediction.prediction_id == "raw-bspline-id"
    assert accepted.prediction.point_confidence == pytest.approx((1.0, 0.25))
    assert invalid_imm.points_xy == ((2_001.0, 0.0), (2_002.0, 0.0))


def test_predictor_falls_back_from_imm_without_covariance_to_bspline() -> None:
    invalid_imm = _candidate("raw-imm-id", "imm").model_copy(
        update={"imm_covariance_xy": ()}
    )
    valid_bspline = _candidate("raw-bspline-id", "bspline")

    predictor = make_snapshot_predictor(
        belief_history=lambda _snapshot, _target_id: ((0, -200.0, 0.0), (100, 0.0, 0.0)),
        horizon_s=60.0,
        sample_step_s=30.0,
        health_config=_health_config(),
        imm_forecaster=lambda _context: invalid_imm,
        bspline_forecaster=lambda _context: valid_bspline,
        short_history_forecaster=lambda _context: pytest.fail(
            "short history must not run after a valid B-spline candidate"
        ),
    )

    accepted = predictor(_snapshot_with_track_history(), "target-01")

    assert accepted.health.status == "degraded"
    assert accepted.health.regime == "bspline"
    assert accepted.health.reason_codes == ("imm_covariance_missing",)
    assert accepted.health.raw_prediction_id == "raw-bspline-id"
    assert accepted.prediction is not None
    assert accepted.prediction.prediction_id == "raw-bspline-id"
    assert accepted.prediction.point_confidence == pytest.approx((1.0, 0.25))
    assert invalid_imm.imm_covariance_xy == ()


def test_predictor_assesses_candidates_using_execution_stage_provenance() -> None:
    spoofed_imm = _candidate("raw-imm-id", "bspline")
    spoofed_bspline = _candidate("raw-bspline-id", "imm")

    predictor = make_snapshot_predictor(
        belief_history=lambda _snapshot, _target_id: ((0, -200.0, 0.0), (100, 0.0, 0.0)),
        horizon_s=60.0,
        sample_step_s=30.0,
        health_config=_health_config(),
        imm_forecaster=lambda _context: spoofed_imm,
        bspline_forecaster=lambda _context: spoofed_bspline,
        short_history_forecaster=lambda _context: pytest.fail(
            "short history must not run after a valid B-spline candidate"
        ),
    )

    accepted = predictor(_snapshot_with_track_history(), "target-01")

    assert accepted.health.status == "degraded"
    assert accepted.health.regime == "bspline"
    assert accepted.health.reason_codes == ("imm_covariance_missing",)
    assert accepted.health.raw_prediction_id == "raw-bspline-id"
    assert accepted.prediction is not None
    assert accepted.prediction.prediction_id == "raw-bspline-id"
    assert accepted.prediction.prediction_regime == "bspline"
    assert accepted.prediction.point_confidence == pytest.approx((1.0, 0.25))
    assert spoofed_imm.prediction_regime == "bspline"
    assert spoofed_bspline.prediction_regime == "imm"


def test_predictor_uses_the_exact_bounded_fallback_order() -> None:
    calls: list[str] = []

    def invalid(stage: str, regime: str):
        def forecast(_context: object) -> PredictedTrackRef:
            calls.append(stage)
            return _candidate(
                f"raw-{stage}-id",
                regime,
                points_xy=((2_001.0, 0.0), (2_002.0, 0.0)),
            )

        return forecast

    def valid_boundary(_context: object) -> PredictedTrackRef:
        calls.append("boundary_recovery")
        return _candidate("raw-boundary-id", "boundary_recovery")

    predictor = make_snapshot_predictor(
        belief_history=lambda _snapshot, _target_id: (),
        horizon_s=60.0,
        sample_step_s=30.0,
        health_config=_health_config(),
        imm_forecaster=invalid("imm", "imm"),
        bspline_forecaster=invalid("bspline", "bspline"),
        short_history_forecaster=invalid("short_history", "short_history"),
        boundary_recovery_forecaster=valid_boundary,
    )

    accepted = predictor(_snapshot_with_track_history(), "target-01")

    assert calls == ["imm", "bspline", "short_history", "boundary_recovery"]
    assert accepted.health.status == "degraded"
    assert accepted.health.regime == "boundary_recovery"
    assert accepted.health.reason_codes == (
        "bspline_point_out_of_bounds",
        "imm_point_out_of_bounds",
        "short_history_point_out_of_bounds",
    )


def test_predictor_returns_unavailable_after_every_bounded_fallback_fails() -> None:
    def invalid(stage: str, regime: str):
        return lambda _context: _candidate(
            f"raw-{stage}-id",
            regime,
            points_xy=((2_001.0, 0.0), (2_002.0, 0.0)),
        )

    predictor = make_snapshot_predictor(
        belief_history=lambda _snapshot, _target_id: (),
        horizon_s=60.0,
        sample_step_s=30.0,
        health_config=_health_config(),
        imm_forecaster=invalid("imm", "imm"),
        bspline_forecaster=invalid("bspline", "bspline"),
        short_history_forecaster=invalid("short-history", "short_history"),
        boundary_recovery_forecaster=invalid("boundary", "boundary_recovery"),
    )

    accepted = predictor(_snapshot_with_track_history(), "target-01")

    assert accepted.prediction is None
    assert accepted.health.status == "unavailable"
    assert accepted.health.regime == "boundary_recovery"
    assert accepted.health.raw_prediction_id == "raw-boundary-id"
    assert accepted.health.reason_codes == (
        "boundary_recovery_point_out_of_bounds",
        "bspline_point_out_of_bounds",
        "imm_point_out_of_bounds",
        "short_history_point_out_of_bounds",
    )


def test_default_boundary_recovery_uses_only_public_belief_and_map_bounds() -> None:
    snapshot = _snapshot_with_track_history()
    snapshot.group_reports[0].belief.mean = (900.0, 0.0, 0.0, 8.0)
    predictor = make_snapshot_predictor(
        belief_history=lambda _snapshot, _target_id: (),
        horizon_s=60.0,
        sample_step_s=30.0,
        max_speed_mps=10.0,
        max_turn_rate_rad_s=0.05,
        health_config=_health_config(),
        imm_forecaster=lambda _context: None,
        bspline_forecaster=lambda _context: None,
        short_history_forecaster=lambda _context: _candidate(
            "raw-short-id",
            "short_history",
            points_xy=((2_001.0, 0.0), (2_002.0, 0.0)),
        ),
    )

    accepted = predictor(snapshot, "target-01")

    assert accepted.health.status == "degraded"
    assert accepted.health.regime == "boundary_recovery"
    assert accepted.prediction is not None
    assert accepted.prediction.source_belief_history_ids == ("raw-observation-9",)
    assert all(-1_000.0 <= x <= 1_000.0 and -1_000.0 <= y <= 1_000.0 for x, y in accepted.prediction.points_xy)
