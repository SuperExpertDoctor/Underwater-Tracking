from __future__ import annotations

import pytest

from underwater_tracking.prediction.imm_forecast import (
    IMMModelStateProjection,
    forecast_imm,
    moment_match_forecasts,
)


def _state(name: str, probability: float, *, velocity: tuple[float, float] = (2.0, 0.0), omega: float = 0.0) -> IMMModelStateProjection:
    covariance = tuple(
        tuple(1.0 if row == column else 0.0 for column in range(5))
        for row in range(5)
    )
    return IMMModelStateProjection(
        model_name=name,
        state_mean=(0.0, 0.0, velocity[0], velocity[1], omega),
        state_covariance=covariance,
        model_probability=probability,
        innovation=(0.1, -0.1),
        likelihood=0.8,
        source_observation_ids=("obs-1",),
    )


def test_forecast_imm_propagates_cv_branch_as_a_bounded_line() -> None:
    result = forecast_imm(
        [_state("CV", 1 / 3), _state("CT_LEFT", 1 / 3), _state("CT_RIGHT", 1 / 3)],
        origin_sim_time_s=100.0,
        horizon_s=30.0,
        sample_step_s=10.0,
        max_speed_mps=4.0,
        max_turn_rate_rad_s=0.1,
    )

    cv = next(branch for branch in result.model_branches if branch.model_name == "CV")
    assert cv.centerline_xy[-1] == pytest.approx((60.0, 0.0))
    assert result.times_s == (110.0, 120.0, 130.0)
    assert sum(result.model_probabilities.values()) == pytest.approx(1.0)


def test_forecast_imm_keeps_left_and_right_turns_mirrored() -> None:
    result = forecast_imm(
        [_state("CV", 0.0), _state("CT_LEFT", 0.5), _state("CT_RIGHT", 0.5)],
        origin_sim_time_s=0.0,
        horizon_s=20.0,
        sample_step_s=10.0,
        max_speed_mps=4.0,
        max_turn_rate_rad_s=0.05,
    )
    left = next(branch for branch in result.model_branches if branch.model_name == "CT_LEFT")
    right = next(branch for branch in result.model_branches if branch.model_name == "CT_RIGHT")

    assert left.centerline_xy[-1][1] > 0.0
    assert right.centerline_xy[-1][1] < 0.0
    assert left.centerline_xy[-1][0] == pytest.approx(right.centerline_xy[-1][0])
    assert left.centerline_xy[-1][1] == pytest.approx(-right.centerline_xy[-1][1])


def test_moment_match_includes_internal_covariance_and_branch_disagreement() -> None:
    means, covariances = moment_match_forecasts(
        {
            "CV": [((0.0, 0.0), ((1.0, 0.0), (0.0, 1.0)))],
            "CT_LEFT": [((4.0, 0.0), ((1.0, 0.0), (0.0, 1.0)))],
        },
        {"CV": 0.5, "CT_LEFT": 0.5},
    )

    assert means[0] == pytest.approx((2.0, 0.0))
    assert covariances[0][0] > 1.0
    assert covariances[0][1] == pytest.approx(0.0)


def test_forecast_imm_records_speed_and_turn_rate_clipping() -> None:
    result = forecast_imm(
        [
            _state("CV", 1 / 3, velocity=(20.0, 0.0), omega=2.0),
            _state("CT_LEFT", 1 / 3, velocity=(20.0, 0.0), omega=2.0),
            _state("CT_RIGHT", 1 / 3, velocity=(20.0, 0.0), omega=-2.0),
        ],
        origin_sim_time_s=0.0,
        horizon_s=10.0,
        sample_step_s=10.0,
        max_speed_mps=4.0,
        max_turn_rate_rad_s=0.1,
    )

    assert result.clipping_records
    assert any("speed" in record for record in result.clipping_records)
    assert any("turn" in record for record in result.clipping_records)
    assert hypot(result.model_branches[0].centerline_xy[0]) <= 40.0


def hypot(point: tuple[float, float]) -> float:
    return (point[0] ** 2 + point[1] ** 2) ** 0.5
