from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from underwater_tracking.config.models import PredictionHealthConfig
from underwater_tracking.domain.agent_models import PredictedTrackRef
from underwater_tracking.prediction.health import (
    assess_prediction,
    effective_radius_limit_m,
)


MAP_BOUNDS = (-10_000.0, 10_000.0, -10_000.0, 10_000.0)


def health_config() -> PredictionHealthConfig:
    return PredictionHealthConfig(
        hard_stale_s=900,
        max_clipped_point_fraction=0.20,
        max_corridor_radius_m=6_000.0,
        max_corridor_map_fraction=0.25,
        minimum_point_confidence=0.02,
        coordinate_tolerance_m=0.000001,
    )


def valid_prediction() -> PredictedTrackRef:
    return PredictedTrackRef(
        prediction_id="raw-prediction-7",
        target_id="target-7",
        sim_time_s=90,
        horizon_s=90.0,
        sample_step_s=30.0,
        times_s=(130.0, 160.0, 190.0),
        points_xy=((0.0, 0.0), (300.0, 0.0), (600.0, 0.0)),
        corridor_radius_m=(100.0, 200.0, 300.0),
        prediction_regime="imm",
        imm_covariance_xy=(
            (10.0, 0.0, 0.0, 10.0),
            (20.0, 0.0, 0.0, 20.0),
            (30.0, 0.0, 0.0, 30.0),
        ),
    )


@pytest.mark.parametrize(
    ("mutation", "point_confidence", "reason"),
    [
        ({"times_s": (130.0, float("nan"), 190.0)}, (0.9, 0.8, 0.7), "non_finite_time"),
        ({"points_xy": ((float("nan"), 0.0),)}, (0.9,), "non_finite_point"),
        ({"corridor_radius_m": (float("nan"),)}, (0.9,), "non_finite_radius"),
        (
            {"imm_covariance_xy": ((float("nan"), 0.0, 0.0, 1.0),)},
            (0.9, 0.8, 0.7),
            "non_finite_covariance",
        ),
        ({"points_xy": ((20_001.0, 0.0),)}, (0.9,), "point_out_of_bounds"),
        ({"corridor_radius_m": (6_001.0,)}, (0.9,), "corridor_radius_exceeded"),
        (
            {"clipping_records": tuple(str(index) for index in range(3))},
            (0.9, 0.8, 0.7),
            "excessive_clipping",
        ),
        ({"times_s": (160.0, 130.0, 190.0)}, (0.9, 0.8, 0.7), "non_monotonic_time"),
        ({"points_xy": ((0.0, 0.0), (600.0, 0.0), (1_200.0, 0.0))}, (0.9, 0.8, 0.7), "speed_exceeded"),
        ({"points_xy": ((0.0, 0.0), (300.0, 0.0), (300.0, 300.0))}, (0.9, 0.8, 0.7), "turn_rate_exceeded"),
        ({}, (0.9, 1.1, 0.7), "confidence_out_of_range"),
        ({}, (0.9, 0.7, 0.8), "confidence_increased"),
        ({}, (0.9, 0.8, 0.01), "confidence_below_floor"),
        ({"sim_time_s": 101}, (0.9, 0.8, 0.7), "source_track_in_future"),
        ({"sim_time_s": 0}, (0.9, 0.8, 0.7), "source_track_stale"),
        ({"points_xy": ((0.0, 0.0),)}, (0.9, 0.8, 0.7), "array_length_mismatch"),
        ({}, (0.9, 0.8), "array_length_mismatch"),
        (
            {
                "times_s": (),
                "points_xy": (),
                "corridor_radius_m": (),
                "imm_covariance_xy": (),
            },
            (),
            "empty_prediction",
        ),
    ],
)
def test_assess_prediction_reports_machine_readable_reasons(
    mutation: Mapping[str, Any],
    point_confidence: tuple[float, ...],
    reason: str,
) -> None:
    prediction = valid_prediction().model_copy(update=mutation)

    health = assess_prediction(
        prediction,
        snapshot_sim_time_s=1_000 if reason == "source_track_stale" else 100,
        map_bounds_xy=MAP_BOUNDS,
        config=health_config(),
        max_speed_mps=15.0,
        max_turn_rate_rad_s=0.05,
        point_confidence=point_confidence,
    )

    assert health.status != "valid"
    assert reason in health.reason_codes


def test_assess_prediction_returns_all_reasons_in_sorted_order_without_repair() -> None:
    prediction = valid_prediction().model_copy(
        update={
            "points_xy": ((20_001.0, 0.0),),
            "corridor_radius_m": (6_001.0,),
        }
    )

    health = assess_prediction(
        prediction,
        snapshot_sim_time_s=100,
        map_bounds_xy=MAP_BOUNDS,
        config=health_config(),
        max_speed_mps=15.0,
        max_turn_rate_rad_s=0.05,
        point_confidence=(1.1,),
    )

    assert health.reason_codes == tuple(sorted(health.reason_codes))
    assert health.raw_prediction_id == "raw-prediction-7"
    assert health.maximum_radius_m == 6_001.0
    assert prediction.points_xy == ((20_001.0, 0.0),)
    assert prediction.corridor_radius_m == (6_001.0,)


def test_assess_prediction_accepts_a_complete_bounded_candidate() -> None:
    health = assess_prediction(
        valid_prediction(),
        snapshot_sim_time_s=100,
        map_bounds_xy=MAP_BOUNDS,
        config=health_config(),
        max_speed_mps=15.0,
        max_turn_rate_rad_s=0.05,
        point_confidence=(0.9, 0.8, 0.7),
    )

    assert health.status == "valid"
    assert health.regime == "imm"
    assert health.reason_codes == ()
    assert health.source_track_age_s == 10.0
    assert health.clipped_point_fraction == 0.0
    assert health.maximum_radius_m == 300.0
    assert health.raw_prediction_id == "raw-prediction-7"


def test_clipped_fraction_counts_unique_forecast_points_across_imm_branches() -> None:
    prediction = valid_prediction().model_copy(
        update={
            "clipping_records": (
                "CV:speed@0",
                "CT_LEFT:speed@0",
                "CT_RIGHT:turn@0",
            )
        }
    )
    config = health_config().model_copy(update={"max_clipped_point_fraction": 0.34})

    health = assess_prediction(
        prediction,
        snapshot_sim_time_s=100,
        map_bounds_xy=MAP_BOUNDS,
        config=config,
        max_speed_mps=15.0,
        max_turn_rate_rad_s=0.05,
        point_confidence=(0.9, 0.8, 0.7),
    )

    assert health.clipped_point_fraction == pytest.approx(1 / 3)
    assert "excessive_clipping" not in health.reason_codes


def test_effective_radius_limit_uses_the_shorter_map_dimension() -> None:
    assert effective_radius_limit_m(
        (-10_000.0, 10_000.0, -8_000.0, 8_000.0), health_config()
    ) == 4_000.0
