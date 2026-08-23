import math
from collections.abc import Callable, Sequence

import pytest

from underwater_tracking.config.models import TrajectoryDiffConfig
from underwater_tracking.domain.agent_models import PredictedTrackRef
from underwater_tracking.prediction.diff import (
    compare_predicted_tracks,
    jensen_shannon_distance,
)


TIMES = tuple(float(value) for value in range(30, 661, 30))
CONFIG = TrajectoryDiffConfig()


def prediction(
    prediction_id: str,
    sim_time_s: int,
    times_s: Sequence[float],
    path: Callable[[float], tuple[float, float]],
    *,
    radius: float = 10.0,
    evidence: tuple[str, ...] = ("O1",),
    regime: str = "short_history",
    probabilities: dict[str, float] | None = None,
) -> PredictedTrackRef:
    return PredictedTrackRef(
        prediction_id=prediction_id,
        target_id="T1",
        sim_time_s=sim_time_s,
        horizon_s=max(times_s) - sim_time_s,
        sample_step_s=30.0,
        times_s=tuple(times_s),
        points_xy=tuple(path(time_s) for time_s in times_s),
        corridor_radius_m=(radius,) * len(times_s),
        source_belief_history_ids=evidence,
        fallback_used=regime != "bspline",
        prediction_regime=regime,
        imm_model_probabilities=probabilities or {"cv": 0.8, "left_turn": 0.1, "right_turn": 0.1},
    )


def test_equal_absolute_path_with_rolled_window_has_zero_diff() -> None:
    old = prediction("P1", 0, TIMES[:-1], lambda time_s: (2.0 * time_s, 0.0))
    new = prediction(
        "P2",
        30,
        TIMES[1:],
        lambda time_s: (2.0 * time_s, 0.0),
        evidence=("O2",),
    )

    result = compare_predicted_tracks(old, new, CONFIG)

    assert result.status == "comparable"
    assert result.absolute_rms_m == pytest.approx(0.0)
    assert result.normalized_rms == pytest.approx(0.0)
    assert result.exceeded is False
    assert result.overlap_start_s == 60.0
    assert result.overlap_end_s == 630.0


def test_both_normalized_and_absolute_gates_are_required() -> None:
    old_wide = prediction("P1", 0, TIMES, lambda time_s: (time_s, 0.0), radius=1_000.0)
    absolute_only = prediction(
        "P2",
        30,
        TIMES,
        lambda time_s: (time_s + 300.0, 0.0),
        radius=1_000.0,
        evidence=("O2",),
    )
    assert not compare_predicted_tracks(old_wide, absolute_only, CONFIG).exceeded

    old_narrow = old_wide.model_copy(update={"corridor_radius_m": (1.0,) * len(TIMES)})
    normalized_only = prediction(
        "P3",
        30,
        TIMES,
        lambda time_s: (time_s + 100.0, 0.0),
        radius=1.0,
        evidence=("O3",),
    )
    assert not compare_predicted_tracks(old_narrow, normalized_only, CONFIG).exceeded

    both = prediction(
        "P4",
        30,
        TIMES,
        lambda time_s: (time_s + 300.0, 0.0),
        radius=1.0,
        evidence=("O4",),
    )
    result = compare_predicted_tracks(old_narrow, both, CONFIG)
    assert result.exceeded
    assert result.absolute_rms_m == pytest.approx(300.0)
    assert result.normalized_rms is not None
    assert result.normalized_rms > 2.45


def test_expected_non_comparable_states_reset_the_baseline() -> None:
    current = prediction("P2", 30, TIMES, lambda time_s: (time_s, 0.0))
    assert compare_predicted_tracks(None, current, CONFIG).status == "first_prediction"

    same_evidence = current.model_copy(update={"prediction_id": "P3"})
    assert compare_predicted_tracks(current, same_evidence, CONFIG).status == "no_new_evidence"

    changed_regime = current.model_copy(
        update={
            "prediction_id": "P4",
            "prediction_regime": "bspline",
            "source_belief_history_ids": ("O2",),
        }
    )
    assert (
        compare_predicted_tracks(current, changed_regime, CONFIG).status == "predictor_regime_reset"
    )


def test_invalid_and_short_overlap_have_explicit_status() -> None:
    old = prediction("P1", 0, TIMES, lambda time_s: (time_s, 0.0))
    short = prediction(
        "P2",
        600,
        (630.0, 660.0),
        lambda time_s: (time_s, 0.0),
        evidence=("O2",),
    )
    assert compare_predicted_tracks(old, short, CONFIG).status == "insufficient_overlap"

    invalid = old.model_copy(
        update={
            "prediction_id": "P3",
            "source_belief_history_ids": ("O3",),
            "points_xy": ((*old.points_xy[:-1], (math.nan, 0.0))),
        }
    )
    result = compare_predicted_tracks(old, invalid, CONFIG)
    assert result.status == "invalid_prediction"
    assert result.reason is not None
    assert "finite" in result.reason


def test_jensen_shannon_distance_is_symmetric_and_bounded() -> None:
    left = {"cv": 0.8, "left_turn": 0.1, "right_turn": 0.1}
    right = {"cv": 0.1, "left_turn": 0.8, "right_turn": 0.1}

    forward = jensen_shannon_distance(left, right)
    reverse = jensen_shannon_distance(right, left)

    assert forward == pytest.approx(reverse)
    assert forward is not None
    assert 0.0 <= forward <= 1.0
    assert jensen_shannon_distance({}, right) is None
