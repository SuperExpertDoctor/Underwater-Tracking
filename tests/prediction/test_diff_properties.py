import math

import numpy as np
import pytest

from underwater_tracking.config.models import TrajectoryDiffConfig
from underwater_tracking.domain.agent_models import PredictedTrackRef
from underwater_tracking.prediction.diff import compare_predicted_tracks


def _track(
    prediction_id: str,
    points: np.ndarray,
    *,
    evidence_id: str,
) -> PredictedTrackRef:
    times = tuple(float(value) for value in range(30, 661, 30))
    return PredictedTrackRef(
        prediction_id=prediction_id,
        target_id="T1",
        sim_time_s=0 if prediction_id == "P1" else 30,
        horizon_s=660.0,
        sample_step_s=30.0,
        times_s=times,
        points_xy=tuple((float(point[0]), float(point[1])) for point in points),
        corridor_radius_m=(20.0,) * len(times),
        source_belief_history_ids=(evidence_id,),
        prediction_regime="short_history",
        imm_model_probabilities={"cv": 0.7, "left_turn": 0.2, "right_turn": 0.1},
    )


def _rigid_transform(
    track: PredictedTrackRef,
    angle: float,
    translation: np.ndarray,
) -> PredictedTrackRef:
    rotation = np.asarray(((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle))))
    points = np.asarray(track.points_xy) @ rotation.T + translation
    return track.model_copy(
        update={"points_xy": tuple((float(point[0]), float(point[1])) for point in points)}
    )


def test_common_rigid_transforms_preserve_scores() -> None:
    times = np.asarray(tuple(float(value) for value in range(30, 661, 30)))
    old = _track("P1", np.column_stack((times, 0.2 * times)), evidence_id="O1")
    new = _track(
        "P2",
        np.column_stack((times + 300.0, 0.2 * times + 50.0)),
        evidence_id="O2",
    )
    config = TrajectoryDiffConfig()
    baseline = compare_predicted_tracks(old, new, config)
    random = np.random.default_rng(42)

    for _ in range(100):
        angle = random.uniform(-math.pi, math.pi)
        translation = random.uniform(-10_000.0, 10_000.0, size=2)
        result = compare_predicted_tracks(
            _rigid_transform(old, angle, translation),
            _rigid_transform(new, angle, translation),
            config,
        )
        assert result.absolute_rms_m == pytest.approx(baseline.absolute_rms_m)
        assert result.normalized_rms == pytest.approx(baseline.normalized_rms)


def test_denser_sampling_does_not_change_constant_offset_score() -> None:
    sparse_times = tuple(float(value) for value in range(30, 661, 30))
    dense_times = tuple(float(value) for value in range(30, 661, 15))

    def build(
        prediction_id: str,
        times: tuple[float, ...],
        offset: float,
        evidence: str,
        step: float,
    ) -> PredictedTrackRef:
        return PredictedTrackRef(
            prediction_id=prediction_id,
            target_id="T1",
            sim_time_s=0 if prediction_id == "P1" else 30,
            horizon_s=660.0,
            sample_step_s=step,
            times_s=times,
            points_xy=tuple((time_s + offset, 0.0) for time_s in times),
            corridor_radius_m=(10.0,) * len(times),
            source_belief_history_ids=(evidence,),
            prediction_regime="short_history",
        )

    old = build("P1", sparse_times, 0.0, "O1", 30.0)
    sparse_new = build("P2", sparse_times, 300.0, "O2", 30.0)
    dense_new = build("P3", dense_times, 300.0, "O3", 15.0)
    config = TrajectoryDiffConfig()

    sparse = compare_predicted_tracks(old, sparse_new, config)
    dense = compare_predicted_tracks(old, dense_new, config)

    assert dense.comparison_step_s == 30.0
    assert dense.absolute_rms_m == pytest.approx(sparse.absolute_rms_m)
    assert dense.normalized_rms == pytest.approx(sparse.normalized_rms)
