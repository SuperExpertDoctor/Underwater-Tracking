import numpy as np
import pytest
from underwater_tracking.prediction.bspline import (
    InsufficientHistoryError,
    predict_track,
)


def _straight_history(speed_x: float = 2.0, speed_y: float = 0.5, span_s: float = 1200.0):
    times = np.arange(0.0, span_s, 30.0)
    positions = np.column_stack([speed_x * times, speed_y * times])
    covariances = np.repeat(np.eye(2)[None, :, :] * 25.0, len(times), axis=0)
    return times, positions, covariances


def test_weighted_bspline_extrapolates_straight_track_with_bounded_speed():
    times = np.arange(0.0, 1200.0, 30.0)
    positions = np.column_stack([2.0 * times, 0.5 * times])
    covariances = np.repeat(np.eye(2)[None, :, :] * 25.0, len(times), axis=0)
    prediction = predict_track(times, positions, covariances, horizon_s=1800,
                               sample_step_s=30, max_speed_mps=3.0,
                               max_turn_rate_rad_s=0.01)
    assert prediction.points_xy.shape == (60, 2)
    speed = np.linalg.norm(np.diff(prediction.points_xy, axis=0), axis=1) / 30.0
    assert np.max(speed) <= 3.0 + 1e-9
    assert np.all(np.diff(prediction.corridor_radius_m) >= 0)


def test_prediction_times_cover_exact_horizon_and_straight_line_is_exact():
    times, positions, covariances = _straight_history()
    prediction = predict_track(times, positions, covariances, horizon_s=1800,
                               sample_step_s=30, max_speed_mps=3.0,
                               max_turn_rate_rad_s=0.01)
    assert prediction.times_s.shape == (60,)
    np.testing.assert_allclose(prediction.times_s[0], times[-1] + 30.0)
    np.testing.assert_allclose(prediction.times_s[-1], times[-1] + 1800.0)
    assert prediction.corridor_radius_m.shape == (60,)
    assert prediction.fallback_used is False
    expected = np.column_stack([2.0 * prediction.times_s, 0.5 * prediction.times_s])
    np.testing.assert_allclose(prediction.points_xy, expected, atol=1e-6)


def test_speed_clipping_bounds_every_step():
    times, positions, covariances = _straight_history(speed_x=4.0, speed_y=0.0)
    prediction = predict_track(times, positions, covariances, horizon_s=900,
                               sample_step_s=30, max_speed_mps=3.0,
                               max_turn_rate_rad_s=0.1)
    step_lengths = np.linalg.norm(np.diff(prediction.points_xy, axis=0), axis=1)
    assert np.max(step_lengths) <= 3.0 * 30.0 + 1e-9


def test_turn_rate_clipping_bounds_heading_change():
    times = np.arange(0.0, 1200.0, 30.0)
    radius, omega = 50.0, 0.02
    angles = omega * times
    positions = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    covariances = np.repeat(np.eye(2)[None, :, :] * 4.0, len(times), axis=0)
    step = 30.0
    prediction = predict_track(times, positions, covariances, horizon_s=600,
                               sample_step_s=step, max_speed_mps=10.0,
                               max_turn_rate_rad_s=0.005)
    segments = np.diff(prediction.points_xy, axis=0)
    headings = np.arctan2(segments[:, 1], segments[:, 0])
    deltas = np.abs(np.diff(headings))
    deltas = np.abs((deltas + np.pi) % (2.0 * np.pi) - np.pi)
    assert np.max(deltas) <= 0.005 * step + 1e-9


def test_insufficient_history_raises():
    few_points = np.arange(0.0, 150.0, 30.0)
    positions = np.column_stack([2.0 * few_points, 0.5 * few_points])
    covariances = np.repeat(np.eye(2)[None, :, :] * 25.0, len(few_points), axis=0)
    with pytest.raises(InsufficientHistoryError):
        predict_track(few_points, positions, covariances, horizon_s=600,
                      sample_step_s=30, max_speed_mps=3.0, max_turn_rate_rad_s=0.01)
    short_span = np.arange(0.0, 900.0, 300.0)
    positions = np.column_stack([2.0 * short_span, 0.5 * short_span])
    covariances = np.repeat(np.eye(2)[None, :, :] * 25.0, len(short_span), axis=0)
    with pytest.raises(InsufficientHistoryError):
        predict_track(short_span, positions, covariances, horizon_s=600,
                      sample_step_s=30, max_speed_mps=3.0, max_turn_rate_rad_s=0.01)
