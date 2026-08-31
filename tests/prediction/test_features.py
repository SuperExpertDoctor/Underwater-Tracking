import numpy as np
import pytest
from underwater_tracking.prediction.features import extract_motion_features


def _turn_arc(start_xy, start_heading, speed, turn_rate, duration_s, step_s):
    """Integrate a constant-turn segment analytically, returning its points."""
    times = np.arange(0.0, duration_s + step_s / 2.0, step_s)
    points = np.empty((len(times), 2))
    for index, elapsed in enumerate(times):
        angle = start_heading + turn_rate * elapsed
        points[index] = (
            start_xy
            + (speed / turn_rate)
            * np.array([np.sin(angle) - np.sin(start_heading),
                        -np.cos(angle) + np.cos(start_heading)])
            if turn_rate != 0.0
            else start_xy + speed * elapsed * np.array(
                [np.cos(start_heading), np.sin(start_heading)]
            )
        )
    return points


def test_straight_transit_features():
    times = np.arange(0.0, 1200.0, 30.0)
    positions = np.column_stack([2.0 * times, 0.5 * times])
    features = extract_motion_features(times, positions)
    np.testing.assert_allclose(features["mean_speed_mps"], np.sqrt(4.25), atol=1e-9)
    np.testing.assert_allclose(features["max_speed_mps"], np.sqrt(4.25), atol=1e-9)
    assert features["heading_change_rad"] < 1e-6
    assert abs(features["signed_turn_rate_mean_rad_s"]) < 1e-9
    assert features["curvature_q50"] < 1e-9
    np.testing.assert_allclose(features["net_displacement_m"], 1170.0 * np.sqrt(4.25),
                               rtol=1e-9)
    assert features["path_efficiency"] > 0.99
    assert features["dwell_fraction"] == 0.0
    assert features["last_window_acceleration_mps2"] < 1e-9


def test_circular_loiter_features():
    times = np.arange(0.0, 1800.0, 30.0)
    radius, omega = 100.0, 0.003
    angles = omega * times
    positions = radius * np.column_stack([np.cos(angles), np.sin(angles)])
    features = extract_motion_features(times, positions)
    assert features["signed_turn_rate_mean_rad_s"] > 0.002
    assert features["curvature_q50"] > 0.002
    assert features["path_efficiency"] < 0.5
    assert features["heading_change_rad"] > 0.1
    assert features["dwell_fraction"] > 0.99
    assert features["last_window_acceleration_mps2"] > 0.0005


def test_sharp_evasion_features():
    step = 30.0
    approach = _turn_arc(np.array([0.0, 0.0]), 0.0, 5.0, 0.0, 300.0, step)
    turn = _turn_arc(np.array([1500.0, 0.0]), 0.0, 5.0, 0.005, 300.0, step)
    escape = _turn_arc(turn[-1], 1.5, 5.0, 0.0, 300.0, step)
    # Drop each segment's duplicated start sample and stitch end-to-end.
    times = np.arange(0.0, 900.0 + step / 2.0, step)
    positions = np.vstack([approach, turn[1:], escape[1:]])
    features = extract_motion_features(times, positions)
    assert features["max_speed_mps"] > 4.9
    assert features["heading_change_rad"] > 1.0
    assert features["signed_turn_rate_mean_rad_s"] > 0.001
    assert features["curvature_q75"] > 0.0005
    assert features["path_efficiency"] < 0.95
    assert features["dwell_fraction"] == 0.0


def test_features_are_deterministic_and_reject_bad_inputs():
    times = np.arange(0.0, 1200.0, 30.0)
    positions = np.column_stack([2.0 * times, 0.5 * times])
    first = extract_motion_features(times, positions)
    second = extract_motion_features(times, positions)
    assert first == second
    with pytest.raises(ValueError):
        extract_motion_features(times[::-1], positions)
    with pytest.raises(ValueError):
        extract_motion_features(times[:2], positions[:1])
