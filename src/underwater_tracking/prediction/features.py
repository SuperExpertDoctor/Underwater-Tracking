"""Deterministic motion-intent features from a track position history.

``extract_motion_features`` reduces a (time, position) history to scalar
kinematic statistics: speed, heading change, signed turn-rate mean,
curvature quantiles, net displacement, path efficiency, dwell fraction,
and last-window acceleration. Every quantity is a pure function of its
inputs, so repeated calls return identical dictionaries. Feature values
are dimensionless or in SI units (radians, meters, seconds).
"""

import numpy as np

from underwater_tracking.tracking.angles import wrap_angle


def extract_motion_features(
    times: np.ndarray,
    positions: np.ndarray,
    stationary_speed_mps: float = 0.5,
    dwell_radius_m: float = 30.0,
    last_window_s: float = 60.0,
) -> dict[str, float]:
    """Return kinematic features that separate straight, loiter, and evasion motion.

    Speed, heading, and turn-rate quantities are computed per segment
    (between consecutive fixes) and time-weighted where a mean is taken.
    A segment is "dwelling" when it both stays within ``dwell_radius_m``
    meters of its start and moves slower than ``stationary_speed_mps``;
    the dwell fraction is that segment time divided by the total track
    time. The last-window acceleration is the mean segment-to-segment
    acceleration magnitude over the final ``last_window_s`` seconds.
    """
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if times.ndim != 1 or positions.shape != (len(times), 2) or len(times) < 3:
        raise ValueError(
            "times must be 1-D and positions (len(times), 2) with at least 3 points"
        )
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing")

    durations = np.diff(times)
    segments = np.diff(positions, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    speeds = lengths / durations
    total_time = float(times[-1] - times[0])
    path_length = float(np.sum(lengths))
    mean_speed = path_length / total_time
    max_speed = float(np.max(speeds))

    headings = np.arctan2(segments[:, 1], segments[:, 0])
    heading_deltas = np.asarray(wrap_angle(np.diff(headings)), dtype=float)
    heading_change = float(np.sum(np.abs(heading_deltas)))
    mid_durations = (durations[:-1] + durations[1:]) / 2.0
    turn_rates = heading_deltas / mid_durations
    signed_turn_rate_mean = float(
        np.sum(turn_rates * mid_durations) / float(np.sum(mid_durations))
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        curvature = np.abs(heading_deltas) / ((lengths[:-1] + lengths[1:]) / 2.0)
    curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0)
    curvature_quantiles = np.quantile(curvature, np.array([0.25, 0.5, 0.75]))

    net_displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    path_efficiency = net_displacement / path_length if path_length > 0.0 else 0.0

    dwelling = (lengths <= dwell_radius_m) & (speeds <= stationary_speed_mps)
    dwell_fraction = float(np.sum(durations[dwelling]) / total_time)

    velocities = segments / durations[:, None]
    segment_mid_times = (times[:-1] + times[1:]) / 2.0
    acceleration_times = (segment_mid_times[:-1] + segment_mid_times[1:]) / 2.0
    acceleration_norms = (
        np.linalg.norm(np.diff(velocities, axis=0), axis=1) / mid_durations
    )
    in_last_window = acceleration_times >= (times[-1] - last_window_s)
    if np.any(in_last_window):
        last_window_acceleration = float(np.mean(acceleration_norms[in_last_window]))
    else:
        last_window_acceleration = float(acceleration_norms[-1])

    return {
        "mean_speed_mps": mean_speed,
        "max_speed_mps": max_speed,
        "heading_change_rad": heading_change,
        "signed_turn_rate_mean_rad_s": signed_turn_rate_mean,
        "curvature_q25": float(curvature_quantiles[0]),
        "curvature_q50": float(curvature_quantiles[1]),
        "curvature_q75": float(curvature_quantiles[2]),
        "net_displacement_m": net_displacement,
        "path_efficiency": float(path_efficiency),
        "dwell_fraction": dwell_fraction,
        "last_window_acceleration_mps2": last_window_acceleration,
    }
