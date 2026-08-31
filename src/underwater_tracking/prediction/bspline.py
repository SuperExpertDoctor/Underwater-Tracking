"""Covariance-weighted cubic smoothing B-spline track prediction.

``predict_track`` fits one ``UnivariateSpline`` per axis over the last
history of filter positions, weighting each point by the inverse square
root of its position-covariance trace so uncertain fixes count less. The
smoothing tolerance is set to the number of history points, the expected
value of the weighted squared-residual sum, which interpolates exactly
through perfectly regular motion and filters measurement noise otherwise.
The splines are sampled one step at a time over the requested horizon and
each inter-point displacement is clipped to the physical limits: step
magnitude to ``max_speed_mps * sample_step_s`` and heading change to
``max_turn_rate_rad_s * sample_step_s``, so wildly non-physical spline
extrapolation is bounded. The corridor radius widens from the final belief
uncertainty plus the fitted-residual scale as the extrapolation horizon
grows. The IMM extrapolation fallback is a later task; ``fallback_used``
is always False here.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.interpolate import UnivariateSpline  # type: ignore[import-untyped]

from underwater_tracking.tracking.angles import wrap_angle

# Minimum history accepted by the spline fit: at least MIN_HISTORY_POINTS
# fixes spanning at least MIN_HISTORY_SPAN_S seconds.
MIN_HISTORY_POINTS = 8
MIN_HISTORY_SPAN_S = 240.0

# Weight floor so a perfectly confident fix never gets infinite weight.
_TRACE_FLOOR = 1e-6
# Corridor floor so a zero covariance never collapses the corridor to zero.
_BASE_SIGMA_FLOOR = 1e-9


class InsufficientHistoryError(RuntimeError):
    """Raised when the position history is too short to fit a prediction spline."""


@dataclass(frozen=True)
class TrackPrediction:
    """Sampled B-spline extrapolation with an expanding uncertainty corridor."""

    times_s: np.ndarray
    points_xy: np.ndarray
    corridor_radius_m: np.ndarray
    fallback_used: bool = False


def predict_track(
    times: np.ndarray,
    positions: np.ndarray,
    covariances: np.ndarray,
    horizon_s: float,
    sample_step_s: float,
    max_speed_mps: float,
    max_turn_rate_rad_s: float,
) -> TrackPrediction:
    """Extrapolate the track ``horizon_s`` seconds ahead with bounded motion.

    ``covariances`` holds one covariance matrix per fix, either the 2x2
    position block or a larger state covariance whose leading 2x2 block is
    the position block. Raises ``InsufficientHistoryError`` when fewer than
    ``MIN_HISTORY_POINTS`` fixes span less than ``MIN_HISTORY_SPAN_S``.
    """
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    covariances = np.asarray(covariances, dtype=float)
    _validate_predict_inputs(
        times, positions, covariances, horizon_s, sample_step_s,
        max_speed_mps, max_turn_rate_rad_s,
    )
    history_span = float(times[-1] - times[0])
    if len(times) < MIN_HISTORY_POINTS or history_span < MIN_HISTORY_SPAN_S:
        raise InsufficientHistoryError(
            f"need at least {MIN_HISTORY_POINTS} history points spanning "
            f"{MIN_HISTORY_SPAN_S:.0f} s to fit a prediction spline "
            f"(got {len(times)} points over {history_span:.1f} s)"
        )

    traces = np.trace(covariances[:, :2, :2], axis1=1, axis2=2)
    weights = 1.0 / np.sqrt(np.maximum(traces, _TRACE_FLOOR))
    smoothing = float(len(times))
    spline_x = UnivariateSpline(times, positions[:, 0], w=weights, s=smoothing)
    spline_y = UnivariateSpline(times, positions[:, 1], w=weights, s=smoothing)

    fitted_history = np.column_stack([spline_x(times), spline_y(times)])
    residual_rms = float(
        np.sqrt(np.mean(np.sum((positions - fitted_history) ** 2, axis=1)))
    )
    base_sigma = float(np.sqrt(np.maximum(traces[-1], _BASE_SIGMA_FLOOR)))

    step_count = int(horizon_s // sample_step_s)
    future_offsets = sample_step_s * np.arange(1, step_count + 1)
    prediction_times = times[-1] + future_offsets
    raw_points = np.column_stack(
        [spline_x(prediction_times), spline_y(prediction_times)]
    )
    initial_heading = float(
        np.arctan2(
            spline_y.derivative(1)(times[-1]), spline_x.derivative(1)(times[-1])
        )
    )
    points_xy = _clip_step_walk(
        raw_points,
        positions[-1],
        initial_heading,
        max_speed_mps * sample_step_s,
        max_turn_rate_rad_s * sample_step_s,
    )
    corridor_radius = base_sigma + residual_rms * np.sqrt(
        1.0 + future_offsets / history_span
    )
    return TrackPrediction(
        times_s=prediction_times,
        points_xy=points_xy,
        corridor_radius_m=corridor_radius,
        fallback_used=False,
    )


def _validate_predict_inputs(
    times: np.ndarray,
    positions: np.ndarray,
    covariances: np.ndarray,
    horizon_s: float,
    sample_step_s: float,
    max_speed_mps: float,
    max_turn_rate_rad_s: float,
) -> None:
    if times.ndim != 1 or positions.shape != (len(times), 2):
        raise ValueError("times must be 1-D and positions must have shape (len(times), 2)")
    if covariances.ndim != 3 or covariances.shape[0] != len(times):
        raise ValueError("covariances must have shape (len(times), n, n)")
    if covariances.shape[1] < 2 or covariances.shape[2] < 2:
        raise ValueError("covariance matrices must be at least 2x2")
    if np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing")
    if horizon_s < sample_step_s or sample_step_s <= 0.0:
        raise ValueError("sample_step_s must be positive and no larger than horizon_s")
    if max_speed_mps <= 0.0 or max_turn_rate_rad_s < 0.0:
        raise ValueError("max_speed_mps must be positive and max_turn_rate_rad_s non-negative")


def _clip_step_walk(
    raw_points: np.ndarray,
    start: np.ndarray,
    initial_heading: float,
    max_step: float,
    max_heading_delta: float,
) -> np.ndarray:
    """Walk the raw samples forward, clipping magnitude then heading per step.

    Each displacement is scaled down to at most ``max_step`` meters, then
    rotated so the heading change from the previous step (or from the fitted
    spline derivative at the start) stays within ``max_heading_delta``.
    """
    clipped = np.empty_like(raw_points)
    previous = np.asarray(start, dtype=float)
    previous_heading = initial_heading
    for index in range(len(raw_points)):
        step = raw_points[index] - previous
        step_norm = float(np.linalg.norm(step))
        if step_norm > max_step:
            step = step * (max_step / step_norm)
            step_norm = max_step
        heading = float(np.arctan2(step[1], step[0]))
        heading_delta = float(wrap_angle(heading - previous_heading))
        if abs(heading_delta) > max_heading_delta:
            heading = previous_heading + math.copysign(max_heading_delta, heading_delta)
            step = np.array([math.cos(heading), math.sin(heading)]) * step_norm
        clipped[index] = previous + step
        previous = np.asarray(clipped[index], dtype=float)
        previous_heading = heading
    return clipped
