"""Time-aligned, uncertainty-aware comparison of consecutive forecasts."""

from __future__ import annotations

from collections.abc import Mapping
from math import isclose, isfinite, sqrt

import numpy as np

from underwater_tracking.config.models import TrajectoryDiffConfig
from underwater_tracking.domain.agent_models import (
    PredictedTrackRef,
    TrajectoryDiffResult,
    TrajectoryDiffStatus,
)


def jensen_shannon_distance(
    previous: Mapping[str, float],
    current: Mapping[str, float],
) -> float | None:
    """Return a symmetric, base-2 Jensen-Shannon distance in ``[0, 1]``."""
    if not previous or not current:
        return None
    labels = sorted(set(previous) | set(current))
    left = np.asarray([previous.get(label, 0.0) for label in labels], dtype=float)
    right = np.asarray([current.get(label, 0.0) for label in labels], dtype=float)
    if (
        not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or np.any(left < 0.0)
        or np.any(right < 0.0)
        or float(left.sum()) <= 0.0
        or float(right.sum()) <= 0.0
    ):
        raise ValueError("model probabilities must be finite, non-negative, and have positive mass")
    left /= left.sum()
    right /= right.sum()
    midpoint = 0.5 * (left + right)

    def divergence(values: np.ndarray) -> float:
        positive = values > 0.0
        return float(np.sum(values[positive] * np.log2(values[positive] / midpoint[positive])))

    distance = sqrt(max(0.0, 0.5 * divergence(left) + 0.5 * divergence(right)))
    return min(1.0, distance)


def compare_predicted_tracks(
    previous: PredictedTrackRef | None,
    current: PredictedTrackRef,
    config: TrajectoryDiffConfig,
) -> TrajectoryDiffResult:
    """Compare forecasts only over their shared absolute-time interval."""
    current_error = _validation_error(current)
    if current_error is not None:
        return _unavailable(previous, current, config, "invalid_prediction", current_error)
    if previous is None:
        return _unavailable(None, current, config, "first_prediction", "no previous prediction")

    previous_error = _validation_error(previous)
    if previous_error is not None:
        return _unavailable(previous, current, config, "invalid_prediction", previous_error)
    if previous.target_id != current.target_id:
        return _unavailable(
            previous,
            current,
            config,
            "target_mismatch",
            f"target changed from {previous.target_id} to {current.target_id}",
        )
    if previous.prediction_regime != current.prediction_regime:
        return _unavailable(
            previous,
            current,
            config,
            "predictor_regime_reset",
            f"prediction regime changed from {previous.prediction_regime} to {current.prediction_regime}",
        )
    if not set(current.source_belief_history_ids).difference(previous.source_belief_history_ids):
        return _unavailable(
            previous,
            current,
            config,
            "no_new_evidence",
            "current prediction contains no new belief evidence",
        )

    overlap_start = max(previous.times_s[0], current.times_s[0])
    overlap_end = min(previous.times_s[-1], current.times_s[-1])
    overlap_duration = max(0.0, overlap_end - overlap_start)
    comparison_step = max(previous.sample_step_s, current.sample_step_s)
    if overlap_duration < config.minimum_overlap_s:
        return _unavailable(
            previous,
            current,
            config,
            "insufficient_overlap",
            f"overlap {overlap_duration:.3f}s is below {config.minimum_overlap_s:.3f}s",
            overlap_start=overlap_start,
            overlap_end=overlap_end,
            comparison_step=comparison_step,
        )

    sample_count = int(np.floor(overlap_duration / comparison_step + 1e-12)) + 1
    sample_times = overlap_start + comparison_step * np.arange(sample_count, dtype=float)
    if sample_count < config.minimum_samples:
        return _unavailable(
            previous,
            current,
            config,
            "insufficient_overlap",
            f"comparison has {sample_count} samples; {config.minimum_samples} required",
            overlap_start=overlap_start,
            overlap_end=overlap_end,
            comparison_step=comparison_step,
            sample_count=sample_count,
        )

    previous_points, previous_radius = _interpolate(previous, sample_times)
    current_points, current_radius = _interpolate(current, sample_times)
    distances = np.linalg.norm(current_points - previous_points, axis=1)
    uncertainty = np.sqrt(previous_radius**2 + current_radius**2 + config.uncertainty_floor_m**2)
    normalized = distances / uncertainty
    weights = np.exp(-(sample_times - sample_times[0]) / config.near_term_decay_s)
    weights /= weights.sum()
    absolute_rms = float(np.sqrt(np.sum(weights * distances**2)))
    normalized_rms = float(np.sqrt(np.sum(weights * normalized**2)))
    maximum_index = int(np.argmax(distances))
    p90 = _weighted_quantile(distances, weights, 0.9)
    exceeded = _meets_threshold(normalized_rms, config.normalized_threshold) and _meets_threshold(
        absolute_rms, config.absolute_floor_m
    )

    return TrajectoryDiffResult(
        **_common_fields(previous, current, config),
        status="comparable",
        overlap_start_s=overlap_start,
        overlap_end_s=overlap_end,
        overlap_duration_s=overlap_duration,
        comparison_step_s=comparison_step,
        sample_count=sample_count,
        absolute_rms_m=absolute_rms,
        normalized_rms=normalized_rms,
        p90_distance_m=p90,
        max_distance_m=float(distances[maximum_index]),
        max_distance_time_s=float(sample_times[maximum_index]),
        js_distance=jensen_shannon_distance(
            previous.imm_model_probabilities,
            current.imm_model_probabilities,
        ),
        previous_leading_model=_leading_model(previous.imm_model_probabilities),
        current_leading_model=_leading_model(current.imm_model_probabilities),
        leading_model_changed=(
            _leading_model(previous.imm_model_probabilities)
            != _leading_model(current.imm_model_probabilities)
        ),
        exceeded=exceeded,
    )


def _validation_error(prediction: PredictedTrackRef) -> str | None:
    count = len(prediction.times_s)
    if count < 2:
        return "prediction must contain at least two samples"
    if len(prediction.points_xy) != count or len(prediction.corridor_radius_m) != count:
        return "prediction times, points, and corridor radii must have equal lengths"
    if not all(isfinite(value) for value in prediction.times_s):
        return "prediction times must be finite"
    if any(right <= left for left, right in zip(prediction.times_s, prediction.times_s[1:])):
        return "prediction times must be strictly increasing"
    if not all(isfinite(value) for point in prediction.points_xy for value in point):
        return "prediction points must be finite"
    if not all(isfinite(value) and value >= 0.0 for value in prediction.corridor_radius_m):
        return "prediction corridor radii must be finite and non-negative"
    try:
        jensen_shannon_distance(
            prediction.imm_model_probabilities,
            prediction.imm_model_probabilities,
        )
    except ValueError as exc:
        return str(exc)
    return None


def _interpolate(
    prediction: PredictedTrackRef,
    sample_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(prediction.points_xy, dtype=float)
    times = np.asarray(prediction.times_s, dtype=float)
    interpolated = np.column_stack(
        (np.interp(sample_times, times, points[:, 0]), np.interp(sample_times, times, points[:, 1]))
    )
    radius = np.interp(sample_times, times, prediction.corridor_radius_m)
    return interpolated, radius


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cumulative, quantile, side="left")), len(values) - 1)
    return float(values[order[index]])


def _meets_threshold(value: float, threshold: float) -> bool:
    return value >= threshold or isclose(value, threshold, rel_tol=1e-12, abs_tol=1e-12)


def _leading_model(probabilities: Mapping[str, float]) -> str | None:
    if not probabilities:
        return None
    return min(probabilities, key=lambda label: (-probabilities[label], label))


def _common_fields(
    previous: PredictedTrackRef | None,
    current: PredictedTrackRef,
    config: TrajectoryDiffConfig,
) -> dict[str, object]:
    return {
        "diff_id": (
            f"{current.prediction_id}:diff-baseline"
            if previous is None
            else f"{current.prediction_id}:diff-from:{previous.prediction_id}"
        ),
        "target_id": current.target_id,
        "previous_prediction_id": None if previous is None else previous.prediction_id,
        "current_prediction_id": current.prediction_id,
        "previous_sim_time_s": None if previous is None else previous.sim_time_s,
        "current_sim_time_s": current.sim_time_s,
        "previous_evidence_ids": ()
        if previous is None
        else tuple(sorted(previous.source_belief_history_ids)),
        "current_evidence_ids": tuple(sorted(current.source_belief_history_ids)),
        "normalized_threshold": config.normalized_threshold,
        "absolute_floor_m": config.absolute_floor_m,
        "reset_normalized_threshold": config.reset_normalized_threshold,
        "reset_absolute_floor_m": config.reset_absolute_floor_m,
        "threshold_schema_version": config.schema_version,
        "confirmation_cycles": config.confirmation_cycles,
    }


def _unavailable(
    previous: PredictedTrackRef | None,
    current: PredictedTrackRef,
    config: TrajectoryDiffConfig,
    status: TrajectoryDiffStatus,
    reason: str,
    *,
    overlap_start: float | None = None,
    overlap_end: float | None = None,
    comparison_step: float | None = None,
    sample_count: int = 0,
) -> TrajectoryDiffResult:
    overlap_duration = (
        max(0.0, overlap_end - overlap_start)
        if overlap_start is not None and overlap_end is not None
        else 0.0
    )
    return TrajectoryDiffResult(
        **_common_fields(previous, current, config),
        status=status,
        reason=reason,
        overlap_start_s=overlap_start,
        overlap_end_s=overlap_end,
        overlap_duration_s=overlap_duration,
        comparison_step_s=comparison_step,
        sample_count=sample_count,
        previous_leading_model=(
            None if previous is None else _leading_model(previous.imm_model_probabilities)
        ),
        current_leading_model=_leading_model(current.imm_model_probabilities),
    )
