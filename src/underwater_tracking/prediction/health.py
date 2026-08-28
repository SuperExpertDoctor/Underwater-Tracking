"""Pure health assessment for public target-track predictions."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from math import atan2, hypot, isfinite, pi
from typing import cast

from underwater_tracking.config.models import PredictionHealthConfig
from underwater_tracking.domain.agent_models import PredictedTrackRef
from underwater_tracking.domain.prediction_models import (
    AcceptedPredictionRegime,
    PredictionHealth,
)


def effective_radius_limit_m(
    map_bounds_xy: tuple[float, float, float, float],
    config: PredictionHealthConfig,
) -> float:
    """Return the configured corridor cap constrained by the map size."""
    min_x, max_x, min_y, max_y = map_bounds_xy
    return min(
        config.max_corridor_radius_m,
        min(max_x - min_x, max_y - min_y) * config.max_corridor_map_fraction,
    )


def assess_prediction(
    prediction: PredictedTrackRef,
    *,
    snapshot_sim_time_s: int,
    map_bounds_xy: tuple[float, float, float, float],
    config: PredictionHealthConfig,
    max_speed_mps: float,
    max_turn_rate_rad_s: float,
    point_confidence: Sequence[float],
) -> PredictionHealth:
    """Assess a candidate without clipping, normalizing, or repairing it."""
    reasons: set[str] = set()
    times = prediction.times_s
    points = prediction.points_xy
    radii = prediction.corridor_radius_m
    covariance = prediction.imm_covariance_xy
    confidence = tuple(float(value) for value in point_confidence)
    point_count = len(points)

    if point_count == 0:
        reasons.add("empty_prediction")
    if len(times) != point_count or len(radii) != point_count or len(confidence) != point_count:
        reasons.add("array_length_mismatch")
    if prediction.prediction_regime == "imm" and not covariance:
        reasons.add("covariance_missing")
    if covariance and len(covariance) != point_count:
        reasons.add("array_length_mismatch")

    if any(not isfinite(value) for value in times):
        reasons.add("non_finite_time")
    if any(not isfinite(coordinate) for point in points for coordinate in point):
        reasons.add("non_finite_point")
    if any(not isfinite(radius) or radius < 0.0 for radius in radii):
        reasons.add("non_finite_radius")
    if any(not isfinite(value) for matrix in covariance for value in matrix):
        reasons.add("non_finite_covariance")

    finite_times = all(isfinite(value) for value in times)
    if finite_times and any(current <= previous for previous, current in pairwise(times)):
        reasons.add("non_monotonic_time")

    min_x, max_x, min_y, max_y = map_bounds_xy
    tolerance = config.coordinate_tolerance_m
    if any(
        isfinite(x)
        and isfinite(y)
        and not (
            min_x - tolerance <= x <= max_x + tolerance
            and min_y - tolerance <= y <= max_y + tolerance
        )
        for x, y in points
    ):
        reasons.add("point_out_of_bounds")

    finite_radii = tuple(radius for radius in radii if isfinite(radius) and radius >= 0.0)
    maximum_radius = max(finite_radii, default=0.0)
    if maximum_radius > effective_radius_limit_m(map_bounds_xy, config):
        reasons.add("corridor_radius_exceeded")

    clipping_records = {
        *prediction.clipping_records,
        *prediction.imm_clipping_records,
    }
    clipped_points = {
        record.rpartition("@")[2] if "@" in record else record
        for record in clipping_records
    }
    clipped_fraction = min(1.0, len(clipped_points) / max(point_count, 1))
    if clipped_fraction > config.max_clipped_point_fraction:
        reasons.add("excessive_clipping")

    _assess_kinematics(
        points,
        sample_step_s=prediction.sample_step_s,
        max_speed_mps=max_speed_mps,
        max_turn_rate_rad_s=max_turn_rate_rad_s,
        reasons=reasons,
    )

    if any(not isfinite(value) or not 0.0 <= value <= 1.0 for value in confidence):
        reasons.add("confidence_out_of_range")
    if all(isfinite(value) for value in confidence) and any(
        current > previous for previous, current in pairwise(confidence)
    ):
        reasons.add("confidence_increased")
    if confidence and isfinite(confidence[-1]) and confidence[-1] < config.minimum_point_confidence:
        reasons.add("confidence_below_floor")

    raw_age = float(snapshot_sim_time_s - prediction.sim_time_s)
    if raw_age < 0.0:
        reasons.add("source_track_in_future")
    elif raw_age > config.hard_stale_s:
        reasons.add("source_track_stale")

    accepted_regimes = {"imm", "bspline", "short_history", "boundary_recovery"}
    if prediction.prediction_regime not in accepted_regimes:
        reasons.add("unsupported_prediction_regime")
    regime = cast(
        AcceptedPredictionRegime,
        prediction.prediction_regime
        if prediction.prediction_regime in accepted_regimes
        else "short_history",
    )

    return PredictionHealth(
        status="valid" if not reasons else "unavailable",
        regime=regime,
        reason_codes=tuple(sorted(reasons)),
        source_track_age_s=max(0.0, raw_age),
        clipped_point_fraction=clipped_fraction,
        maximum_radius_m=maximum_radius,
        raw_prediction_id=prediction.prediction_id,
    )


def _assess_kinematics(
    points_xy: Sequence[tuple[float, float]],
    *,
    sample_step_s: float,
    max_speed_mps: float,
    max_turn_rate_rad_s: float,
    reasons: set[str],
) -> None:
    segment_headings: list[float] = []
    for index in range(1, len(points_xy)):
        previous = points_xy[index - 1]
        current = points_xy[index]
        values = (sample_step_s, *previous, *current)
        if sample_step_s <= 0.0 or not all(isfinite(value) for value in values):
            continue
        delta_x = current[0] - previous[0]
        delta_y = current[1] - previous[1]
        distance = hypot(delta_x, delta_y)
        speed_limit = max_speed_mps * sample_step_s
        if distance > speed_limit + 1.0e-9 * max(1.0, speed_limit):
            reasons.add("speed_exceeded")
        if distance > 0.0:
            segment_headings.append(atan2(delta_y, delta_x))

    for previous_heading, current_heading in pairwise(segment_headings):
        heading_delta = abs((current_heading - previous_heading + pi) % (2.0 * pi) - pi)
        turn_limit = max_turn_rate_rad_s * sample_step_s
        if heading_delta > turn_limit + 1.0e-9 * max(1.0, turn_limit):
            reasons.add("turn_rate_exceeded")


__all__ = ["assess_prediction", "effective_radius_limit_m"]
