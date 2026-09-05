# src/underwater_tracking/prediction/port.py
"""Trajectory-predictor port over the real B-spline prediction module (spec 6.6).

``make_snapshot_predictor`` adapts ``predict_track`` — the real prediction
module — to the carrier's per-target predictor contract (one
``PredictedTrackRef`` per tracked target, sampled from the target's
estimated belief history).
Covariances are not part of the engine's belief
history contract, so each fix is weighted by the group report's current
position-covariance block (the latest belief's uncertainty is the best
available estimate for every fix). When the history is too short for a
spline fit (the module needs at least MIN_HISTORY_POINTS fixes spanning
MIN_HISTORY_SPAN_S), the port degrades to a bounded linear extrapolation of
the last two estimated fixes — or a hold on the current estimate when only
one fix exists — with an expanding uncertainty corridor. The IMM
extrapolation fallback is a later task; this fallback is a pure function of
the snapshot and never invokes the LLM.
"""

from __future__ import annotations

import math
from underwater_tracking.tracking.public_estimate import assess_public_estimate
from underwater_tracking.domain.models import TargetBelief
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from underwater_tracking.config.models import PredictionHealthConfig
from underwater_tracking.domain.agent_models import PredictedTrackRef
from underwater_tracking.domain.execution_models import IMMModelForecast
from underwater_tracking.domain.models import GroupReport, SituationSnapshot
from underwater_tracking.domain.prediction_models import (
    AcceptedPrediction,
    AcceptedPredictionRegime,
    PredictionHealth,
)
from underwater_tracking.prediction.bspline import (
    MIN_HISTORY_POINTS,
    MIN_HISTORY_SPAN_S,
    predict_track,
)
from underwater_tracking.prediction.health import assess_prediction, effective_radius_limit_m
from underwater_tracking.prediction.imm_forecast import forecast_imm

# A belief-history sample: (sim_time_s, x, y) — the engine's public contract.
BeliefSample = tuple[int, float, float]

# Default physical limits mirror the simulated target, not the observer:
# ``tracking.submarine_sprint_speed_mps`` (14 m/s) and
# ``tracking.submarine_turn_rate_rad_s`` (pi/300 rad/s).
_DEFAULT_MAX_SPEED_MPS = 14.0
_DEFAULT_MAX_TURN_RATE_RAD_S = math.pi / 300.0

# Corridor floor so a perfectly confident belief never collapses the
# corridor to zero (same idea as the bspline module's _BASE_SIGMA_FLOOR).
_BASE_SIGMA_FLOOR = 1e-9
_DEFAULT_HEALTH_CONFIG = PredictionHealthConfig()


@dataclass(frozen=True)
class ForecastContext:
    """Estimator-safe inputs shared by deterministic candidate forecasters."""

    prediction_id: str
    snapshot: SituationSnapshot
    target_id: str
    report: GroupReport | None
    samples: tuple[BeliefSample, ...]
    position_block: np.ndarray
    horizon_s: float
    sample_step_s: float
    max_speed_mps: float
    max_turn_rate_rad_s: float
    health_config: PredictionHealthConfig = _DEFAULT_HEALTH_CONFIG


class CandidateForecaster(Protocol):
    def __call__(self, context: ForecastContext) -> PredictedTrackRef | None: ...


def make_snapshot_predictor(
    *,
    belief_history: Callable[[SituationSnapshot, str], Sequence[BeliefSample]],
    horizon_s: float,
    sample_step_s: float,
    max_speed_mps: float = _DEFAULT_MAX_SPEED_MPS,
    max_turn_rate_rad_s: float = _DEFAULT_MAX_TURN_RATE_RAD_S,
    health_config: PredictionHealthConfig = _DEFAULT_HEALTH_CONFIG,
    imm_forecaster: CandidateForecaster | None = None,
    bspline_forecaster: CandidateForecaster | None = None,
    short_history_forecaster: CandidateForecaster | None = None,
    boundary_recovery_forecaster: CandidateForecaster | None = None,
) -> Callable[[SituationSnapshot, str], AcceptedPrediction]:
    """Build a predictor that accepts only bounded estimator-safe candidates.

    ``belief_history`` returns the target's estimated position history as
    ``(sim_time_s, x, y)`` samples. There is deliberately no simulator-truth
    history port: operational prediction must remain reproducible from public
    observations and estimator output. The returned predictor is pure in the
    snapshot (same snapshot and history always yield the same ``PredictedTrackRef``).
    The default physical limits mirror the configured target limits,
    ``tracking.submarine_sprint_speed_mps`` (14 m/s) and
    ``tracking.submarine_turn_rate_rad_s`` (pi/300 rad/s). Predictions
    with fewer than ``MIN_HISTORY_POINTS`` fixes spanning
    ``MIN_HISTORY_SPAN_S`` are served by the documented short-history
    fallback instead of failing the planning cycle. Every generated candidate
    is assessed without geometry repair in this order: IMM, B-spline,
    short-history, boundary recovery, unavailable.
    """

    forecasters: tuple[tuple[AcceptedPredictionRegime, CandidateForecaster], ...] = (
        ("imm", imm_forecaster or _default_imm_forecaster),
        ("bspline", bspline_forecaster or _default_bspline_forecaster),
        ("short_history", short_history_forecaster or _default_short_history_forecaster),
        (
            "boundary_recovery",
            boundary_recovery_forecaster or _default_boundary_recovery_forecaster,
        ),
    )

    def predict(snapshot: SituationSnapshot, target_id: str) -> AcceptedPrediction:
        samples = tuple(belief_history(snapshot, target_id))
        report = _group_report(snapshot, target_id)
        estimate_health = (
            assess_public_estimate(report.belief, snapshot.sim_time_s)
            if report is not None and isinstance(report.belief, TargetBelief) else None
        )
        if estimate_health is not None and estimate_health.status in {"expired", "unavailable"}:
            return _unavailable_prediction(
                reason_codes=estimate_health.reason_codes,
                source_track_age_s=estimate_health.source_age_s or 0.0,
            )
        covariance = report.belief.covariance if report is not None else ()
        position_block = _position_block(covariance)
        prediction_id = f"{snapshot.scenario_id}:track:{target_id}:{snapshot.snapshot_revision}"
        context = ForecastContext(
            prediction_id=prediction_id,
            snapshot=snapshot,
            target_id=target_id,
            report=report,
            samples=samples,
            position_block=position_block,
            horizon_s=horizon_s,
            sample_step_s=sample_step_s,
            max_speed_mps=max_speed_mps,
            max_turn_rate_rad_s=max_turn_rate_rad_s,
            health_config=health_config,
        )
        map_bounds = snapshot.map_bounds_xy
        if map_bounds is None:
            return _unavailable_prediction(
                reason_codes=("map_bounds_unavailable",),
                source_track_age_s=0.0,
            )

        upstream_reasons: list[str] = []
        last_health: PredictionHealth | None = None
        imm_candidate: PredictedTrackRef | None = None
        for regime, forecaster in forecasters:
            candidate = forecaster(context)
            if candidate is None:
                upstream_reasons.append(f"{regime}_unavailable")
                continue
            candidate = candidate.model_copy(update={"prediction_regime": regime})
            if estimate_health is not None:
                candidate = candidate.model_copy(update={
                    "source_track_revision": estimate_health.source_track_revision,
                    "last_observed_at_s": estimate_health.last_observed_at_s,
                    "valid_until_s": estimate_health.valid_until_s,
                    "generated_at_s": float(snapshot.sim_time_s),
                    "prediction_revision": max(1, snapshot.snapshot_revision),
                })
            if regime == "imm":
                imm_candidate = candidate
            confidence = candidate.point_confidence or _point_confidence(
                candidate.corridor_radius_m,
                _leading_model_probability(candidate),
                health_config.minimum_point_confidence,
            )
            candidate = candidate.model_copy(update={"point_confidence": confidence})
            candidate_health = assess_prediction(
                candidate,
                snapshot_sim_time_s=snapshot.sim_time_s,
                map_bounds_xy=map_bounds,
                config=health_config,
                max_speed_mps=max_speed_mps,
                max_turn_rate_rad_s=max_turn_rate_rad_s,
                point_confidence=confidence,
            )
            last_health = candidate_health
            if candidate_health.status == "valid":
                if regime == "imm":
                    # IMM acceptance must not short-circuit the independent
                    # historical cubic B-spline artifact used by the UI.
                    bspline_candidate = _safe_auxiliary_forecast(
                        forecasters[1][1], context
                    )
                    candidate = _attach_prediction_components(
                        candidate,
                        imm_candidate=candidate,
                        bspline_candidate=bspline_candidate,
                    )
                elif regime == "bspline":
                    candidate = _attach_prediction_components(
                        candidate,
                        imm_candidate=imm_candidate,
                        bspline_candidate=candidate,
                    )
                if estimate_health is not None and estimate_health.status == "degraded":
                    upstream_reasons.append("estimate_extrapolated")
                accepted_status = "valid" if regime == "imm" and not upstream_reasons else "degraded"
                return AcceptedPrediction(
                    prediction=candidate,
                    health=candidate_health.model_copy(
                        update={
                            "status": accepted_status,
                            "regime": regime,
                            "reason_codes": tuple(sorted(upstream_reasons)),
                        }
                    ),
                )
            upstream_reasons.extend(
                f"{regime}_{reason}" for reason in candidate_health.reason_codes
            )

        if last_health is None:
            return _unavailable_prediction(reason_codes=tuple(sorted(upstream_reasons)))
        return AcceptedPrediction(
            prediction=None,
            health=last_health.model_copy(
                update={
                    "status": "unavailable",
                    "regime": "boundary_recovery",
                    "reason_codes": tuple(sorted(upstream_reasons)),
                }
            ),
        )

    return predict


def _safe_auxiliary_forecast(
    forecaster: CandidateForecaster,
    context: ForecastContext,
) -> PredictedTrackRef | None:
    """Keep a display-only forecast failure out of the physical decision path."""
    try:
        return forecaster(context)
    except Exception:  # noqa: BLE001 - display evidence is best effort
        return None


def _prediction_component(
    candidate: PredictedTrackRef | None,
    component: Literal["imm", "bspline"],
) -> tuple[tuple[float, ...], tuple[tuple[float, float], ...], tuple[float, ...]] | None:
    if candidate is None:
        return None
    if component == "imm":
        times = candidate.imm_times_s or candidate.times_s
        points = candidate.imm_centerline_xy or candidate.points_xy
        radii = candidate.imm_corridor_radius_m or candidate.corridor_radius_m
    else:
        times = candidate.bspline_times_s or candidate.times_s
        points = candidate.bspline_centerline_xy or candidate.points_xy
        radii = ()
    if len(times) != len(points) or not points:
        return None
    if component == "imm" and len(radii) != len(points):
        return None
    return (
        tuple(float(value) for value in times),
        tuple((float(x), float(y)) for x, y in points),
        tuple(float(value) for value in radii),
    )


def _attach_prediction_components(
    prediction: PredictedTrackRef,
    *,
    imm_candidate: PredictedTrackRef | None,
    bspline_candidate: PredictedTrackRef | None,
) -> PredictedTrackRef:
    updates: dict[str, object] = {}
    imm = _prediction_component(imm_candidate, "imm")
    if imm is not None:
        updates.update(
            imm_times_s=imm[0],
            imm_centerline_xy=imm[1],
            imm_corridor_radius_m=imm[2],
        )
    bspline = _prediction_component(bspline_candidate, "bspline")
    if bspline is not None:
        updates.update(bspline_times_s=bspline[0], bspline_centerline_xy=bspline[1])
    return prediction.model_copy(update=updates) if updates else prediction


def _default_imm_forecaster(context: ForecastContext) -> PredictedTrackRef | None:
    span = (
        context.samples[-1][0] - context.samples[0][0]
        if len(context.samples) >= 2
        else 0.0
    )
    if len(context.samples) < MIN_HISTORY_POINTS or span < MIN_HISTORY_SPAN_S:
        return None
    return _imm_prediction_ref(
        context.prediction_id,
        context.snapshot,
        context.target_id,
        context.report,
        context.samples,
        context.position_block,
        context.horizon_s,
        context.sample_step_s,
        context.max_speed_mps,
        context.max_turn_rate_rad_s,
    )


def _default_bspline_forecaster(context: ForecastContext) -> PredictedTrackRef | None:
    span = (
        context.samples[-1][0] - context.samples[0][0]
        if len(context.samples) >= 2
        else 0.0
    )
    if len(context.samples) < MIN_HISTORY_POINTS or span < MIN_HISTORY_SPAN_S:
        return None
    try:
        raw = predict_track(
            np.asarray([sample[0] for sample in context.samples], dtype=float),
            np.asarray([[sample[1], sample[2]] for sample in context.samples], dtype=float),
            np.repeat(
                context.position_block[np.newaxis, :, :],
                len(context.samples),
                axis=0,
            ),
            context.horizon_s,
            context.sample_step_s,
            max_speed_mps=context.max_speed_mps,
            max_turn_rate_rad_s=context.max_turn_rate_rad_s,
        )
    except (TypeError, ValueError, RuntimeError, FloatingPointError):
        return None
    times, points = _rebase_prediction_if_stale(
        context.snapshot,
        context.report,
        context.samples,
        raw.times_s,
        raw.points_xy,
        context.sample_step_s,
        context.max_speed_mps,
    )
    return _ref_from_prediction(
        context.prediction_id,
        context.snapshot,
        context.target_id,
        context.report,
        times,
        points,
        raw.corridor_radius_m,
        raw.fallback_used,
        context.horizon_s,
        context.sample_step_s,
    )


def _default_short_history_forecaster(context: ForecastContext) -> PredictedTrackRef:
    return _short_history_ref(
        context.prediction_id,
        context.snapshot,
        context.target_id,
        context.report,
        context.samples,
        context.position_block,
        context.horizon_s,
        context.sample_step_s,
        context.max_speed_mps,
    )


def _default_boundary_recovery_forecaster(context: ForecastContext) -> PredictedTrackRef | None:
    map_bounds = context.snapshot.map_bounds_xy
    if context.report is None or map_bounds is None or len(context.report.belief.mean) < 2:
        return None
    raw_position = (
        float(context.report.belief.mean[0]),
        float(context.report.belief.mean[1]),
    )
    min_x, max_x, min_y, max_y = map_bounds
    position = _project_map_anchor(raw_position, map_bounds)
    if position is None:
        return None
    center = ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5)
    velocity = _public_velocity(
        context.report,
        context.samples,
        max_speed_mps=context.max_speed_mps,
    )
    speed = min(context.max_speed_mps, math.hypot(*velocity))
    if speed <= 1e-12:
        speed = context.max_speed_mps * 0.5
        heading = math.atan2(center[1] - position[1], center[0] - position[0])
    else:
        heading = math.atan2(velocity[1], velocity[0])
    steps = max(1, int(context.horizon_s // context.sample_step_s))
    max_delta = max(0.0, context.max_turn_rate_rad_s * context.sample_step_s)
    turn_steps = (
        max(1, math.ceil(math.pi / max_delta))
        if max_delta > 1.0e-12
        else steps
    )
    boundary_guard_distance = speed * context.sample_step_s * (turn_steps + 1)
    holding_for_boundary_turn = _boundary_turn_required(
        position,
        heading,
        map_bounds,
        boundary_guard_distance,
    )
    times: list[float] = []
    points: list[tuple[float, float]] = []
    for index in range(steps):
        desired = math.atan2(center[1] - position[1], center[0] - position[0])
        heading_delta = _signed_angle_delta(heading, desired)
        heading += max(-max_delta, min(max_delta, heading_delta))
        distance_to_center = math.hypot(center[0] - position[0], center[1] - position[1])
        distance = min(speed * context.sample_step_s, distance_to_center)
        if holding_for_boundary_turn:
            holding_for_boundary_turn = _boundary_turn_required(
                position,
                heading,
                map_bounds,
                boundary_guard_distance,
            )
            if holding_for_boundary_turn:
                distance = 0.0
        candidate = (
            position[0] + math.cos(heading) * distance,
            position[1] + math.sin(heading) * distance,
        )
        if not _point_inside_map(candidate, map_bounds):
            return None
        position = candidate
        times.append(context.snapshot.sim_time_s + (index + 1) * context.sample_step_s)
        points.append(position)
    source_ids = tuple(context.report.belief.source_observation_ids)
    corridor_limit = max(
        0.0,
        effective_radius_limit_m(map_bounds, context.health_config),
    )
    corridor_base = min(_base_sigma(context.position_block), corridor_limit)
    corridor = tuple(
        min(
            corridor_limit,
            corridor_base + 10.0 * (step_index / max(steps, 1)),
        )
        for step_index in range(steps)
    )
    return PredictedTrackRef(
        prediction_id=context.prediction_id,
        target_id=context.target_id,
        sim_time_s=_source_track_sim_time_s(
            context.snapshot,
            context.report,
            context.samples,
        ),
        horizon_s=context.horizon_s,
        sample_step_s=context.sample_step_s,
        times_s=tuple(times),
        points_xy=tuple(points),
        corridor_radius_m=corridor,
        source_belief_history_ids=source_ids,
        fallback_used=True,
        fallback_reason=(
            "map-projected public-track boundary recovery"
            if position != raw_position
            else "bounded public-track boundary recovery"
        ),
        prediction_regime="boundary_recovery",
        imm_model_probabilities=_model_probabilities(context.report),
    )


def _signed_angle_delta(current: float, desired: float) -> float:
    return (desired - current + math.pi) % (2.0 * math.pi) - math.pi


def _project_map_anchor(
    position: tuple[float, float],
    map_bounds: tuple[float, float, float, float],
) -> tuple[float, float] | None:
    if not all(math.isfinite(value) for value in (*position, *map_bounds)):
        return None
    min_x, max_x, min_y, max_y = map_bounds
    if min_x > max_x or min_y > max_y:
        return None
    return (
        min(max(position[0], min_x), max_x),
        min(max(position[1], min_y), max_y),
    )


def _boundary_turn_required(
    position: tuple[float, float],
    heading: float,
    map_bounds: tuple[float, float, float, float],
    guard_distance: float,
) -> bool:
    min_x, max_x, min_y, max_y = map_bounds
    x, y = position
    return any(
        (
            distance <= guard_distance
            and component < 1.0e-12
        )
        for distance, component in (
            (x - min_x, math.cos(heading)),
            (max_x - x, -math.cos(heading)),
            (y - min_y, math.sin(heading)),
            (max_y - y, -math.sin(heading)),
        )
    )


def _point_inside_map(
    point: tuple[float, float],
    map_bounds: tuple[float, float, float, float],
) -> bool:
    min_x, max_x, min_y, max_y = map_bounds
    return min_x <= point[0] <= max_x and min_y <= point[1] <= max_y


def _leading_model_probability(prediction: PredictedTrackRef) -> float:
    return max(prediction.imm_model_probabilities.values(), default=1.0)


def _point_confidence(
    radii_m: Sequence[float],
    leading_model_probability: float,
    minimum_confidence: float,
) -> tuple[float, ...]:
    if not radii_m:
        return ()
    finite_positive = tuple(
        float(radius) for radius in radii_m if math.isfinite(radius) and radius > 0.0
    )
    base_radius = min(finite_positive, default=1.0)
    leading_probability = max(0.0, min(1.0, float(leading_model_probability)))
    result: list[float] = []
    previous = 1.0
    for radius in radii_m:
        raw = (
            leading_probability
            if radius <= 0.0
            else leading_probability * (base_radius / float(radius)) ** 2
        )
        value = max(minimum_confidence, min(previous, max(0.0, min(1.0, raw))))
        result.append(value)
        previous = value
    return tuple(result)


def _unavailable_prediction(
    *,
    reason_codes: tuple[str, ...],
    source_track_age_s: float = 0.0,
) -> AcceptedPrediction:
    return AcceptedPrediction(
        prediction=None,
        health=PredictionHealth(
            status="unavailable",
            regime="boundary_recovery",
            reason_codes=reason_codes,
            source_track_age_s=source_track_age_s,
            clipped_point_fraction=0.0,
            maximum_radius_m=0.0,
            raw_prediction_id=None,
        ),
    )


def _imm_prediction_ref(
    prediction_id: str,
    snapshot: SituationSnapshot,
    target_id: str,
    report: GroupReport | None,
    samples: Sequence[BeliefSample],
    position_block: np.ndarray,
    horizon_s: float,
    sample_step_s: float,
    max_speed_mps: float,
    max_turn_rate_rad_s: float,
) -> PredictedTrackRef | None:
    """Propagate and mix the three operational IMM motion hypotheses."""
    if report is None:
        return None
    states = _imm_model_states(
        report,
        position_block,
        samples,
        max_speed_mps,
    )
    if states is None:
        return None
    try:
        forecast = forecast_imm(
            states=states,
            origin_sim_time_s=float(snapshot.sim_time_s),
            horizon_s=horizon_s,
            sample_step_s=sample_step_s,
            max_speed_mps=max_speed_mps,
            max_turn_rate_rad_s=max_turn_rate_rad_s,
        )
    except (TypeError, ValueError, FloatingPointError):
        return None
    raw_probabilities = _model_probabilities(report)
    return PredictedTrackRef(
        prediction_id=prediction_id,
        target_id=target_id,
        sim_time_s=_source_track_sim_time_s(snapshot, report, samples),
        horizon_s=horizon_s,
        sample_step_s=sample_step_s,
        times_s=forecast.times_s,
        points_xy=forecast.centerline_xy,
        corridor_radius_m=forecast.corridor_radius_m,
        imm_times_s=forecast.times_s,
        imm_centerline_xy=forecast.centerline_xy,
        imm_corridor_radius_m=forecast.corridor_radius_m,
        source_belief_history_ids=tuple(report.belief.source_observation_ids),
        clipping_records=forecast.clipping_records,
        fallback_used=False,
        prediction_regime="imm",
        imm_model_probabilities=raw_probabilities,
        imm_model_states=states,
        imm_covariance_xy=forecast.covariance_xy,
        imm_clipping_records=forecast.clipping_records,
    )


def _imm_model_states(
    report: GroupReport,
    position_block: np.ndarray,
    samples: Sequence[BeliefSample],
    max_speed_mps: float,
) -> tuple[IMMModelForecast, ...] | None:
    """Build complete five-state IMM projections from a public group report."""
    probabilities = _canonical_model_probabilities(report)
    total = sum(probabilities.values())
    if total <= 1e-12:
        return None
    probabilities = {name: value / total for name, value in probabilities.items()}
    mean = np.zeros(5, dtype=float)
    belief_mean = tuple(float(value) for value in report.belief.mean)
    mean[: min(len(belief_mean), 5)] = belief_mean[:5]
    if len(belief_mean) < 4:
        mean[2:4] = _public_velocity(report, samples, max_speed_mps=max_speed_mps)
    speed = math.hypot(float(mean[2]), float(mean[3]))
    if speed > max_speed_mps and speed > 1e-12:
        mean[2:4] *= max_speed_mps / speed
    covariance = _imm_state_covariance(report.belief.covariance, position_block)
    source_ids = tuple(report.belief.source_observation_ids)
    return tuple(
        IMMModelForecast(
            model_name=name,
            state_mean=tuple(float(value) for value in mean),
            state_covariance=tuple(
                tuple(float(value) for value in row) for row in covariance
            ),
            model_probability=probabilities[name],
            innovation=(),
            likelihood=1.0,
            source_observation_ids=source_ids,
        )
        for name in ("CV", "CT_LEFT", "CT_RIGHT")
    )


def _canonical_model_probabilities(report: GroupReport) -> dict[str, float]:
    result = {"CV": 0.0, "CT_LEFT": 0.0, "CT_RIGHT": 0.0}
    for label, probability in report.belief.model_probabilities.items():
        normalized = str(label).casefold().replace("-", "_")
        if normalized in {"cv", "constant_velocity"}:
            result["CV"] += float(probability)
        elif normalized in {"left", "left_turn", "ct_left"}:
            result["CT_LEFT"] += float(probability)
        elif normalized in {"right", "right_turn", "ct_right"}:
            result["CT_RIGHT"] += float(probability)
    return result


def _imm_state_covariance(
    raw_covariance: Sequence[Sequence[float]],
    position_block: np.ndarray,
) -> np.ndarray:
    """Embed legacy position/state covariance into the five-state turn model."""
    covariance = np.eye(5, dtype=float)
    raw = np.asarray(raw_covariance, dtype=float)
    if raw.ndim == 2 and raw.shape[0] == raw.shape[1]:
        size = min(raw.shape[0], 5)
        covariance[:size, :size] = raw[:size, :size]
    covariance[:2, :2] = position_block
    covariance[4, 4] = max(float(covariance[4, 4]), 1e-6)
    covariance = (covariance + covariance.T) * 0.5
    values, vectors = np.linalg.eigh(covariance)
    return np.asarray((vectors * np.maximum(values, 1e-9)) @ vectors.T, dtype=float)


def _group_report(snapshot: SituationSnapshot, target_id: str) -> GroupReport | None:
    """The target's group report, or None when the target is not tracked."""
    for report in snapshot.group_reports:
        if report.target_id == target_id:
            return report
    return None


def _position_block(covariance: Sequence[Sequence[float]]) -> np.ndarray:
    """The 2x2 position block of the belief covariance (zeros when empty)."""
    block = np.zeros((2, 2), dtype=float)
    for row in range(2):
        for column in range(2):
            if row < len(covariance) and column < len(covariance[row]):
                block[row, column] = float(covariance[row][column])
    return block


def _base_sigma(position_block: np.ndarray) -> float:
    """Root-mean-square position uncertainty from the covariance trace."""
    trace = float(position_block[0, 0] + position_block[1, 1])
    return max(math.sqrt(max(trace, 0.0) / 2.0), _BASE_SIGMA_FLOOR)


def _corridor(base_sigma: float, steps: int) -> tuple[float, ...]:
    """Expanding corridor: base uncertainty grows linearly with the step."""
    return tuple(base_sigma + 10.0 * (step_index / max(steps, 1)) for step_index in range(steps))


def _short_history_ref(
    prediction_id: str,
    snapshot: SituationSnapshot,
    target_id: str,
    report: GroupReport | None,
    samples: Sequence[BeliefSample],
    position_block: np.ndarray,
    horizon_s: float,
    sample_step_s: float,
    max_speed_mps: float,
) -> PredictedTrackRef:
    """Deterministic short-history fallback (no spline fit possible).

    The last two estimated fixes are extrapolated linearly, one sample step
    at a time, with the latest belief's position uncertainty as the base
    corridor; a single fix holds the current estimate. ``fallback_used`` is
    always True and ``fallback_reason`` documents the case, mirroring the
    bspline module's later IMM fallback slot.
    """
    source_ids: tuple[str, ...] = ()
    if report is not None:
        source_ids = tuple(report.belief.source_observation_ids)
    if report is None:
        mean = (0.0, 0.0)
    else:
        mean = (float(report.belief.mean[0]), float(report.belief.mean[1]))
    horizon_steps = max(1, int(horizon_s // sample_step_s))
    velocity = _public_velocity(report, samples, max_speed_mps=max_speed_mps)
    if len(samples) >= 2:
        last_t, last_x, last_y = samples[-1]
        elapsed = max(0.0, float(snapshot.sim_time_s) - float(last_t))
        anchor_time = max(float(last_t), float(snapshot.sim_time_s))
        anchor_x = float(last_x) + velocity[0] * elapsed
        anchor_y = float(last_y) + velocity[1] * elapsed
        times = tuple(anchor_time + (index + 1) * sample_step_s for index in range(horizon_steps))
        points = tuple(
            (
                anchor_x + velocity[0] * (index + 1) * sample_step_s,
                anchor_y + velocity[1] * (index + 1) * sample_step_s,
            )
            for index in range(horizon_steps)
        )
    else:
        anchor_t = samples[0][0] if samples else snapshot.sim_time_s
        anchor_xy = (float(samples[0][1]), float(samples[0][2])) if samples else mean
        elapsed = max(0.0, float(snapshot.sim_time_s) - float(anchor_t))
        anchor_t = max(float(anchor_t), float(snapshot.sim_time_s))
        anchor_xy = (
            anchor_xy[0] + velocity[0] * elapsed,
            anchor_xy[1] + velocity[1] * elapsed,
        )
        times = tuple(
            float(anchor_t) + (index + 1) * sample_step_s for index in range(horizon_steps)
        )
        points = tuple(anchor_xy for _ in range(horizon_steps))
    base_sigma = _base_sigma(position_block)
    corridor = _corridor(base_sigma, horizon_steps)
    return PredictedTrackRef(
        prediction_id=prediction_id,
        target_id=target_id,
        sim_time_s=_source_track_sim_time_s(snapshot, report, samples),
        horizon_s=horizon_s,
        sample_step_s=sample_step_s,
        times_s=times,
        points_xy=points,
        corridor_radius_m=corridor,
        source_belief_history_ids=source_ids,
        fallback_used=True,
        fallback_reason=(
            f"belief history has fewer than {MIN_HISTORY_POINTS} fixes"
            f" spanning {MIN_HISTORY_SPAN_S:.0f} s (got {len(samples)});"
            " linear short-history extrapolation used"
        ),
        prediction_regime="short_history",
        imm_model_probabilities=_model_probabilities(report),
    )


def _public_velocity(
    report: GroupReport | None,
    samples: Sequence[BeliefSample],
    *,
    max_speed_mps: float,
) -> tuple[float, float]:
    """Estimate stale-track velocity from public belief data only."""
    velocity: tuple[float, float] | None = None
    if report is not None and len(report.belief.mean) >= 4:
        velocity = (float(report.belief.mean[2]), float(report.belief.mean[3]))
    elif len(samples) >= 2:
        previous = samples[-2]
        latest = samples[-1]
        delta_s = max(float(latest[0] - previous[0]), 1.0)
        velocity = (
            (float(latest[1]) - float(previous[1])) / delta_s,
            (float(latest[2]) - float(previous[2])) / delta_s,
        )
    if velocity is None:
        return (0.0, 0.0)
    speed = math.hypot(*velocity)
    if speed <= max_speed_mps or speed <= 1e-12:
        return velocity
    scale = max_speed_mps / speed
    return velocity[0] * scale, velocity[1] * scale


def _source_track_sim_time_s(
    snapshot: SituationSnapshot,
    report: GroupReport | None,
    samples: Sequence[BeliefSample] = (),
) -> int:
    if report is not None:
        belief_time = getattr(report.belief, "sim_time_s", None)
        if belief_time is not None:
            return int(belief_time)
    if samples:
        return int(samples[-1][0])
    return int(snapshot.sim_time_s)


def _rebase_prediction_if_stale(
    snapshot: SituationSnapshot,
    report: GroupReport | None,
    samples: Sequence[BeliefSample],
    times: np.ndarray,
    points: np.ndarray,
    sample_step_s: float,
    max_speed_mps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Move a prediction whose last public fix predates the current cycle."""
    if not samples or float(samples[-1][0]) >= float(snapshot.sim_time_s):
        return times, points
    last_t, last_x, last_y = samples[-1]
    elapsed = float(snapshot.sim_time_s) - float(last_t)
    velocity = _public_velocity(
        report,
        samples,
        max_speed_mps=max_speed_mps,
    )
    anchor = np.asarray(
        [float(last_x) + velocity[0] * elapsed, float(last_y) + velocity[1] * elapsed],
        dtype=float,
    )
    stale_anchor = np.asarray([float(last_x), float(last_y)], dtype=float)
    rebased_points = np.asarray(points, dtype=float) + (anchor - stale_anchor)
    rebased_times = float(snapshot.sim_time_s) + sample_step_s * np.arange(
        1, len(times) + 1, dtype=float
    )
    return rebased_times, rebased_points


def _ref_from_prediction(
    prediction_id: str,
    snapshot: SituationSnapshot,
    target_id: str,
    report: GroupReport | None,
    times: np.ndarray,
    points: np.ndarray,
    corridor: np.ndarray,
    fallback_used: bool,
    horizon_s: float,
    sample_step_s: float,
) -> PredictedTrackRef:
    """One ``PredictedTrackRef`` mirroring the spline prediction arrays."""
    source_ids: tuple[str, ...] = ()
    if report is not None:
        source_ids = tuple(report.belief.source_observation_ids)
    return PredictedTrackRef(
        prediction_id=prediction_id,
        target_id=target_id,
        sim_time_s=_source_track_sim_time_s(snapshot, report),
        horizon_s=horizon_s,
        sample_step_s=sample_step_s,
        times_s=tuple(float(value) for value in times),
        points_xy=tuple((float(x), float(y)) for x, y in points),
        corridor_radius_m=tuple(float(value) for value in corridor),
        bspline_times_s=tuple(float(value) for value in times),
        bspline_centerline_xy=tuple((float(x), float(y)) for x, y in points),
        spline_degree=3,
        source_belief_history_ids=source_ids,
        fallback_used=fallback_used,
        prediction_regime="short_history" if fallback_used else "bspline",
        imm_model_probabilities=_model_probabilities(report),
    )


def _model_probabilities(report: GroupReport | None) -> dict[str, float]:
    if report is None:
        return {}
    return {
        str(label): float(probability)
        for label, probability in sorted(report.belief.model_probabilities.items())
    }
