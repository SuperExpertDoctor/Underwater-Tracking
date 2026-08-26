# src/underwater_tracking/prediction/port.py
"""Trajectory-predictor port over the real B-spline prediction module (spec 6.6).

``make_snapshot_predictor`` adapts ``predict_track`` — the real prediction
module — to the carrier's per-target predictor contract (one
``PredictedTrackRef`` per tracked target, sampled only from the target's
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
from collections.abc import Callable, Sequence

import numpy as np

from underwater_tracking.domain.agent_models import PredictedTrackRef
from underwater_tracking.domain.models import GroupReport, SituationSnapshot
from underwater_tracking.prediction.bspline import (
    MIN_HISTORY_POINTS,
    MIN_HISTORY_SPAN_S,
    predict_track,
)

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


def make_snapshot_predictor(
    *,
    belief_history: Callable[[SituationSnapshot, str], Sequence[BeliefSample]],
    horizon_s: float,
    sample_step_s: float,
    max_speed_mps: float = _DEFAULT_MAX_SPEED_MPS,
    max_turn_rate_rad_s: float = _DEFAULT_MAX_TURN_RATE_RAD_S,
) -> Callable[[SituationSnapshot, str], PredictedTrackRef]:
    """One deterministic per-target predictor over the B-spline module.

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
    fallback instead of failing the planning cycle.
    """

    def predict(snapshot: SituationSnapshot, target_id: str) -> PredictedTrackRef:
        samples = list(belief_history(snapshot, target_id))
        report = _group_report(snapshot, target_id)
        covariance = report.belief.covariance if report is not None else ()
        position_block = _position_block(covariance)
        prediction_id = f"{snapshot.scenario_id}:track:{target_id}:{snapshot.snapshot_revision}"
        span = samples[-1][0] - samples[0][0] if len(samples) >= 2 else 0.0
        if len(samples) >= MIN_HISTORY_POINTS and span >= MIN_HISTORY_SPAN_S:
            imm_prediction = _imm_prediction_ref(
                prediction_id,
                snapshot,
                target_id,
                report,
                samples,
                position_block,
                horizon_s,
                sample_step_s,
                max_speed_mps,
                max_turn_rate_rad_s,
            )
            if imm_prediction is not None:
                return imm_prediction
            prediction = predict_track(
                np.asarray([sample[0] for sample in samples], dtype=float),
                np.asarray([[sample[1], sample[2]] for sample in samples], dtype=float),
                np.repeat(position_block[np.newaxis, :, :], len(samples), axis=0),
                horizon_s,
                sample_step_s,
                max_speed_mps=max_speed_mps,
                max_turn_rate_rad_s=max_turn_rate_rad_s,
            )
            times, points = _rebase_prediction_if_stale(
                snapshot,
                report,
                samples,
                prediction.times_s,
                prediction.points_xy,
                sample_step_s,
                max_speed_mps,
            )
            return _ref_from_prediction(
                prediction_id,
                snapshot,
                target_id,
                report,
                times,
                points,
                prediction.corridor_radius_m,
                prediction.fallback_used,
                horizon_s,
                sample_step_s,
            )
        return _short_history_ref(
            prediction_id,
            snapshot,
            target_id,
            report,
            samples,
            position_block,
            horizon_s,
            sample_step_s,
            max_speed_mps,
        )

    return predict


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
    raw_probabilities = _model_probabilities(report)
    weights = {
        "cv": sum(
            probability
            for label, probability in raw_probabilities.items()
            if label.casefold() in {"cv", "constant_velocity"}
        ),
        "left_turn": sum(
            probability
            for label, probability in raw_probabilities.items()
            if label.casefold() in {"left_turn", "left", "ct_left"}
        ),
        "right_turn": sum(
            probability
            for label, probability in raw_probabilities.items()
            if label.casefold() in {"right_turn", "right", "ct_right"}
        ),
    }
    total = sum(weights.values())
    if total <= 1e-12:
        return None
    weights = {label: probability / total for label, probability in weights.items()}
    velocity = _public_velocity(report, samples, max_speed_mps=max_speed_mps)
    speed = math.hypot(*velocity)
    if speed <= 1e-9:
        return None
    last_t, last_x, last_y = samples[-1]
    stale_s = max(0.0, float(snapshot.sim_time_s) - float(last_t))
    anchor = (
        float(last_x) + velocity[0] * stale_s,
        float(last_y) + velocity[1] * stale_s,
    )
    heading = math.atan2(velocity[1], velocity[0])
    horizon_steps = max(1, int(horizon_s // sample_step_s))
    times = tuple(
        float(snapshot.sim_time_s) + (index + 1) * sample_step_s
        for index in range(horizon_steps)
    )
    model_points: dict[str, tuple[tuple[float, float], ...]] = {}
    for label, turn_rate in (
        ("cv", 0.0),
        ("left_turn", max_turn_rate_rad_s),
        ("right_turn", -max_turn_rate_rad_s),
    ):
        propagated: list[tuple[float, float]] = []
        for index in range(horizon_steps):
            elapsed = (index + 1) * sample_step_s
            if abs(turn_rate) <= 1e-12:
                point = (
                    anchor[0] + velocity[0] * elapsed,
                    anchor[1] + velocity[1] * elapsed,
                )
            else:
                point = (
                    anchor[0]
                    + speed
                    / turn_rate
                    * (math.sin(heading + turn_rate * elapsed) - math.sin(heading)),
                    anchor[1]
                    - speed
                    / turn_rate
                    * (math.cos(heading + turn_rate * elapsed) - math.cos(heading)),
                )
            propagated.append(point)
        model_points[label] = tuple(propagated)
    points = tuple(
        (
            sum(
                weights[label] * model_points[label][index][0]
                for label in weights
            ),
            sum(
                weights[label] * model_points[label][index][1]
                for label in weights
            ),
        )
        for index in range(horizon_steps)
    )
    base_sigma = _base_sigma(position_block)
    corridor = tuple(
        base_sigma
        + math.sqrt(
            sum(
                weights[label]
                * (
                    (model_points[label][index][0] - points[index][0]) ** 2
                    + (model_points[label][index][1] - points[index][1]) ** 2
                )
                for label in weights
            )
        )
        for index in range(horizon_steps)
    )
    return PredictedTrackRef(
        prediction_id=prediction_id,
        target_id=target_id,
        sim_time_s=snapshot.sim_time_s,
        horizon_s=horizon_s,
        sample_step_s=sample_step_s,
        times_s=times,
        points_xy=points,
        corridor_radius_m=corridor,
        source_belief_history_ids=tuple(report.belief.source_observation_ids),
        fallback_used=False,
        prediction_regime="imm",
        imm_model_probabilities=raw_probabilities,
    )


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
        sim_time_s=snapshot.sim_time_s,
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
        sim_time_s=snapshot.sim_time_s,
        horizon_s=horizon_s,
        sample_step_s=sample_step_s,
        times_s=tuple(float(value) for value in times),
        points_xy=tuple((float(x), float(y)) for x, y in points),
        corridor_radius_m=tuple(float(value) for value in corridor),
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
