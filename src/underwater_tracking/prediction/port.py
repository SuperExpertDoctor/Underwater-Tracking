# src/underwater_tracking/prediction/port.py
"""Trajectory-predictor port over the real B-spline prediction module (spec 6.6).

``make_snapshot_predictor`` adapts ``predict_track`` — the real prediction
module — to the carrier's per-target predictor contract (one
``PredictedTrackRef`` per tracked target, sampled from the target's
estimated belief history). Covariances are not part of the engine's belief
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

# Default physical limits mirror the simulation configuration: the
# ``tracking.uuv_max_speed_mps`` knob (4 m/s max speed) and the
# ``tracking.uuv_max_turn_rate_rad_s`` knob (pi/60 rad/s max turn rate).
_DEFAULT_MAX_SPEED_MPS = 4.0
_DEFAULT_MAX_TURN_RATE_RAD_S = math.pi / 60.0

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

    ``belief_history`` must return the target's estimated position history
    as ``(sim_time_s, x, y)`` samples; the returned predictor is pure in
    the snapshot (same snapshot and history always yield the same
    ``PredictedTrackRef``). The default physical limits mirror the
    configured ``tracking.uuv_max_speed_mps`` (4 m/s) and
    ``tracking.uuv_max_turn_rate_rad_s`` (pi/60 rad/s) knobs. Predictions
    with fewer than ``MIN_HISTORY_POINTS`` fixes spanning
    ``MIN_HISTORY_SPAN_S`` are served by the documented short-history
    fallback instead of failing the planning cycle.
    """

    def predict(snapshot: SituationSnapshot, target_id: str) -> PredictedTrackRef:
        samples = list(belief_history(snapshot, target_id))
        report = _group_report(snapshot, target_id)
        covariance = report.belief.covariance if report is not None else ()
        position_block = _position_block(covariance)
        prediction_id = (
            f"{snapshot.scenario_id}:track:{target_id}:{snapshot.snapshot_revision}"
        )
        span = samples[-1][0] - samples[0][0] if len(samples) >= 2 else 0.0
        if len(samples) >= MIN_HISTORY_POINTS and span >= MIN_HISTORY_SPAN_S:
            prediction = predict_track(
                np.asarray([sample[0] for sample in samples], dtype=float),
                np.asarray(
                    [[sample[1], sample[2]] for sample in samples], dtype=float
                ),
                np.repeat(
                    position_block[np.newaxis, :, :], len(samples), axis=0
                ),
                horizon_s,
                sample_step_s,
                max_speed_mps=max_speed_mps,
                max_turn_rate_rad_s=max_turn_rate_rad_s,
            )
            return _ref_from_prediction(
                prediction_id,
                snapshot,
                target_id,
                report,
                prediction.times_s,
                prediction.points_xy,
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
        )

    return predict


def _group_report(
    snapshot: SituationSnapshot, target_id: str
) -> GroupReport | None:
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
    return tuple(
        base_sigma + 10.0 * (step_index / max(steps, 1)) for step_index in range(steps)
    )


def _short_history_ref(
    prediction_id: str,
    snapshot: SituationSnapshot,
    target_id: str,
    report: GroupReport | None,
    samples: Sequence[BeliefSample],
    position_block: np.ndarray,
    horizon_s: float,
    sample_step_s: float,
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
    if len(samples) >= 2:
        last_t, last_x, last_y = samples[-1]
        prev_t, prev_x, prev_y = samples[-2]
        delta = max(float(last_t - prev_t), 1.0)
        step_x = (float(last_x) - float(prev_x)) / delta * sample_step_s
        step_y = (float(last_y) - float(prev_y)) / delta * sample_step_s
        times = tuple(
            float(last_t) + (index + 1) * sample_step_s for index in range(horizon_steps)
        )
        points = tuple(
            (float(last_x) + step_x * (index + 1), float(last_y) + step_y * (index + 1))
            for index in range(horizon_steps)
        )
    else:
        anchor_t = samples[0][0] if samples else snapshot.sim_time_s
        anchor_xy = (float(samples[0][1]), float(samples[0][2])) if samples else mean
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
    )


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
    )
