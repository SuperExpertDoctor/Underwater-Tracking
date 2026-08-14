"""Deterministic robust receding-horizon waypoint planning for a UUV group.

``plan_group_waypoints`` builds a short rolling waypoint sequence for
every UUV in a tracking group. Each horizon step poses one joint
assignment problem: pick one candidate waypoint per UUV from a discrete
lattice of relative bearings (every 15 degrees) and standoff radii
(spread between the sensor minimum and maximum range), subject to
per-step motion, scenario-boundary, and pairwise-separation
constraints. Joint choices are scored by the lexicographic tuple

    (worst_case_fim_min_eigenvalue, worst_case_logdet,
     -energy_cost, -waypoint_change_cost)

where the information metrics are the minimum over every target sigma
point. A deterministic beam search expands one UUV at a time and keeps
the ``beam_width`` most promising partial assignments by ascending
negated-score key with a coordinate tie-break; if the beam result ever
violates the separation constraint and a feasible joint assignment
exists, a bounded exhaustive search certifies a feasible replacement.
The first waypoint of each UUV's sequence is the committed waypoint
(``WaypointPlan.waypoints_xy``). Scoring calls the shared batched FIM
evaluator ``fim.bearing_fim_batch`` (the same bearing-only operator as
``fim.bearing_fim`` and the same reductions as ``fim.fim_metrics``,
batched over joint candidates and sigma points at once), so the
planner always reflects any change to the FIM conventions in
``planning/fim.py``; ``tests/planning/test_waypoints.py`` pins the
planner scores against the per-call functions.

``WaypointPlan.separation_violated`` is ``True`` whenever the returned
sequence contains a step whose waypoints are genuinely closer than
``min_separation_m`` (beyond the float-boundary tolerance). This can
happen when no feasible joint exists, or when the exhaustive
certification search is capped (see ``_EXHAUSTIVE_LIMIT``); the
planner still returns the best fallback so callers can degrade
gracefully instead of crashing.

All functions are pure: no randomness, no state. Inputs are only ever
reordered by the optional ``uuv_ids`` label, so permuting the input
rows changes the result by UUV id, never by array index.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from underwater_tracking.planning.fim import bearing_fim_batch

# Smallest squared standoff (m^2) between a waypoint and any target
# sigma point, mirroring the coincident-observer guard in
# ``fim.bearing_fim``.
_MIN_RANGE_SQUARED = 1e-12

# Bearing lattice step (degrees): relative bearings every 15 degrees.
_BEARING_STEP_DEG = 15.0

# Default scenario bounds ``(xmin, xmax, ymin, ymax)``: the simulation
# box used across the foundation plan.
_DEFAULT_BOUNDS: tuple[float, float, float, float] = (
    -5000.0,
    5000.0,
    -5000.0,
    5000.0,
)

# Cap on the exhaustive certification search (product of per-UUV
# candidate counts). Groups are 2-4 UUVs with at most ~120 lattice
# candidates each, but the cap keeps the repair path bounded.
_EXHAUSTIVE_LIMIT = 200_000

# Absolute tolerance (m^2) on the step-length and separation bounds.
# Lattice points at radii spaced 300 m apart are exactly 600 m apart, so
# a candidate can sit exactly on ``max_step_m``; the squared-distance
# comparison then flips with float rounding (order ~1e-9 m^2) and makes
# feasibility frame-dependent under rigid transforms. The tolerance
# admits such boundary cases consistently (max_step + 5e-7 m worst
# case), which the constraint test tolerates.
_BOUND_TOLERANCE_SQUARED = 1e-6


@dataclass(frozen=True)
class WaypointPlan:
    """First committed waypoint per UUV plus the full short sequence.

    ``waypoints_xy`` has shape ``(n_uuvs, 2)``: the first point of each
    UUV's rolling sequence, in the same row order as the input
    ``uuv_positions``. ``sequence_xy`` has shape
    ``(n_uuvs, horizon_steps, 2)``: the full short sequence each UUV
    would follow if the plan is re-executed with rolling first-point
    commitment. ``separation_violated`` is ``True`` whenever any step of
    ``sequence_xy`` places two UUVs genuinely closer than
    ``min_separation_m`` (beyond the float-boundary tolerance): either
    no feasible joint exists for that step, or the exhaustive
    certification search was capped and could not find one.
    """

    waypoints_xy: np.ndarray
    sequence_xy: np.ndarray
    separation_violated: bool = False


def plan_group_waypoints(
    uuv_positions: np.ndarray,
    target_sigma_points: np.ndarray,
    previous_waypoints: np.ndarray | None,
    max_step_m: float,
    min_separation_m: float,
    bearing_variance: float,
    beam_width: int,
    *,
    uuv_ids: Sequence[str] | None = None,
    min_range_m: float = 500.0,
    max_range_m: float = 4000.0,
    range_bins: int = 5,
    horizon_steps: int = 3,
    bounds: tuple[float, float, float, float] = _DEFAULT_BOUNDS,
) -> WaypointPlan:
    """Plan a robust short waypoint sequence for a UUV group.

    ``uuv_positions`` has shape ``(n_uuvs, 2)``, ``target_sigma_points``
    has shape ``(n_sigma, 2)``, and ``previous_waypoints`` either is
    ``None`` or has the same shape as ``uuv_positions``. ``bounds`` is
    ``(xmin, xmax, ymin, ymax)``. ``uuv_ids`` optionally labels each
    input row; expansion always proceeds in sorted id order so results
    are invariant to input row permutation. The returned plan's
    ``waypoints_xy`` holds the first (committed) waypoint per UUV; the
    full rolling sequence is available as ``sequence_xy``.
    """
    uuv_positions = np.asarray(uuv_positions, dtype=float)
    target_sigma_points = np.asarray(target_sigma_points, dtype=float)
    previous = (
        None if previous_waypoints is None else np.asarray(previous_waypoints, dtype=float)
    )
    _validate_inputs(
        uuv_positions,
        target_sigma_points,
        previous,
        max_step_m,
        min_separation_m,
        bearing_variance,
        beam_width,
        uuv_ids,
        min_range_m,
        max_range_m,
        range_bins,
        horizon_steps,
        bounds,
    )
    order = _expansion_order(uuv_ids, uuv_positions.shape[0])
    positions = uuv_positions[order]
    previous_in_order = None if previous is None else previous[order]

    steps: list[np.ndarray] = []
    current = positions
    current_previous = previous_in_order
    separation_violated = False
    for _ in range(horizon_steps):
        waypoints = _beam_search(
            current,
            target_sigma_points,
            current_previous,
            max_step_m,
            min_separation_m,
            bearing_variance,
            beam_width,
            min_range_m,
            max_range_m,
            range_bins,
            bounds,
        )
        steps.append(waypoints)
        if _violates_separation(waypoints, min_separation_m):
            separation_violated = True
        current = waypoints
        # Only the first step has a reference for the change cost.
        current_previous = None
    sequence = np.stack(steps, axis=1)
    input_order = np.argsort(np.asarray(order, dtype=np.int64))
    sequence_in_input_order = sequence[input_order]
    return WaypointPlan(
        waypoints_xy=sequence_in_input_order[:, 0, :],
        sequence_xy=sequence_in_input_order,
        separation_violated=separation_violated,
    )


def _beam_search(
    positions: np.ndarray,
    sigma_points: np.ndarray,
    previous: np.ndarray | None,
    max_step_m: float,
    min_separation_m: float,
    bearing_variance: float,
    beam_width: int,
    min_range_m: float,
    max_range_m: float,
    range_bins: int,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    """Jointly choose one waypoint per UUV for a single horizon step."""
    n_uuvs = positions.shape[0]
    lattice = _candidate_lattice(
        sigma_points.mean(axis=0), min_range_m, max_range_m, range_bins
    )
    candidates_by_uuv = [
        _feasible_candidates(positions[i], lattice, sigma_points, max_step_m, bounds)
        for i in range(n_uuvs)
    ]
    for i, candidates in enumerate(candidates_by_uuv):
        if candidates.shape[0] == 0:
            if _min_squared_distance(positions[i], sigma_points) < _MIN_RANGE_SQUARED:
                raise ValueError(
                    "uuv has no feasible waypoint and its position coincides with a "
                    "target sigma point"
                )
            # Nothing reachable: hold position as the only option.
            candidates_by_uuv[i] = positions[i][None, :].copy()

    beam: list[tuple[np.ndarray, np.ndarray]] = [(np.empty((0,)), np.empty((0, 2)))]
    for i in range(n_uuvs):
        beam_arrays = np.stack([entry[1] for entry in beam])
        chosen_sets = [
            candidates_by_uuv[i][
                _separated_mask(candidates_by_uuv[i], partial, min_separation_m)
            ]
            for partial in beam_arrays
        ]
        counts = np.asarray([candidates.shape[0] for candidates in chosen_sets])
        if counts.sum() > 0:
            expanded = np.repeat(beam_arrays, counts, axis=0)
            chosen_all = np.concatenate(chosen_sets, axis=0)
            joint = np.concatenate([expanded, chosen_all[:, None, :]], axis=1)
            scores = _score_joint_batch(
                joint, sigma_points, positions[: i + 1], previous, bearing_variance
            )
            # Ascending key of the negated score tuple: minimize
            # ``-min_eig`` (maximize min_eig), then ``-logdet``, then the
            # positive costs, then coordinates as the final tie-break.
            keys = np.concatenate(
                [-_score_key_columns(scores), joint.reshape(joint.shape[0], -1)],
                axis=1,
            )
            beam = _keep_best(keys, joint, beam_width)
        else:
            # Every candidate collides with every partial; keep the
            # partials extended with the current position, unconstrained.
            current_position = positions[i][None, :]
            joint = np.concatenate(
                [beam_arrays, np.repeat(current_position, beam_arrays.shape[0], axis=0)[:, None, :]],
                axis=1,
            )
            scores = _score_joint_batch(
                joint, sigma_points, positions[: i + 1], previous, bearing_variance
            )
            keys = np.concatenate(
                [-_score_key_columns(scores), joint.reshape(joint.shape[0], -1)],
                axis=1,
            )
            beam = [
                (keys[index], joint[index]) for index in range(joint.shape[0])
            ]

    result = beam[0][1]
    # Consistent with ``_separated_mask``: only pairs genuinely closer
    # than ``min_separation_m`` (beyond the bound tolerance) certify.
    if _violates_separation(result, min_separation_m):
        repaired = _best_feasible_joint(
            candidates_by_uuv,
            sigma_points,
            positions,
            previous,
            bearing_variance,
            min_separation_m,
        )
        if repaired is not None:
            result = repaired
    return result


def _candidate_lattice(
    target_mean: np.ndarray,
    min_range_m: float,
    max_range_m: float,
    range_bins: int,
) -> np.ndarray:
    """Waypoint lattice: every 15 degrees, radii from min to max range."""
    bearings = np.deg2rad(np.arange(0.0, 360.0, _BEARING_STEP_DEG))
    radii = np.linspace(min_range_m, max_range_m, range_bins)
    x = np.outer(np.cos(bearings), radii).ravel()
    y = np.outer(np.sin(bearings), radii).ravel()
    lattice: np.ndarray = np.column_stack([x, y])
    centered: np.ndarray = lattice + target_mean
    return centered


def _feasible_candidates(
    position: np.ndarray,
    lattice: np.ndarray,
    sigma_points: np.ndarray,
    max_step_m: float,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    """Lattice points within step range, inside the scenario box, and
    not coincident with any sigma point."""
    delta = lattice - position
    within_step = (
        np.sum(delta * delta, axis=1)
        <= max_step_m * max_step_m + _BOUND_TOLERANCE_SQUARED
    )
    in_bounds = (
        (lattice[:, 0] >= bounds[0])
        & (lattice[:, 0] <= bounds[1])
        & (lattice[:, 1] >= bounds[2])
        & (lattice[:, 1] <= bounds[3])
    )
    squared_to_sigma = np.sum((lattice[:, None, :] - sigma_points[None, :, :]) ** 2, axis=2)
    away_from_sigma = np.min(squared_to_sigma, axis=1) >= _MIN_RANGE_SQUARED
    feasible: np.ndarray = lattice[within_step & in_bounds & away_from_sigma]
    return feasible


def _score_joint_batch(
    waypoints: np.ndarray,
    sigma_points: np.ndarray,
    positions: np.ndarray,
    previous: np.ndarray | None,
    bearing_variance: float,
) -> np.ndarray:
    """The brief's score tuple for a batch of joint assignments.

    ``waypoints`` has shape ``(P, k, 2)`` with one row per joint
    assignment; returns a ``(P, 4)`` array of ``(worst_min_eigenvalue,
    worst_logdet, -energy_cost, -waypoint_change_cost)`` rows. The
    information columns come from ``fim.bearing_fim_batch`` (the same
    bearing-only operator and worst-case reduction as
    ``fim.bearing_fim``/``fim.fim_metrics``); costs are the squared
    energy spent reaching the waypoints and the squared deviation from
    the previous committed waypoints.
    """
    worst_min_eigenvalue, worst_logdet = bearing_fim_batch(
        waypoints, sigma_points, bearing_variance
    )
    energy_cost = np.sum((waypoints - positions[None, :, :]) ** 2, axis=(1, 2))
    if previous is None:
        change_cost = np.zeros(waypoints.shape[0])
    else:
        # ``previous`` rows align with the assigned UUV prefix only.
        previous_rows = previous[: waypoints.shape[1]]
        change_cost = np.sum((waypoints - previous_rows[None, :, :]) ** 2, axis=(1, 2))
    return np.stack(
        [worst_min_eigenvalue, worst_logdet, -energy_cost, -change_cost], axis=1
    )


def _score_key_columns(scores: np.ndarray) -> np.ndarray:
    """Round score columns onto a fine grid before lexicographic keying.

    The FIM metrics are eigendecompositions of sums of rank-one
    observer terms, so congruent joint assignments (an observer
    permutation of the same waypoint set) compute identical metrics up
    to floating-point summation rounding of order 1e-19. Keying on the
    raw floats would let that rounding pick the winner, and the
    rounding is frame-dependent under rigid transforms. Snapping onto a
    grid far above the rounding noise (1e-10, versus ~1e-19) and far
    below the information differences that separate genuinely distinct
    lattice geometries in the scenario space (order 1e-5 or larger)
    makes congruent joints tie on the information columns, so the exact
    energy and change-cost columns -- and finally the coordinate
    tie-break -- decide deterministically.
    """
    return np.round(scores, 10)


def _keep_best(
    keys: np.ndarray,
    joint: np.ndarray,
    beam_width: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deduplicate by key and keep the ``beam_width`` best entries.

    ``np.unique`` sorts the full keys lexicographically (negated score
    first, then ascending waypoint coordinates), so the first
    occurrences of the smallest rows are the best partial assignments.
    """
    _, indices = np.unique(keys, axis=0, return_index=True)
    selected = indices[:beam_width]
    return [(keys[index], joint[index]) for index in selected]


def _separated_mask(
    candidates: np.ndarray,
    partial: np.ndarray,
    min_separation_m: float,
) -> np.ndarray:
    """Boolean mask: candidates at least ``min_separation_m`` from every
    waypoint already assigned in ``partial``."""
    if partial.shape[0] == 0:
        return np.ones(candidates.shape[0], dtype=bool)
    squared = np.sum(
        (candidates[:, None, :] - partial[None, :, :]) ** 2, axis=2
    )
    separated: np.ndarray = (
        np.min(squared, axis=1)
        >= min_separation_m * min_separation_m - _BOUND_TOLERANCE_SQUARED
    )
    return separated


def _best_feasible_joint(
    candidates_by_uuv: list[np.ndarray],
    sigma_points: np.ndarray,
    positions: np.ndarray,
    previous: np.ndarray | None,
    bearing_variance: float,
    min_separation_m: float,
) -> np.ndarray | None:
    """Exhaustive (bounded) best joint assignment respecting separation.

    Used only to certify the beam result: whenever a feasible joint
    assignment exists, returns the best one by the same score key, and
    ``None`` otherwise. Skipped when the candidate product exceeds
    ``_EXHAUSTIVE_LIMIT``.
    """
    n_uuvs = positions.shape[0]
    total = math.prod(int(candidates.shape[0]) for candidates in candidates_by_uuv)
    if total > _EXHAUSTIVE_LIMIT:
        return None
    leaves: list[np.ndarray] = []

    def visit(uuv: int, partial: np.ndarray) -> None:
        if uuv == n_uuvs:
            leaves.append(partial)
            return
        chosen = candidates_by_uuv[uuv][
            _separated_mask(candidates_by_uuv[uuv], partial, min_separation_m)
        ]
        for candidate in chosen:
            visit(uuv + 1, np.concatenate([partial, candidate[None, :]]))

    visit(0, np.empty((0, 2)))
    if not leaves:
        return None
    joint = np.stack(leaves)
    scores = _score_joint_batch(joint, sigma_points, positions, previous, bearing_variance)
    keys = np.concatenate(
        [-_score_key_columns(scores), joint.reshape(joint.shape[0], -1)], axis=1
    )
    best_index = int(np.lexsort(keys.T[::-1])[0])
    best_joint: np.ndarray = joint[best_index]
    return best_joint


def _violates_separation(positions: np.ndarray, min_separation_m: float) -> bool:
    """True when a pair of waypoints is genuinely closer than
    ``min_separation_m`` (beyond the float-boundary tolerance admitted
    for step/separation feasibility)."""
    return _min_pairwise_distance(positions) < min_separation_m - math.sqrt(
        _BOUND_TOLERANCE_SQUARED
    )


def _min_pairwise_distance(positions: np.ndarray) -> float:
    if positions.shape[0] < 2:
        return float("inf")
    minimum = float("inf")
    for i in range(positions.shape[0]):
        for j in range(i + 1, positions.shape[0]):
            squared = float(np.sum((positions[i] - positions[j]) ** 2))
            minimum = min(minimum, math.sqrt(squared))
    return minimum


def _min_squared_distance(position: np.ndarray, points: np.ndarray) -> float:
    squared = np.sum((points - position) ** 2, axis=1)
    return float(np.min(squared))


def _expansion_order(uuv_ids: Sequence[str] | None, n_uuvs: int) -> list[int]:
    if uuv_ids is None:
        return list(range(n_uuvs))
    return sorted(range(n_uuvs), key=lambda i: uuv_ids[i])


def _validate_inputs(
    uuv_positions: np.ndarray,
    target_sigma_points: np.ndarray,
    previous: np.ndarray | None,
    max_step_m: float,
    min_separation_m: float,
    bearing_variance: float,
    beam_width: int,
    uuv_ids: Sequence[str] | None,
    min_range_m: float,
    max_range_m: float,
    range_bins: int,
    horizon_steps: int,
    bounds: tuple[float, float, float, float],
) -> None:
    if uuv_positions.ndim != 2 or uuv_positions.shape[1] != 2:
        raise ValueError(f"uuv_positions must have shape (n, 2), got {uuv_positions.shape}")
    n_uuvs = uuv_positions.shape[0]
    if n_uuvs < 1:
        raise ValueError("at least one uuv position is required")
    if target_sigma_points.ndim != 2 or target_sigma_points.shape[1] != 2:
        raise ValueError(
            f"target_sigma_points must have shape (n, 2), got {target_sigma_points.shape}"
        )
    if target_sigma_points.shape[0] < 1:
        raise ValueError("at least one target sigma point is required")
    if previous is not None and previous.shape != (n_uuvs, 2):
        raise ValueError(
            f"previous_waypoints must have shape (n, 2), got {previous.shape}"
        )
    if not max_step_m > 0.0:
        raise ValueError("max_step_m must be strictly positive")
    if not min_separation_m >= 0.0:
        raise ValueError("min_separation_m must be non-negative")
    if not bearing_variance > 0.0:
        raise ValueError("bearing_variance must be strictly positive")
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1")
    if uuv_ids is not None and (len(uuv_ids) != n_uuvs or len(set(uuv_ids)) != n_uuvs):
        raise ValueError("uuv_ids must be unique and match the uuv count")
    if not min_range_m > 0.0:
        raise ValueError("min_range_m must be strictly positive")
    if not max_range_m >= min_range_m:
        raise ValueError("max_range_m must not be below min_range_m")
    if range_bins < 1:
        raise ValueError("range_bins must be at least 1")
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be at least 1")
    xmin, xmax, ymin, ymax = bounds
    if not (xmin < xmax and ymin < ymax):
        raise ValueError("bounds must satisfy xmin < xmax and ymin < ymax")
