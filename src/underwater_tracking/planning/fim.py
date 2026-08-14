"""Deterministic bearing-only Fisher information metrics.

``bearing_fim`` accumulates the 2x2 information matrix of a set of
bearing measurements about a target. Each observer contributes its
outer-product Jacobian ``H.T @ H`` scaled by the inverse bearing
variance, where ``H = [-dy/r^2, dx/r^2]`` is the gradient of the
bearing angle with respect to the target position. ``fim_metrics``
reduces the matrix to the minimum eigenvalue, the log-determinant, and
the condition number, all of which the group quality calculator and the
waypoint planner consume. ``bearing_fim_batch`` evaluates the same
operator and reductions over a batch of joint assignments, returning
per-joint worst-case metrics; the waypoint planner's joint scorer
consumes it so the planner always reflects any change to the FIM
conventions. All functions are pure: no randomness, no state.
"""

from dataclasses import dataclass

import numpy as np

# Smallest squared standoff (m^2) accepted before an observer is treated
# as coincident with the target.
_MIN_RANGE_SQUARED = 1e-12


@dataclass(frozen=True)
class FimMetrics:
    """Scalar observability summary of a 2x2 Fisher information matrix.

    ``condition_number`` is ``inf`` for a degenerate (rank-deficient)
    matrix; ``logdet`` is ``-inf`` in that case.
    """

    min_eigenvalue: float
    logdet: float
    condition_number: float


def bearing_fim(
    target: np.ndarray,
    observer_positions: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """Accumulate the bearing-only Fisher information matrix.

    ``target`` has shape ``(2,)``; ``observer_positions`` has shape
    ``(n, 2)`` with one row per observer; ``variances`` has shape
    ``(n,)`` with the bearing variance (rad^2) of each observer.
    Returns a symmetric positive-semidefinite ``(2, 2)`` matrix.
    """
    target = np.asarray(target, dtype=float)
    observer_positions = np.asarray(observer_positions, dtype=float)
    variances = np.asarray(variances, dtype=float)
    _validate_bearing_fim_inputs(target, observer_positions, variances)
    dx = observer_positions[:, 0] - target[0]
    dy = observer_positions[:, 1] - target[1]
    range_squared = dx * dx + dy * dy
    jacobian = np.column_stack([-dy / range_squared, dx / range_squared])
    # ``optimize=False`` pins the direct per-term einsum accumulation so
    # ``bearing_fim_batch`` can mirror it bit-for-bit.
    fim: np.ndarray = np.einsum(
        "ki,kj,k->ij", jacobian, jacobian, 1.0 / variances, optimize=False
    )
    return fim


def bearing_fim_batch(
    waypoints: np.ndarray,
    sigma_points: np.ndarray,
    bearing_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Worst-case FIM metrics over a batch of joint assignments.

    ``waypoints`` has shape ``(P, k, 2)`` with one row per joint
    assignment of ``k`` observers; ``sigma_points`` has shape
    ``(n, 2)``. For every joint and every sigma point the 2x2
    bearing-only information matrix of the ``k`` observers about that
    target is accumulated with the exact per-term arithmetic of
    ``bearing_fim`` (the inverse-variance weight multiplies each
    rank-one term inside the einsum contraction, so both paths are
    bit-identical even for degenerate matrices), then reduced with the
    same conventions as ``fim_metrics``, including its full-rank
    predicate. Returns the per-joint worst case over sigma points as a
    ``(min_eigenvalue, logdet)`` tuple of ``(P,)`` arrays; ``logdet``
    is ``-inf`` for a singular matrix, matching ``fim_metrics``. The
    waypoint planner's joint scorer consumes this batched form.
    """
    waypoints = np.asarray(waypoints, dtype=float)
    sigma_points = np.asarray(sigma_points, dtype=float)
    _validate_bearing_fim_batch_inputs(waypoints, sigma_points, bearing_variance)
    dx = waypoints[:, :, 0][:, None, :] - sigma_points[:, 0][None, :, None]
    dy = waypoints[:, :, 1][:, None, :] - sigma_points[:, 1][None, :, None]
    range_squared = dx * dx + dy * dy
    jacobian_x = -dy / range_squared
    jacobian_y = dx / range_squared
    # The weight multiplies each rank-one term inside the contraction,
    # mirroring ``bearing_fim``'s ``einsum("ki,kj,k->ij", ...)`` operand
    # by operand, so the batch matrices are bit-identical to the
    # per-call ones. Summing first and scaling after (``einsum(...) *
    # weight``) instead leaves a different rounding pattern, which
    # amplifies into disagreement for near-singular matrices.
    weights = np.full(jacobian_x.shape, 1.0 / bearing_variance)
    fim_xx = np.einsum(
        "pmk,pmk,pmk->pm", jacobian_x, jacobian_x, weights, optimize=False
    )
    fim_yy = np.einsum(
        "pmk,pmk,pmk->pm", jacobian_y, jacobian_y, weights, optimize=False
    )
    fim_xy = np.einsum(
        "pmk,pmk,pmk->pm", jacobian_x, jacobian_y, weights, optimize=False
    )
    fim: np.ndarray = np.empty(fim_xx.shape + (2, 2))
    fim[:, :, 0, 0] = fim_xx
    fim[:, :, 1, 1] = fim_yy
    fim[:, :, 0, 1] = fim_xy
    fim[:, :, 1, 0] = fim_xy
    eigenvalues: np.ndarray = np.linalg.eigvalsh(fim)
    min_eigenvalue: np.ndarray = np.maximum(0.0, eigenvalues[:, :, 0])
    worst_min_eigenvalue: np.ndarray = np.min(min_eigenvalue, axis=1)
    # Singular matrices (e.g. single-observer prefixes) legitimately
    # produce ``logdet = -inf``; silence the divide-by-zero inside
    # slogdet, which is expected here.
    with np.errstate(divide="ignore", invalid="ignore"):
        sign, logdet = np.linalg.slogdet(fim)
    # Same full-rank convention as ``fim_metrics``: the maximum
    # eigenvalue is compared against the clamped minimum, exactly as
    # the per-call path does.
    max_eigenvalue = np.maximum(min_eigenvalue, eigenvalues[:, :, 1])
    finite_logdet = _is_finite_logdet(sign, max_eigenvalue)
    logdet_values: np.ndarray = np.where(finite_logdet, logdet, float("-inf"))
    worst_logdet: np.ndarray = np.min(logdet_values, axis=1)
    return worst_min_eigenvalue, worst_logdet


def fim_metrics(fim: np.ndarray) -> FimMetrics:
    """Reduce a 2x2 information matrix to scalar observability metrics.

    Tiny negative eigenvalues caused by floating-point rounding are
    clamped to zero so a positive-semidefinite input never reports a
    negative minimum eigenvalue. The condition number is ``inf`` when
    the minimum eigenvalue is zero; ``logdet`` follows ``slogdet`` and
    is ``-inf`` for a singular matrix.
    """
    fim = np.asarray(fim, dtype=float)
    if fim.shape != (2, 2):
        raise ValueError(f"fim must have shape (2, 2), got {fim.shape}")
    eigenvalues = np.linalg.eigvalsh(fim)
    min_eigenvalue = max(0.0, float(eigenvalues[0]))
    max_eigenvalue = max(min_eigenvalue, float(eigenvalues[1]))
    # Singular matrices (e.g. rank-one geometries) legitimately produce
    # ``logdet = -inf``; silence the divide-by-zero inside slogdet, which
    # is expected here (mirroring the batch path).
    with np.errstate(divide="ignore", invalid="ignore"):
        sign, logdet = np.linalg.slogdet(fim)
    if bool(_is_finite_logdet(np.asarray(sign), np.asarray(max_eigenvalue))):
        logdet_value = float(logdet)
    else:
        logdet_value = float("-inf")
    if min_eigenvalue > 0.0:
        condition_number = max_eigenvalue / min_eigenvalue
    else:
        condition_number = float("inf")
    return FimMetrics(
        min_eigenvalue=min_eigenvalue,
        logdet=logdet_value,
        condition_number=condition_number,
    )


def _is_finite_logdet(sign: np.ndarray, max_eigenvalue: np.ndarray) -> np.ndarray:
    """Full-rank predicate shared by the batch and per-call reductions.

    The log-determinant of a symmetric information matrix is only
    meaningful when the determinant sign is positive AND the maximum
    eigenvalue is positive; degenerate (rank-deficient, or indefinite
    as an artifact of floating-point rounding) matrices collapse to
    ``-inf``. ``fim_metrics`` and ``bearing_fim_batch`` both route
    their degenerate decision through this predicate so the two paths
    agree for every matrix.
    """
    return (sign > 0.0) & (max_eigenvalue > 0.0)


def _validate_bearing_fim_batch_inputs(
    waypoints: np.ndarray,
    sigma_points: np.ndarray,
    bearing_variance: float,
) -> None:
    if waypoints.ndim != 3 or waypoints.shape[2] != 2:
        raise ValueError(f"waypoints must have shape (P, k, 2), got {waypoints.shape}")
    if waypoints.shape[0] < 1:
        raise ValueError("at least one joint assignment is required")
    if waypoints.shape[1] < 1:
        raise ValueError("at least one observer per joint is required")
    if sigma_points.ndim != 2 or sigma_points.shape[1] != 2:
        raise ValueError(
            f"sigma_points must have shape (n, 2), got {sigma_points.shape}"
        )
    if sigma_points.shape[0] < 1:
        raise ValueError("at least one sigma point is required")
    if not bearing_variance > 0.0:
        raise ValueError("bearing_variance must be strictly positive")
    dx = waypoints[:, :, 0][:, None, :] - sigma_points[:, 0][None, :, None]
    dy = waypoints[:, :, 1][:, None, :] - sigma_points[:, 1][None, :, None]
    if np.any(dx * dx + dy * dy < _MIN_RANGE_SQUARED):
        raise ValueError("an observer position coincides with a target sigma point")


def _validate_bearing_fim_inputs(
    target: np.ndarray,
    observer_positions: np.ndarray,
    variances: np.ndarray,
) -> None:
    if target.shape != (2,):
        raise ValueError(f"target must have shape (2,), got {target.shape}")
    if observer_positions.ndim != 2 or observer_positions.shape[1] != 2:
        raise ValueError(
            f"observer_positions must have shape (n, 2), got {observer_positions.shape}"
        )
    if observer_positions.shape[0] < 1:
        raise ValueError("at least one observer position is required")
    if variances.shape != (len(observer_positions),):
        raise ValueError(
            "variances must have shape (n,) matching the observer count, "
            f"got {variances.shape}"
        )
    if np.any(variances <= 0.0):
        raise ValueError("bearing variances must be strictly positive")
    dx = observer_positions[:, 0] - target[0]
    dy = observer_positions[:, 1] - target[1]
    if np.any(dx * dx + dy * dy < _MIN_RANGE_SQUARED):
        raise ValueError("an observer position coincides with the target")
