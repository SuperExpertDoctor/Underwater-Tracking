"""Deterministic bearing-only Fisher information metrics.

``bearing_fim`` accumulates the 2x2 information matrix of a set of
bearing measurements about a target. Each observer contributes its
outer-product Jacobian ``H.T @ H`` scaled by the inverse bearing
variance, where ``H = [-dy/r^2, dx/r^2]`` is the gradient of the
bearing angle with respect to the target position. ``fim_metrics``
reduces the matrix to the minimum eigenvalue, the log-determinant, and
the condition number, all of which the group quality calculator and the
waypoint planner consume. Both functions are pure: no randomness, no
state.
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
    fim: np.ndarray = np.einsum("ki,kj,k->ij", jacobian, jacobian, 1.0 / variances)
    return fim


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
    sign, logdet = np.linalg.slogdet(fim)
    if sign <= 0.0 or max_eigenvalue <= 0.0:
        logdet_value = float("-inf")
    else:
        logdet_value = float(logdet)
    if min_eigenvalue > 0.0:
        condition_number = max_eigenvalue / min_eigenvalue
    else:
        condition_number = float("inf")
    return FimMetrics(
        min_eigenvalue=min_eigenvalue,
        logdet=logdet_value,
        condition_number=condition_number,
    )


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
