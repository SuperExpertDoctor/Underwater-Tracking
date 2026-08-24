"""Public-prior temporal search envelopes for bearing-only UUV control.

The controller never needs hidden target state to start a search.  It turns a
validated public prior into a deterministic sequence of sigma-point clouds:
the cloud advances at a declared bounded sweep speed and its radius grows
with elapsed time.  The resulting points are consumed by the existing robust
FIM waypoint planner, so every committed motion remains subject to its
step, separation, and map-boundary checks.

The uncertainty-aware rolling-search design is informed by Yu, Ma, and Zhang,
"Cooperative search decision-making and path planning methods for multi-UUV
search missions under uncertainties" (Ocean Engineering, 2025,
doi:10.1016/j.oceaneng.2025.122875).  This module implements only the
transferable public-prior uncertainty envelope consumed by this repository's
robust FIM planner; it does not claim to reproduce that paper's complete
DRHO-AC probability-map or A-star navigation system.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite, sqrt

import numpy as np


def public_temporal_sigma_points(
    center_xy: Sequence[float],
    covariance_xy: Sequence[Sequence[float]],
    *,
    elapsed_s: float,
    horizon_s: float,
    search_direction_xy: Sequence[float],
    sweep_speed_mps: float,
    radial_growth_mps: float,
    temporal_slices: int = 3,
) -> np.ndarray:
    """Build a bounded temporal sigma-point envelope from public inputs.

    ``elapsed_s`` is measured from the prior issue time and ``horizon_s`` is
    the remaining validity horizon.  Each temporal slice contains a center
    plus four principal-axis points.  The principal axes come only from the
    public covariance; the center motion and uncertainty inflation are
    explicit bounded planning parameters.
    """
    center = np.asarray(tuple(center_xy), dtype=float)
    covariance = np.asarray(covariance_xy, dtype=float)
    direction = np.asarray(tuple(search_direction_xy), dtype=float)
    _validate_inputs(
        center,
        covariance,
        direction,
        elapsed_s,
        horizon_s,
        sweep_speed_mps,
        radial_growth_mps,
        temporal_slices,
    )
    direction /= np.linalg.norm(direction)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-9)
    axis_a = eigenvectors[:, 0]
    axis_b = eigenvectors[:, 1]
    # A 95% two-dimensional ellipse scale keeps the cloud conservative while
    # leaving the growth term explicit and auditable.
    chi_square_radius = sqrt(5.991464547107979)
    times = np.linspace(0.0, horizon_s, temporal_slices)
    current_center = center + direction * sweep_speed_mps * elapsed_s
    points: list[np.ndarray] = []
    for future_s in times:
        total_elapsed_s = elapsed_s + float(future_s)
        temporal_center = current_center + direction * sweep_speed_mps * float(future_s)
        growth = radial_growth_mps * total_elapsed_s
        radius_a = chi_square_radius * sqrt(eigenvalues[0]) + growth
        radius_b = chi_square_radius * sqrt(eigenvalues[1]) + growth
        points.extend(
            (
                temporal_center,
                temporal_center + axis_a * radius_a,
                temporal_center - axis_a * radius_a,
                temporal_center + axis_b * radius_b,
                temporal_center - axis_b * radius_b,
            )
        )
    return np.asarray(points, dtype=float)


def _validate_inputs(
    center: np.ndarray,
    covariance: np.ndarray,
    direction: np.ndarray,
    elapsed_s: float,
    horizon_s: float,
    sweep_speed_mps: float,
    radial_growth_mps: float,
    temporal_slices: int,
) -> None:
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError("center_xy must contain two finite values")
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        raise ValueError("covariance_xy must be a finite 2x2 matrix")
    if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12):
        raise ValueError("covariance_xy must be symmetric")
    if np.any(np.linalg.eigvalsh(covariance) <= 0.0):
        raise ValueError("covariance_xy must be positive definite")
    if direction.shape != (2,) or not np.isfinite(direction).all() or np.linalg.norm(direction) <= 0.0:
        raise ValueError("search_direction_xy must be a non-zero finite vector")
    if not all(
        isfinite(value) and value >= 0.0
        for value in (elapsed_s, horizon_s, sweep_speed_mps, radial_growth_mps)
    ):
        raise ValueError("temporal search parameters must be finite and non-negative")
    if temporal_slices < 1:
        raise ValueError("temporal_slices must be at least one")
