"""Deterministic Gaussian probability mass for executable task regions."""

from __future__ import annotations

from collections.abc import Sequence
from math import exp, isfinite, pi, sqrt
from typing import cast

from scipy.integrate import quad  # type: ignore[import-untyped]
from scipy.special import ndtr  # type: ignore[import-untyped]


def _rectangle_bounds(
    polygon_xy: Sequence[tuple[float, float]],
) -> tuple[float, float, float, float] | None:
    try:
        points = tuple((float(point[0]), float(point[1])) for point in polygon_xy)
    except (IndexError, TypeError, ValueError):
        return None
    if len(points) != 4 or len(set(points)) != 4:
        return None
    if not all(isfinite(value) for point in points for value in point):
        return None
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    if max_x <= min_x or max_y <= min_y:
        return None
    corners = {
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
    }
    if set(points) != corners:
        return None
    return min_x, max_x, min_y, max_y


def _covariance_values(
    covariance_xy: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[float, float, float] | None:
    try:
        if len(covariance_xy) != 2 or any(len(row) != 2 for row in covariance_xy):
            return None
        variance_x = float(covariance_xy[0][0])
        covariance = float(covariance_xy[0][1])
        reverse_covariance = float(covariance_xy[1][0])
        variance_y = float(covariance_xy[1][1])
    except (IndexError, TypeError, ValueError):
        return None
    if not all(
        isfinite(value)
        for value in (variance_x, covariance, reverse_covariance, variance_y)
    ):
        return None
    symmetry_tolerance = 1e-9 * max(
        1.0,
        abs(covariance),
        abs(reverse_covariance),
    )
    if abs(covariance - reverse_covariance) > symmetry_tolerance:
        return None
    determinant = variance_x * variance_y - covariance * covariance
    if variance_x <= 0.0 or variance_y <= 0.0 or determinant <= 0.0:
        return None
    return variance_x, covariance, variance_y


def gaussian_probability_in_axis_aligned_region(
    *,
    mean_xy: tuple[float, float],
    covariance_xy: tuple[tuple[float, float], tuple[float, float]],
    polygon_xy: Sequence[tuple[float, float]],
) -> float | None:
    """Return the probability mass of a 2-D Gaussian inside a rectangle.

    The integral is evaluated over ``x`` using the conditional normal CDF of
    ``y | x``. Invalid geometry, non-finite inputs, non-symmetric covariance,
    and non-positive-definite covariance return ``None``.
    """
    try:
        mean_x = float(mean_xy[0])
        mean_y = float(mean_xy[1])
    except (IndexError, TypeError, ValueError):
        return None
    if not isfinite(mean_x) or not isfinite(mean_y):
        return None
    bounds = _rectangle_bounds(polygon_xy)
    covariance = _covariance_values(covariance_xy)
    if bounds is None or covariance is None:
        return None
    min_x, max_x, min_y, max_y = bounds
    variance_x, covariance_xy_value, variance_y = covariance
    conditional_variance_y = variance_y - covariance_xy_value**2 / variance_x
    if conditional_variance_y <= 0.0 or not isfinite(conditional_variance_y):
        return None
    conditional_std_y = sqrt(conditional_variance_y)
    marginal_std_x = sqrt(variance_x)

    def integrand(x_value: float) -> float:
        standardized_x = (x_value - mean_x) / marginal_std_x
        marginal_density = exp(-0.5 * standardized_x**2) / (
            marginal_std_x * sqrt(2.0 * pi)
        )
        conditional_mean_y = mean_y + covariance_xy_value / variance_x * (
            x_value - mean_x
        )
        upper = float(ndtr((max_y - conditional_mean_y) / conditional_std_y))
        lower = float(ndtr((min_y - conditional_mean_y) / conditional_std_y))
        return marginal_density * max(0.0, upper - lower)

    try:
        probability, _ = quad(
            integrand,
            min_x,
            max_x,
            epsabs=1e-10,
            epsrel=1e-10,
            limit=100,
        )
    except (ArithmeticError, ValueError, RuntimeError):
        return None
    probability = cast(float, probability)
    if not isfinite(probability):
        return None
    return max(0.0, min(1.0, float(probability)))


__all__ = ["gaussian_probability_in_axis_aligned_region"]
