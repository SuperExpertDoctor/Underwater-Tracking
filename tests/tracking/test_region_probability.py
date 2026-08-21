from __future__ import annotations

from math import isclose

from underwater_tracking.tracking.region_probability import (
    gaussian_probability_in_axis_aligned_region,
)


def _square(
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> tuple[tuple[float, float], ...]:
    return (
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
    )


def test_tight_distribution_inside_region_has_near_unity_mass() -> None:
    probability = gaussian_probability_in_axis_aligned_region(
        mean_xy=(0.0, 0.0),
        covariance_xy=((0.01, 0.0), (0.0, 0.01)),
        polygon_xy=_square(-1.0, 1.0, -1.0, 1.0),
    )

    assert probability is not None and probability > 0.99


def test_far_distribution_has_near_zero_mass() -> None:
    probability = gaussian_probability_in_axis_aligned_region(
        mean_xy=(10.0, 10.0),
        covariance_xy=((1.0, 0.0), (0.0, 1.0)),
        polygon_xy=_square(-1.0, 1.0, -1.0, 1.0),
    )

    assert probability is not None and probability < 0.01


def test_boundary_centered_distribution_and_correlated_covariance_are_supported() -> None:
    centered = gaussian_probability_in_axis_aligned_region(
        mean_xy=(0.0, 0.0),
        covariance_xy=((1.0, 0.0), (0.0, 1.0)),
        polygon_xy=_square(-1.0, 1.0, -1.0, 1.0),
    )
    correlated = gaussian_probability_in_axis_aligned_region(
        mean_xy=(0.0, 0.0),
        covariance_xy=((1.0, 0.8), (0.8, 1.0)),
        polygon_xy=_square(-1.0, 1.0, -1.0, 1.0),
    )

    assert centered is not None and 0.45 < centered < 0.55
    assert correlated is not None and 0.0 < correlated < 1.0
    assert not isclose(centered, correlated)


def test_probability_is_repeatable() -> None:
    kwargs = {
        "mean_xy": (0.3, -0.2),
        "covariance_xy": ((2.0, 0.4), (0.4, 1.0)),
        "polygon_xy": _square(-1.0, 2.0, -2.0, 1.0),
    }

    assert gaussian_probability_in_axis_aligned_region(
        **kwargs
    ) == gaussian_probability_in_axis_aligned_region(**kwargs)


def test_invalid_polygon_mean_and_covariance_return_none() -> None:
    cases = (
        {
            "mean_xy": (0.0, 0.0),
            "covariance_xy": ((1.0, 0.0), (0.0, 1.0)),
            "polygon_xy": ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        },
        {
            "mean_xy": (float("nan"), 0.0),
            "covariance_xy": ((1.0, 0.0), (0.0, 1.0)),
            "polygon_xy": _square(-1.0, 1.0, -1.0, 1.0),
        },
        {
            "mean_xy": (0.0, 0.0),
            "covariance_xy": ((1.0, 0.2), (0.0, 1.0)),
            "polygon_xy": _square(-1.0, 1.0, -1.0, 1.0),
        },
        {
            "mean_xy": (0.0, 0.0),
            "covariance_xy": ((1.0, 1.1), (1.1, 1.0)),
            "polygon_xy": _square(-1.0, 1.0, -1.0, 1.0),
        },
    )

    for case in cases:
        assert gaussian_probability_in_axis_aligned_region(**case) is None
