import numpy as np

from underwater_tracking.planning.search_control import public_temporal_sigma_points


def test_public_temporal_sigma_points_expand_and_follow_only_public_direction() -> None:
    points = public_temporal_sigma_points(
        center_xy=(-4200.0, -6200.0),
        covariance_xy=((360_000.0, 0.0), (0.0, 360_000.0)),
        elapsed_s=300.0,
        horizon_s=900.0,
        search_direction_xy=(1.0, 0.0),
        sweep_speed_mps=4.0,
        radial_growth_mps=2.0,
        temporal_slices=3,
    )

    assert points.shape == (15, 2)
    assert np.isfinite(points).all()
    # The temporal mean advances only along the declared public search axis.
    assert points[:, 1].mean() == -6200.0
    assert points[:, 0].mean() > -4200.0

    first_slice = points[:5]
    last_slice = points[-5:]
    first_radius = np.max(np.linalg.norm(first_slice - first_slice.mean(axis=0), axis=1))
    last_radius = np.max(np.linalg.norm(last_slice - last_slice.mean(axis=0), axis=1))
    assert last_radius > first_radius


def test_public_temporal_sigma_points_are_deterministic_and_bounded_by_time() -> None:
    kwargs = {
        "center_xy": (10.0, 20.0),
        "covariance_xy": ((9.0, 2.0), (2.0, 16.0)),
        "elapsed_s": 100.0,
        "horizon_s": 400.0,
        "search_direction_xy": (3.0, 4.0),
        "sweep_speed_mps": 5.0,
        "radial_growth_mps": 1.0,
        "temporal_slices": 4,
    }

    first = public_temporal_sigma_points(**kwargs)
    second = public_temporal_sigma_points(**kwargs)

    np.testing.assert_array_equal(first, second)
    assert first[:, 0].mean() > 10.0
    assert first[:, 1].mean() > 20.0
