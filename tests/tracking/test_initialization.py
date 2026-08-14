import numpy as np
import pytest
from underwater_tracking.tracking.initialization import (
    InsufficientGeometryError,
    initialize_from_bearings,
)


def test_gauss_newton_initialization_recovers_crossing_bearings():
    result = initialize_from_bearings(
        origins=np.array([[0.0, 0.0], [1000.0, 0.0], [0.0, 1000.0]]),
        bearings=np.array([0.7853981634, 2.3561944902, -0.7853981634]),
        variances=np.full(3, 1e-4),
        prior=np.array([400.0, 600.0]),
    )
    np.testing.assert_allclose(result.position_xy, [500.0, 500.0], atol=2.0)
    assert np.all(np.linalg.eigvalsh(result.covariance_xy) > 0)


def test_near_parallel_bearings_raise_insufficient_geometry():
    with pytest.raises(InsufficientGeometryError):
        initialize_from_bearings(
            origins=np.array([[0.0, 0.0], [1000.0, 0.0], [0.0, 1000.0]]),
            bearings=np.full(3, 0.5),
            variances=np.full(3, 1e-4),
            prior=np.array([400.0, 600.0]),
        )
