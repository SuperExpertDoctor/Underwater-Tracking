import numpy as np
import pytest
from underwater_tracking.planning.fim import bearing_fim, fim_metrics


def test_crossing_geometry_has_more_information_than_collinear_geometry():
    target = np.array([0.0, 0.0])
    crossing = bearing_fim(target, np.array([[1000.0, 0.0], [0.0, 1000.0]]), np.full(2, 1e-3))
    collinear = bearing_fim(target, np.array([[1000.0, 0.0], [2000.0, 0.0]]), np.full(2, 1e-3))
    assert fim_metrics(crossing).min_eigenvalue > fim_metrics(collinear).min_eigenvalue


def test_fim_metrics_reports_consistent_eigenvalue_logdet_condition():
    target = np.array([0.0, 0.0])
    fim = bearing_fim(target, np.array([[1000.0, 0.0], [0.0, 1000.0]]), np.full(2, 1e-3))
    metrics = fim_metrics(fim)
    eigenvalues = np.linalg.eigvalsh(fim)
    assert metrics.min_eigenvalue == pytest.approx(float(eigenvalues[0]), rel=1e-9)
    assert metrics.condition_number == pytest.approx(
        float(eigenvalues[-1] / eigenvalues[0]), rel=1e-6
    )
    assert metrics.logdet == pytest.approx(
        float(np.sum(np.log(eigenvalues))), rel=1e-9
    )


def test_collinear_geometry_is_rank_one_and_clamped_to_zero():
    target = np.array([0.0, 0.0])
    fim = bearing_fim(target, np.array([[1000.0, 0.0], [2000.0, 0.0]]), np.full(2, 1e-3))
    metrics = fim_metrics(fim)
    assert metrics.min_eigenvalue == 0.0
    assert metrics.condition_number == float("inf")
    assert metrics.logdet == float("-inf")


def test_fim_is_symmetric_positive_semidefinite():
    target = np.array([100.0, -200.0])
    observers = np.array(
        [[-1500.0, 800.0], [1200.0, 600.0], [-300.0, -1800.0], [900.0, -100.0]]
    )
    fim = bearing_fim(target, observers, np.full(4, 2.5e-4))
    np.testing.assert_allclose(fim, fim.T, atol=1e-18)
    assert float(np.min(np.linalg.eigvalsh(fim))) >= 0.0


def test_bearing_fim_removes_rounding_negative_eigenvalues():
    """A one-observer FIM stays PSD at ill-conditioned floating-point scales."""
    target = np.array([0.0, 0.0625])
    observers = np.array([[0.125, 0.00390625]])
    fim = bearing_fim(target, observers, np.array([1e-6]))

    assert float(np.min(np.linalg.eigvalsh(fim))) >= 0.0


def test_fim_scales_inversely_with_squared_standoff_and_variance():
    target = np.array([0.0, 0.0])
    near = bearing_fim(target, np.array([[1000.0, 0.0]]), np.array([1e-3]))
    far = bearing_fim(target, np.array([[2000.0, 0.0]]), np.array([1e-3]))
    np.testing.assert_allclose(near, far * 4.0, rtol=1e-9)
    certain = bearing_fim(target, np.array([[1000.0, 0.0]]), np.array([1e-3]))
    uncertain = bearing_fim(target, np.array([[1000.0, 0.0]]), np.array([2e-3]))
    np.testing.assert_allclose(certain, uncertain * 2.0, rtol=1e-9)


def test_bearing_fim_rejects_coincident_observer_and_invalid_variance():
    target = np.array([0.0, 0.0])
    with pytest.raises(ValueError):
        bearing_fim(target, np.array([[0.0, 0.0], [1000.0, 0.0]]), np.full(2, 1e-3))
    with pytest.raises(ValueError):
        bearing_fim(target, np.array([[1000.0, 0.0]]), np.array([0.0]))
