import numpy as np
from underwater_tracking.tracking.imm import build_default_imm
from underwater_tracking.tracking.models import bearing_measurement, constant_turn
from underwater_tracking.tracking.uif import UnscentedInformationFilter

INITIAL_MEAN = np.array([500.0, 500.0, 2.0, 0.0, 0.0])
INITIAL_COVARIANCE = np.diag([40_000.0, 40_000.0, 25.0, 25.0, 0.01])
PROCESS_NOISE = np.diag([0.01, 0.01, 0.001, 0.001, 1e-6])


def test_imm_uif_reduces_position_covariance_with_crossing_bearings() -> None:
    imm = build_default_imm(mean=INITIAL_MEAN, covariance=INITIAL_COVARIANCE)
    before = np.trace(imm.mixed_covariance[:2, :2])
    imm.predict(30.0)
    imm.update(
        observer_positions=np.array([[0.0, 0.0], [1000.0, 0.0]]),
        bearings=np.array([0.785398, 2.356194]), variances=np.array([1e-3, 1e-3]),
    )
    assert np.trace(imm.mixed_covariance[:2, :2]) < before
    assert abs(sum(imm.model_probabilities) - 1.0) < 1e-12


def test_angle_wrapping_across_pi_keeps_measurement_accepted() -> None:
    # Target is nearly due-west of the observer: its bearing sits just below
    # +pi while the measurement sits just above -pi. An unwrapped residual
    # would be ~2pi and the measurement would be gated out.
    filt = UnscentedInformationFilter(
        mean=np.array([-500.0, 10.0, 0.0, 0.0, 0.0]),
        covariance=np.diag([25.0, 25.0, 1.0, 1.0, 0.01]),
        process_noise=PROCESS_NOISE,
    )
    before = np.trace(filt.covariance[:2, :2])
    nis = filt.update_bearings(
        observer_positions=np.array([[0.0, 0.0]]),
        bearings=np.array([-np.pi + 0.01]),
        variances=np.array([1e-3]),
    )
    assert nis[0] < 6.635
    assert np.trace(filt.covariance[:2, :2]) < before


def test_bearing_beyond_nis_gate_is_rejected_and_inflates_covariance() -> None:
    filt = UnscentedInformationFilter(
        mean=INITIAL_MEAN, covariance=INITIAL_COVARIANCE, process_noise=PROCESS_NOISE
    )
    before = filt.covariance.copy()
    nis = filt.update_bearings(
        observer_positions=np.array([[0.0, 0.0]]),
        bearings=np.array([3.0]),
        variances=np.array([1e-3]),
    )
    assert nis[0] > 6.635
    np.testing.assert_allclose(filt.covariance, before * 1.1)


def test_huber_band_inflates_measurement_variance() -> None:
    # With a tiny state covariance the predicted measurement spread is
    # negligible, so the innovation variance is ~R = 1.0 and a measurement
    # at 2.55 rad lands in the Huber band (2.5, sqrt(6.635)). The likelihood
    # must be computed with the variance inflated by 2.55 / 2.5.
    filt = UnscentedInformationFilter(
        mean=np.array([100.0, 0.0, 0.0, 0.0, 0.0]),
        covariance=np.diag([1e-3, 1e-3, 1e-3, 1e-3, 1e-8]),
        process_noise=PROCESS_NOISE,
    )
    nis = filt.update_bearings(
        observer_positions=np.array([[0.0, 0.0]]),
        bearings=np.array([2.55]),
        variances=np.array([1.0]),
    )
    assert 2.5**2 < nis[0] < 6.635
    inflated_variance = 1.0 * (2.55 / 2.5)
    expected_log_likelihood = -0.5 * (
        2.55**2 / inflated_variance + np.log(2.0 * np.pi * inflated_variance)
    )
    np.testing.assert_allclose(filt.log_likelihood, expected_log_likelihood, atol=1e-4)


def test_missed_update_inflates_covariance() -> None:
    filt = UnscentedInformationFilter(
        mean=INITIAL_MEAN, covariance=INITIAL_COVARIANCE, process_noise=PROCESS_NOISE
    )
    before = filt.covariance.copy()
    nis = filt.update_bearings(
        observer_positions=np.empty((0, 2)), bearings=np.array([]), variances=np.array([])
    )
    assert nis == []
    np.testing.assert_allclose(filt.covariance, before * 1.1)
    assert filt.log_likelihood == 0.0


def test_synthetic_turn_track_has_finite_consistent_outputs() -> None:
    dt = 30.0
    observers = np.array([[0.0, 0.0], [1000.0, 0.0]])
    true_state = INITIAL_MEAN.copy()
    imm = build_default_imm(mean=INITIAL_MEAN, covariance=INITIAL_COVARIANCE)
    for step in range(200):
        turn_rate = 0.002 if step >= 100 else 0.0
        true_state = constant_turn(true_state, dt, turn_rate)
        observers = observers + np.array([1.0 * dt, 0.0])
        bearings = np.array(
            [
                bearing_measurement(true_state, observers[0]),
                bearing_measurement(true_state, observers[1]),
            ]
        )
        imm.predict(dt)
        imm.update(observers, bearings, np.full(2, 1e-3))
        assert np.all(np.isfinite(imm.mixed_mean))
        assert np.all(np.isfinite(imm.mixed_covariance))
        assert np.all(np.linalg.eigvalsh(imm.mixed_covariance) > 0.0)
        assert abs(float(sum(imm.model_probabilities)) - 1.0) < 1e-12
