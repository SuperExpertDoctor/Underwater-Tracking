"""Scaled unscented filter for scalar bearing measurements in information form.

The filter propagates a mean/covariance belief through sigma points
(alpha=0.3, beta=2, kappa=0) and refreshes the information form
(``Y = pinv(P)``, ``y = Y @ x``) after every update. Bearing measurements
are scalar and processed sequentially with circular statistics: means use
``atan2(sum(w*sin z), sum(w*cos z))`` and every residual is wrapped to
``[-pi, pi)``. Each measurement is gated at a chi-square NIS threshold;
residuals in the Huber band ``(huber_threshold, sqrt(nis_gate))`` get their
variance inflated, and rejected or missed detections leave the belief
predict-only with an inflated covariance.
"""

from collections.abc import Callable

import numpy as np

from underwater_tracking.tracking.angles import wrap_angle
from underwater_tracking.tracking.models import bearing_measurement


class UnscentedInformationFilter:
    """Deterministic unscented information filter over a 5-D turn state.

    State layout is ``[x, y, vx, vy, omega]``; the motion model is injected
    per call through ``predict(transition, dt)`` so the same filter works
    under any commanded turn rate.
    """

    def __init__(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        process_noise: np.ndarray,
        alpha: float = 0.3,
        beta: float = 2.0,
        kappa: float = 0.0,
        nis_gate: float = 6.635,
        huber_threshold: float = 2.5,
        missed_update_inflation: float = 1.1,
    ) -> None:
        self.mean = np.asarray(mean, dtype=float)
        self.covariance = np.asarray(covariance, dtype=float)
        self.process_noise = np.asarray(process_noise, dtype=float)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.nis_gate = nis_gate
        self.huber_threshold = huber_threshold
        self.missed_update_inflation = missed_update_inflation
        # Log likelihood of the accepted measurements from the last update.
        self.log_likelihood = 0.0
        self._refresh_information()

    def sigma_points(self) -> np.ndarray:
        """Return the 2n+1 scaled sigma points around the current belief."""
        dimension = len(self.mean)
        scaling = self.alpha**2 * (dimension + self.kappa) - dimension
        spread = np.linalg.cholesky((dimension + scaling) * self.covariance)
        points = np.empty((2 * dimension + 1, dimension))
        points[0] = self.mean
        points[1 : dimension + 1] = self.mean + spread.T
        points[dimension + 1 :] = self.mean - spread.T
        return points

    def predict(
        self, transition: Callable[[np.ndarray, float], np.ndarray], dt: float
    ) -> None:
        """Propagate the sigma points and add ``process_noise * dt``."""
        mapped = np.asarray([transition(point, dt) for point in self.sigma_points()])
        mean_weights = self._mean_weights()
        self.mean = np.sum(mapped * mean_weights[:, None], axis=0)
        deviations = mapped - self.mean
        self.covariance = (
            np.sum(
                self._covariance_weights()[:, None, None]
                * (deviations[:, :, None] * deviations[:, None, :]),
                axis=0,
            )
            + self.process_noise * dt
        )
        self._refresh_information()

    def update_bearings(
        self,
        observer_positions: np.ndarray,
        bearings: np.ndarray,
        variances: np.ndarray,
    ) -> list[float]:
        """Update sequentially and return one NIS value per measurement."""
        positions = np.asarray(observer_positions, dtype=float)
        bearing_values = np.asarray(bearings, dtype=float)
        measurement_variances = np.asarray(variances, dtype=float)
        self.log_likelihood = 0.0
        if len(bearing_values) == 0:
            self._inflate_covariance()
            return []
        nis_values: list[float] = []
        for index in range(len(bearing_values)):
            variance = float(measurement_variances[index])
            innovation_variance, innovation, cross = self._measurement_statistics(
                positions[index], float(bearing_values[index]), variance
            )
            nis = float(innovation**2 / innovation_variance)
            nis_values.append(nis)
            if nis > self.nis_gate:
                # Rejected detection: predict-only with covariance inflation.
                self._inflate_covariance()
                continue
            normalized = float(abs(innovation) / np.sqrt(innovation_variance))
            if normalized > self.huber_threshold:
                # Huber reweighting: inflate the measurement variance.
                effective_variance = variance * (normalized / self.huber_threshold)
                innovation_variance = (
                    innovation_variance - variance + effective_variance
                )
            gain = cross / innovation_variance
            self.mean = self.mean + gain * innovation
            self.covariance = (
                self.covariance - np.outer(gain, gain) * innovation_variance
            )
            self._refresh_information()
            self.log_likelihood += float(
                -0.5
                * (
                    innovation**2 / innovation_variance
                    + np.log(2.0 * np.pi * innovation_variance)
                )
            )
        return nis_values

    def set_state(self, mean: np.ndarray, covariance: np.ndarray) -> None:
        """Replace the belief (used by IMM mixing) and refresh the information form."""
        self.mean = np.asarray(mean, dtype=float)
        self.covariance = np.asarray(covariance, dtype=float)
        self._refresh_information()

    def _measurement_statistics(
        self, observer_xy: np.ndarray, bearing: float, variance: float
    ) -> tuple[float, float, np.ndarray]:
        """Return (innovation variance, wrapped innovation, cross-covariance)."""
        points = self.sigma_points()
        predicted = np.asarray(
            [bearing_measurement(point, observer_xy) for point in points]
        )
        mean_weights = self._mean_weights()
        circular_mean = np.arctan2(
            np.sum(mean_weights * np.sin(predicted)),
            np.sum(mean_weights * np.cos(predicted)),
        )
        residuals = wrap_angle(predicted - circular_mean)
        covariance_weights = self._covariance_weights()
        innovation_variance = float(
            np.sum(covariance_weights * residuals**2) + variance
        )
        innovation = float(wrap_angle(bearing - circular_mean))
        cross = np.asarray(
            np.sum(
                covariance_weights[:, None]
                * (points - self.mean)
                * residuals[:, None],
                axis=0,
            ),
            dtype=float,
        )
        return innovation_variance, innovation, cross

    def _mean_weights(self) -> np.ndarray:
        dimension = len(self.mean)
        scaling = self.alpha**2 * (dimension + self.kappa) - dimension
        weights = np.full(2 * dimension + 1, 1.0 / (2.0 * (dimension + scaling)))
        weights[0] = scaling / (dimension + scaling)
        return weights

    def _covariance_weights(self) -> np.ndarray:
        weights = self._mean_weights()
        weights[0] += 1.0 - self.alpha**2 + self.beta
        return weights

    def _inflate_covariance(self) -> None:
        self.covariance = self.covariance * self.missed_update_inflation
        self._refresh_information()

    def _refresh_information(self) -> None:
        self.information_matrix = np.linalg.pinv(self.covariance)
        self.information_vector = self.information_matrix @ self.mean
