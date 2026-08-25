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
from numpy.typing import NDArray

from underwater_tracking.tracking.angles import wrap_angle
from underwater_tracking.tracking.models import bearing_measurement

COVARIANCE_ABSOLUTE_EIGENVALUE_FLOOR = 1e-12
COVARIANCE_RELATIVE_EIGENVALUE_FLOOR = 1e-12
# Nonlinear bearing updates can accumulate a small indefinite component before
# the next information refresh. Keep internal repair bounded while allowing
# the long-running local-perception path to recover from that numeric drift.
MAX_INTERNAL_NEGATIVE_EIGENVALUE_RATIO = 0.30
MAX_NUMERICAL_NEGATIVE_EIGENVALUE_RATIO = 1e-10
FloatArray = NDArray[np.float64]


def stabilize_covariance(
    covariance: FloatArray,
    *,
    dimension: int | None = None,
    name: str = "covariance",
    allow_projection: bool = False,
) -> FloatArray:
    """Return a finite, symmetric positive-definite covariance matrix.

    Public state-entry paths reject materially non-positive matrices. Internal
    nonlinear updates may opt into a bounded projection because the UKF
    subtraction can leave a covariance indefinite even when its inputs were
    valid. The negative-eigenvalue limit prevents this recovery path from
    silently accepting an arbitrary corrupted state.
    """
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty square matrix")
    if dimension is not None and matrix.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape {(dimension, dimension)}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    symmetric = (matrix + matrix.T) * 0.5
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} eigendecomposition failed") from exc
    if not np.all(np.isfinite(eigenvalues)):
        raise ValueError(f"{name} eigenvalues must be finite")
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    floor = max(
        COVARIANCE_ABSOLUTE_EIGENVALUE_FLOOR,
        scale * COVARIANCE_RELATIVE_EIGENVALUE_FLOOR,
    )
    if float(np.min(eigenvalues)) >= floor:
        return symmetric
    negative_ratio = max(0.0, -float(np.min(eigenvalues))) / scale
    if (
        negative_ratio > MAX_NUMERICAL_NEGATIVE_EIGENVALUE_RATIO
        and (not allow_projection or negative_ratio > MAX_INTERNAL_NEGATIVE_EIGENVALUE_RATIO)
    ):
        raise ValueError(
            f"{name} is not positive definite (negative eigenvalue ratio "
            f"{negative_ratio:.3g})"
        )
    clipped = np.maximum(eigenvalues, floor)
    repaired = np.asarray((eigenvectors * clipped) @ eigenvectors.T, dtype=np.float64)
    repaired = (repaired + repaired.T) * 0.5
    if not np.all(np.isfinite(repaired)):
        raise ValueError(f"{name} repair produced non-finite values")
    return repaired


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
        self.mean = np.asarray(mean, dtype=np.float64)
        self.covariance = np.asarray(covariance, dtype=np.float64)
        self.process_noise = np.asarray(process_noise, dtype=np.float64)
        self._validate_dimensions()
        self.covariance = stabilize_covariance(
            self.covariance, dimension=self.mean.size, name="covariance"
        )
        if not np.all(np.isfinite(self.process_noise)):
            raise ValueError("process_noise must contain only finite values")
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
        self._validate_dimensions()
        dimension = len(self.mean)
        self.covariance = stabilize_covariance(
            self.covariance,
            dimension=dimension,
            name="covariance",
            allow_projection=True,
        )
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
        positions = np.asarray(observer_positions, dtype=np.float64)
        bearing_values = np.asarray(bearings, dtype=np.float64)
        measurement_variances = np.asarray(variances, dtype=np.float64)
        self._validate_measurements(positions, bearing_values, measurement_variances)
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
            cross = self._bounded_scalar_cross_covariance(
                cross, innovation_variance
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
        candidate_mean = np.asarray(mean, dtype=np.float64)
        candidate_covariance = np.asarray(covariance, dtype=np.float64)
        self._validate_state_arrays(candidate_mean, candidate_covariance)
        candidate_covariance = stabilize_covariance(
            candidate_covariance, dimension=candidate_mean.size, name="covariance"
        )
        self.mean = candidate_mean
        self.covariance = candidate_covariance
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

    def _bounded_scalar_cross_covariance(
        self,
        cross: np.ndarray,
        innovation_variance: float,
    ) -> np.ndarray:
        """Keep the scalar UKF covariance downdate positive definite."""
        if not np.isfinite(innovation_variance) or innovation_variance <= 0.0:
            raise ValueError("innovation variance must be finite and positive")
        try:
            whitened = np.linalg.solve(self.covariance, cross)
        except np.linalg.LinAlgError:
            whitened = np.linalg.pinv(self.covariance) @ cross
        leverage = float(cross @ whitened)
        limit = innovation_variance * (1.0 - 1e-9)
        if leverage <= limit:
            return cross
        if not np.isfinite(leverage) or leverage <= 0.0:
            raise ValueError("cross covariance leverage must be finite and positive")
        return np.asarray(cross * np.sqrt(limit / leverage), dtype=np.float64)

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
        self._validate_dimensions()
        self.covariance = stabilize_covariance(
            self.covariance,
            dimension=self.mean.size,
            name="covariance",
            allow_projection=True,
        )
        self.information_matrix = np.linalg.pinv(self.covariance)
        self.information_vector = self.information_matrix @ self.mean

    @staticmethod
    def _validate_measurements(
        observer_positions: np.ndarray,
        bearings: np.ndarray,
        variances: np.ndarray,
    ) -> None:
        if observer_positions.ndim != 2 or observer_positions.shape[1:] != (2,):
            raise ValueError("observer_positions must have shape (n, 2)")
        if bearings.ndim != 1 or variances.ndim != 1:
            raise ValueError("bearings and variances must be one-dimensional")
        if observer_positions.shape[0] != bearings.size or bearings.size != variances.size:
            raise ValueError("observer positions, bearings, and variances must have equal length")
        if not (
            np.all(np.isfinite(observer_positions))
            and np.all(np.isfinite(bearings))
            and np.all(np.isfinite(variances))
        ):
            raise ValueError("measurements must contain only finite values")
        if np.any(variances <= 0.0):
            raise ValueError("measurement variances must be positive")

    def _validate_dimensions(self) -> None:
        self._validate_state_arrays(self.mean, self.covariance)

    def _validate_state_arrays(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> None:
        if mean.ndim != 1 or mean.size == 0:
            raise ValueError("mean must be a non-empty one-dimensional vector")
        if not np.all(np.isfinite(mean)):
            raise ValueError("mean must contain only finite values")
        expected_shape = (mean.size, mean.size)
        if covariance.shape != expected_shape:
            raise ValueError(
                "covariance shape must match the square of the mean dimension"
            )
        if self.process_noise.shape != expected_shape:
            raise ValueError(
                "process_noise shape must match the square of the mean dimension"
            )
        if not np.all(np.isfinite(self.process_noise)):
            raise ValueError("process_noise must contain only finite values")
