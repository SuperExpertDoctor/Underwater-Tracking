# src/underwater_tracking/tracking/initialization.py
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares  # type: ignore[import-untyped]
from underwater_tracking.tracking.angles import wrap_angle


class InsufficientGeometryError(RuntimeError):
    """Raised when no pair of bearing lines-of-sight crosses enough to triangulate."""


@dataclass(frozen=True)
class InitializationResult:
    position_xy: np.ndarray
    covariance_xy: np.ndarray
    residual_norm: float


def initialize_from_bearings(
    origins: np.ndarray,
    bearings: np.ndarray,
    variances: np.ndarray,
    prior: np.ndarray,
    minimum_crossing_sine: float = 0.15,
) -> InitializationResult:
    origins = np.asarray(origins, dtype=float)
    bearings = np.asarray(bearings, dtype=float)
    variances = np.asarray(variances, dtype=float)

    bearing_diffs = bearings[:, None] - bearings[None, :]
    if not np.any(np.abs(np.sin(bearing_diffs)) >= minimum_crossing_sine):
        raise InsufficientGeometryError(
            "no bearing pair crosses (max |sin(delta_bearing)| below "
            f"{minimum_crossing_sine}); cannot initialize a track"
        )

    def residual(position: np.ndarray) -> np.ndarray:
        predicted = np.arctan2(position[1] - origins[:, 1], position[0] - origins[:, 0])
        return np.asarray(wrap_angle(predicted - bearings) / np.sqrt(variances))

    fit = least_squares(residual, np.asarray(prior, dtype=float), method="trf")
    information = fit.jac.T @ fit.jac
    covariance = np.linalg.pinv(information)
    return InitializationResult(fit.x, covariance, float(np.linalg.norm(fit.fun)))
