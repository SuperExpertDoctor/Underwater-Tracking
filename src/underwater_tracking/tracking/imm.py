"""Interacting multiple model (IMM) estimator over a turn-rate model library.

Three unscented information filters run the constant-velocity, left-turn,
and right-turn models. Each 30 s cycle mixes the per-model beliefs with the
Markov transition matrix, predicts each model forward, updates from the
bearings, and combines the model likelihoods (from accepted innovations)
into normalized posterior model probabilities. Upper layers read the blended
``mixed_mean`` / ``mixed_covariance`` belief.
"""

from collections.abc import Callable
from math import exp

import numpy as np

from underwater_tracking.domain.execution_models import IMMModelForecast
from underwater_tracking.tracking.models import constant_turn
from underwater_tracking.tracking.uif import UnscentedInformationFilter, stabilize_covariance

# Per-second process noise for the [x, y, vx, vy, omega] state; predict adds Q*dt.
# The omega entry is the turn-rate variance growth per second.
DEFAULT_PROCESS_NOISE = np.diag([0.01, 0.01, 0.001, 0.001, 1e-4])

# Markov transition matrix, row i (from model i) -> column j (to model j).
DEFAULT_TRANSITION_MATRIX = np.array(
    [
        [0.94, 0.03, 0.03],
        [0.08, 0.88, 0.04],
        [0.08, 0.04, 0.88],
    ]
)

MODEL_ORDER = ("cv", "left_turn", "right_turn")
# Commanded turn rates are radians per second; constant_turn multiplies them by
# the prediction interval in seconds to obtain the heading change.
DEFAULT_COMMANDED_TURNS = (0.0, 0.0105, -0.0105)


def _turn_transition(turn_rate: float) -> Callable[[np.ndarray, float], np.ndarray]:
    def transition(state: np.ndarray, dt: float) -> np.ndarray:
        return constant_turn(state, dt, turn_rate)

    return transition


class ImmEstimator:
    """Interacting multiple model estimator over heterogeneous turn models."""

    def __init__(
        self,
        filters: dict[str, UnscentedInformationFilter],
        transition_matrix: np.ndarray,
        model_probabilities: np.ndarray,
        commanded_turns: dict[str, float],
    ) -> None:
        self.filters = dict(filters)
        self.transition_matrix = np.asarray(transition_matrix, dtype=float)
        self.commanded_turns = dict(commanded_turns)
        self._model_probabilities = np.asarray(model_probabilities, dtype=float)
        self._mixed_mean: np.ndarray
        self._mixed_covariance: np.ndarray
        self._refresh_mixed_output()

    @property
    def mixed_mean(self) -> np.ndarray:
        return self._mixed_mean

    @property
    def mixed_covariance(self) -> np.ndarray:
        return self._mixed_covariance

    @property
    def model_probabilities(self) -> np.ndarray:
        return self._model_probabilities

    def model_state_projections(
        self, source_observation_ids: tuple[str, ...] = ()
    ) -> tuple[IMMModelForecast, ...]:
        """Return immutable, serializable state for every IMM branch."""
        projections: list[IMMModelForecast] = []
        for index, (name, model) in enumerate(self.filters.items()):
            likelihood = exp(min(700.0, max(-745.0, float(model.log_likelihood))))
            projections.append(
                IMMModelForecast(
                    model_name={
                        "cv": "CV",
                        "left_turn": "CT_LEFT",
                        "right_turn": "CT_RIGHT",
                    }.get(name, name.upper()),
                    state_mean=tuple(float(value) for value in model.mean),
                    state_covariance=tuple(
                        tuple(float(value) for value in row) for row in model.covariance
                    ),
                    model_probability=float(self._model_probabilities[index]),
                    innovation=tuple(float(value) for value in model.last_innovations),
                    likelihood=likelihood,
                    source_observation_ids=source_observation_ids,
                )
            )
        return tuple(projections)

    @property
    def model_states(self) -> tuple[IMMModelForecast, ...]:
        """Compatibility accessor for consumers expecting a state property."""
        return self.model_state_projections()

    def predict(self, dt: float) -> None:
        """Run IMM interaction/mixing, then predict every model forward."""
        self._interact_and_mix()
        for name, model in self.filters.items():
            model.predict(_turn_transition(self.commanded_turns[name]), dt)
        self._refresh_mixed_output()

    def update(
        self,
        observer_positions: np.ndarray,
        bearings: np.ndarray,
        variances: np.ndarray,
    ) -> list[list[float]]:
        """Update every model, combine likelihoods, return per-model NIS lists."""
        nis_by_model: list[list[float]] = []
        log_likelihoods: list[float] = []
        for model in self.filters.values():
            nis_by_model.append(
                model.update_bearings(observer_positions, bearings, variances)
            )
            log_likelihoods.append(model.log_likelihood)
        self._update_model_probabilities(log_likelihoods)
        self._refresh_mixed_output()
        return nis_by_model

    def _interact_and_mix(self) -> None:
        means = np.stack([model.mean for model in self.filters.values()])
        covariances = np.stack([model.covariance for model in self.filters.values()])
        for j, model in enumerate(self.filters.values()):
            mixing_weights = self._model_probabilities * self.transition_matrix[:, j]
            mixing_weights = mixing_weights / float(np.sum(mixing_weights))
            mixed_mean = np.sum(mixing_weights[:, None] * means, axis=0)
            deviations = means - mixed_mean
            mixed_covariance = np.sum(
                mixing_weights[:, None, None]
                * (
                    covariances
                    + deviations[:, :, None] * deviations[:, None, :]
                ),
                axis=0,
            )
            model.set_state(mixed_mean, mixed_covariance)

    def _update_model_probabilities(self, log_likelihoods: list[float]) -> None:
        priors = self.transition_matrix.T @ self._model_probabilities
        log_weights = np.asarray(log_likelihoods, dtype=float) + np.log(priors)
        log_weights = log_weights - float(np.max(log_weights))
        posterior = np.exp(log_weights)
        self._model_probabilities = posterior / float(np.sum(posterior))

    def _refresh_mixed_output(self) -> None:
        means = np.stack([model.mean for model in self.filters.values()])
        covariances = np.stack([model.covariance for model in self.filters.values()])
        probabilities = self._model_probabilities
        self._mixed_mean = np.sum(probabilities[:, None] * means, axis=0)
        deviations = means - self._mixed_mean
        self._mixed_covariance = np.sum(
            probabilities[:, None, None]
            * (covariances + deviations[:, :, None] * deviations[:, None, :]),
            axis=0,
        )
        self._mixed_covariance = stabilize_covariance(
            self._mixed_covariance,
            dimension=self._mixed_mean.size,
            name="mixed covariance",
        )


def build_default_imm(
    mean: np.ndarray, covariance: np.ndarray
) -> ImmEstimator:
    """Build the default cv/left-turn/right-turn IMM over a shared initial belief."""
    filters = {
        name: UnscentedInformationFilter(mean, covariance, DEFAULT_PROCESS_NOISE)
        for name in MODEL_ORDER
    }
    return ImmEstimator(
        filters,
        DEFAULT_TRANSITION_MATRIX,
        np.full(len(MODEL_ORDER), 1.0 / len(MODEL_ORDER)),
        dict(zip(MODEL_ORDER, DEFAULT_COMMANDED_TURNS)),
    )
