"""Full-horizon IMM branch propagation and moment matching."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import atan2, ceil, cos, hypot, isfinite, pi, sin, sqrt
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from underwater_tracking.domain.execution_models import (
    ExecutionModel,
    IMMModelForecast,
)


class IMMModelStateProjection(IMMModelForecast):
    """Checkpoint-safe projection of one estimator model state."""

    @model_validator(mode="before")
    @classmethod
    def pad_legacy_four_state(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        normalized = dict(value)
        mean = normalized.get("state_mean")
        covariance = normalized.get("state_covariance")
        if isinstance(mean, (tuple, list)) and len(mean) == 4:
            normalized["state_mean"] = (*mean, 0.0)
        if isinstance(covariance, (tuple, list)) and len(covariance) == 4:
            rows = [list(row) for row in covariance]
            if all(len(row) == 4 for row in rows):
                normalized["state_covariance"] = tuple(
                    (*row, 0.0) for row in rows
                ) + ((0.0, 0.0, 0.0, 0.0, 1.0),)
        return normalized


class IMMBranchForecast(ExecutionModel):
    """One model's complete position and state forecast."""

    model_name: Literal["CV", "CT_LEFT", "CT_RIGHT"]
    model_probability: float = Field(ge=0, le=1, allow_inf_nan=False)
    times_s: tuple[float, ...] = Field(min_length=1)
    centerline_xy: tuple[tuple[float, float], ...] = Field(min_length=1)
    covariance_xy: tuple[tuple[float, float, float, float], ...] = Field(min_length=1)
    state_means: tuple[tuple[float, ...], ...] = Field(min_length=1)
    state_covariances: tuple[tuple[tuple[float, ...], ...], ...] = Field(min_length=1)
    innovation: tuple[float, ...] = ()
    likelihood: float = Field(ge=0, allow_inf_nan=False)
    source_observation_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def arrays_match(self) -> IMMBranchForecast:
        size = len(self.times_s)
        if len(self.centerline_xy) != size or len(self.covariance_xy) != size:
            raise ValueError("IMM branch forecast arrays must have equal lengths")
        if len(self.state_means) != size or len(self.state_covariances) != size:
            raise ValueError("IMM branch state arrays must have equal lengths")
        return self


class IMMForecastResult(ExecutionModel):
    """Complete moment-matched IMM prediction over an absolute time grid."""

    origin_sim_time_s: float = Field(ge=0, allow_inf_nan=False)
    times_s: tuple[float, ...] = Field(min_length=1)
    centerline_xy: tuple[tuple[float, float], ...] = Field(min_length=1)
    covariance_xy: tuple[tuple[float, float, float, float], ...] = Field(min_length=1)
    corridor_radius_m: tuple[float, ...] = Field(min_length=1)
    model_branches: tuple[IMMBranchForecast, ...] = Field(min_length=3)
    model_probabilities: dict[str, float]
    clipping_records: tuple[str, ...] = ()
    source_observation_ids: tuple[str, ...] = ()
    prediction_regime: Literal["imm"] = "imm"

    @model_validator(mode="after")
    def result_arrays_match(self) -> IMMForecastResult:
        size = len(self.times_s)
        if len(self.centerline_xy) != size:
            raise ValueError("IMM result centerline and times must have equal lengths")
        if len(self.covariance_xy) != size or len(self.corridor_radius_m) != size:
            raise ValueError("IMM result arrays must have equal lengths")
        if set(self.model_probabilities) != {
            branch.model_name for branch in self.model_branches
        }:
            raise ValueError("IMM result probabilities must name every branch")
        if not isfinite(sum(self.model_probabilities.values())):
            raise ValueError("IMM result probabilities must be finite")
        return self


def _canonical_model_name(name: str) -> str:
    normalized = name.casefold().replace("-", "_")
    if normalized in {"cv", "constant_velocity"}:
        return "CV"
    if normalized in {"ct_left", "left", "left_turn"}:
        return "CT_LEFT"
    if normalized in {"ct_right", "right", "right_turn"}:
        return "CT_RIGHT"
    raise ValueError(f"unknown IMM model {name!r}")


def _normalise_state(value: Any, name: str) -> IMMModelStateProjection:
    if isinstance(value, IMMModelStateProjection):
        return value.model_copy(update={"model_name": _canonical_model_name(value.model_name)})
    if isinstance(value, IMMModelForecast):
        return IMMModelStateProjection.model_validate(
            {**value.model_dump(mode="python"), "model_name": _canonical_model_name(value.model_name)}
        )
    if isinstance(value, Mapping):
        return IMMModelStateProjection.model_validate(
            {**value, "model_name": _canonical_model_name(str(value.get("model_name", name)))}
        )
    raise TypeError(f"IMM state for {name!r} must be a model projection")


def _state_sequence(
    model_states: Mapping[str, Any] | Sequence[Any] | Any,
) -> tuple[IMMModelStateProjection, ...]:
    if hasattr(model_states, "filters") and hasattr(model_states, "model_probabilities"):
        probabilities = tuple(float(value) for value in model_states.model_probabilities)
        names = tuple(model_states.filters)
        return tuple(
            _normalise_state(
                {
                    "model_name": name,
                    "state_mean": model_states.filters[name].mean,
                    "state_covariance": model_states.filters[name].covariance,
                    "model_probability": probabilities[index],
                    "innovation": getattr(model_states.filters[name], "last_innovations", ()),
                    "likelihood": float(np.exp(model_states.filters[name].log_likelihood)),
                },
                name,
            )
            for index, name in enumerate(names)
        )
    if isinstance(model_states, Mapping):
        return tuple(_normalise_state(value, str(name)) for name, value in model_states.items())
    return tuple(
        _normalise_state(value, str(getattr(value, "model_name", index)))
        for index, value in enumerate(model_states)
    )


def _normalise_probabilities(
    states: Sequence[IMMModelStateProjection],
    supplied: Mapping[str, float] | Sequence[float] | None,
) -> dict[str, float]:
    if supplied is None:
        raw = {
            state.model_name: float(state.model_probability)
            for state in states
        }
    elif isinstance(supplied, Mapping):
        raw = {
            _canonical_model_name(str(name)): float(value)
            for name, value in supplied.items()
        }
    else:
        if len(supplied) != len(states):
            raise ValueError("IMM probabilities must match model state count")
        raw = {
            state.model_name: float(value)
            for state, value in zip(states, supplied)
        }
    if set(raw) != {"CV", "CT_LEFT", "CT_RIGHT"}:
        raise ValueError("full IMM forecast requires CV, CT_LEFT and CT_RIGHT")
    if any(not isfinite(value) or value < 0.0 for value in raw.values()):
        raise ValueError("IMM probabilities must be finite and non-negative")
    total = sum(raw.values())
    if total <= 0.0:
        raise ValueError("IMM probabilities must contain positive mass")
    return {name: value / total for name, value in sorted(raw.items())}


def _stabilize_position_covariance(covariance: np.ndarray) -> np.ndarray:
    symmetric = (covariance + covariance.T) * 0.5
    values, vectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(values, 1e-9)
    return np.asarray((vectors * clipped) @ vectors.T, dtype=float)


def _propagate_state(state: np.ndarray, dt: float, turn_rate: float) -> np.ndarray:
    x, y, vx, vy, _ = state
    speed = hypot(float(vx), float(vy))
    if abs(turn_rate) <= 1e-12:
        return np.asarray([x + vx * dt, y + vy * dt, vx, vy, 0.0], dtype=float)
    heading = atan2(float(vy), float(vx)) if speed > 1e-12 else 0.0
    next_heading = heading + turn_rate * dt
    return np.asarray(
        [
            x + speed / turn_rate * (sin(next_heading) - sin(heading)),
            y - speed / turn_rate * (cos(next_heading) - cos(heading)),
            speed * cos(next_heading),
            speed * sin(next_heading),
            turn_rate,
        ],
        dtype=float,
    )


def _propagate_covariance(
    covariance: np.ndarray,
    state: np.ndarray,
    dt: float,
    turn_rate: float,
    process_noise: np.ndarray,
) -> np.ndarray:
    angle = turn_rate * dt
    rotation = np.asarray([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
    transition = np.eye(5)
    transition[0, 2] = dt
    transition[1, 3] = dt
    transition[2:4, 2:4] = rotation
    transition[2, 4] = -float(state[3]) * dt
    transition[3, 4] = float(state[2]) * dt
    result = transition @ covariance @ transition.T + process_noise * dt
    symmetric = (result + result.T) * 0.5
    values, vectors = np.linalg.eigh(symmetric)
    return np.asarray((vectors * np.maximum(values, 1e-9)) @ vectors.T, dtype=float)


def moment_match_forecasts(
    model_forecasts: Mapping[str, Sequence[Any]] | Sequence[Any],
    model_probabilities: Mapping[str, float] | Sequence[float],
) -> tuple[tuple[tuple[float, float], ...], tuple[tuple[float, float, float, float], ...]]:
    """Moment-match per-model position means and 2-D covariances.

    Each sequence may contain ``(mean, covariance)`` pairs, an
    ``IMMBranchForecast``, or a mapping with ``centerline_xy`` and
    ``covariance_xy`` fields. The returned covariance includes both each
    branch's internal covariance and its disagreement with the mixed mean.
    """
    if isinstance(model_forecasts, Mapping):
        named = {
            _canonical_model_name(str(name)): tuple(values)
            for name, values in model_forecasts.items()
        }
    else:
        named = {
            _canonical_model_name(str(getattr(branch, "model_name", index))): tuple(
                getattr(branch, "centerline_xy", branch)
            )
            for index, branch in enumerate(model_forecasts)
        }
    if isinstance(model_probabilities, Mapping):
        raw_probabilities = {
            _canonical_model_name(str(name)): float(value)
            for name, value in model_probabilities.items()
        }
    else:
        if len(model_probabilities) != len(named):
            raise ValueError("moment-match probabilities must match model forecasts")
        raw_probabilities = {
            name: float(value) for name, value in zip(sorted(named), model_probabilities)
        }
    names = tuple(sorted(named))
    if set(raw_probabilities) != set(names):
        raise ValueError("moment-match probabilities must name every model")
    total = sum(raw_probabilities.values())
    if total <= 0.0 or any(value < 0.0 or not isfinite(value) for value in raw_probabilities.values()):
        raise ValueError("moment-match probabilities must have positive finite mass")
    probabilities = {name: raw_probabilities[name] / total for name in names}

    def series_for(values: Sequence[Any]) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        result: list[tuple[np.ndarray, np.ndarray]] = []
        for item in values:
            if isinstance(item, Mapping):
                mean = item.get("centerline_xy", item.get("mean"))
                covariance = item.get("covariance_xy", item.get("covariance"))
            elif hasattr(item, "centerline_xy"):
                mean = item.centerline_xy
                covariance = item.covariance_xy
            else:
                mean, covariance = item
            mean_array = np.asarray(mean, dtype=float)
            covariance_array = np.asarray(covariance, dtype=float)
            if mean_array.shape != (2,) or covariance_array.shape != (2, 2):
                raise ValueError("moment-match positions and covariances must be 2-D")
            result.append((mean_array, covariance_array))
        return tuple(result)

    series = {name: series_for(named[name]) for name in names}
    lengths = {len(values) for values in series.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise ValueError("all IMM branches must have the same non-empty horizon")
    means: list[tuple[float, float]] = []
    covariances: list[tuple[float, float, float, float]] = []
    for index in range(next(iter(lengths))):
        mixed_mean = sum(
            probabilities[name] * series[name][index][0] for name in names
        )
        mixed_covariance = sum(
            (
                probabilities[name]
                * (
                    series[name][index][1]
                    + np.outer(
                        series[name][index][0] - mixed_mean,
                        series[name][index][0] - mixed_mean,
                    )
                )
            )
            for name in names
        )
        stable = _stabilize_position_covariance(np.asarray(mixed_covariance, dtype=float))
        means.append((float(mixed_mean[0]), float(mixed_mean[1])))
        covariances.append(
            (float(stable[0, 0]), float(stable[0, 1]), float(stable[1, 0]), float(stable[1, 1]))
        )
    return tuple(means), tuple(covariances)


def forecast_imm(
    model_states: Mapping[str, Any] | Sequence[Any] | Any | None = None,
    origin_sim_time_s: float = 0.0,
    horizon_s: float = 1800.0,
    sample_step_s: float = 30.0,
    max_speed_mps: float = 14.0,
    max_turn_rate_rad_s: float = pi / 300.0,
    *,
    states: Mapping[str, Any] | Sequence[Any] | Any | None = None,
    model_probabilities: Mapping[str, float] | Sequence[float] | None = None,
    commanded_turns: Mapping[str, float] | None = None,
    process_noise: Sequence[Sequence[float]] | None = None,
) -> IMMForecastResult:
    """Propagate CV/CT_LEFT/CT_RIGHT and return their moment-matched forecast."""
    selected_states = states if states is not None else model_states
    if selected_states is None:
        raise ValueError("IMM model states are required")
    if horizon_s <= 0.0 or sample_step_s <= 0.0:
        raise ValueError("IMM horizon and sample step must be positive")
    if max_speed_mps <= 0.0 or max_turn_rate_rad_s <= 0.0:
        raise ValueError("IMM physical limits must be positive")
    projections = _state_sequence(selected_states)
    if len(projections) != 3 or len({state.model_name for state in projections}) != 3:
        raise ValueError("full IMM forecast requires three distinct model states")
    probabilities = _normalise_probabilities(projections, model_probabilities)
    noise = np.asarray(
        process_noise
        if process_noise is not None
        else np.diag([0.01, 0.01, 0.001, 0.001, 1e-4]),
        dtype=float,
    )
    if noise.shape != (5, 5) or not np.all(np.isfinite(noise)):
        raise ValueError("IMM process_noise must be a finite 5x5 matrix")
    steps = max(1, ceil(horizon_s / sample_step_s))
    times = tuple(float(origin_sim_time_s + (index + 1) * sample_step_s) for index in range(steps))
    clipping: list[str] = []
    branches: list[IMMBranchForecast] = []
    branch_inputs: dict[str, tuple[tuple[tuple[float, float], np.ndarray], ...]] = {}
    for projection in sorted(projections, key=lambda item: item.model_name):
        name = projection.model_name
        state = np.asarray(projection.state_mean, dtype=float)
        covariance = np.asarray(projection.state_covariance, dtype=float)
        if state.shape != (5,) or covariance.shape != (5, 5):
            raise ValueError("IMM state projections must use the five-state turn model")
        speed = hypot(float(state[2]), float(state[3]))
        if speed > max_speed_mps:
            state[2:4] *= max_speed_mps / speed
            clipping.append(f"{name}:speed@0")
        requested_turn = {
            "CV": 0.0,
            "CT_LEFT": max_turn_rate_rad_s,
            "CT_RIGHT": -max_turn_rate_rad_s,
        }[name]
        if commanded_turns is not None:
            requested_turn = float(
                commanded_turns.get(name, commanded_turns.get(name.casefold(), requested_turn))
            )
        elif abs(float(state[4])) > 1e-12:
            requested_turn = float(state[4])
        turn = max(-max_turn_rate_rad_s, min(max_turn_rate_rad_s, requested_turn))
        if turn != requested_turn:
            clipping.append(f"{name}:turn@0")
        branch_points: list[tuple[float, float]] = []
        branch_covariances: list[tuple[float, float, float, float]] = []
        state_means: list[tuple[float, ...]] = []
        state_covariances: list[tuple[tuple[float, ...], ...]] = []
        pairs: list[tuple[tuple[float, float], np.ndarray]] = []
        for index in range(steps):
            state = _propagate_state(state, sample_step_s, turn)
            speed = hypot(float(state[2]), float(state[3]))
            if speed > max_speed_mps + 1e-9:
                state[2:4] *= max_speed_mps / speed
                clipping.append(f"{name}:speed@{index + 1}")
            covariance = _propagate_covariance(
                covariance, state, sample_step_s, turn, noise
            )
            position_covariance = _stabilize_position_covariance(covariance[:2, :2])
            branch_points.append((float(state[0]), float(state[1])))
            branch_covariances.append(
                (
                    float(position_covariance[0, 0]),
                    float(position_covariance[0, 1]),
                    float(position_covariance[1, 0]),
                    float(position_covariance[1, 1]),
                )
            )
            state_means.append(tuple(float(value) for value in state))
            state_covariances.append(
                tuple(tuple(float(value) for value in row) for row in covariance)
            )
            pairs.append((branch_points[-1], position_covariance))
        branch_inputs[name] = tuple(pairs)
        branches.append(
            IMMBranchForecast(
                model_name=name,
                model_probability=probabilities[name],
                times_s=times,
                centerline_xy=tuple(branch_points),
                covariance_xy=tuple(branch_covariances),
                state_means=tuple(state_means),
                state_covariances=tuple(state_covariances),
                innovation=projection.innovation,
                likelihood=projection.likelihood,
                source_observation_ids=projection.source_observation_ids,
            )
        )
    centerline, covariance_xy = moment_match_forecasts(branch_inputs, probabilities)
    corridor_radius = tuple(
        max(
            1e-9,
            sqrt(
                max(
                    np.linalg.eigvalsh(
                        np.asarray(
                            [[covariance[0], covariance[1]], [covariance[2], covariance[3]]],
                            dtype=float,
                        )
                    )
                )
            ),
        )
        for covariance in covariance_xy
    )
    return IMMForecastResult(
        origin_sim_time_s=origin_sim_time_s,
        times_s=times,
        centerline_xy=centerline,
        covariance_xy=covariance_xy,
        corridor_radius_m=corridor_radius,
        model_branches=tuple(branches),
        model_probabilities=probabilities,
        clipping_records=tuple(clipping),
        source_observation_ids=tuple(
            sorted(
                {
                    observation_id
                    for projection in projections
                    for observation_id in projection.source_observation_ids
                }
            )
        ),
    )


__all__ = [
    "IMMBranchForecast",
    "IMMForecastResult",
    "IMMModelStateProjection",
    "forecast_imm",
    "moment_match_forecasts",
]
