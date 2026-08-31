"""Deterministic target-intent inference and bounded semantic revision.

The classifier consumes only executed target-track samples and public IMM
state.  Its result is therefore available before any provider call and can be
used as the stable planning baseline when an LLM is slow, unavailable, or
returns an invalid suggestion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import hypot
from typing import Any

import numpy as np
from pydantic import Field

from underwater_tracking.domain.execution_models import (
    ExecutionModel,
    IntentLabel,
)
from underwater_tracking.prediction.features import extract_motion_features

INTENT_LABELS: tuple[IntentLabel, ...] = (
    "transit",
    "patrol",
    "loiter",
    "evade",
    "approach",
    "withdraw",
    "unknown",
)
RULE_VERSION = "deterministic-intent-v1"


class MotionIntentFeatures(ExecutionModel):
    """Kinematic and IMM features used by the deterministic classifier."""

    target_id: str = ""
    sim_time_s: float = Field(default=0.0, ge=0)
    mean_speed_mps: float = 0.0
    max_speed_mps: float = 0.0
    acceleration_mps2: float = 0.0
    heading_change_rad: float = 0.0
    signed_turn_rate_rad_s: float = 0.0
    curvature_q75: float = 0.0
    net_displacement_m: float = 0.0
    path_efficiency: float = 0.0
    dwell_fraction: float = Field(default=0.0, ge=0, le=1)
    dwell_duration_s: float = Field(default=0.0, ge=0)
    boundary_distance_m: float | None = Field(default=None, ge=0)
    boundary_approach_rate_mps: float = 0.0
    leading_model: str = "CV"
    leading_model_probability: float = Field(default=0.0, ge=0, le=1)
    model_probability_change: float = 0.0
    evidence_ids: tuple[str, ...] = ()


class IntentLatchState(ExecutionModel):
    """Checkpointable state for semantic hysteresis and LLM confirmation."""

    current_label: IntentLabel = "unknown"
    candidate_label: IntentLabel | None = None
    candidate_count: int = Field(default=0, ge=0)
    intent_revision: int = Field(default=0, ge=0)
    last_prediction_revision: int = Field(default=0, ge=0)
    llm_candidate_label: IntentLabel | None = None
    llm_candidate_count: int = Field(default=0, ge=0)


class ConfirmedIntentRevision(ExecutionModel):
    """The current deterministic intent, optionally replaced by a gated LLM revision."""

    target_id: str
    intent_label: IntentLabel
    confidence: float = Field(ge=0, le=1)
    intent_revision: int = Field(ge=1)
    prediction_revision: int = Field(ge=1)
    rule_version: str = RULE_VERSION
    features: Mapping[str, Any] = Field(default_factory=dict)
    thresholds: Mapping[str, float] = Field(default_factory=dict)
    evidence_ids: tuple[str, ...] = Field(min_length=1)
    latch_state: IntentLatchState
    llm_revision_accepted: bool = False
    llm_revision_reason: str | None = None

    @property
    def label(self) -> IntentLabel:
        """Compatibility accessor used by planners that call the result a label."""

        return self.intent_label


class DeterministicIntentClassifier:
    """Classify target motion with enter/exit thresholds and a cycle latch."""

    def __init__(
        self,
        *,
        confirmation_cycles: int = 2,
        stationary_speed_mps: float = 0.5,
        dwell_enter_fraction: float = 0.50,
        dwell_exit_fraction: float = 0.30,
        boundary_rate_enter_mps: float = 1.0,
        boundary_rate_exit_mps: float = 0.35,
        evade_acceleration_enter_mps2: float = 0.75,
        evade_acceleration_exit_mps2: float = 0.35,
        evade_curvature_enter: float = 0.008,
        evade_curvature_exit: float = 0.004,
        transit_efficiency_enter: float = 0.75,
        patrol_efficiency_exit: float = 0.65,
        llm_min_confidence: float = 0.70,
        llm_min_margin: float = 0.15,
    ) -> None:
        if confirmation_cycles < 1:
            raise ValueError("confirmation_cycles must be positive")
        if not 0.0 <= dwell_exit_fraction < dwell_enter_fraction <= 1.0:
            raise ValueError("dwell exit threshold must be below dwell enter threshold")
        if boundary_rate_exit_mps < 0.0 or boundary_rate_enter_mps <= boundary_rate_exit_mps:
            raise ValueError("boundary exit threshold must be below boundary enter threshold")
        if evade_acceleration_exit_mps2 < 0.0 or evade_acceleration_enter_mps2 <= evade_acceleration_exit_mps2:
            raise ValueError("evasion acceleration exit threshold must be below enter threshold")
        if evade_curvature_exit < 0.0 or evade_curvature_enter <= evade_curvature_exit:
            raise ValueError("evasion curvature exit threshold must be below enter threshold")
        self.confirmation_cycles = confirmation_cycles
        self.stationary_speed_mps = stationary_speed_mps
        self.dwell_enter_fraction = dwell_enter_fraction
        self.dwell_exit_fraction = dwell_exit_fraction
        self.boundary_rate_enter_mps = boundary_rate_enter_mps
        self.boundary_rate_exit_mps = boundary_rate_exit_mps
        self.evade_acceleration_enter_mps2 = evade_acceleration_enter_mps2
        self.evade_acceleration_exit_mps2 = evade_acceleration_exit_mps2
        self.evade_curvature_enter = evade_curvature_enter
        self.evade_curvature_exit = evade_curvature_exit
        self.transit_efficiency_enter = transit_efficiency_enter
        self.patrol_efficiency_exit = patrol_efficiency_exit
        self.llm_min_confidence = llm_min_confidence
        self.llm_min_margin = llm_min_margin

    @property
    def thresholds(self) -> dict[str, float]:
        """Return the exact thresholds recorded with every conclusion."""

        return {
            "stationary_speed_mps": self.stationary_speed_mps,
            "dwell_enter_fraction": self.dwell_enter_fraction,
            "dwell_exit_fraction": self.dwell_exit_fraction,
            "boundary_rate_enter_mps": self.boundary_rate_enter_mps,
            "boundary_rate_exit_mps": self.boundary_rate_exit_mps,
            "evade_acceleration_enter_mps2": self.evade_acceleration_enter_mps2,
            "evade_acceleration_exit_mps2": self.evade_acceleration_exit_mps2,
            "evade_curvature_enter": self.evade_curvature_enter,
            "evade_curvature_exit": self.evade_curvature_exit,
            "transit_efficiency_enter": self.transit_efficiency_enter,
            "patrol_efficiency_exit": self.patrol_efficiency_exit,
        }

    def features_from_history(
        self,
        target_id: str,
        history: Sequence[Any],
        *,
        boundary: tuple[float, float, float, float] | None = None,
        imm_model_probabilities: Mapping[str, float] | None = None,
        source_evidence_ids: Sequence[str] = (),
    ) -> MotionIntentFeatures:
        """Extract kinematics from ``(sim_time_s, x, y)`` or sample objects."""
        samples = _normalize_history(history)
        if len(samples) < 3:
            raise ValueError("at least three target-track samples are required")
        times = np.asarray([sample[0] for sample in samples], dtype=float)
        positions = np.asarray([[sample[1], sample[2]] for sample in samples], dtype=float)
        extracted = extract_motion_features(
            times,
            positions,
            stationary_speed_mps=self.stationary_speed_mps,
        )
        boundary_distance = None
        boundary_rate = 0.0
        if boundary is not None:
            distances = tuple(_distance_to_boundary(point, boundary) for point in positions)
            boundary_distance = distances[-1]
            boundary_rate = (distances[-1] - distances[0]) / float(times[-1] - times[0])
        model, model_probability, probability_change = _leading_model(
            imm_model_probabilities
        )
        return MotionIntentFeatures(
            target_id=target_id,
            sim_time_s=float(times[-1]),
            mean_speed_mps=extracted["mean_speed_mps"],
            max_speed_mps=extracted["max_speed_mps"],
            acceleration_mps2=extracted["last_window_acceleration_mps2"],
            heading_change_rad=extracted["heading_change_rad"],
            signed_turn_rate_rad_s=extracted["signed_turn_rate_mean_rad_s"],
            curvature_q75=extracted["curvature_q75"],
            net_displacement_m=extracted["net_displacement_m"],
            path_efficiency=extracted["path_efficiency"],
            dwell_fraction=extracted["dwell_fraction"],
            dwell_duration_s=extracted["dwell_fraction"]
            * float(times[-1] - times[0]),
            boundary_distance_m=boundary_distance,
            boundary_approach_rate_mps=boundary_rate,
            leading_model=model,
            leading_model_probability=model_probability,
            model_probability_change=probability_change,
            evidence_ids=tuple(sorted(set(source_evidence_ids))),
        )

    def classify_history(
        self,
        target_id: str,
        history: Sequence[Any],
        *,
        prediction_revision: int,
        prior: IntentLatchState | None = None,
        boundary: tuple[float, float, float, float] | None = None,
        imm_model_probabilities: Mapping[str, float] | None = None,
        source_evidence_ids: Sequence[str] = (),
    ) -> ConfirmedIntentRevision:
        features = self.features_from_history(
            target_id,
            history,
            boundary=boundary,
            imm_model_probabilities=imm_model_probabilities,
            source_evidence_ids=source_evidence_ids,
        )
        return self.classify(features, prior=prior, prediction_revision=prediction_revision)

    def classify(
        self,
        features: MotionIntentFeatures | Mapping[str, Any],
        *,
        prior: IntentLatchState | None = None,
        prediction_revision: int = 1,
    ) -> ConfirmedIntentRevision:
        """Classify one feature vector and apply the deterministic latch."""
        normalized = (
            features
            if isinstance(features, MotionIntentFeatures)
            else MotionIntentFeatures.model_validate(features)
        )
        if prediction_revision < 1:
            raise ValueError("prediction_revision must be positive")
        state = prior or IntentLatchState()
        candidate = self._candidate(normalized, state.current_label)
        evidence_ids = normalized.evidence_ids or (
            f"intent:{normalized.target_id or 'target'}:{prediction_revision}",
        )
        if state.current_label == "unknown" and candidate != "unknown":
            current_label = candidate
            intent_revision = max(1, state.intent_revision + 1)
            next_state = IntentLatchState(
                current_label=current_label,
                intent_revision=intent_revision,
                last_prediction_revision=prediction_revision,
            )
        elif candidate == state.current_label or candidate == "unknown":
            next_state = state.model_copy(
                update={
                    "candidate_label": None,
                    "candidate_count": 0,
                    "last_prediction_revision": prediction_revision,
                }
            )
            current_label = state.current_label
            intent_revision = max(1, state.intent_revision)
        else:
            count = (
                state.candidate_count + 1
                if state.candidate_label == candidate
                else 1
            )
            accepted = count >= self.confirmation_cycles
            current_label = candidate if accepted else state.current_label
            intent_revision = (
                max(1, state.intent_revision + 1)
                if accepted
                else max(1, state.intent_revision)
            )
            next_state = IntentLatchState(
                current_label=current_label,
                candidate_label=None if accepted else candidate,
                candidate_count=0 if accepted else count,
                intent_revision=intent_revision,
                last_prediction_revision=prediction_revision,
                llm_candidate_label=state.llm_candidate_label,
                llm_candidate_count=state.llm_candidate_count,
            )
        confidence = self._confidence(normalized, current_label, candidate)
        return ConfirmedIntentRevision(
            target_id=normalized.target_id,
            intent_label=current_label,
            confidence=confidence,
            intent_revision=intent_revision,
            prediction_revision=prediction_revision,
            features=normalized.model_dump(
                mode="python", exclude={"target_id", "evidence_ids"}
            ),
            thresholds=self.thresholds,
            evidence_ids=evidence_ids,
            latch_state=next_state,
        )

    def accept_llm_revision(
        self,
        baseline: ConfirmedIntentRevision,
        *,
        proposed_label: str,
        confidence: float,
        evidence_ids: Sequence[str],
        allowed_evidence_ids: Sequence[str],
        prediction_revision: int,
        runner_up_confidence: float = 0.0,
    ) -> ConfirmedIntentRevision:
        """Apply an LLM label only after two current, evidence-backed calls."""
        reason: str | None = None
        try:
            label = _normalize_label(proposed_label)
        except ValueError:
            label = baseline.intent_label
            reason = "invalid_label"
        supplied = set(evidence_ids)
        allowed = set(allowed_evidence_ids)
        if reason is None and prediction_revision != baseline.prediction_revision:
            reason = "stale_prediction_revision"
        if reason is None and confidence < self.llm_min_confidence:
            reason = "low_confidence"
        if reason is None and confidence - runner_up_confidence < self.llm_min_margin:
            reason = "insufficient_margin"
        if reason is None and not supplied or (reason is None and not supplied <= allowed):
            reason = "unresolved_evidence"
        if reason is None and label == baseline.intent_label:
            reason = "same_as_baseline"
        state = baseline.latch_state
        if reason is not None:
            reset = state.model_copy(update={"llm_candidate_label": None, "llm_candidate_count": 0})
            return baseline.model_copy(
                update={"latch_state": reset, "llm_revision_accepted": False, "llm_revision_reason": reason}
            )
        count = state.llm_candidate_count + 1 if state.llm_candidate_label == label else 1
        if count < max(2, self.confirmation_cycles):
            pending = state.model_copy(
                update={"llm_candidate_label": label, "llm_candidate_count": count}
            )
            return baseline.model_copy(
                update={
                    "latch_state": pending,
                    "llm_revision_accepted": False,
                    "llm_revision_reason": "awaiting_consecutive_confirmation",
                }
            )
        revised_state = state.model_copy(
            update={
                "current_label": label,
                "candidate_label": None,
                "candidate_count": 0,
                "intent_revision": baseline.intent_revision + 1,
                "llm_candidate_label": label,
                "llm_candidate_count": count,
            }
        )
        return baseline.model_copy(
            update={
                "intent_label": label,
                "confidence": confidence,
                "intent_revision": baseline.intent_revision + 1,
                "evidence_ids": tuple(sorted(set(baseline.evidence_ids) | supplied)),
                "latch_state": revised_state,
                "llm_revision_accepted": True,
                "llm_revision_reason": None,
            }
        )

    def _candidate(
        self, features: MotionIntentFeatures, current_label: IntentLabel
    ) -> IntentLabel:
        rate = features.boundary_approach_rate_mps
        if features.boundary_distance_m is not None:
            if rate <= -self.boundary_rate_enter_mps:
                return "approach"
            if rate >= self.boundary_rate_enter_mps:
                return "withdraw"
            if current_label == "approach" and rate <= self.boundary_rate_exit_mps:
                return "approach"
            if current_label == "withdraw" and rate >= -self.boundary_rate_exit_mps:
                return "withdraw"
        if features.dwell_fraction >= self.dwell_enter_fraction:
            return "loiter"
        if current_label == "loiter" and features.dwell_fraction >= self.dwell_exit_fraction:
            return "loiter"
        evasion_signal = (
            features.acceleration_mps2 >= self.evade_acceleration_enter_mps2
            or features.curvature_q75 >= self.evade_curvature_enter
        )
        if current_label == "evade":
            evasion_signal = (
                features.acceleration_mps2 >= self.evade_acceleration_exit_mps2
                or features.curvature_q75 >= self.evade_curvature_exit
            )
        if evasion_signal and features.path_efficiency >= 0.20:
            return "evade"
        if features.path_efficiency >= self.transit_efficiency_enter and abs(
            features.signed_turn_rate_rad_s
        ) < self.boundary_rate_exit_mps / 100.0:
            return "transit"
        if (
            features.heading_change_rad > 0.5
            or features.path_efficiency < self.patrol_efficiency_exit
        ):
            return "patrol"
        return "unknown"

    @staticmethod
    def _confidence(
        features: MotionIntentFeatures,
        current_label: IntentLabel,
        candidate: IntentLabel,
    ) -> float:
        if current_label == "unknown":
            return 0.0
        if current_label != candidate:
            return 0.5
        if current_label == "loiter":
            return min(1.0, 0.5 + features.dwell_fraction / 2.0)
        if current_label in {"approach", "withdraw"}:
            return min(1.0, 0.5 + abs(features.boundary_approach_rate_mps) / 10.0)
        if current_label == "transit":
            return min(1.0, 0.5 + features.path_efficiency / 2.0)
        return min(1.0, 0.5 + features.heading_change_rad / 6.0)


def _normalize_history(history: Sequence[Any]) -> tuple[tuple[float, float, float], ...]:
    normalized: list[tuple[float, float, float]] = []
    for sample in history:
        if isinstance(sample, Mapping):
            time_s = sample.get("sim_time_s")
            position = sample.get("position_xy")
            if position is None:
                position = (sample.get("x"), sample.get("y"))
        elif hasattr(sample, "sim_time_s"):
            time_s = sample.sim_time_s
            position = sample.position_xy
        else:
            time_s, x, y = sample[:3]
            position = (x, y)
        if time_s is None or position is None or len(position) != 2:
            raise ValueError("target-track samples must contain time and a 2-D position")
        normalized.append((float(time_s), float(position[0]), float(position[1])))
    return tuple(normalized)


def _distance_to_boundary(
    point: Sequence[float], boundary: tuple[float, float, float, float]
) -> float:
    xmin, xmax, ymin, ymax = boundary
    x, y = float(point[0]), float(point[1])
    if xmin > xmax or ymin > ymax:
        raise ValueError("boundary extents must be ordered")
    if xmin <= x <= xmax and ymin <= y <= ymax:
        return min(x - xmin, xmax - x, y - ymin, ymax - y)
    dx = max(xmin - x, 0.0, x - xmax)
    dy = max(ymin - y, 0.0, y - ymax)
    return hypot(dx, dy)


def _leading_model(
    probabilities: Mapping[str, float] | None,
) -> tuple[str, float, float]:
    if not probabilities:
        return "CV", 0.0, 0.0
    values = sorted(
        ((str(name), float(value)) for name, value in probabilities.items()),
        key=lambda item: (-item[1], item[0]),
    )
    if not values:
        return "CV", 0.0, 0.0
    leading, probability = values[0]
    second = values[1][1] if len(values) > 1 else 0.0
    return leading.upper(), probability, probability - second


def _normalize_label(value: str) -> IntentLabel:
    label = str(value).casefold()
    if label not in INTENT_LABELS:
        raise ValueError(f"unsupported intent label {value!r}")
    return label  # type: ignore[return-value]


__all__ = [
    "ConfirmedIntentRevision",
    "DeterministicIntentClassifier",
    "IntentLatchState",
    "MotionIntentFeatures",
]
