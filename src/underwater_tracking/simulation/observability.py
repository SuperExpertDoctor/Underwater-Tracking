"""Truth-safe observability feedback for the multi-UUV tracking layer.

This module is deliberately independent from the simulation engine.  It accepts
only estimator-visible track, bearing, innovation, and UUV freshness data.  It
returns bounded metrics and evidence-only event hypotheses for a higher-level
LLM or operator.  It never emits a waypoint, sensor command, actuator command,
or other low-level control artifact.

The metric and hypothesis vocabulary follows the local
``uuv_observability_feedback_mvp`` contract while using immutable Python models
that can be adapted to the live system later.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
import math
from pathlib import Path
from types import MappingProxyType
from typing import cast

import numpy as np
import yaml  # type: ignore[import-untyped]


METRIC_NAMES = (
    "geometry_od",
    "fim_min_eigenvalue",
    "fim_logdet",
    "crlb_position_rmse_m",
    "posterior_rmse_m",
    "covariance_area_95_m2",
    "innovation_rms_rad",
    "active_sensor_count",
)

EVENT_HYPOTHESES = (
    "CLOCK_RESET",
    "TARGET_STOPPED",
    "TARGET_SIGNAL_LOST_AFTER_STOP",
    "UUV_SENSOR_OR_COMM_FAILURE",
    "DECOY_OR_NEW_TARGET",
    "TARGET_MANEUVER",
    "GEOMETRY_DEGRADED",
    "ISOLATED_BAD_MEASUREMENT",
    "OBSERVABILITY_CHANGE_UNCLASSIFIED",
)

_FORBIDDEN_FIELDS = frozenset(
    {
        "ground_truth",
        "target_position_truth",
        "target_truth",
        "targettruth",
        "is_decoy",
        "role_label",
        "scenario_truth_label",
        "control_command",
        "low_level_control",
        "command",
        "waypoint",
        "thruster",
        "actuator",
    }
)


class MetricDirection(str, Enum):
    HIGHER_IS_BETTER = "HIGHER_IS_BETTER"
    LOWER_IS_BETTER = "LOWER_IS_BETTER"


class MetricStatus(str, Enum):
    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    SINGULAR_FIM = "SINGULAR_FIM"
    INVALID_COVARIANCE = "INVALID_COVARIANCE"
    NO_VALID_INNOVATION = "NO_VALID_INNOVATION"
    NO_VALID_DATA = "NO_VALID_DATA"
    UNKNOWN = "UNKNOWN"


class ReportType(str, Enum):
    PERIODIC = "PERIODIC"
    URGENT = "URGENT"
    RECOVERY = "RECOVERY"


class ReportSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class ObservabilityConfigError(ValueError):
    """Raised when the observability YAML violates its contract."""


class TruthSafetyError(ValueError):
    """Raised when an input contains truth labels or low-level control data."""


class FrameValidationError(ValueError):
    """Raised when estimator-visible frame data is malformed."""


class WindowTimeError(ValueError):
    """Raised when a non-reset frame moves the explicit clock backward."""


class DuplicateFrameError(ValueError):
    """Raised when a frame sequence identifier is reused."""


def _finite(value: object, field_path: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FrameValidationError(f"{field_path} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise FrameValidationError(f"{field_path} must be finite")
    if minimum is not None and result < minimum:
        raise FrameValidationError(f"{field_path} must be >= {minimum}")
    return result


def _config_number(
    value: object,
    field_path: str,
    *,
    minimum: float = 0.0,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ObservabilityConfigError(f"{field_path} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ObservabilityConfigError(f"{field_path} must be finite")
    if result < minimum or (strictly_positive and result == 0.0):
        relation = ">" if strictly_positive else ">="
        raise ObservabilityConfigError(f"{field_path} must be {relation} {minimum}")
    return result


def _identifier(value: object, field_path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FrameValidationError(f"{field_path} must be a nonempty string")
    return value


def _config_mapping(value: object, field_path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ObservabilityConfigError(f"{field_path} must be an object")
    return cast(Mapping[str, object], value)


def _frame_mapping(value: object, field_path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FrameValidationError(f"{field_path} must be an object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, field_path: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FrameValidationError(f"{field_path} must be an array")
    return value


def _required_keys(
    mapping: Mapping[str, object],
    expected: set[str],
    field_path: str,
) -> None:
    actual = set(mapping)
    extra = actual - expected
    if extra:
        raise FrameValidationError(
            f"{field_path}.{sorted(str(item) for item in extra)[0]} is not allowed"
        )
    missing = expected - actual
    if missing:
        raise FrameValidationError(
            f"{field_path}.{sorted(str(item) for item in missing)[0]} is required"
        )


def _scan_forbidden_fields(value: object, field_path: str = "$") -> None:
    if isinstance(value, Mapping):
        for raw_name, child in value.items():
            name = str(raw_name)
            if name.lower() in _FORBIDDEN_FIELDS:
                raise TruthSafetyError(f"{field_path}.{name} is not allowed online")
            _scan_forbidden_fields(child, f"{field_path}.{name}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _scan_forbidden_fields(child, f"{field_path}[{index}]")


def _pair(value: object, field_path: str) -> tuple[float, float]:
    values = _sequence(value, field_path)
    if len(values) != 2:
        raise FrameValidationError(f"{field_path} must contain two values")
    return (
        _finite(values[0], f"{field_path}[0]"),
        _finite(values[1], f"{field_path}[1]"),
    )


def _covariance(
    value: object,
    field_path: str,
) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    values = _sequence(value, field_path)
    if len(values) != 4:
        raise FrameValidationError(f"{field_path} must contain four values")
    result = tuple(_finite(item, f"{field_path}[{index}]") for index, item in enumerate(values))
    if not math.isclose(result[1], result[2], rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise FrameValidationError(f"{field_path} must be symmetric")
    return cast(tuple[float, float, float, float], result)


def _normalized_angle(value: object, field_path: str) -> float:
    result = _finite(value, field_path)
    if result < -math.pi or result >= math.pi:
        raise FrameValidationError(f"{field_path} must be normalized to [-pi, pi)")
    return result


def wrap_angle_rad(value: float) -> float:
    """Return an angle in ``[-pi, pi)`` without exposing truth labels."""

    result = (float(value) + math.pi) % (2.0 * math.pi) - math.pi
    return -math.pi if result >= math.pi else result


@dataclass(frozen=True, slots=True)
class TrackEstimate:
    track_id: str
    estimated_position_xy_m: tuple[float, float]
    estimated_velocity_xy_mps: tuple[float, float]
    position_covariance_2x2: tuple[float, float, float, float] | None
    association_confidence: float | None
    association_entropy: float | None
    lifecycle_state: str

    @classmethod
    def from_mapping(cls, raw: object, field_path: str = "tracks[]") -> TrackEstimate:
        mapping = _frame_mapping(raw, field_path)
        _required_keys(
            mapping,
            {
                "track_id",
                "estimated_position_xy_m",
                "estimated_velocity_xy_mps",
                "position_covariance_2x2",
                "association_confidence",
                "association_entropy",
                "lifecycle_state",
            },
            field_path,
        )
        confidence = mapping["association_confidence"]
        entropy = mapping["association_entropy"]
        return cls(
            track_id=_identifier(mapping["track_id"], f"{field_path}.track_id"),
            estimated_position_xy_m=_pair(
                mapping["estimated_position_xy_m"],
                f"{field_path}.estimated_position_xy_m",
            ),
            estimated_velocity_xy_mps=_pair(
                mapping["estimated_velocity_xy_mps"],
                f"{field_path}.estimated_velocity_xy_mps",
            ),
            position_covariance_2x2=_covariance(
                mapping["position_covariance_2x2"],
                f"{field_path}.position_covariance_2x2",
            ),
            association_confidence=(
                None
                if confidence is None
                else _finite(confidence, f"{field_path}.association_confidence", minimum=0.0)
            ),
            association_entropy=(
                None
                if entropy is None
                else _finite(entropy, f"{field_path}.association_entropy", minimum=0.0)
            ),
            lifecycle_state=_identifier(
                mapping["lifecycle_state"], f"{field_path}.lifecycle_state"
            ),
        )

    def __post_init__(self) -> None:
        _identifier(self.track_id, "track_id")
        if len(self.estimated_position_xy_m) != 2:
            raise FrameValidationError("estimated_position_xy_m must contain two values")
        if len(self.estimated_velocity_xy_mps) != 2:
            raise FrameValidationError("estimated_velocity_xy_mps must contain two values")
        for index, value in enumerate(self.estimated_position_xy_m):
            _finite(value, f"estimated_position_xy_m[{index}]")
        for index, value in enumerate(self.estimated_velocity_xy_mps):
            _finite(value, f"estimated_velocity_xy_mps[{index}]")
        if self.position_covariance_2x2 is not None:
            _covariance(self.position_covariance_2x2, "position_covariance_2x2")
        if self.association_confidence is not None:
            _finite(self.association_confidence, "association_confidence", minimum=0.0)
            if self.association_confidence > 1.0:
                raise FrameValidationError("association_confidence must be <= 1")
        if self.association_entropy is not None:
            _finite(self.association_entropy, "association_entropy", minimum=0.0)
        _identifier(self.lifecycle_state, "lifecycle_state")


@dataclass(frozen=True, slots=True)
class UuvState:
    uuv_id: str
    position_xy_m: tuple[float, float]
    heading_rad: float
    state_age_sec: float
    communication_age_sec: float
    valid: bool

    @classmethod
    def from_mapping(cls, raw: object, field_path: str = "uuvs[]") -> UuvState:
        mapping = _frame_mapping(raw, field_path)
        _required_keys(
            mapping,
            {
                "uuv_id",
                "position_xy_m",
                "heading_rad",
                "state_age_sec",
                "communication_age_sec",
                "valid",
            },
            field_path,
        )
        valid = mapping["valid"]
        if not isinstance(valid, bool):
            raise FrameValidationError(f"{field_path}.valid must be boolean")
        return cls(
            uuv_id=_identifier(mapping["uuv_id"], f"{field_path}.uuv_id"),
            position_xy_m=_pair(mapping["position_xy_m"], f"{field_path}.position_xy_m"),
            heading_rad=_normalized_angle(mapping["heading_rad"], f"{field_path}.heading_rad"),
            state_age_sec=_finite(mapping["state_age_sec"], f"{field_path}.state_age_sec", minimum=0.0),
            communication_age_sec=_finite(
                mapping["communication_age_sec"],
                f"{field_path}.communication_age_sec",
                minimum=0.0,
            ),
            valid=valid,
        )

    def __post_init__(self) -> None:
        _identifier(self.uuv_id, "uuv_id")
        _pair(self.position_xy_m, "position_xy_m")
        _normalized_angle(self.heading_rad, "heading_rad")
        _finite(self.state_age_sec, "state_age_sec", minimum=0.0)
        _finite(self.communication_age_sec, "communication_age_sec", minimum=0.0)
        if not isinstance(self.valid, bool):
            raise FrameValidationError("valid must be boolean")


@dataclass(frozen=True, slots=True)
class BearingObservation:
    uuv_id: str
    candidate_track_id: str
    sequence_id: int
    bearing_rad: float
    bearing_variance_rad2: float
    measurement_age_sec: float
    valid: bool

    @classmethod
    def from_mapping(
        cls,
        raw: object,
        field_path: str = "bearing_observations[]",
    ) -> BearingObservation:
        mapping = _frame_mapping(raw, field_path)
        _required_keys(
            mapping,
            {
                "uuv_id",
                "candidate_track_id",
                "sequence_id",
                "bearing_rad",
                "bearing_variance_rad2",
                "measurement_age_sec",
                "valid",
            },
            field_path,
        )
        sequence_id = mapping["sequence_id"]
        valid = mapping["valid"]
        if isinstance(sequence_id, bool) or not isinstance(sequence_id, int) or sequence_id < 0:
            raise FrameValidationError(f"{field_path}.sequence_id must be a nonnegative integer")
        if not isinstance(valid, bool):
            raise FrameValidationError(f"{field_path}.valid must be boolean")
        return cls(
            uuv_id=_identifier(mapping["uuv_id"], f"{field_path}.uuv_id"),
            candidate_track_id=_identifier(
                mapping["candidate_track_id"], f"{field_path}.candidate_track_id"
            ),
            sequence_id=sequence_id,
            bearing_rad=_normalized_angle(mapping["bearing_rad"], f"{field_path}.bearing_rad"),
            bearing_variance_rad2=_finite(
                mapping["bearing_variance_rad2"],
                f"{field_path}.bearing_variance_rad2",
                minimum=1.0e-12,
            ),
            measurement_age_sec=_finite(
                mapping["measurement_age_sec"],
                f"{field_path}.measurement_age_sec",
                minimum=0.0,
            ),
            valid=valid,
        )

    def __post_init__(self) -> None:
        _identifier(self.uuv_id, "uuv_id")
        _identifier(self.candidate_track_id, "candidate_track_id")
        if isinstance(self.sequence_id, bool) or not isinstance(self.sequence_id, int):
            raise FrameValidationError("sequence_id must be an integer")
        if self.sequence_id < 0:
            raise FrameValidationError("sequence_id must be nonnegative")
        _normalized_angle(self.bearing_rad, "bearing_rad")
        _finite(self.bearing_variance_rad2, "bearing_variance_rad2", minimum=1.0e-12)
        _finite(self.measurement_age_sec, "measurement_age_sec", minimum=0.0)
        if not isinstance(self.valid, bool):
            raise FrameValidationError("valid must be boolean")


@dataclass(frozen=True, slots=True)
class InnovationSample:
    track_id: str
    uuv_id: str
    innovation_rad: float
    innovation_variance_rad2: float | None

    @classmethod
    def from_mapping(
        cls,
        raw: object,
        field_path: str = "innovations[]",
    ) -> InnovationSample:
        mapping = _frame_mapping(raw, field_path)
        _required_keys(
            mapping,
            {"track_id", "uuv_id", "innovation_rad", "innovation_variance_rad2"},
            field_path,
        )
        variance = mapping["innovation_variance_rad2"]
        return cls(
            track_id=_identifier(mapping["track_id"], f"{field_path}.track_id"),
            uuv_id=_identifier(mapping["uuv_id"], f"{field_path}.uuv_id"),
            innovation_rad=_finite(mapping["innovation_rad"], f"{field_path}.innovation_rad"),
            innovation_variance_rad2=(
                None
                if variance is None
                else _finite(variance, f"{field_path}.innovation_variance_rad2", minimum=0.0)
            ),
        )

    def __post_init__(self) -> None:
        _identifier(self.track_id, "track_id")
        _identifier(self.uuv_id, "uuv_id")
        _finite(self.innovation_rad, "innovation_rad")
        if self.innovation_variance_rad2 is not None:
            _finite(self.innovation_variance_rad2, "innovation_variance_rad2", minimum=0.0)


@dataclass(frozen=True, slots=True)
class InputFrame:
    """Estimator-visible input; no truth or control fields are representable."""

    timestamp_sec: float
    frame_sequence_id: int
    frame_id: str
    tracks: tuple[TrackEstimate, ...]
    uuvs: tuple[UuvState, ...]
    bearing_observations: tuple[BearingObservation, ...]
    innovations: tuple[InnovationSample, ...]

    @classmethod
    def from_mapping(cls, raw: object) -> InputFrame:
        _scan_forbidden_fields(raw)
        mapping = _frame_mapping(raw, "$")
        _required_keys(
            mapping,
            {
                "timestamp_sec",
                "frame_sequence_id",
                "frame_id",
                "tracks",
                "uuvs",
                "bearing_observations",
                "innovations",
            },
            "$",
        )
        sequence_id = mapping["frame_sequence_id"]
        if isinstance(sequence_id, bool) or not isinstance(sequence_id, int) or sequence_id < 0:
            raise FrameValidationError("frame_sequence_id must be a nonnegative integer")
        frame_id = mapping["frame_id"]
        if frame_id != "map":
            raise FrameValidationError("frame_id must be 'map'")
        tracks_raw = _sequence(mapping["tracks"], "$.tracks")
        uuvs_raw = _sequence(mapping["uuvs"], "$.uuvs")
        observations_raw = _sequence(
            mapping["bearing_observations"], "$.bearing_observations"
        )
        innovations_raw = _sequence(mapping["innovations"], "$.innovations")
        return cls(
            timestamp_sec=_finite(mapping["timestamp_sec"], "$.timestamp_sec", minimum=0.0),
            frame_sequence_id=sequence_id,
            frame_id="map",
            tracks=tuple(
                TrackEstimate.from_mapping(item, f"$.tracks[{index}]")
                for index, item in enumerate(tracks_raw)
            ),
            uuvs=tuple(
                UuvState.from_mapping(item, f"$.uuvs[{index}]")
                for index, item in enumerate(uuvs_raw)
            ),
            bearing_observations=tuple(
                BearingObservation.from_mapping(
                    item, f"$.bearing_observations[{index}]"
                )
                for index, item in enumerate(observations_raw)
            ),
            innovations=tuple(
                InnovationSample.from_mapping(item, f"$.innovations[{index}]")
                for index, item in enumerate(innovations_raw)
            ),
        )

    def __post_init__(self) -> None:
        _finite(self.timestamp_sec, "timestamp_sec", minimum=0.0)
        if isinstance(self.frame_sequence_id, bool) or not isinstance(self.frame_sequence_id, int):
            raise FrameValidationError("frame_sequence_id must be an integer")
        if self.frame_sequence_id < 0:
            raise FrameValidationError("frame_sequence_id must be nonnegative")
        if self.frame_id != "map":
            raise FrameValidationError("frame_id must be 'map'")
        track_ids = [track.track_id for track in self.tracks]
        uuv_ids = [uuv.uuv_id for uuv in self.uuvs]
        if len(set(track_ids)) != len(track_ids):
            raise FrameValidationError("track ids must be unique")
        if len(set(uuv_ids)) != len(uuv_ids):
            raise FrameValidationError("UUV ids must be unique")
        known_tracks = set(track_ids)
        known_uuvs = set(uuv_ids)
        for observation in self.bearing_observations:
            if observation.uuv_id not in known_uuvs:
                raise FrameValidationError("bearing observation references unknown UUV")
            if observation.candidate_track_id not in known_tracks:
                raise FrameValidationError("bearing observation references unknown track")
        for innovation in self.innovations:
            if innovation.uuv_id not in known_uuvs:
                raise FrameValidationError("innovation references unknown UUV")
            if innovation.track_id not in known_tracks:
                raise FrameValidationError("innovation references unknown track")


@dataclass(frozen=True, slots=True)
class TimingConfig:
    calculation_period_sec: float
    window_duration_sec: float
    periodic_feedback_sec: float
    warmup_sec: float
    soft_confirmation_samples: int
    event_cooldown_sec: float
    recovery_stable_sec: float
    baseline_window_sec: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> TimingConfig:
        names = (
            "calculation_period_sec",
            "window_duration_sec",
            "periodic_feedback_sec",
            "warmup_sec",
            "event_cooldown_sec",
            "recovery_stable_sec",
            "baseline_window_sec",
        )
        for name in names:
            if name not in raw:
                raise ObservabilityConfigError(f"timing.{name} is required")
        confirmation = raw.get("soft_confirmation_samples")
        if isinstance(confirmation, bool) or not isinstance(confirmation, int) or confirmation <= 0:
            raise ObservabilityConfigError(
                "timing.soft_confirmation_samples must be a positive integer"
            )
        values = {
            name: _config_number(
                raw[name],
                f"timing.{name}",
                strictly_positive=name in {
                    "calculation_period_sec",
                    "window_duration_sec",
                    "periodic_feedback_sec",
                    "baseline_window_sec",
                },
            )
            for name in names
        }
        return cls(soft_confirmation_samples=confirmation, **values)

    def __post_init__(self) -> None:
        for name in (
            "calculation_period_sec",
            "window_duration_sec",
            "periodic_feedback_sec",
            "baseline_window_sec",
        ):
            _config_number(getattr(self, name), f"timing.{name}", strictly_positive=True)
        for name in ("warmup_sec", "event_cooldown_sec", "recovery_stable_sec"):
            _config_number(getattr(self, name), f"timing.{name}")
        if isinstance(self.soft_confirmation_samples, bool) or not isinstance(
            self.soft_confirmation_samples, int
        ) or self.soft_confirmation_samples <= 0:
            raise ObservabilityConfigError(
                "timing.soft_confirmation_samples must be a positive integer"
            )


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    robust_z_threshold: float
    cusum_drift_sigma: float
    cusum_threshold_sigma: float
    min_range_squared_m2: float
    fim_rank_tolerance: float
    max_state_age_sec: float
    max_communication_age_sec: float
    max_measurement_age_sec: float
    stop_speed_threshold_mps: float
    stop_confirmation_sec: float
    maneuver_speed_change_mps: float
    maneuver_heading_change_rad: float
    association_confidence_drop_threshold: float
    association_entropy_rise_threshold: float
    innovation_absolute_threshold_rad: float
    isolated_innovation_ratio: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> DetectionConfig:
        names = (
            "robust_z_threshold",
            "cusum_drift_sigma",
            "cusum_threshold_sigma",
            "min_range_squared_m2",
            "fim_rank_tolerance",
            "max_state_age_sec",
            "max_communication_age_sec",
            "max_measurement_age_sec",
            "stop_speed_threshold_mps",
            "stop_confirmation_sec",
            "maneuver_speed_change_mps",
            "maneuver_heading_change_rad",
            "association_confidence_drop_threshold",
            "association_entropy_rise_threshold",
            "innovation_absolute_threshold_rad",
            "isolated_innovation_ratio",
        )
        for name in names:
            if name not in raw:
                raise ObservabilityConfigError(f"detection.{name} is required")
        return cls(
            **{
                name: _config_number(
                    raw[name],
                    f"detection.{name}",
                    strictly_positive=name
                    in {
                        "robust_z_threshold",
                        "cusum_threshold_sigma",
                        "min_range_squared_m2",
                        "stop_confirmation_sec",
                        "maneuver_speed_change_mps",
                        "maneuver_heading_change_rad",
                        "association_confidence_drop_threshold",
                        "association_entropy_rise_threshold",
                        "innovation_absolute_threshold_rad",
                        "isolated_innovation_ratio",
                    },
                )
                for name in names
            }
        )


@dataclass(frozen=True, slots=True)
class MetricThreshold:
    unit: str
    direction: MetricDirection
    warning: float
    critical: float

    @classmethod
    def from_mapping(cls, metric_name: str, raw: object) -> MetricThreshold:
        mapping = _config_mapping(raw, f"metric_thresholds.{metric_name}")
        expected = {"unit", "direction", "warning", "critical"}
        if set(mapping) != expected:
            raise ObservabilityConfigError(
                f"metric_thresholds.{metric_name} must contain {sorted(expected)}"
            )
        unit = mapping["unit"]
        direction_raw = mapping["direction"]
        if not isinstance(unit, str) or not unit:
            raise ObservabilityConfigError(f"metric_thresholds.{metric_name}.unit is invalid")
        try:
            direction = MetricDirection(direction_raw)
        except (TypeError, ValueError) as exc:
            raise ObservabilityConfigError(
                f"metric_thresholds.{metric_name}.direction is invalid"
            ) from exc
        warning = _config_number(
            mapping["warning"], f"metric_thresholds.{metric_name}.warning", minimum=-math.inf
        )
        critical = _config_number(
            mapping["critical"], f"metric_thresholds.{metric_name}.critical", minimum=-math.inf
        )
        if direction is MetricDirection.HIGHER_IS_BETTER and critical >= warning:
            raise ObservabilityConfigError(
                f"metric_thresholds.{metric_name} requires critical < warning"
            )
        if direction is MetricDirection.LOWER_IS_BETTER and warning >= critical:
            raise ObservabilityConfigError(
                f"metric_thresholds.{metric_name} requires warning < critical"
            )
        return cls(unit=unit, direction=direction, warning=warning, critical=critical)


@dataclass(frozen=True, slots=True)
class MetricsConfig:
    """Metric calculation settings derived from the supervisor YAML."""

    min_range_squared_m2: float
    fim_rank_tolerance: float
    max_state_age_sec: float
    max_communication_age_sec: float
    max_measurement_age_sec: float

    @classmethod
    def from_detection(cls, detection: DetectionConfig) -> MetricsConfig:
        return cls(
            min_range_squared_m2=detection.min_range_squared_m2,
            fim_rank_tolerance=detection.fim_rank_tolerance,
            max_state_age_sec=detection.max_state_age_sec,
            max_communication_age_sec=detection.max_communication_age_sec,
            max_measurement_age_sec=detection.max_measurement_age_sec,
        )


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    schema_version: str
    timing: TimingConfig
    detection: DetectionConfig
    metric_thresholds: Mapping[str, MetricThreshold]
    expected_uuv_count: int
    coordinate_frame: str

    @classmethod
    def from_mapping(cls, raw: object) -> ObservabilityConfig:
        root = _config_mapping(raw, "$")
        expected = {"schema_version", "timing", "detection", "system", "metric_thresholds"}
        if set(root) != expected:
            raise ObservabilityConfigError(
                f"configuration root must contain exactly {sorted(expected)}"
            )
        schema_version = root["schema_version"]
        if not isinstance(schema_version, str) or not schema_version:
            raise ObservabilityConfigError("schema_version must be nonempty")
        system = _config_mapping(root["system"], "system")
        if set(system) != {"expected_uuv_count", "coordinate_frame"}:
            raise ObservabilityConfigError(
                "system must contain exactly expected_uuv_count and coordinate_frame"
            )
        expected_uuv_count = system["expected_uuv_count"]
        if (
            isinstance(expected_uuv_count, bool)
            or not isinstance(expected_uuv_count, int)
            or expected_uuv_count <= 0
        ):
            raise ObservabilityConfigError("system.expected_uuv_count must be positive")
        coordinate_frame = system["coordinate_frame"]
        if coordinate_frame != "map":
            raise ObservabilityConfigError("system.coordinate_frame must be 'map'")
        thresholds_raw = _config_mapping(root["metric_thresholds"], "metric_thresholds")
        if tuple(thresholds_raw) != METRIC_NAMES:
            if set(thresholds_raw) != set(METRIC_NAMES):
                raise ObservabilityConfigError(
                    "metric_thresholds must define exactly the eight observability metrics"
                )
        thresholds = {
            name: MetricThreshold.from_mapping(name, thresholds_raw[name])
            for name in METRIC_NAMES
        }
        for name, threshold in thresholds.items():
            expected_unit, expected_direction = _METRIC_METADATA[name]
            if threshold.unit != expected_unit or threshold.direction is not expected_direction:
                raise ObservabilityConfigError(
                    f"metric_thresholds.{name} metadata does not match metric contract"
                )
        return cls(
            schema_version=schema_version,
            timing=TimingConfig.from_mapping(_config_mapping(root["timing"], "timing")),
            detection=DetectionConfig.from_mapping(
                _config_mapping(root["detection"], "detection")
            ),
            metric_thresholds=dict(thresholds),
            expected_uuv_count=expected_uuv_count,
            coordinate_frame="map",
        )

    @property
    def metric_names(self) -> tuple[str, ...]:
        return METRIC_NAMES

    @property
    def metrics(self) -> MetricsConfig:
        return MetricsConfig.from_detection(self.detection)

    def with_timing(self, **updates: object) -> ObservabilityConfig:
        allowed = set(self.timing.__dataclass_fields__)
        unknown = set(updates) - allowed
        if unknown:
            raise ObservabilityConfigError(
                f"unknown timing override: {sorted(unknown)[0]}"
            )
        values = {name: getattr(self.timing, name) for name in allowed}
        values.update(updates)
        return replace(self, timing=TimingConfig(**values))


def load_observability_config(path: str | Path) -> ObservabilityConfig:
    """Load and validate the explicit observability configuration."""

    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ObservabilityConfigError(f"cannot read {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ObservabilityConfigError(f"invalid YAML in {config_path}") from exc
    return ObservabilityConfig.from_mapping(raw)


_METRIC_METADATA: dict[str, tuple[str, MetricDirection]] = {
    "geometry_od": ("1", MetricDirection.HIGHER_IS_BETTER),
    "fim_min_eigenvalue": ("1/m2", MetricDirection.HIGHER_IS_BETTER),
    "fim_logdet": ("log(1/m4)", MetricDirection.HIGHER_IS_BETTER),
    "crlb_position_rmse_m": ("m", MetricDirection.LOWER_IS_BETTER),
    "posterior_rmse_m": ("m", MetricDirection.LOWER_IS_BETTER),
    "covariance_area_95_m2": ("m2", MetricDirection.LOWER_IS_BETTER),
    "innovation_rms_rad": ("rad", MetricDirection.LOWER_IS_BETTER),
    "active_sensor_count": ("count", MetricDirection.HIGHER_IS_BETTER),
}


@dataclass(frozen=True, slots=True)
class MetricValue:
    name: str
    unit: str
    direction: MetricDirection
    value: float | int | None
    valid: bool
    reason: str

    def __post_init__(self) -> None:
        if self.name not in _METRIC_METADATA:
            raise ValueError(f"unknown metric: {self.name}")
        unit, direction = _METRIC_METADATA[self.name]
        if self.unit != unit or self.direction is not direction:
            raise ValueError(f"metric metadata does not match {self.name}")
        if not self.reason:
            raise ValueError("metric reason must be nonempty")
        if self.valid != (self.value is not None):
            raise ValueError("metric valid flag and value disagree")
        if self.value is not None:
            _finite(self.value, f"metric.{self.name}.value")


@dataclass(frozen=True, slots=True)
class TrackMetricFrame:
    track_id: str
    timestamp_sec: float
    active_uuv_ids: tuple[str, ...]
    stale_uuv_ids: tuple[str, ...]
    association_confidence: float | None
    association_entropy: float | None
    metrics: Mapping[str, MetricValue]


def _valid_metric(name: str, value: float | int) -> MetricValue:
    unit, direction = _METRIC_METADATA[name]
    return MetricValue(name, unit, direction, value, True, MetricStatus.OK.value)


def _invalid_metric(name: str, status: MetricStatus) -> MetricValue:
    unit, direction = _METRIC_METADATA[name]
    return MetricValue(name, unit, direction, None, False, status.value)


def _uuv_is_fresh(uuv: UuvState, config: MetricsConfig) -> bool:
    return bool(
        uuv.valid
        and uuv.state_age_sec <= config.max_state_age_sec
        and uuv.communication_age_sec <= config.max_communication_age_sec
    )


def _fim_metric_values(
    fim: np.ndarray,
    active_sensor_count: int,
    tolerance: float,
) -> dict[str, MetricValue]:
    names = ("geometry_od", "fim_min_eigenvalue", "fim_logdet", "crlb_position_rmse_m")
    if fim.shape != (2, 2) or not np.all(np.isfinite(fim)):
        return {name: _invalid_metric(name, MetricStatus.NO_VALID_DATA) for name in names}
    try:
        eigenvalues = np.linalg.eigvalsh(fim)
    except np.linalg.LinAlgError:
        return {name: _invalid_metric(name, MetricStatus.NO_VALID_DATA) for name in names}
    if not np.all(np.isfinite(eigenvalues)):
        return {name: _invalid_metric(name, MetricStatus.NO_VALID_DATA) for name in names}
    lambda_min = max(0.0, float(eigenvalues[0]))
    lambda_max = max(0.0, float(eigenvalues[1]))
    singular = (
        active_sensor_count < 2
        or lambda_max <= 0.0
        or lambda_min <= tolerance * max(lambda_max, 1.0)
    )
    metrics = {
        "geometry_od": _valid_metric(
            "geometry_od",
            0.0 if singular else math.sqrt(lambda_min / lambda_max),
        ),
        "fim_min_eigenvalue": _valid_metric("fim_min_eigenvalue", lambda_min),
    }
    if singular:
        metrics["fim_logdet"] = _invalid_metric("fim_logdet", MetricStatus.SINGULAR_FIM)
        metrics["crlb_position_rmse_m"] = _invalid_metric(
            "crlb_position_rmse_m", MetricStatus.SINGULAR_FIM
        )
        return metrics
    logdet = math.log(lambda_min) + math.log(lambda_max)
    crlb = math.sqrt((1.0 / lambda_min) + (1.0 / lambda_max))
    metrics["fim_logdet"] = (
        _valid_metric("fim_logdet", logdet)
        if math.isfinite(logdet)
        else _invalid_metric("fim_logdet", MetricStatus.NO_VALID_DATA)
    )
    metrics["crlb_position_rmse_m"] = (
        _valid_metric("crlb_position_rmse_m", crlb)
        if math.isfinite(crlb)
        else _invalid_metric("crlb_position_rmse_m", MetricStatus.NO_VALID_DATA)
    )
    return metrics


def _posterior_metric_values(
    track: TrackEstimate,
) -> dict[str, MetricValue]:
    if track.position_covariance_2x2 is None:
        return {
            "posterior_rmse_m": _invalid_metric("posterior_rmse_m", MetricStatus.NO_VALID_DATA),
            "covariance_area_95_m2": _invalid_metric(
                "covariance_area_95_m2", MetricStatus.NO_VALID_DATA
            ),
        }
    covariance = np.asarray(
        (
            track.position_covariance_2x2[:2],
            track.position_covariance_2x2[2:],
        ),
        dtype=float,
    )
    try:
        eigenvalues = np.linalg.eigvalsh(covariance)
    except np.linalg.LinAlgError:
        eigenvalues = np.asarray((math.nan, math.nan), dtype=float)
    if not np.all(np.isfinite(eigenvalues)) or float(eigenvalues[0]) < -1.0e-12:
        return {
            "posterior_rmse_m": _invalid_metric(
                "posterior_rmse_m", MetricStatus.INVALID_COVARIANCE
            ),
            "covariance_area_95_m2": _invalid_metric(
                "covariance_area_95_m2", MetricStatus.INVALID_COVARIANCE
            ),
        }
    lambda_min = max(0.0, float(eigenvalues[0]))
    lambda_max = max(0.0, float(eigenvalues[1]))
    posterior_rmse = math.sqrt(lambda_min + lambda_max)
    area = math.pi * 5.991 * math.sqrt(lambda_min * lambda_max)
    if not math.isfinite(posterior_rmse) or not math.isfinite(area):
        return {
            "posterior_rmse_m": _invalid_metric(
                "posterior_rmse_m", MetricStatus.INVALID_COVARIANCE
            ),
            "covariance_area_95_m2": _invalid_metric(
                "covariance_area_95_m2", MetricStatus.INVALID_COVARIANCE
            ),
        }
    return {
        "posterior_rmse_m": _valid_metric("posterior_rmse_m", posterior_rmse),
        "covariance_area_95_m2": _valid_metric("covariance_area_95_m2", area),
    }


def compute_track_metrics(
    frame: InputFrame,
    track_id: str,
    config: MetricsConfig,
) -> TrackMetricFrame:
    """Compute all eight metrics from estimator-visible observations only."""

    if not isinstance(frame, InputFrame):
        raise TypeError("frame must be InputFrame")
    if not isinstance(config, MetricsConfig):
        raise TypeError("config must be MetricsConfig")
    track = next((item for item in frame.tracks if item.track_id == track_id), None)
    if track is None:
        raise KeyError(f"unknown track: {track_id}")
    uuv_by_id = {uuv.uuv_id: uuv for uuv in frame.uuvs}
    observations_by_uuv: dict[str, list[BearingObservation]] = {}
    for observation in frame.bearing_observations:
        if observation.candidate_track_id == track_id:
            observations_by_uuv.setdefault(observation.uuv_id, []).append(observation)

    target_x, target_y = track.estimated_position_xy_m
    fim = np.zeros((2, 2), dtype=float)
    active_ids: set[str] = set()
    stale_ids: set[str] = set()
    for uuv_id, observations in observations_by_uuv.items():
        uuv = uuv_by_id[uuv_id]
        eligible = [
            item
            for item in observations
            if item.valid and item.measurement_age_sec <= config.max_measurement_age_sec
        ]
        candidate: BearingObservation | None = (
            max(eligible, key=lambda item: item.sequence_id) if eligible else None
        )
        if candidate is None or not _uuv_is_fresh(uuv, config):
            stale_ids.add(uuv_id)
            continue
        delta_x = target_x - uuv.position_xy_m[0]
        delta_y = target_y - uuv.position_xy_m[1]
        range_squared = delta_x * delta_x + delta_y * delta_y
        if range_squared < config.min_range_squared_m2:
            stale_ids.add(uuv_id)
            continue
        gradient = np.asarray((-delta_y / range_squared, delta_x / range_squared))
        contribution = np.outer(gradient, gradient) / candidate.bearing_variance_rad2
        if not np.all(np.isfinite(contribution)):
            stale_ids.add(uuv_id)
            continue
        fim += contribution
        active_ids.add(uuv_id)

    metrics = _fim_metric_values(fim, len(active_ids), config.fim_rank_tolerance)
    metrics.update(_posterior_metric_values(track))
    innovations = [
        wrap_angle_rad(item.innovation_rad)
        for item in frame.innovations
        if item.track_id == track_id and item.uuv_id in active_ids
    ]
    if innovations:
        rms = math.sqrt(math.fsum(value * value for value in innovations) / len(innovations))
        metrics["innovation_rms_rad"] = (
            _valid_metric("innovation_rms_rad", rms)
            if math.isfinite(rms)
            else _invalid_metric("innovation_rms_rad", MetricStatus.NO_VALID_INNOVATION)
        )
    else:
        metrics["innovation_rms_rad"] = _invalid_metric(
            "innovation_rms_rad", MetricStatus.NO_VALID_INNOVATION
        )
    metrics["active_sensor_count"] = _valid_metric(
        "active_sensor_count", len(active_ids)
    )
    ordered_metrics = {name: metrics[name] for name in METRIC_NAMES}
    fresh_without_observation = {
        uuv.uuv_id for uuv in frame.uuvs if not _uuv_is_fresh(uuv, config)
    }
    stale_ids.update(fresh_without_observation)
    stale_ids.difference_update(active_ids)
    return TrackMetricFrame(
        track_id=track_id,
        timestamp_sec=frame.timestamp_sec,
        active_uuv_ids=tuple(sorted(active_ids)),
        stale_uuv_ids=tuple(sorted(stale_ids)),
        association_confidence=track.association_confidence,
        association_entropy=track.association_entropy,
        metrics=MappingProxyType(ordered_metrics),
    )


@dataclass(frozen=True, slots=True)
class MetricSummary:
    unit: str
    direction: MetricDirection
    instant: float | int | None
    instant_valid: bool
    mean_window: float | None
    mean_valid: bool
    worst_window: float | int | None
    worst_valid: bool
    trend_per_sec: float | None
    trend_valid: bool
    valid_fraction: float
    sample_count: int
    status: MetricStatus
    reason: str


@dataclass(frozen=True, slots=True)
class _WindowSample:
    timestamp_sec: float
    metric: MetricValue


class _MetricWindowBank:
    def __init__(
        self,
        duration_sec: float,
        thresholds: Mapping[str, MetricThreshold],
    ) -> None:
        self._duration_sec = duration_sec
        self._thresholds = thresholds
        self._samples: dict[tuple[str, str], deque[_WindowSample]] = {}
        self._latest_timestamp: float | None = None

    def reset(self) -> None:
        self._samples.clear()
        self._latest_timestamp = None

    def _accept_timestamp(self, timestamp_sec: float) -> None:
        if self._latest_timestamp is not None and timestamp_sec < self._latest_timestamp:
            raise WindowTimeError(
                f"timestamp moved backward from {self._latest_timestamp} to {timestamp_sec}"
            )
        self._latest_timestamp = timestamp_sec

    def _prune(self, samples: deque[_WindowSample], timestamp_sec: float) -> None:
        left = timestamp_sec - self._duration_sec
        while len(samples) > 1 and samples[1].timestamp_sec <= left:
            samples.popleft()

    def add(self, track_id: str, timestamp_sec: float, metric: MetricValue) -> None:
        self._accept_timestamp(timestamp_sec)
        if metric.name not in self._thresholds:
            raise KeyError(metric.name)
        samples = self._samples.setdefault((track_id, metric.name), deque())
        if samples and samples[-1].timestamp_sec == timestamp_sec:
            samples[-1] = _WindowSample(timestamp_sec, metric)
        else:
            samples.append(_WindowSample(timestamp_sec, metric))
        self._prune(samples, timestamp_sec)

    @staticmethod
    def _status(metric: MetricValue, threshold: MetricThreshold) -> tuple[MetricStatus, str]:
        if not metric.valid or metric.value is None:
            try:
                status = MetricStatus(metric.reason)
            except ValueError:
                status = MetricStatus.NO_VALID_DATA
            return status, status.value
        value = float(metric.value)
        if threshold.direction is MetricDirection.HIGHER_IS_BETTER:
            status = (
                MetricStatus.CRITICAL
                if value <= threshold.critical
                else MetricStatus.WARNING
                if value <= threshold.warning
                else MetricStatus.OK
            )
        else:
            status = (
                MetricStatus.CRITICAL
                if value >= threshold.critical
                else MetricStatus.WARNING
                if value >= threshold.warning
                else MetricStatus.OK
            )
        return status, status.value

    def summarize(self, track_id: str, metric_name: str, timestamp_sec: float) -> MetricSummary:
        self._accept_timestamp(timestamp_sec)
        threshold = self._thresholds[metric_name]
        samples = self._samples.get((track_id, metric_name), deque())
        self._prune(samples, timestamp_sec)
        if not samples:
            return MetricSummary(
                unit=threshold.unit,
                direction=threshold.direction,
                instant=None,
                instant_valid=False,
                mean_window=None,
                mean_valid=False,
                worst_window=None,
                worst_valid=False,
                trend_per_sec=None,
                trend_valid=False,
                valid_fraction=0.0,
                sample_count=0,
                status=MetricStatus.NO_VALID_DATA,
                reason=MetricStatus.NO_VALID_DATA.value,
            )
        ordered = list(samples)
        latest = ordered[-1].metric
        status, reason = self._status(latest, threshold)
        left = timestamp_sec - self._duration_sec
        weighted: list[tuple[float, float]] = []
        valid_values: list[float] = []
        valid_intervals = 0.0
        for index, sample in enumerate(ordered):
            start = max(left, sample.timestamp_sec)
            next_timestamp = (
                ordered[index + 1].timestamp_sec
                if index + 1 < len(ordered)
                else timestamp_sec
            )
            end = min(timestamp_sec, next_timestamp)
            duration = max(0.0, end - start)
            if duration > 0.0 and sample.metric.valid and sample.metric.value is not None:
                value = float(sample.metric.value)
                weighted.append((duration, value))
                valid_values.append(value)
                valid_intervals += duration
        valid_fraction = min(1.0, max(0.0, valid_intervals / self._duration_sec))
        mean = (
            math.fsum(duration * value for duration, value in weighted)
            / math.fsum(duration for duration, _ in weighted)
            if weighted
            else None
        )
        worst: float | None
        if not valid_values:
            worst = None
        elif threshold.direction is MetricDirection.HIGHER_IS_BETTER:
            worst = min(valid_values)
        else:
            worst = max(valid_values)
        first = next((sample.metric.value for sample in ordered if sample.metric.valid), None)
        last = next(
            (sample.metric.value for sample in reversed(ordered) if sample.metric.valid),
            None,
        )
        first_timestamp = next(
            (sample.timestamp_sec for sample in ordered if sample.metric.valid), None
        )
        last_timestamp = next(
            (sample.timestamp_sec for sample in reversed(ordered) if sample.metric.valid),
            None,
        )
        trend = (
            (float(last) - float(first)) / (float(last_timestamp) - float(first_timestamp))
            if first is not None
            and last is not None
            and first_timestamp is not None
            and last_timestamp is not None
            and last_timestamp > first_timestamp
            else None
        )
        return MetricSummary(
            unit=threshold.unit,
            direction=threshold.direction,
            instant=latest.value,
            instant_valid=latest.valid,
            mean_window=mean,
            mean_valid=mean is not None,
            worst_window=worst,
            worst_valid=worst is not None,
            trend_per_sec=trend,
            trend_valid=trend is not None,
            valid_fraction=valid_fraction,
            sample_count=len(ordered),
            status=status,
            reason=reason,
        )


@dataclass(frozen=True, slots=True)
class EvidenceEvent:
    event_id: str
    track_id: str
    hypothesis: str
    timestamp_sec: float
    severity: ReportSeverity
    confidence: float
    evidence: tuple[str, ...]
    metric_names: tuple[str, ...] = ()
    recovery: bool = False

    def __post_init__(self) -> None:
        if self.hypothesis not in EVENT_HYPOTHESES:
            raise ValueError(f"unknown evidence hypothesis: {self.hypothesis}")
        if not self.event_id or not self.track_id or not self.evidence:
            raise ValueError("event identifiers and evidence must be nonempty")
        if not 0.0 <= self.confidence <= 1.0 or not math.isfinite(self.confidence):
            raise ValueError("event confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class TrackFeedback:
    track_id: str
    risk_level: ReportSeverity
    active_sensor_count: int
    stale_uuv_ids: tuple[str, ...]
    metrics: Mapping[str, MetricSummary]


@dataclass(frozen=True, slots=True)
class ObservabilityReport:
    report_id: str
    report_type: ReportType
    severity: ReportSeverity
    timestamp_sec: float
    tracks: tuple[TrackFeedback, ...]
    events: tuple[EvidenceEvent, ...]

    def to_public_dict(self) -> dict[str, object]:
        """Serialize only the evidence boundary consumed by an LLM/operator."""

        tracks: list[dict[str, object]] = []
        for track in self.tracks:
            metrics: dict[str, object] = {}
            for name, summary in track.metrics.items():
                metrics[name] = {
                    "unit": summary.unit,
                    "direction": summary.direction.value,
                    "instant": summary.instant,
                    "instant_valid": summary.instant_valid,
                    "mean_window": summary.mean_window,
                    "mean_valid": summary.mean_valid,
                    "worst_window": summary.worst_window,
                    "worst_valid": summary.worst_valid,
                    "trend_per_sec": summary.trend_per_sec,
                    "trend_valid": summary.trend_valid,
                    "valid_fraction": summary.valid_fraction,
                    "sample_count": summary.sample_count,
                    "status": summary.status.value,
                    "reason": summary.reason,
                }
            tracks.append(
                {
                    "track_id": track.track_id,
                    "risk_level": track.risk_level.value,
                    "active_sensor_count": track.active_sensor_count,
                    "stale_uuv_ids": list(track.stale_uuv_ids),
                    "metrics": metrics,
                }
            )
        events = [
            {
                "event_id": event.event_id,
                "track_id": event.track_id,
                "hypothesis": event.hypothesis,
                "timestamp_sec": event.timestamp_sec,
                "severity": event.severity.value,
                "confidence": event.confidence,
                "evidence": list(event.evidence),
                "metric_names": list(event.metric_names),
                "recovery": event.recovery,
            }
            for event in self.events
        ]
        return {
            "schema_version": "observability-feedback-1",
            "report_id": self.report_id,
            "report_type": self.report_type.value,
            "severity": self.severity.value,
            "timestamp_sec": self.timestamp_sec,
            "tracks": tracks,
            "events": events,
        }


@dataclass(slots=True)
class _TrackState:
    previous_active_sensor_count: int | None = None
    previous_speed_mps: float | None = None
    previous_heading_rad: float | None = None
    previous_association_confidence: float | None = None
    previous_association_entropy: float | None = None
    low_speed_since_sec: float | None = None
    candidate_counts: dict[str, int] = field(default_factory=dict)
    active_events: dict[str, EvidenceEvent] = field(default_factory=dict)
    clear_since_sec: dict[str, float] = field(default_factory=dict)
    cooldown_until_sec: dict[str, float] = field(default_factory=dict)


class ObservabilitySupervisor:
    """Windowed evidence supervisor with no policy or low-level control output."""

    def __init__(self, config: ObservabilityConfig) -> None:
        if not isinstance(config, ObservabilityConfig):
            raise TypeError("config must be ObservabilityConfig")
        self._config = config
        self._window = _MetricWindowBank(
            config.timing.window_duration_sec,
            config.metric_thresholds,
        )
        self._states: dict[str, _TrackState] = {}
        self._last_timestamp_sec: float | None = None
        self._next_periodic_sec: float | None = None
        self._seen_frame_ids: set[int] = set()
        self._report_sequence = 0
        self._event_sequence = 0
        self._nominal_track_ids: frozenset[str] | None = None
        self._previous_track_count: int | None = None

    @property
    def config(self) -> ObservabilityConfig:
        return self._config

    def reset(self) -> None:
        self._window.reset()
        self._states.clear()
        self._last_timestamp_sec = None
        self._next_periodic_sec = None
        self._seen_frame_ids.clear()
        self._nominal_track_ids = None
        self._previous_track_count = None

    def _new_report_id(self) -> str:
        self._report_sequence += 1
        return f"report-{self._report_sequence:06d}"

    def _new_event_id(self, track_id: str, hypothesis: str) -> str:
        self._event_sequence += 1
        return f"event-{self._event_sequence:06d}-{track_id}-{hypothesis.lower()}"

    def _track_risk(self, summaries: Mapping[str, MetricSummary]) -> ReportSeverity:
        statuses = {summary.status for summary in summaries.values()}
        if MetricStatus.CRITICAL in statuses:
            return ReportSeverity.CRITICAL
        if MetricStatus.WARNING in statuses or MetricStatus.UNKNOWN in statuses:
            return ReportSeverity.WARNING
        if statuses - {MetricStatus.OK}:
            return ReportSeverity.WARNING
        return ReportSeverity.INFO

    def _track_feedback(
        self,
        metric_frame: TrackMetricFrame,
        summaries: Mapping[str, MetricSummary],
    ) -> TrackFeedback:
        active = metric_frame.metrics["active_sensor_count"].value
        if active is None:
            raise RuntimeError("active_sensor_count must always be present")
        return TrackFeedback(
            track_id=metric_frame.track_id,
            risk_level=self._track_risk(summaries),
            active_sensor_count=int(active),
            stale_uuv_ids=metric_frame.stale_uuv_ids,
            metrics=summaries,
        )

    def _periodic_due(self, timestamp_sec: float) -> bool:
        if self._next_periodic_sec is None or timestamp_sec < self._next_periodic_sec:
            return False
        while self._next_periodic_sec <= timestamp_sec:
            self._next_periodic_sec += self._config.timing.periodic_feedback_sec
        return True

    def _condition_evidence(
        self,
        frame: InputFrame,
        metric_frame: TrackMetricFrame,
        summaries: Mapping[str, MetricSummary],
        track: TrackEstimate,
        state: _TrackState,
    ) -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
        detection = self._config.detection
        active_count = len(metric_frame.active_uuv_ids)
        speed = math.hypot(*track.estimated_velocity_xy_mps)
        if speed <= detection.stop_speed_threshold_mps:
            if state.low_speed_since_sec is None:
                state.low_speed_since_sec = frame.timestamp_sec
            low_speed_duration = frame.timestamp_sec - state.low_speed_since_sec
        else:
            state.low_speed_since_sec = None
            low_speed_duration = 0.0
        stale = metric_frame.stale_uuv_ids
        conditions: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}

        if state.previous_active_sensor_count is not None and active_count < state.previous_active_sensor_count:
            conditions["UUV_SENSOR_OR_COMM_FAILURE"] = (
                ("active sensor count decreased",),
                ("active_sensor_count",),
            )
        elif stale:
            conditions["UUV_SENSOR_OR_COMM_FAILURE"] = (
                ("UUV state or communication data is stale",),
                ("active_sensor_count",),
            )
        if low_speed_duration >= detection.stop_confirmation_sec:
            conditions["TARGET_STOPPED"] = (
                ("low speed persisted through the stop confirmation interval",),
                (),
            )
            if active_count == 0:
                conditions["TARGET_SIGNAL_LOST_AFTER_STOP"] = (
                    ("active sensor count is zero after confirmed low speed",),
                    ("active_sensor_count",),
                )

        speed_change = (
            None
            if state.previous_speed_mps is None
            else abs(speed - state.previous_speed_mps)
        )
        heading = math.atan2(*reversed(track.estimated_velocity_xy_mps)) if speed > 1.0e-12 else None
        heading_change = (
            None
            if heading is None or state.previous_heading_rad is None
            else abs(wrap_angle_rad(heading - state.previous_heading_rad))
        )
        innovation = summaries["innovation_rms_rad"]
        innovation_high = bool(
            innovation.instant_valid
            and innovation.instant is not None
            and float(innovation.instant) >= detection.innovation_absolute_threshold_rad
        )
        motion_changed = bool(
            (speed_change is not None and speed_change >= detection.maneuver_speed_change_mps)
            or (
                heading_change is not None
                and heading_change >= detection.maneuver_heading_change_rad
            )
        )
        if motion_changed and innovation_high:
            conditions["TARGET_MANEUVER"] = (
                ("estimated motion changed while innovation residuals were elevated",),
                ("innovation_rms_rad",),
            )

        confidence_change = (
            None
            if track.association_confidence is None
            or state.previous_association_confidence is None
            else track.association_confidence - state.previous_association_confidence
        )
        entropy_change = (
            None
            if track.association_entropy is None or state.previous_association_entropy is None
            else track.association_entropy - state.previous_association_entropy
        )
        association_evidence: list[str] = []
        if self._previous_track_count is not None and len(frame.tracks) > self._previous_track_count:
            association_evidence.append("candidate track count increased")
        if confidence_change is not None and confidence_change <= -detection.association_confidence_drop_threshold:
            association_evidence.append("association confidence decreased")
        if entropy_change is not None and entropy_change >= detection.association_entropy_rise_threshold:
            association_evidence.append("association entropy increased")
        if association_evidence:
            conditions["DECOY_OR_NEW_TARGET"] = (
                tuple(association_evidence),
                (),
            )

        innovations_by_uuv: dict[str, float] = {}
        for sample in frame.innovations:
            if sample.track_id == track.track_id:
                innovations_by_uuv[sample.uuv_id] = max(
                    innovations_by_uuv.get(sample.uuv_id, 0.0),
                    abs(wrap_angle_rad(sample.innovation_rad)),
                )
        elevated = [
            uuv_id
            for uuv_id, value in innovations_by_uuv.items()
            if value >= detection.innovation_absolute_threshold_rad
        ]
        if len(elevated) == 1 and len(innovations_by_uuv) >= 2:
            other_values = [value for key, value in innovations_by_uuv.items() if key != elevated[0]]
            reference = float(np.median(other_values)) if other_values else 0.0
            if reference <= 1.0e-12 or innovations_by_uuv[elevated[0]] >= detection.isolated_innovation_ratio * reference:
                conditions["ISOLATED_BAD_MEASUREMENT"] = (
                    (f"{elevated[0]} innovation was isolated from peer innovations",),
                    ("innovation_rms_rad",),
                )

        geometry_names = ("geometry_od", "fim_min_eigenvalue")
        uncertainty_names = ("crlb_position_rmse_m", "covariance_area_95_m2")
        degraded_geometry = [
            name
            for name in geometry_names
            if summaries[name].status in {MetricStatus.WARNING, MetricStatus.CRITICAL, MetricStatus.SINGULAR_FIM}
        ]
        degraded_uncertainty = [
            name
            for name in uncertainty_names
            if summaries[name].status
            in {
                MetricStatus.WARNING,
                MetricStatus.CRITICAL,
                MetricStatus.SINGULAR_FIM,
                MetricStatus.INVALID_COVARIANCE,
                MetricStatus.NO_VALID_DATA,
            }
        ]
        if degraded_geometry and degraded_uncertainty:
            names = tuple(dict.fromkeys(degraded_geometry + degraded_uncertainty))
            conditions["GEOMETRY_DEGRADED"] = (
                tuple(f"{name} crossed its configured evidence threshold" for name in names),
                names,
            )
        unclassified_names = tuple(
            name
            for name, summary in summaries.items()
            if summary.status
            in {
                MetricStatus.CRITICAL,
                MetricStatus.SINGULAR_FIM,
                MetricStatus.INVALID_COVARIANCE,
                MetricStatus.NO_VALID_DATA,
            }
            and name != "active_sensor_count"
        )
        if unclassified_names and not conditions:
            names = tuple(
                name
                for name, summary in summaries.items()
                if name in unclassified_names
            )
            conditions["OBSERVABILITY_CHANGE_UNCLASSIFIED"] = (
                ("one or more critical or invalid observability metrics were observed",),
                names,
            )
        return conditions

    @staticmethod
    def _event_confidence(evidence_count: int, severity: ReportSeverity) -> float:
        base = 0.55 + min(0.24, 0.08 * evidence_count)
        if severity is ReportSeverity.CRITICAL:
            base += 0.08
        return min(0.95, base)

    def _emit_episode(
        self,
        track_id: str,
        hypothesis: str,
        timestamp_sec: float,
        evidence: tuple[str, ...],
        metric_names: tuple[str, ...],
        severity: ReportSeverity,
    ) -> EvidenceEvent | None:
        state = self._states[track_id]
        cooldown_until = state.cooldown_until_sec.get(hypothesis, -math.inf)
        if timestamp_sec < cooldown_until:
            return None
        event = EvidenceEvent(
            event_id=self._new_event_id(track_id, hypothesis),
            track_id=track_id,
            hypothesis=hypothesis,
            timestamp_sec=timestamp_sec,
            severity=severity,
            confidence=self._event_confidence(len(evidence), severity),
            evidence=evidence,
            metric_names=metric_names,
        )
        state.active_events[hypothesis] = event
        state.clear_since_sec.pop(hypothesis, None)
        state.cooldown_until_sec[hypothesis] = (
            timestamp_sec + self._config.timing.event_cooldown_sec
        )
        return event

    def _update_events(
        self,
        frame: InputFrame,
        metric_frames: Mapping[str, TrackMetricFrame],
        summaries_by_track: Mapping[str, Mapping[str, MetricSummary]],
    ) -> tuple[EvidenceEvent, ...]:
        urgent: list[EvidenceEvent] = []
        current_track_count = len(frame.tracks)
        for track in sorted(frame.tracks, key=lambda item: item.track_id):
            state = self._states.setdefault(track.track_id, _TrackState())
            conditions = (
                {}
                if frame.timestamp_sec < self._config.timing.warmup_sec
                else self._condition_evidence(
                    frame,
                    metric_frames[track.track_id],
                    summaries_by_track[track.track_id],
                    track,
                    state,
                )
            )
            known_hypotheses = set(state.active_events) | set(state.candidate_counts) | set(conditions)
            for hypothesis in sorted(known_hypotheses):
                condition = conditions.get(hypothesis)
                if condition is not None:
                    state.clear_since_sec.pop(hypothesis, None)
                    state.candidate_counts[hypothesis] = state.candidate_counts.get(hypothesis, 0) + 1
                    if hypothesis not in state.active_events and state.candidate_counts[hypothesis] >= self._config.timing.soft_confirmation_samples:
                        names = condition[1]
                        status = summaries_by_track[track.track_id]
                        severity = (
                            ReportSeverity.CRITICAL
                            if any(
                                status[name].status is MetricStatus.CRITICAL
                                for name in names
                                if name in status
                            )
                            else ReportSeverity.WARNING
                        )
                        event = self._emit_episode(
                            track.track_id,
                            hypothesis,
                            frame.timestamp_sec,
                            condition[0],
                            names,
                            severity,
                        )
                        if event is not None:
                            urgent.append(event)
                elif hypothesis in state.active_events:
                    clear_since = state.clear_since_sec.setdefault(hypothesis, frame.timestamp_sec)
                    if frame.timestamp_sec - clear_since >= self._config.timing.recovery_stable_sec:
                        active = state.active_events.pop(hypothesis)
                        urgent.append(
                            replace(
                                active,
                                timestamp_sec=frame.timestamp_sec,
                                severity=ReportSeverity.INFO,
                                evidence=("observability evidence returned stable",),
                                recovery=True,
                            )
                        )
                        state.candidate_counts.pop(hypothesis, None)
                        state.clear_since_sec.pop(hypothesis, None)
                else:
                    state.candidate_counts.pop(hypothesis, None)
            speed = math.hypot(*track.estimated_velocity_xy_mps)
            state.previous_active_sensor_count = len(metric_frames[track.track_id].active_uuv_ids)
            state.previous_speed_mps = speed
            state.previous_heading_rad = (
                math.atan2(track.estimated_velocity_xy_mps[1], track.estimated_velocity_xy_mps[0])
                if speed > 1.0e-12
                else None
            )
            state.previous_association_confidence = track.association_confidence
            state.previous_association_entropy = track.association_entropy
        self._previous_track_count = current_track_count
        if self._nominal_track_ids is None:
            self._nominal_track_ids = frozenset(track.track_id for track in frame.tracks)
        return tuple(urgent)

    def _build_report(
        self,
        report_type: ReportType,
        severity: ReportSeverity,
        timestamp_sec: float,
        tracks: tuple[TrackFeedback, ...],
        events: tuple[EvidenceEvent, ...],
    ) -> ObservabilityReport:
        return ObservabilityReport(
            report_id=self._new_report_id(),
            report_type=report_type,
            severity=severity,
            timestamp_sec=timestamp_sec,
            tracks=tracks,
            events=events,
        )

    def process_frame(self, frame: InputFrame) -> tuple[ObservabilityReport, ...]:
        """Process one explicit-time frame and return ordered evidence reports."""

        if not isinstance(frame, InputFrame):
            raise TypeError("frame must be InputFrame")
        if frame.frame_sequence_id in self._seen_frame_ids:
            raise DuplicateFrameError(
                f"frame_sequence_id is duplicated: {frame.frame_sequence_id}"
            )
        self._seen_frame_ids.add(frame.frame_sequence_id)
        timestamp = frame.timestamp_sec
        if self._last_timestamp_sec is not None and timestamp < self._last_timestamp_sec:
            self._window.reset()
            self._states.clear()
            self._seen_frame_ids.clear()
            self._seen_frame_ids.add(frame.frame_sequence_id)
            self._nominal_track_ids = None
            self._previous_track_count = None
            self._next_periodic_sec = timestamp + self._config.timing.periodic_feedback_sec
            self._last_timestamp_sec = timestamp
            event = EvidenceEvent(
                event_id=self._new_event_id("system", "CLOCK_RESET"),
                track_id="system",
                hypothesis="CLOCK_RESET",
                timestamp_sec=timestamp,
                severity=ReportSeverity.WARNING,
                confidence=0.71,
                evidence=("timestamp moved backward",),
            )
            return (
                self._build_report(
                    ReportType.URGENT,
                    ReportSeverity.WARNING,
                    timestamp,
                    (),
                    (event,),
                ),
            )
        if self._last_timestamp_sec is None:
            self._next_periodic_sec = timestamp + self._config.timing.periodic_feedback_sec
        if not frame.tracks:
            raise FrameValidationError("a non-reset frame must contain at least one track")
        if len(frame.uuvs) > self._config.expected_uuv_count:
            raise FrameValidationError("frame contains more UUVs than configured")
        metric_frames = {
            track.track_id: compute_track_metrics(frame, track.track_id, self._config.metrics)
            for track in sorted(frame.tracks, key=lambda item: item.track_id)
        }
        summaries_by_track: dict[str, Mapping[str, MetricSummary]] = {}
        feedback_tracks: list[TrackFeedback] = []
        for track_id, metric_frame in metric_frames.items():
            for metric in metric_frame.metrics.values():
                self._window.add(track_id, timestamp, metric)
            summaries = {
                name: self._window.summarize(track_id, name, timestamp)
                for name in METRIC_NAMES
            }
            summaries_by_track[track_id] = MappingProxyType(summaries)
            feedback_tracks.append(self._track_feedback(metric_frame, summaries))
        self._last_timestamp_sec = timestamp
        urgent_events = self._update_events(frame, metric_frames, summaries_by_track)
        tracks = tuple(feedback_tracks)
        reports: list[ObservabilityReport] = []
        recoveries = tuple(event for event in urgent_events if event.recovery)
        new_events = tuple(event for event in urgent_events if not event.recovery)
        if new_events:
            severity = (
                ReportSeverity.CRITICAL
                if any(event.severity is ReportSeverity.CRITICAL for event in new_events)
                else ReportSeverity.WARNING
            )
            reports.append(
                self._build_report(ReportType.URGENT, severity, timestamp, tracks, new_events)
            )
        if recoveries:
            reports.append(
                self._build_report(
                    ReportType.RECOVERY,
                    ReportSeverity.INFO,
                    timestamp,
                    tracks,
                    recoveries,
                )
            )
        if self._periodic_due(timestamp):
            periodic_severity = max(
                (track.risk_level for track in tracks),
                default=ReportSeverity.INFO,
                key=lambda level: {
                    ReportSeverity.INFO: 0,
                    ReportSeverity.WARNING: 1,
                    ReportSeverity.CRITICAL: 2,
                }[level],
            )
            reports.append(
                self._build_report(
                    ReportType.PERIODIC,
                    periodic_severity,
                    timestamp,
                    tracks,
                    (),
                )
            )
        return tuple(reports)


__all__ = [
    "BearingObservation",
    "DetectionConfig",
    "DuplicateFrameError",
    "EVENT_HYPOTHESES",
    "EvidenceEvent",
    "FrameValidationError",
    "InputFrame",
    "InnovationSample",
    "METRIC_NAMES",
    "MetricDirection",
    "MetricStatus",
    "MetricSummary",
    "MetricThreshold",
    "MetricValue",
    "MetricsConfig",
    "ObservabilityConfig",
    "ObservabilityConfigError",
    "ObservabilityReport",
    "ObservabilitySupervisor",
    "ReportSeverity",
    "ReportType",
    "TimingConfig",
    "TrackEstimate",
    "TrackFeedback",
    "TrackMetricFrame",
    "TruthSafetyError",
    "UuvState",
    "WindowTimeError",
    "compute_track_metrics",
    "load_observability_config",
    "wrap_angle_rad",
]
