from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from math import atan2, hypot, pi
from collections.abc import Callable

from underwater_tracking.domain.observations import PassiveSonarObservation
from underwater_tracking.domain.platforms import SonarCapability

_MIN_DETECTION_CONFIDENCE = 0.05
_MAX_DETECTION_CONFIDENCE = 0.98
_CONFIDENCE_NOISE_STDDEV = 0.075
_MIN_SNR_DB = -12.0
_MAX_SNR_DB = 12.0
_SNR_NOISE_STDDEV_DB = 1.5
_DEFAULT_PD_NEAR = 0.95
_DEFAULT_PD_FAR = 0.40


@dataclass(frozen=True, slots=True)
class SonarNode:
    platform_id: str
    position_xy: tuple[float, float]
    capability: SonarCapability


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return hypot(left[0] - right[0], left[1] - right[1])


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _range_fraction(distance: float, maximum_range_m: float) -> float:
    return _clamp(distance / maximum_range_m, 0.0, 1.0)


def default_pd_curve(range_fraction: float) -> float:
    """Default passive-sonar probability of detection versus normalized range."""
    fraction = _clamp(range_fraction, 0.0, 1.0)
    return _clamp(
        _DEFAULT_PD_NEAR + (_DEFAULT_PD_FAR - _DEFAULT_PD_NEAR) * fraction,
        0.02,
        0.98,
    )


def _sample_detection_confidence(range_fraction: float, rng: random.Random) -> float:
    mean_confidence = 0.9 - 0.7 * range_fraction
    return _clamp(
        mean_confidence + rng.gauss(0.0, _CONFIDENCE_NOISE_STDDEV),
        _MIN_DETECTION_CONFIDENCE,
        _MAX_DETECTION_CONFIDENCE,
    )


def _sample_snr_db(range_fraction: float, rng: random.Random) -> float:
    mean_snr_db = 10.0 - 18.0 * range_fraction
    return _clamp(
        mean_snr_db + rng.gauss(0.0, _SNR_NOISE_STDDEV_DB),
        _MIN_SNR_DB,
        _MAX_SNR_DB,
    )


def _resolve_quality_rng(
    measurement_rng: random.Random, quality_rng: random.Random | None
) -> random.Random:
    """Return a deterministic quality stream without advancing measurements."""
    if quality_rng is not None:
        return quality_rng
    state_digest = hashlib.sha256(repr(measurement_rng.getstate()).encode("utf-8")).digest()
    return random.Random(state_digest)


def _resolve_detection_rng(
    measurement_rng: random.Random, detection_rng: random.Random | None
) -> random.Random:
    if detection_rng is not None:
        return detection_rng
    state_digest = hashlib.sha256(
        b"detection:" + repr(measurement_rng.getstate()).encode("utf-8")
    ).digest()
    return random.Random(state_digest)


def _resolve_clutter_rng(
    measurement_rng: random.Random, clutter_rng: random.Random | None
) -> random.Random:
    if clutter_rng is not None:
        return clutter_rng
    state_digest = hashlib.sha256(
        b"clutter:" + repr(measurement_rng.getstate()).encode("utf-8")
    ).digest()
    return random.Random(state_digest)


def _wrapped_noisy_bearing(
    origin: tuple[float, float],
    target: tuple[float, float],
    sigma_rad: float,
    rng: random.Random,
) -> float:
    bearing = atan2(target[1] - origin[1], target[0] - origin[0])
    return (bearing + rng.gauss(0.0, sigma_rad) + pi) % (2.0 * pi) - pi


def make_passive_observation(
    *,
    scenario_id: str,
    sim_time_s: int,
    observer: SonarNode,
    target_id: str,
    target_xy: tuple[float, float],
    rng: random.Random,
    quality_rng: random.Random | None = None,
    detection_rng: random.Random | None = None,
    pd_curve: Callable[[float], float] = default_pd_curve,
) -> PassiveSonarObservation | None:
    distance = _distance(observer.position_xy, target_xy)
    if distance > observer.capability.passive_range_m:
        return None
    range_fraction = _range_fraction(distance, observer.capability.passive_range_m)
    probability = _clamp(float(pd_curve(range_fraction)), 0.0, 1.0)
    if _resolve_detection_rng(rng, detection_rng).random() >= probability:
        return None
    resolved_quality_rng = _resolve_quality_rng(rng, quality_rng)
    confidence = _sample_detection_confidence(range_fraction, resolved_quality_rng)
    snr_db = _sample_snr_db(range_fraction, resolved_quality_rng)
    return PassiveSonarObservation(
        observation_id=f"passive:{observer.platform_id}:{target_id}:{sim_time_s}",
        scenario_id=scenario_id,
        sim_time_s=sim_time_s,
        observer_id=observer.platform_id,
        target_id=target_id,
        azimuth_rad=_wrapped_noisy_bearing(
            observer.position_xy,
            target_xy,
            observer.capability.passive_bearing_variance_rad2**0.5,
            rng,
        ),
        variance_rad2=observer.capability.passive_bearing_variance_rad2,
        detection_confidence=confidence,
        snr_db=snr_db,
        observer_position_xy=observer.position_xy,
    )


def make_passive_observations(
    *,
    scenario_id: str,
    sim_time_s: int,
    observer: SonarNode,
    target_id: str,
    target_xy: tuple[float, float],
    rng: random.Random,
    quality_rng: random.Random | None = None,
    detection_rng: random.Random | None = None,
    clutter_rng: random.Random | None = None,
    clutter_sensitivity: float = 0.0,
    pd_curve: Callable[[float], float] = default_pd_curve,
) -> tuple[PassiveSonarObservation, ...]:
    """Generate a true passive detection plus at most one marked clutter hit."""
    if not 0.0 <= clutter_sensitivity <= 1.0:
        raise ValueError("clutter_sensitivity must be between 0 and 1")
    observations: list[PassiveSonarObservation] = []
    true_observation = make_passive_observation(
        scenario_id=scenario_id,
        sim_time_s=sim_time_s,
        observer=observer,
        target_id=target_id,
        target_xy=target_xy,
        rng=rng,
        quality_rng=quality_rng,
        detection_rng=detection_rng,
        pd_curve=pd_curve,
    )
    if true_observation is not None:
        observations.append(true_observation)
    resolved_clutter_rng = _resolve_clutter_rng(rng, clutter_rng)
    if resolved_clutter_rng.random() >= clutter_sensitivity:
        return tuple(observations)
    resolved_quality_rng = _resolve_quality_rng(rng, quality_rng)
    clutter_id = f"clutter:{observer.platform_id}:{target_id}:{sim_time_s}"
    observations.append(
        PassiveSonarObservation(
            observation_id=f"passive:{clutter_id}",
            scenario_id=scenario_id,
            sim_time_s=sim_time_s,
            observer_id=observer.platform_id,
            target_id=clutter_id,
            azimuth_rad=resolved_clutter_rng.uniform(-pi, pi),
            variance_rad2=observer.capability.passive_bearing_variance_rad2 * 4.0,
            detection_confidence=_sample_detection_confidence(1.0, resolved_quality_rng),
            snr_db=_sample_snr_db(1.0, resolved_quality_rng),
            is_false_alarm=True,
        )
    )
    return tuple(observations)
