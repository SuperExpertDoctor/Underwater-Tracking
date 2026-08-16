from __future__ import annotations

import random
from dataclasses import dataclass
from math import atan2, hypot, pi

from underwater_tracking.domain.observations import (
    ActiveTransmission,
    MultistaticObservation,
    PassiveSonarObservation,
)
from underwater_tracking.domain.platforms import SonarCapability

_MIN_DETECTION_CONFIDENCE = 0.05
_MAX_DETECTION_CONFIDENCE = 0.98
_CONFIDENCE_NOISE_STDDEV = 0.075
_MIN_SNR_DB = -12.0
_MAX_SNR_DB = 12.0
_SNR_NOISE_STDDEV_DB = 1.5


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
) -> PassiveSonarObservation | None:
    distance = _distance(observer.position_xy, target_xy)
    if distance > observer.capability.passive_range_m:
        return None
    range_fraction = _range_fraction(distance, observer.capability.passive_range_m)
    confidence = _sample_detection_confidence(range_fraction, rng)
    snr_db = _sample_snr_db(range_fraction, rng)
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
    )


def make_multistatic_observations(
    *,
    scenario_id: str,
    sim_time_s: int,
    emitter: SonarNode,
    receivers: tuple[SonarNode, ...],
    target_id: str,
    target_xy: tuple[float, float],
    rng: random.Random,
) -> tuple[ActiveTransmission, tuple[MultistaticObservation, ...]]:
    if not emitter.capability.active_capable:
        raise ValueError(f"platform {emitter.platform_id!r} cannot emit active sonar")
    emitter_leg = _distance(emitter.position_xy, target_xy)
    if emitter_leg > emitter.capability.active_source_range_m:
        raise ValueError("target is outside emitter active-source range")
    transmission = ActiveTransmission(
        transmission_id=f"ping:{emitter.platform_id}:{target_id}:{sim_time_s}",
        scenario_id=scenario_id,
        sim_time_s=sim_time_s,
        emitter_id=emitter.platform_id,
        target_id=target_id,
    )
    observations: list[MultistaticObservation] = []
    for receiver in sorted(receivers, key=lambda node: node.platform_id):
        receiver_leg = _distance(receiver.position_xy, target_xy)
        if receiver_leg > receiver.capability.active_receive_range_m:
            continue
        range_sigma = receiver.capability.active_range_sigma_m
        bearing_sigma = receiver.capability.active_bearing_sigma_rad
        confidence = _sample_detection_confidence(
            _range_fraction(receiver_leg, receiver.capability.active_receive_range_m),
            rng,
        )
        observations.append(
            MultistaticObservation(
                observation_id=(
                    f"active:{emitter.platform_id}:{receiver.platform_id}:"
                    f"{target_id}:{sim_time_s}"
                ),
                transmission_id=transmission.transmission_id,
                scenario_id=scenario_id,
                sim_time_s=sim_time_s,
                emitter_id=emitter.platform_id,
                receiver_id=receiver.platform_id,
                target_id=target_id,
                bistatic_range_m=max(
                    1e-6,
                    emitter_leg + receiver_leg + rng.gauss(0.0, range_sigma),
                ),
                receiver_azimuth_rad=_wrapped_noisy_bearing(
                    receiver.position_xy,
                    target_xy,
                    bearing_sigma,
                    rng,
                ),
                range_variance_m2=range_sigma**2,
                bearing_variance_rad2=bearing_sigma**2,
                detection_confidence=confidence,
            )
        )
    return transmission, tuple(observations)
