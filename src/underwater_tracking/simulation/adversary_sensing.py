"""Target-owned local platform sensing.

Private world coordinates enter this module only to decide whether a platform
is inside the target's sensor boundary. The result contains noisy local
estimates and never carries the private coordinates onward.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import atan2, hypot, isfinite, pi
import random
from collections.abc import Sequence
from typing import Literal

from underwater_tracking.domain.adversary_models import (
    LocalPlatformDetection,
    TargetPlatformKind,
)

TargetSensorMode = Literal["active", "passive"]


@dataclass(frozen=True, slots=True)
class ExposedPlatform:
    """Private candidate state consumed only by the target sensor boundary."""

    platform_id: str
    platform_kind: TargetPlatformKind
    position_xy: tuple[float, float]
    sensor_mode: TargetSensorMode
    relay_available: bool

    def __post_init__(self) -> None:
        if not self.platform_id:
            raise ValueError("exposed platform id must be non-empty")
        if self.platform_kind not in ("carrier", "mother_ship", "uuv"):
            raise ValueError("target platform kind must be carrier, mother_ship, or uuv")
        if self.sensor_mode not in ("active", "passive"):
            raise ValueError("target sensor mode must be active or passive")
        if len(self.position_xy) != 2 or not all(isfinite(value) for value in self.position_xy):
            raise ValueError("exposed platform position must be finite")


@dataclass(frozen=True, slots=True)
class TargetLocalSensingResult:
    """One target observation-boundary result."""

    detections: tuple[LocalPlatformDetection, ...]
    acquired_platform_ids: frozenset[str]
    lost_platform_ids: frozenset[str]
    audible_active_emitter_ids: frozenset[str]


def _stable_seed(seed: int, target_id: str, platform_id: str, sim_time_s: int) -> int:
    material = f"{seed}:{target_id}:{platform_id}:{sim_time_s}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big")


def _wrap_angle(angle_rad: float) -> float:
    return (angle_rad + pi) % (2.0 * pi) - pi


def update_local_platform_detections(
    *,
    target_id: str,
    target_position_xy: tuple[float, float],
    target_heading_rad: float,
    detection_range_m: float,
    release_margin_m: float,
    candidates: Sequence[ExposedPlatform],
    previous_ids: frozenset[str],
    sim_time_s: int,
    seed: int,
) -> TargetLocalSensingResult:
    """Update the target's hysteretic local platform whitelist.

    Acquisition uses ``detection_range_m`` while release uses the larger
    hysteresis boundary. Active-emitter audibility always uses acquisition
    range, even when a retained detection is outside that strict boundary.
    """
    if not target_id:
        raise ValueError("target_id must be non-empty")
    if detection_range_m <= 0.0 or not isfinite(detection_range_m):
        raise ValueError("detection_range_m must be finite and positive")
    if release_margin_m < 0.0 or not isfinite(release_margin_m):
        raise ValueError("release_margin_m must be finite and non-negative")
    if len(target_position_xy) != 2 or not all(isfinite(value) for value in target_position_xy):
        raise ValueError("target position must be finite")
    if not isfinite(target_heading_rad) or sim_time_s < 0:
        raise ValueError("target heading and observation time must be valid")

    candidate_by_id: dict[str, ExposedPlatform] = {}
    for candidate in candidates:
        if candidate.platform_id in candidate_by_id:
            raise ValueError(f"duplicate exposed platform {candidate.platform_id!r}")
        candidate_by_id[candidate.platform_id] = candidate

    release_range_m = detection_range_m + release_margin_m
    retained_ids: set[str] = set()
    distances: dict[str, float] = {}
    for platform_id, candidate in sorted(candidate_by_id.items()):
        distance_m = hypot(
            candidate.position_xy[0] - target_position_xy[0],
            candidate.position_xy[1] - target_position_xy[1],
        )
        distances[platform_id] = distance_m
        if distance_m <= detection_range_m or (
            platform_id in previous_ids and distance_m <= release_range_m
        ):
            retained_ids.add(platform_id)

    acquired = frozenset(retained_ids - set(previous_ids))
    lost = frozenset(set(previous_ids) - retained_ids)
    detections: list[LocalPlatformDetection] = []
    for platform_id in sorted(retained_ids):
        candidate = candidate_by_id[platform_id]
        distance_m = distances[platform_id]
        rng = random.Random(_stable_seed(seed, target_id, platform_id, sim_time_s))
        range_sigma_m = max(2.0, detection_range_m * 0.01)
        bearing_sigma_rad = 0.01
        estimated_range_m = max(0.0, distance_m + rng.gauss(0.0, range_sigma_m))
        relative_bearing_rad = _wrap_angle(
            atan2(
                candidate.position_xy[1] - target_position_xy[1],
                candidate.position_xy[0] - target_position_xy[0],
            )
            - target_heading_rad
            + rng.gauss(0.0, bearing_sigma_rad)
        )
        confidence = max(
            0.0,
            min(1.0, 0.98 - 0.35 * min(distance_m / detection_range_m, 1.0)),
        )
        detections.append(
            LocalPlatformDetection(
                platform_id=platform_id,
                platform_kind=candidate.platform_kind,
                observed_at_s=sim_time_s,
                estimated_range_m=estimated_range_m,
                relative_bearing_rad=relative_bearing_rad,
                confidence=confidence,
                sensor_mode=candidate.sensor_mode,
                relay_available=candidate.relay_available,
            )
        )

    audible_active_emitters = frozenset(
        platform_id
        for platform_id, candidate in sorted(candidate_by_id.items())
        if candidate.sensor_mode == "active" and distances[platform_id] <= detection_range_m
    )
    return TargetLocalSensingResult(
        detections=tuple(detections),
        acquired_platform_ids=acquired,
        lost_platform_ids=lost,
        audible_active_emitter_ids=audible_active_emitters,
    )


__all__ = [
    "ExposedPlatform",
    "TargetLocalSensingResult",
    "update_local_platform_detections",
]
