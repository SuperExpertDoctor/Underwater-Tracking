"""Target-owned local platform sensing.

Private world coordinates enter this module only to decide whether a platform
is inside the target's sensor boundary. The result contains noisy local
estimates and never carries the private coordinates onward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from math import atan2, isfinite, pi, sqrt
import random
from collections.abc import Sequence
from typing import Literal

from underwater_tracking.domain.adversary_models import (
    AdversaryTrigger,
    LocalPlatformDetection,
    TargetLocalContact,
    ThreatLevel,
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
    depth_m: float = 0.0

    def __post_init__(self) -> None:
        if not self.platform_id:
            raise ValueError("exposed platform id must be non-empty")
        if self.platform_kind not in ("carrier", "mother_ship", "uuv"):
            raise ValueError("target platform kind must be carrier, mother_ship, or uuv")
        if self.sensor_mode not in ("active", "passive"):
            raise ValueError("target sensor mode must be active or passive")
        if len(self.position_xy) != 2 or not all(isfinite(value) for value in self.position_xy):
            raise ValueError("exposed platform position must be finite")
        if self.depth_m < 0.0 or not isfinite(self.depth_m):
            raise ValueError("exposed platform depth must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class TargetLocalSensingResult:
    """One target observation-boundary result."""

    detections: tuple[LocalPlatformDetection, ...]
    acquired_platform_ids: frozenset[str]
    lost_platform_ids: frozenset[str]
    audible_active_emitter_ids: frozenset[str]


@dataclass(slots=True)
class TargetContactMemory:
    """Episode memory built exclusively from target-local detections."""

    target_id: str
    ttl_s: int = 120
    _contacts: dict[str, TargetLocalContact] = field(default_factory=dict, init=False)
    _range_buckets: dict[str, int] = field(default_factory=dict, init=False)
    _emitter_ids: frozenset[str] = field(default_factory=frozenset, init=False)

    def update(
        self,
        result: TargetLocalSensingResult,
        sim_time_s: int,
    ) -> tuple[AdversaryTrigger, ...]:
        if sim_time_s < 0:
            raise ValueError("sim_time_s must be non-negative")
        by_id = {detection.platform_id: detection for detection in result.detections}
        previous_active = {
            contact_id
            for contact_id, contact in self._contacts.items()
            if contact.status == "active"
        }
        triggers: list[AdversaryTrigger] = []
        for platform_id, detection in sorted(by_id.items()):
            threat = _threat_level(detection)
            previous = self._contacts.get(platform_id)
            reacquired = previous is not None and previous.status == "lost"
            if previous is None or reacquired:
                self._contacts[platform_id] = TargetLocalContact(
                    platform_id=platform_id,
                    platform_kind=detection.platform_kind,
                    first_seen_s=sim_time_s,
                    last_seen_s=sim_time_s,
                    estimated_range_m=detection.estimated_range_m,
                    relative_bearing_rad=detection.relative_bearing_rad,
                    threat_level=threat,
                    status="active",
                )
                triggers.append(
                    _trigger(
                        self.target_id,
                        platform_id,
                        sim_time_s,
                        "target_detection_acquired",
                        "local contact acquired",
                    )
                )
            else:
                previous_threat = previous.threat_level
                current_bucket = int(detection.estimated_range_m // 250.0)
                previous_bucket = self._range_buckets.get(
                    platform_id,
                    int(previous.estimated_range_m // 250.0),
                )
                if current_bucket != previous_bucket:
                    triggers.append(
                        _trigger(
                            self.target_id,
                            platform_id,
                            sim_time_s,
                            "target_contact_range_changed",
                            "local contact range bucket changed",
                        )
                    )
                if threat != previous_threat:
                    triggers.append(
                        _trigger(
                            self.target_id,
                            platform_id,
                            sim_time_s,
                            "target_contact_threat_changed",
                            "local contact threat level changed",
                        )
                    )
                self._contacts[platform_id] = previous.model_copy(
                    update={
                        "last_seen_s": sim_time_s,
                        "estimated_range_m": detection.estimated_range_m,
                        "relative_bearing_rad": detection.relative_bearing_rad,
                        "threat_level": threat,
                        "status": "active",
                    }
                )
            self._range_buckets[platform_id] = int(detection.estimated_range_m // 250.0)

        for platform_id in sorted(previous_active - set(by_id)):
            previous = self._contacts[platform_id]
            self._contacts[platform_id] = previous.model_copy(
                update={"last_seen_s": sim_time_s, "status": "lost"}
            )
            triggers.append(
                _trigger(
                    self.target_id,
                    platform_id,
                    sim_time_s,
                    "target_detection_lost",
                    "local contact lost",
                )
            )

        new_emitters = result.audible_active_emitter_ids - self._emitter_ids
        for platform_id in sorted(new_emitters):
            triggers.append(
                _trigger(
                    self.target_id,
                    platform_id,
                    sim_time_s,
                    "target_active_emitter_acquired",
                    "new active emitter is audible locally",
                )
            )
        self._emitter_ids = result.audible_active_emitter_ids
        self._expire(sim_time_s)
        return tuple(triggers)

    def active(self, sim_time_s: int) -> tuple[TargetLocalContact, ...]:
        self._expire(sim_time_s)
        return tuple(
            contact
            for _, contact in sorted(self._contacts.items())
            if contact.status == "active"
        )

    def context(self, sim_time_s: int) -> tuple[TargetLocalContact, ...]:
        self._expire(sim_time_s)
        return tuple(contact for _, contact in sorted(self._contacts.items()))

    def _expire(self, sim_time_s: int) -> None:
        expired = tuple(
            platform_id
            for platform_id, contact in self._contacts.items()
            if contact.status == "lost" and sim_time_s - contact.last_seen_s > self.ttl_s
        )
        for platform_id in expired:
            self._contacts.pop(platform_id, None)
            self._range_buckets.pop(platform_id, None)


def _threat_level(detection: LocalPlatformDetection) -> ThreatLevel:
    risk = detection.confidence + (0.2 if detection.sensor_mode == "active" else 0.0)
    if detection.relay_available and risk >= 0.9:
        return "critical"
    if risk >= 0.75:
        return "high"
    if risk >= 0.4:
        return "medium"
    return "low"


def _trigger(
    target_id: str,
    platform_id: str,
    sim_time_s: int,
    event_type: str,
    summary: str,
) -> AdversaryTrigger:
    return AdversaryTrigger(
        trigger_id=f"{event_type}:{target_id}:{platform_id}:{sim_time_s}",
        event_type=event_type,
        sim_time_s=sim_time_s,
        severity="tactical",
        summary=summary,
    )


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
    target_depth_m: float = 0.0,
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
    if target_depth_m < 0.0 or not isfinite(target_depth_m):
        raise ValueError("target depth must be finite and non-negative")

    candidate_by_id: dict[str, ExposedPlatform] = {}
    for candidate in candidates:
        if candidate.platform_id in candidate_by_id:
            raise ValueError(f"duplicate exposed platform {candidate.platform_id!r}")
        candidate_by_id[candidate.platform_id] = candidate

    release_range_m = detection_range_m + release_margin_m
    retained_ids: set[str] = set()
    distances: dict[str, float] = {}
    for platform_id, candidate in sorted(candidate_by_id.items()):
        distance_m = sqrt(
            (candidate.position_xy[0] - target_position_xy[0]) ** 2
            + (candidate.position_xy[1] - target_position_xy[1]) ** 2
            + (candidate.depth_m - target_depth_m) ** 2
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
    "TargetContactMemory",
    "TargetLocalSensingResult",
    "update_local_platform_detections",
]
