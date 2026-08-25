"""Target-owned local platform sensing.

Private world coordinates enter this module only to decide whether a platform
is inside the target's sensor boundary. The result contains noisy local
estimates and never carries the private coordinates onward.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
from itertools import pairwise
from math import atan2, hypot, isfinite, pi, sqrt
import random
from typing import Literal

from underwater_tracking.domain.adversary_models import (
    AdversaryTrigger,
    LocalPlatformDetection,
    TargetLocalContact,
    ThreatLevel,
    TargetPlatformKind,
    UUVTrackingPattern,
    UUVTrackingPatternType,
    UUVTrajectoryPoint,
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
    uuv_status: Literal["active", "unavailable", "track", "scan"] | None = None

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
        if self.platform_kind != "uuv" and self.uuv_status is not None:
            raise ValueError("only exposed UUVs can carry an operational status")


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
    material = f"{seed}:{target_id}:{platform_id}:{sim_time_s}".encode()
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
                uuv_status=candidate.uuv_status,
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


def extract_uuv_tracking_patterns(
    trajectory_cache: Mapping[
        str,
        Sequence[UUVTrajectoryPoint | tuple[int, tuple[float, float]]],
    ],
    *,
    target_position_xy: tuple[float, float],
    target_velocity_xy: tuple[float, float],
) -> tuple[UUVTrackingPattern, ...]:
    """Infer target-side tracking semantics from observed UUV trajectories."""

    normalized: dict[str, tuple[tuple[int, tuple[float, float]], ...]] = {}
    for uuv_id, raw_points in sorted(trajectory_cache.items()):
        points: list[tuple[int, tuple[float, float]]] = []
        for raw in raw_points:
            if isinstance(raw, UUVTrajectoryPoint):
                points.append((raw.observed_at_s, raw.estimated_position_xy))
            else:
                observed_at_s, position_xy = raw
                points.append(
                    (
                        int(observed_at_s),
                        (float(position_xy[0]), float(position_xy[1])),
                    )
                )
        if points:
            normalized[uuv_id] = tuple(sorted(points))

    patterns: list[UUVTrackingPattern] = []
    distance_change_by_id: dict[str, float] = {}
    last_position_by_id: dict[str, tuple[float, float]] = {}
    target_speed = hypot(*target_velocity_xy)
    direction = (
        (target_velocity_xy[0] / target_speed, target_velocity_xy[1] / target_speed)
        if target_speed > 1e-9
        else (1.0, 0.0)
    )

    def append_pattern(
        pattern_type: UUVTrackingPatternType,
        uuv_ids: tuple[str, ...],
        confidence: float,
        summary: str,
    ) -> None:
        relevant = [normalized[uuv_id] for uuv_id in uuv_ids]
        patterns.append(
            UUVTrackingPattern(
                pattern_type=pattern_type,
                uuv_ids=uuv_ids,
                first_observed_s=min(points[0][0] for points in relevant),
                last_observed_s=max(points[-1][0] for points in relevant),
                confidence=max(0.0, min(1.0, confidence)),
                semantic_summary=summary,
            )
        )

    for uuv_id, points in normalized.items():
        if len(points) < 2:
            continue
        distances = tuple(
            hypot(position[0] - target_position_xy[0], position[1] - target_position_xy[1])
            for _, position in points
        )
        change = distances[-1] - distances[0]
        distance_change_by_id[uuv_id] = change
        last_position_by_id[uuv_id] = points[-1][1]
        material_change = max(200.0, distances[0] * 0.15)
        if change <= -material_change:
            append_pattern(
                "tracking_approach",
                (uuv_id,),
                min(0.95, 0.55 + abs(change) / max(distances[0], 1.0) * 0.4),
                "UUV range is persistently decreasing, suggesting tracking acquisition.",
            )
        elif change >= material_change:
            append_pattern(
                "tracking_disengagement",
                (uuv_id,),
                min(0.95, 0.55 + change / max(distances[0], 1.0) * 0.3),
                "UUV range is persistently increasing and motion correlation is weakening.",
            )

        gaps = tuple(
            later[0] - earlier[0] for earlier, later in pairwise(points)
            if later[0] > earlier[0]
        )
        if gaps and max(gaps) >= max(90, min(gaps) * 3):
            append_pattern(
                "tracking_reacquisition",
                (uuv_id,),
                0.78,
                "UUV observations resume after a material detection gap.",
            )

        distance_deltas = tuple(
            later - earlier for earlier, later in pairwise(distances)
        )
        signs = tuple(1 if delta > 80.0 else -1 if delta < -80.0 else 0 for delta in distance_deltas)
        reversals = sum(
            1 for earlier, later in pairwise(signs)
            if earlier and later and earlier != later
        )
        if reversals >= 2:
            append_pattern(
                "intermittent_tracking",
                (uuv_id,),
                0.72,
                "UUV repeatedly closes and opens range, indicating intermittent tracking.",
            )

        start_time, start = points[-2]
        end_time, end = points[-1]
        dt_s = max(1, end_time - start_time)
        velocity = ((end[0] - start[0]) / dt_s, (end[1] - start[1]) / dt_s)
        velocity_speed = hypot(*velocity)
        parallel = (
            (velocity[0] * direction[0] + velocity[1] * direction[1]) / velocity_speed
            if velocity_speed > 1e-9
            else 0.0
        )
        relative = (end[0] - target_position_xy[0], end[1] - target_position_xy[1])
        along_track = relative[0] * direction[0] + relative[1] * direction[1]
        cross_track = relative[0] * -direction[1] + relative[1] * direction[0]
        stable_range = abs(change) <= max(250.0, distances[0] * 0.2)
        if parallel >= 0.8 and stable_range and along_track < -250.0:
            append_pattern(
                "stable_trailing",
                (uuv_id,),
                0.8,
                "UUV remains behind the target with correlated heading and speed.",
            )
        elif parallel >= 0.8 and stable_range and abs(cross_track) >= 300.0:
            append_pattern(
                "accompanying_tracking",
                (uuv_id,),
                0.76,
                "UUV holds a lateral offset while moving with the target.",
            )
        if target_speed > 1e-9 and along_track > 250.0 and change < -100.0:
            append_pattern(
                "intercept_tracking",
                (uuv_id,),
                0.74,
                "UUV is moving toward a position ahead of the target trajectory.",
            )

    moving_ids = tuple(sorted(distance_change_by_id))
    if len(moving_ids) >= 2:
        approaching = tuple(
            uuv_id for uuv_id in moving_ids if distance_change_by_id[uuv_id] < -200.0
        )
        disengaging = tuple(
            uuv_id for uuv_id in moving_ids if distance_change_by_id[uuv_id] > 200.0
        )
        if approaching and disengaging:
            relay_ids = tuple(sorted((approaching[0], disengaging[0])))
            append_pattern(
                "relay_tracking",
                relay_ids,
                0.82,
                "One UUV closes as another disengages, suggesting a tracking handoff.",
            )
        if len(approaching) >= 2:
            append_pattern(
                "multi_uuv_coordinated_tracking",
                tuple(sorted(approaching)),
                0.8,
                "Multiple UUVs concurrently close range from distinct positions.",
            )

        cross_by_id = {
            uuv_id: (
                (last_position_by_id[uuv_id][0] - target_position_xy[0]) * -direction[1]
                + (last_position_by_id[uuv_id][1] - target_position_xy[1]) * direction[0]
            )
            for uuv_id in moving_ids
        }
        left = tuple(uuv_id for uuv_id in moving_ids if cross_by_id[uuv_id] >= 300.0)
        right = tuple(uuv_id for uuv_id in moving_ids if cross_by_id[uuv_id] <= -300.0)
        if left and right:
            flank_ids = tuple(sorted((left[0], right[0])))
            append_pattern(
                "flank_envelope_tracking",
                flank_ids,
                0.84,
                "UUVs occupy opposite flanks around the target movement axis.",
            )

    return tuple(
        sorted(
            patterns,
            key=lambda item: (item.pattern_type, item.uuv_ids),
        )
    )


__all__ = [
    "ExposedPlatform",
    "TargetContactMemory",
    "TargetLocalSensingResult",
    "extract_uuv_tracking_patterns",
    "update_local_platform_detections",
]
