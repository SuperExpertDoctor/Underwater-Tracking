"""Pure metrics for deterministic UUV tracking and coverage audit traces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
from itertools import pairwise
import json
from math import hypot, isfinite
from numbers import Real
from typing import cast

import numpy as np

from underwater_tracking.planning.coverage import coverage_gap_area_m2
from underwater_tracking.verification.live_demo import validate_uuv_only_frame

Point = tuple[float, float]

_RUNTIME_POLICY = {
    "region_count": 4,
    "task_group_size": 3,
    "task_region_side_m": 2_000.0,
    "target_detection_radius_m": 1_000.0,
    "uuv_active_detection_radius_m": 600.0,
    "uuv_passive_detection_radius_m": 600.0,
    "region_entry_probability_threshold": 0.70,
    "region_transition_confirm_cycles": 2,
    "max_uuv_mileage_m": 50_000.0,
    "dedicated_release_remaining_mileage_m": 7_000.0,
}
_RUNTIME_LIFECYCLE_SENSOR_MODES = {
    "entering": "active",
    "active_scan": "active",
    "passive_track": "passive",
    "dedicated_track": "passive",
    "dedicated_release_pending": "passive",
    "exiting": "passive",
    "disappeared": "off",
}


def deterministic_trace_digest(trace: Mapping[str, object]) -> str:
    """Return a SHA-256 digest of the trace's canonical JSON representation."""
    payload = json.dumps(
        trace,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _point(value: object) -> Point | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    if len(value) < 2:
        return None
    try:
        point = float(value[0]), float(value[1])
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(isfinite(coordinate) for coordinate in point):
        raise ValueError("point coordinates must be finite")
    return point


def _required_points(values: object, *, field: str) -> tuple[Point, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError(f"{field} must be a sequence of points")
    result: list[Point] = []
    for value in values:
        point = _point(value)
        if point is None:
            raise ValueError(f"{field} must contain two-coordinate numeric points")
        result.append(point)
    return tuple(result)


def _points_by_id(items: object, *, id_field: str) -> dict[str, Point]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return {}
    result: dict[str, Point] = {}
    for raw in items:
        if not isinstance(raw, Mapping):
            continue
        identifier = raw.get(id_field)
        point = _point(raw.get("position_xy"))
        if isinstance(identifier, str) and point is not None:
            result[identifier] = point
    return result


def _deployed_points_by_id(items: object) -> dict[str, Point]:
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return {}
    return _points_by_id(
        tuple(
            raw
            for raw in items
            if isinstance(raw, Mapping)
            and raw.get("deployment_state") == "deployed"
        ),
        id_field="platform_id",
    )


def _finite_time_s(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    result = float(value)
    if not isfinite(result):
        raise ValueError("sim_time_s must be finite")
    return result


def target_position_errors_m(
    frames: Sequence[Mapping[str, object]],
    target_id: str,
) -> tuple[float, ...]:
    """Pair same-frame estimates and truth and return position errors in metres."""
    errors: list[float] = []
    seen_track_times: set[float] = set()
    for frame in frames:
        if not isinstance(frame, Mapping):
            continue
        frame_time = _finite_time_s(frame.get("sim_time_s"))
        if frame_time is None:
            continue
        truth = _points_by_id(frame.get("target_truth"), id_field="target_id").get(
            target_id
        )
        if truth is None:
            continue
        tracks = frame.get("tracks")
        if not isinstance(tracks, Sequence) or isinstance(tracks, (str, bytes)):
            continue
        for raw in tracks:
            if not isinstance(raw, Mapping) or raw.get("target_id") != target_id:
                continue
            track_time = _finite_time_s(raw.get("sim_time_s"))
            if (
                track_time is None
                or track_time != frame_time
                or track_time in seen_track_times
            ):
                continue
            estimate = _point(raw.get("mean"))
            if estimate is None:
                continue
            errors.append(hypot(estimate[0] - truth[0], estimate[1] - truth[1]))
            seen_track_times.add(track_time)
            break
    return tuple(errors)


def minimum_pairwise_separation_m(
    frames: Sequence[Mapping[str, object]],
) -> float | None:
    """Return the minimum same-frame separation between deployed UUVs."""
    minimum: float | None = None
    for frame in frames:
        if not isinstance(frame, Mapping):
            continue
        deployed = tuple(_deployed_points_by_id(frame.get("uuvs")).values())
        for index, left in enumerate(deployed):
            for right in deployed[index + 1 :]:
                distance = hypot(left[0] - right[0], left[1] - right[1])
                minimum = distance if minimum is None else min(minimum, distance)
    return minimum


def command_motion_counts(
    frames: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    """Count commanded UUV intervals and those with observable motion."""
    commanded = 0
    moved = 0
    for current, following in pairwise(frames):
        if not isinstance(current, Mapping) or not isinstance(following, Mapping):
            continue
        commands = current.get("waypoint_commands")
        if not isinstance(commands, Mapping):
            continue
        commanded_ids = {
            uuv_id
            for by_target in commands.values()
            if isinstance(by_target, Mapping)
            for uuv_id in by_target
            if isinstance(uuv_id, str)
        }
        before = _deployed_points_by_id(current.get("uuvs"))
        after = _deployed_points_by_id(following.get("uuvs"))
        for uuv_id in sorted(commanded_ids & before.keys() & after.keys()):
            commanded += 1
            if hypot(
                after[uuv_id][0] - before[uuv_id][0],
                after[uuv_id][1] - before[uuv_id][1],
            ) > 1.0e-9:
                moved += 1
    return {"commanded_intervals": commanded, "moved_intervals": moved}


def waypoint_visit_fraction(
    trajectory: Sequence[Point],
    route: Sequence[Point],
    *,
    numerical_tolerance_m: float = 1.0e-6,
) -> float | None:
    """Return the fraction of planned waypoints physically visited."""
    if not isfinite(numerical_tolerance_m) or numerical_tolerance_m < 0.0:
        raise ValueError("numerical_tolerance_m must be finite and non-negative")
    trajectory_points = _required_points(trajectory, field="trajectory")
    route_points = _required_points(route, field="route")
    if not route_points:
        return None
    visited = sum(
        any(
            hypot(sample[0] - point[0], sample[1] - point[1])
            <= numerical_tolerance_m
            for sample in trajectory_points
        )
        for point in route_points
    )
    return visited / len(route_points)


def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
    x, y = point
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    tolerance = 1.0e-9 * max(1.0, abs(dx), abs(dy))
    cross_product = (x - x1) * dy - (y - y1) * dx
    if abs(cross_product) > tolerance:
        return False
    return (
        min(x1, x2) - tolerance <= x <= max(x1, x2) + tolerance
        and min(y1, y2) - tolerance <= y <= max(y1, y2) + tolerance
    )


def _point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = point
    inside = False
    for start, end in zip(polygon, (*polygon[1:], polygon[0])):
        if _point_on_segment(point, start, end):
            return True
        x1, y1 = start
        x2, y2 = end
        if (y1 > y) != (y2 > y):
            crossing_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < crossing_x:
                inside = not inside
    return inside


def _polygon_area_twice(polygon: Sequence[Point]) -> float:
    return sum(
        start[0] * end[1] - end[0] * start[1]
        for start, end in zip(polygon, (*polygon[1:], polygon[0]))
    )


def sampled_footprint_fraction(
    polygon: Sequence[Point],
    emissions: Sequence[tuple[Point, float]],
    *,
    samples_per_axis: int = 81,
) -> float | None:
    """Estimate the actively insonified polygon fraction on a fixed grid."""
    if samples_per_axis < 2:
        raise ValueError("samples_per_axis must be at least two")
    polygon_points = _required_points(polygon, field="polygon")
    if not isinstance(emissions, Sequence) or isinstance(emissions, (str, bytes)):
        raise TypeError("emissions must be a sequence")
    validated_emissions: list[tuple[Point, float]] = []
    for emission in emissions:
        if not isinstance(emission, Sequence) or isinstance(emission, (str, bytes)):
            raise TypeError("each emission must contain a center and radius")
        if len(emission) < 2:
            raise ValueError("each emission must contain a center and radius")
        center = _point(emission[0])
        if center is None:
            raise ValueError("emission center must contain two numeric coordinates")
        try:
            radius = float(emission[1])
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("emission radius must be finite and positive") from error
        if not isfinite(radius) or radius <= 0.0:
            raise ValueError("emission radius must be finite and positive")
        validated_emissions.append((center, radius))
    if not validated_emissions:
        return None
    if len(polygon_points) < 3 or _polygon_area_twice(polygon_points) == 0.0:
        return None
    min_x = min(point[0] for point in polygon_points)
    max_x = max(point[0] for point in polygon_points)
    min_y = min(point[1] for point in polygon_points)
    max_y = max(point[1] for point in polygon_points)
    candidates = [
        (float(x), float(y))
        for x in np.linspace(min_x, max_x, samples_per_axis)
        for y in np.linspace(min_y, max_y, samples_per_axis)
        if _point_in_polygon((float(x), float(y)), polygon_points)
    ]
    if not candidates:
        return None
    covered = sum(
        any(
            hypot(point[0] - center[0], point[1] - center[1]) <= radius
            for center, radius in validated_emissions
        )
        for point in candidates
    )
    return covered / len(candidates)


def percentile_summary(values: Sequence[float]) -> dict[str, float] | None:
    """Summarize a non-empty finite metric series."""
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    if not bool(np.all(np.isfinite(array))):
        raise ValueError("values must be finite")
    return {
        "rmse": float(np.sqrt(np.mean(np.square(array)))),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def _runtime_point(value: object) -> Point | None:
    if isinstance(value, Mapping):
        x = value.get("x")
        y = value.get("y")
        if (
            isinstance(x, Real)
            and not isinstance(x, bool)
            and isinstance(y, Real)
            and not isinstance(y, bool)
        ):
            point = float(x), float(y)
            return point if all(isfinite(coordinate) for coordinate in point) else None
    return _point(value)


def _runtime_sequence(value: object) -> tuple[object, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    return tuple(value)


def _runtime_region_ids(target_id: str) -> tuple[str, ...]:
    return tuple(f"{target_id}:task:{index:02d}" for index in range(1, 5))


def _runtime_policy_value(
    policy: Mapping[str, object],
    field: str,
    expected: object,
    violations: list[str],
) -> object:
    value = policy.get(field)
    if value != expected:
        violations.append(f"policy_{field}_mismatch")
    return value


def _runtime_square_is_valid(
    region: Mapping[str, object],
    *,
    expected_side_m: float,
) -> bool:
    raw_geometry = region.get("geometry")
    if not isinstance(raw_geometry, Sequence) or isinstance(
        raw_geometry, (str, bytes)
    ):
        return False
    points = tuple(
        _runtime_point(value)
        for value in raw_geometry
    )
    if len(points) != 4 or any(point is None for point in points):
        return False
    corners = tuple(point for point in points if point is not None)
    if len(set(corners)) != 4:
        return False
    min_x = min(point[0] for point in corners)
    max_x = max(point[0] for point in corners)
    min_y = min(point[1] for point in corners)
    max_y = max(point[1] for point in corners)
    expected_corners = {
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
    }
    if set(corners) != expected_corners:
        return False
    tolerance = 1.0e-6
    if (
        abs(max_x - min_x - expected_side_m) > tolerance
        or abs(max_y - min_y - expected_side_m) > tolerance
    ):
        return False
    declared_side = region.get("side_length_m")
    if declared_side is None:
        return True
    return (
        isinstance(declared_side, Real)
        and not isinstance(declared_side, bool)
        and abs(float(declared_side) - expected_side_m) <= tolerance
    )


def _runtime_static_coverage_gap_m2(
    trace: Mapping[str, object],
    *,
    detection_radius_m: float,
    violations: list[str],
) -> float | None:
    raw_regions = trace.get("regions")
    raw_routes = trace.get("routes")
    if not isinstance(raw_regions, Mapping) or not isinstance(raw_routes, Mapping):
        violations.append("coverage_projection_missing")
        return None
    gaps: list[float] = []
    for region_id, raw_region in sorted(raw_regions.items()):
        if not isinstance(region_id, str) or not isinstance(raw_region, Mapping):
            violations.append("coverage_projection_invalid")
            continue
        polygon_values = raw_region.get("polygon")
        polygon_items = (
            polygon_values
            if isinstance(polygon_values, Sequence)
            and not isinstance(polygon_values, (str, bytes))
            else ()
        )
        polygon = tuple(
            point
            for value in polygon_items
            if (point := _runtime_point(value)) is not None
        )
        raw_by_uuv = raw_routes.get(region_id)
        by_uuv = (
            raw_by_uuv
            if isinstance(raw_by_uuv, Mapping)
            else {}
        )
        routes: dict[str, Sequence[Point]] = {}
        for uuv_id, raw_route in by_uuv.items():
            if (
                not isinstance(uuv_id, str)
                or not isinstance(raw_route, Sequence)
                or isinstance(raw_route, (str, bytes))
            ):
                violations.append(f"coverage_route_invalid:{region_id}")
                continue
            route = tuple(
                point
                for value in raw_route
                if (point := _runtime_point(value)) is not None
            )
            if len(route) != len(raw_route):
                violations.append(f"coverage_route_invalid:{region_id}:{uuv_id}")
                continue
            routes[uuv_id] = route
        if len(polygon) != len(polygon_items) or not routes:
            violations.append(f"coverage_projection_invalid:{region_id}")
            continue
        try:
            gap = coverage_gap_area_m2(polygon, routes, detection_radius_m)
        except (TypeError, ValueError):
            violations.append(f"coverage_projection_invalid:{region_id}")
            continue
        gaps.append(float(gap))
        if gap > 1.0e-6:
            violations.append(f"coverage_path_incomplete:{region_id}")
    return max(gaps, default=None)


def audit_runtime_execution_trace(
    trace: Mapping[str, object],
) -> dict[str, object]:
    """Audit the authoritative published execution projection in a trace.

    The runner stores evaluation truth beside, rather than inside, the published
    frame. This function intentionally reads only ``operational_frame`` and its
    recorded transport hash for runtime acceptance metrics.
    """
    raw_frames = trace.get("frames")
    frames = (
        tuple(raw_frames)
        if isinstance(raw_frames, Sequence)
        and not isinstance(raw_frames, (str, bytes))
        else ()
    )
    runtime_entries = tuple(
        (
            cast(Mapping[str, object], frame),
            frame.get("operational_frame"),
        )
        for frame in frames
        if isinstance(frame, Mapping)
        and isinstance(frame.get("operational_frame"), Mapping)
    )
    runtime_frames = tuple(entry[1] for entry in runtime_entries)
    if not runtime_frames:
        return {
            "available": False,
            "checked": False,
            "valid": None,
            "frame_count": 0,
            "violations": [],
            "region_side_m": None,
            "target_detection_radius_m": None,
            "uuv_detection_radius_m": None,
            "task_group_size": None,
            "max_coverage_gap_area_m2": None,
            "active_ping_count_during_passive": None,
            "tracking_owner_gap_frames": None,
            "max_visible_uuv_count": None,
            "runtime_scan_regions_with_pings": None,
            "max_runtime_scan_coverage": None,
            "runtime_entry_confirmation_count": None,
            "execution_revision_monotonic": None,
            "execution_revision_advanced": None,
            "max_runtime_group_count": None,
            "runtime_replacement_pair_count": None,
            "replacement_contract_valid": None,
            "active_ping_count": None,
            "active_echo_count": None,
            "active_ping_cadence_valid": None,
            "active_echo_source_valid": None,
            "entry_evidence_public": None,
            "entry_confirmation_progression_valid": None,
            "tracking_owner_observed": None,
            "max_tracking_owner_count": None,
            "handoff_atomic_valid": None,
            "canonical_transport_valid": None,
            "execution_frame_contract_valid": None,
        }

    violations: list[str] = []
    verification = trace.get("verification_evidence")
    public_observation_ids = {
        value
        for value in (
            _runtime_sequence(verification.get("public_observation_ids"))
            if isinstance(verification, Mapping)
            else ()
        )
        or ()
        if isinstance(value, str) and value
    }
    policy_values: dict[str, object] = {}
    active_ping_count_during_passive = 0
    tracking_owner_gap_frames = 0
    max_visible_uuv_count = 0
    max_runtime_group_count = 0
    max_tracking_owner_count = 0
    boundary_check_count = 0
    boundary_violation_count = 0
    ping_region_by_event_id: dict[str, str] = {}
    ping_times_by_source: dict[tuple[str, str], list[float]] = {}
    active_ping_event_ids: set[str] = set()
    active_echo_event_ids: set[str] = set()
    active_echo_source_ids: set[str] = set()
    scan_regions_with_pings: set[str] = set()
    expected_scan_region_ids: set[str] = set()
    scan_state_by_key: dict[
        tuple[str, object, object],
        tuple[float, float, int, int],
    ] = {}
    geometry_revision_by_region: dict[str, int] = {}
    execution_revisions: list[int] = []
    replacement_pairs: set[tuple[str, str, str]] = set()
    entry_cycles_by_region: dict[str, dict[int, int]] = {}
    max_runtime_scan_coverage = 0.0
    max_entry_confirmation_count = 0
    initial_valid_until_s: float | None = None
    final_sim_time_s: float | None = None
    entry_evidence_public = True
    entry_confirmation_progression_valid = True
    replacement_contract_valid = True
    handoff_atomic_valid = True
    canonical_transport_valid = True
    execution_frame_contract_valid = True
    tracking_owner_observed = False
    previous_owner_id: str | None = None
    previous_tracking_control: Mapping[str, object] | None = None
    for index, (raw_trace_frame, raw_frame) in enumerate(runtime_entries):
        frame = cast(Mapping[str, object], raw_frame)
        frame_violations = validate_uuv_only_frame(frame)
        if frame_violations:
            execution_frame_contract_valid = False
        violations.extend(f"frame_{index}:{violation}" for violation in frame_violations)
        try:
            canonical_frame = json.dumps(
                frame,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError):
            violations.append(f"frame_{index}:transport_payload_invalid")
            canonical_frame = ""
        expected_hash = sha256(canonical_frame.encode("utf-8")).hexdigest()
        if raw_trace_frame.get("transport_hash") != expected_hash:
            canonical_transport_valid = False
            violations.append(f"frame_{index}:transport_hash_mismatch")

        execution = frame.get("execution")
        if not isinstance(execution, Mapping):
            execution_frame_contract_valid = False
            violations.append(f"frame_{index}:execution_missing")
            continue
        execution_revision = execution.get("execution_revision")
        if (
            not isinstance(execution_revision, int)
            or isinstance(execution_revision, bool)
            or execution_revision < 1
        ):
            violations.append(f"frame_{index}:execution_revision_invalid")
        else:
            if execution_revisions:
                previous_revision = execution_revisions[-1]
                if execution_revision < previous_revision:
                    violations.append(
                        f"frame_{index}:execution_revision_nonmonotonic:"
                        f"{previous_revision}->{execution_revision}"
                    )
                elif execution_revision > previous_revision + 1:
                    violations.append(
                        f"frame_{index}:execution_revision_gap:"
                        f"{previous_revision}->{execution_revision}"
                    )
            execution_revisions.append(execution_revision)
        valid_until_s = execution.get("valid_until_s")
        if (
            initial_valid_until_s is None
            and isinstance(valid_until_s, Real)
            and not isinstance(valid_until_s, bool)
            and isfinite(float(valid_until_s))
        ):
            initial_valid_until_s = float(valid_until_s)
        policy = execution.get("tracking_policy")
        if not isinstance(policy, Mapping):
            violations.append(f"frame_{index}:tracking_policy_missing")
            policy = {}
        for field, expected in _RUNTIME_POLICY.items():
            value = _runtime_policy_value(policy, field, expected, violations)
            if field not in policy_values:
                policy_values[field] = value
            elif policy_values[field] != value:
                violations.append(f"frame_{index}:policy_{field}_changed")

        target_id = execution.get("target_id")
        expected_region_ids = (
            _runtime_region_ids(target_id) if isinstance(target_id, str) else ()
        )
        expected_scan_region_ids.update(expected_region_ids)
        regions = tuple(
            value
            for value in execution.get("regions", ())
            if isinstance(value, Mapping)
        )
        groups = tuple(
            value
            for value in execution.get("task_groups", ())
            if isinstance(value, Mapping)
        )
        max_runtime_group_count = max(max_runtime_group_count, len(groups))
        region_by_id = {
            region.get("region_id"): region
            for region in regions
            if isinstance(region.get("region_id"), str)
        }
        group_by_id = {
            group.get("group_instance_id"): group
            for group in groups
            if isinstance(group.get("group_instance_id"), str)
        }
        tracking_control = execution.get("tracking_control")
        tracking_mode = (
            tracking_control.get("mode")
            if isinstance(tracking_control, Mapping)
            else None
        )
        owner_id = (
            tracking_control.get("tracking_owner_group_id")
            if isinstance(tracking_control, Mapping)
            else None
        )
        owner = group_by_id.get(owner_id) if isinstance(owner_id, str) else None
        passive_groups = tuple(
            group
            for group in groups
            if group.get("lifecycle") in {
                "passive_track",
                "dedicated_track",
                "dedicated_release_pending",
            }
        )
        owner_count = sum(
            group.get("ownership_status") == "owner" for group in groups
        )
        max_tracking_owner_count = max(max_tracking_owner_count, owner_count)
        if owner_count > 1:
            violations.append(f"frame_{index}:tracking_owner_not_unique")
        owner_is_valid = (
            isinstance(owner_id, str)
            and owner is not None
            and owner.get("ownership_status") == "owner"
            and len(_runtime_sequence(owner.get("member_uuv_ids")) or ()) == 3
        )
        if (tracking_mode == "dedicated" or passive_groups) and not owner_is_valid:
            tracking_owner_gap_frames += 1
        if owner_is_valid:
            tracking_owner_observed = True
        if (
            previous_owner_id is not None
            and isinstance(owner_id, str)
            and owner_id != previous_owner_id
        ):
            previous_pending_id = (
                previous_tracking_control.get("pending_successor_group_id")
                if previous_tracking_control is not None
                else None
            )
            readiness_fields = (
                "successor_required_uuv_ids",
                "successor_deployed_uuv_ids",
                "successor_healthy_uuv_ids",
                "successor_passive_uuv_ids",
                "successor_observing_uuv_ids",
            )
            readiness_sets = tuple(
                set(_runtime_sequence(previous_tracking_control.get(field)) or ())
                if previous_tracking_control is not None
                else set()
                for field in readiness_fields
            )
            if (
                previous_pending_id != owner_id
                or not readiness_sets
                or len(readiness_sets[0]) != 3
                or any(values != readiness_sets[0] for values in readiness_sets[1:])
            ):
                handoff_atomic_valid = False
                violations.append(f"frame_{index}:tracking_handoff_not_atomic")
        if isinstance(owner_id, str):
            previous_owner_id = owner_id
        previous_tracking_control = (
            tracking_control if isinstance(tracking_control, Mapping) else None
        )

        linked_group_ids: set[str] = set()
        for region_id in expected_region_ids:
            region = region_by_id.get(region_id)
            if region is None:
                violations.append(f"frame_{index}:region_missing:{region_id}")
                continue
            if not _runtime_square_is_valid(
                region,
                expected_side_m=float(_RUNTIME_POLICY["task_region_side_m"]),
            ):
                violations.append(f"frame_{index}:region_not_fixed_square:{region_id}")
            group_id = region.get("task_group_id")
            if isinstance(group_id, str):
                linked_group_ids.add(group_id)
            elif tracking_mode != "dedicated":
                violations.append(f"frame_{index}:region_group_link_missing:{region_id}")

        member_to_group: dict[str, Mapping[str, object]] = {}
        for group in groups:
            group_id = group.get("group_instance_id")
            members = _runtime_sequence(group.get("member_uuv_ids"))
            lifecycle = group.get("lifecycle")
            sensor_mode = group.get("sensor_mode")
            if not isinstance(group_id, str) or not group_id:
                violations.append(f"frame_{index}:group_id_missing")
            if (
                members is None
                or len(members) != 3
                or any(not isinstance(member, str) or not member for member in members)
                or len(set(members)) != 3
            ):
                violations.append(f"frame_{index}:group_member_cardinality_invalid")
            expected_sensor_mode = (
                _RUNTIME_LIFECYCLE_SENSOR_MODES.get(lifecycle)
                if isinstance(lifecycle, str)
                else None
            )
            if expected_sensor_mode != sensor_mode:
                violations.append(f"frame_{index}:group_sensor_lifecycle_mismatch")
            for point_field in ("entry_boundary_point", "exit_boundary_point"):
                if group.get(point_field) is not None:
                    boundary_check_count += 1
                    if _runtime_point(group.get(point_field)) is None:
                        boundary_violation_count += 1
                        violations.append(f"frame_{index}:group_{point_field}_invalid")
            if isinstance(group_id, str) and members is not None:
                for member in members:
                    if isinstance(member, str):
                        if member in member_to_group:
                            violations.append(f"frame_{index}:member_group_duplicate:{member}")
                        member_to_group[member] = group

        runtime_uuvs = tuple(
            value
            for value in (_runtime_sequence(frame.get("uuvs")) or ())
            if isinstance(value, Mapping)
        )
        visible_group_ids = {
            group_id
            for group_id, group in group_by_id.items()
            if group.get("lifecycle") != "disappeared"
        }
        visible_count = sum(
            item.get("physically_exposed") is True
            and item.get("group_instance_id") in visible_group_ids
            for item in runtime_uuvs
        )
        max_visible_uuv_count = max(max_visible_uuv_count, visible_count)
        if len(groups) > 8:
            replacement_contract_valid = False
            violations.append(f"frame_{index}:runtime_group_limit_exceeded")
        if visible_count > 24:
            replacement_contract_valid = False
            violations.append(f"frame_{index}:visible_uuv_limit_exceeded")
        for item in runtime_uuvs:
            uuv_id = item.get("uuv_id")
            group_id = item.get("group_instance_id")
            if item.get("physically_exposed") is True and group_id not in group_by_id:
                violations.append(f"frame_{index}:uuv_group_missing:{uuv_id}")
            current_group = (
                group_by_id.get(group_id) if isinstance(group_id, str) else None
            )
            if current_group is None:
                continue
            if item.get("group_lifecycle") != current_group.get("lifecycle"):
                violations.append(f"frame_{index}:uuv_lifecycle_mismatch:{uuv_id}")
            if item.get("sensor_mode") != current_group.get("sensor_mode"):
                violations.append(f"frame_{index}:uuv_sensor_mode_mismatch:{uuv_id}")

        frame_time = frame.get("sim_time_s")
        if (
            isinstance(frame_time, Real)
            and not isinstance(frame_time, bool)
            and isfinite(float(frame_time))
        ):
            final_sim_time_s = float(frame_time)
        passive_member_ids = {
            member
            for group in passive_groups
            for member in (_runtime_sequence(group.get("member_uuv_ids")) or ())
            if isinstance(member, str)
        }
        raw_events = raw_trace_frame.get("events", frame.get("events", ()))
        for raw_event in raw_events if isinstance(raw_events, Sequence) else ():
            if not isinstance(raw_event, Mapping):
                continue
            event_type = raw_event.get("event_type")
            payload = raw_event.get("payload")
            event_id = raw_event.get("event_id")
            if event_type == "active_echo":
                if not isinstance(event_id, str) or not event_id:
                    violations.append(f"frame_{index}:active_echo_evidence_id_missing")
                    continue
                if event_id in active_echo_event_ids:
                    continue
                active_echo_event_ids.add(event_id)
                source_ping_event_id = (
                    payload.get("source_ping_event_id")
                    if isinstance(payload, Mapping)
                    else None
                )
                if not isinstance(source_ping_event_id, str) or not source_ping_event_id:
                    violations.append(f"frame_{index}:active_echo_source_missing:{event_id}")
                else:
                    active_echo_source_ids.add(source_ping_event_id)
                continue
            if event_type != "active_ping":
                continue
            event_time = raw_event.get("sim_time_s")
            emitter_id = payload.get("emitter_id") if isinstance(payload, Mapping) else None
            contact_id = payload.get("contact_id") if isinstance(payload, Mapping) else None
            emitter_group = member_to_group.get(emitter_id) if isinstance(emitter_id, str) else None
            emitter_region_id = (
                emitter_group.get("region_id")
                if emitter_group is not None
                else None
            )
            if not isinstance(event_id, str) or not event_id:
                violations.append(f"frame_{index}:active_ping_evidence_id_missing")
            elif event_id in active_ping_event_ids:
                pass
            elif not isinstance(emitter_region_id, str):
                violations.append(f"frame_{index}:active_ping_group_source_missing:{event_id}")
            else:
                active_ping_event_ids.add(event_id)
                previous_region_id = ping_region_by_event_id.setdefault(
                    event_id,
                    emitter_region_id,
                )
                if previous_region_id != emitter_region_id:
                    violations.append(f"frame_{index}:active_ping_region_changed:{event_id}")
                if (
                    isinstance(emitter_id, str)
                    and isinstance(contact_id, str)
                    and isinstance(event_time, Real)
                    and not isinstance(event_time, bool)
                    and isfinite(float(event_time))
                ):
                    ping_times_by_source.setdefault(
                        (emitter_id, contact_id), []
                    ).append(float(event_time))
            if isinstance(payload, Mapping):
                configured_range_m = payload.get("configured_range_m")
                if (
                    _runtime_point(payload.get("source_position_xy")) is None
                    or not isinstance(configured_range_m, Real)
                    or isinstance(configured_range_m, bool)
                    or not isfinite(float(configured_range_m))
                    or float(configured_range_m) <= 0.0
                ):
                    violations.append(
                        f"frame_{index}:active_ping_physical_source_missing"
                    )
            if passive_member_ids and event_time == frame_time and (
                emitter_id in passive_member_ids or not isinstance(emitter_id, str)
            ):
                active_ping_count_during_passive += 1

        replacements = _runtime_sequence(execution.get("replacements"))
        if replacements is None:
            replacement_contract_valid = False
            violations.append(f"frame_{index}:replacement_projection_invalid")
        else:
            for replacement in replacements:
                if not isinstance(replacement, Mapping):
                    replacement_contract_valid = False
                    violations.append(f"frame_{index}:replacement_projection_invalid")
                    continue
                replacement_region_id = replacement.get("region_id")
                outgoing_id = replacement.get("outgoing_group_id")
                incoming_id = replacement.get("incoming_group_id")
                outgoing = (
                    group_by_id.get(outgoing_id)
                    if isinstance(outgoing_id, str)
                    else None
                )
                incoming = (
                    group_by_id.get(incoming_id)
                    if isinstance(incoming_id, str)
                    else None
                )
                replacement_region = (
                    region_by_id.get(replacement_region_id)
                    if isinstance(replacement_region_id, str)
                    else None
                )
                source_revision = replacement.get("source_geometry_revision")
                target_revision = replacement.get("target_geometry_revision")
                valid_replacement = (
                    isinstance(replacement_region_id, str)
                    and isinstance(outgoing_id, str)
                    and isinstance(incoming_id, str)
                    and outgoing is not None
                    and incoming is not None
                    and replacement_region is not None
                    and incoming.get("source_group_instance_id") == outgoing_id
                    and outgoing.get("region_id") == replacement_region_id
                    and incoming.get("region_id") == replacement_region_id
                    and isinstance(source_revision, int)
                    and not isinstance(source_revision, bool)
                    and isinstance(target_revision, int)
                    and not isinstance(target_revision, bool)
                    and source_revision < target_revision
                    and replacement_region.get("geometry_revision") == target_revision
                )
                if not valid_replacement:
                    replacement_contract_valid = False
                    violations.append(
                        f"frame_{index}:replacement_pair_invalid:{replacement_region_id}"
                    )
                else:
                    replacement_pairs.add(
                        (
                            cast(str, replacement_region_id),
                            cast(str, outgoing_id),
                            cast(str, incoming_id),
                        )
                    )

        for region_id in expected_region_ids:
            region = region_by_id.get(region_id)
            if region is None:
                continue
            geometry_revision = region.get("geometry_revision")
            if (
                not isinstance(geometry_revision, int)
                or isinstance(geometry_revision, bool)
                or geometry_revision < 1
            ):
                violations.append(
                    f"frame_{index}:geometry_revision_invalid:{region_id}"
                )
            else:
                previous_geometry_revision = geometry_revision_by_region.get(region_id)
                if (
                    previous_geometry_revision is not None
                    and geometry_revision < previous_geometry_revision
                ):
                    violations.append(
                        f"frame_{index}:geometry_revision_nonmonotonic:{region_id}:"
                        f"{previous_geometry_revision}->{geometry_revision}"
                    )
                geometry_revision_by_region[region_id] = geometry_revision
            coverage = region.get("coverage")
            route_progress = region.get("route_progress")
            scan_round = region.get("scan_round")
            ping_count = region.get("ping_count")
            threshold = region.get("scan_completion_threshold")
            completed = region.get("scan_completed")
            evidence_ids = _runtime_sequence(region.get("scan_evidence_ids"))
            numeric_scan_state = (
                isinstance(coverage, Real)
                and not isinstance(coverage, bool)
                and isfinite(float(coverage))
                and 0.0 <= float(coverage) <= 1.0
                and isinstance(route_progress, Real)
                and not isinstance(route_progress, bool)
                and isfinite(float(route_progress))
                and 0.0 <= float(route_progress) <= 1.0
                and isinstance(scan_round, int)
                and not isinstance(scan_round, bool)
                and scan_round >= 0
                and isinstance(ping_count, int)
                and not isinstance(ping_count, bool)
                and ping_count >= 0
                and isinstance(threshold, Real)
                and not isinstance(threshold, bool)
                and isfinite(float(threshold))
                and 0.0 < float(threshold) <= 1.0
                and isinstance(completed, bool)
                and evidence_ids is not None
                and len(evidence_ids) == len(set(evidence_ids))
                and all(isinstance(value, str) and value for value in evidence_ids)
            )
            if not numeric_scan_state:
                violations.append(f"frame_{index}:runtime_scan_state_invalid:{region_id}")
            else:
                coverage_value = float(cast(Real, coverage))
                route_progress_value = float(cast(Real, route_progress))
                threshold_value = float(cast(Real, threshold))
                scan_round_value = cast(int, scan_round)
                ping_count_value = cast(int, ping_count)
                completed_value = cast(bool, completed)
                scan_evidence_ids = cast(tuple[object, ...], evidence_ids)
                max_runtime_scan_coverage = max(
                    max_runtime_scan_coverage,
                    coverage_value,
                )
                if completed_value != (coverage_value >= threshold_value):
                    violations.append(
                        f"frame_{index}:runtime_scan_completion_mismatch:{region_id}"
                    )
                if ping_count_value < len(scan_evidence_ids):
                    violations.append(
                        f"frame_{index}:runtime_scan_ping_count_mismatch:{region_id}"
                    )
                if ping_count_value > 0:
                    scan_regions_with_pings.add(region_id)
                for evidence_id in scan_evidence_ids:
                    if (
                        not isinstance(evidence_id, str)
                        or ping_region_by_event_id.get(evidence_id) != region_id
                    ):
                        violations.append(
                            f"frame_{index}:runtime_scan_evidence_unresolved:"
                            f"{region_id}:{evidence_id}"
                        )
                state_key = (
                    region_id,
                    region.get("geometry_revision"),
                    region.get("task_group_id"),
                )
                previous_state = scan_state_by_key.get(state_key)
                if previous_state is not None:
                    (
                        previous_coverage,
                        previous_route_progress,
                        previous_round,
                        previous_ping_count,
                    ) = previous_state
                    if scan_round_value < previous_round or (
                        scan_round_value == previous_round
                        and (
                            coverage_value < previous_coverage
                            or route_progress_value < previous_route_progress
                            or ping_count_value < previous_ping_count
                        )
                    ):
                        violations.append(
                            "frame_"
                            f"{index}:runtime_scan_state_nonmonotonic:{region_id}"
                        )
                scan_state_by_key[state_key] = (
                    coverage_value,
                    route_progress_value,
                    scan_round_value,
                    ping_count_value,
                )

            entry_probability = region.get("entry_probability")
            confirmations = region.get("entry_confirmations")
            confirmation_required = region.get("entry_confirmation_required")
            observation_cycle_s = region.get("entry_observation_cycle_s")
            reset_reason = region.get("entry_reset_reason")
            entry_evidence_ids = _runtime_sequence(region.get("entry_evidence_ids"))
            if (
                (
                    entry_probability is not None
                    and (
                        not isinstance(entry_probability, Real)
                        or isinstance(entry_probability, bool)
                        or not isfinite(float(entry_probability))
                        or not 0.0 <= float(entry_probability) <= 1.0
                    )
                )
                or not isinstance(confirmations, int)
                or isinstance(confirmations, bool)
                or confirmations < 0
                or not isinstance(confirmation_required, int)
                or isinstance(confirmation_required, bool)
                or confirmation_required < 1
                or (
                    observation_cycle_s is not None
                    and (
                        not isinstance(observation_cycle_s, int)
                        or isinstance(observation_cycle_s, bool)
                        or observation_cycle_s < 0
                        or (
                            isinstance(frame_time, int)
                            and observation_cycle_s > frame_time
                        )
                    )
                )
                or reset_reason
                not in {
                    None,
                    "missing_probability",
                    "non_finite_probability",
                    "below_threshold",
                    "simultaneous_exit",
                }
                or entry_evidence_ids is None
                or len(entry_evidence_ids) != len(set(entry_evidence_ids))
                or any(
                    not isinstance(evidence_id, str) or not evidence_id
                    for evidence_id in entry_evidence_ids
                )
            ):
                violations.append(f"frame_{index}:runtime_entry_state_invalid:{region_id}")
            elif confirmations > 0:
                if confirmations > confirmation_required:
                    violations.append(
                        f"frame_{index}:runtime_entry_confirmation_overflow:{region_id}"
                    )
                if not isinstance(observation_cycle_s, int):
                    violations.append(
                        f"frame_{index}:runtime_entry_cycle_missing:{region_id}"
                    )
                else:
                    max_entry_confirmation_count = max(
                        max_entry_confirmation_count,
                        confirmations,
                    )
                if not entry_evidence_ids:
                    violations.append(
                        f"frame_{index}:runtime_entry_evidence_missing:{region_id}"
                    )
                elif not set(entry_evidence_ids).issubset(public_observation_ids):
                    entry_evidence_public = False
                    violations.append(
                        f"frame_{index}:runtime_entry_evidence_not_public:{region_id}"
                    )
                if isinstance(observation_cycle_s, int):
                    level_cycles = entry_cycles_by_region.setdefault(region_id, {})
                    level_cycles.setdefault(
                        confirmations,
                        observation_cycle_s,
                    )

        if tracking_mode == "regional" and linked_group_ids and any(
            group_id not in linked_group_ids
            and group.get("lifecycle") not in {"exiting", "disappeared"}
            and group_id
            not in {
                candidate.get("source_group_instance_id")
                for candidate in groups
            }
            and group.get("source_group_instance_id") not in linked_group_ids
            for group_id, group in group_by_id.items()
        ):
            violations.append(f"frame_{index}:unlinked_live_group")

    policy_side = policy_values.get("task_region_side_m")
    target_radius = policy_values.get("target_detection_radius_m")
    uuv_radius = policy_values.get("uuv_active_detection_radius_m")
    max_gap = (
        _runtime_static_coverage_gap_m2(
            trace,
            detection_radius_m=float(uuv_radius),
            violations=violations,
        )
        if isinstance(uuv_radius, Real) and not isinstance(uuv_radius, bool)
        else None
    )
    if max_gap is not None and max_gap > 1.0e-6:
        violations.append("coverage_gap_exceeds_tolerance")
    execution_revision_monotonic = not any(
        "execution_revision_nonmonotonic" in violation
        or "execution_revision_gap" in violation
        or "execution_revision_invalid" in violation
        or "geometry_revision_nonmonotonic" in violation
        or "geometry_revision_invalid" in violation
        for violation in violations
    )
    execution_revision_advanced = (
        bool(execution_revisions)
        and max(execution_revisions) > min(execution_revisions)
    )
    if (
        initial_valid_until_s is not None
        and final_sim_time_s is not None
        and final_sim_time_s >= initial_valid_until_s
        and not execution_revision_advanced
    ):
        violations.append("execution_revision_did_not_advance_when_due")
        execution_revision_monotonic = False

    ping_interval = trace.get("sensor_ping_interval_s")
    active_ping_cadence_valid = (
        isinstance(ping_interval, Real)
        and not isinstance(ping_interval, bool)
        and isfinite(float(ping_interval))
        and float(ping_interval) > 0.0
    )
    if active_ping_cadence_valid:
        interval_s = float(cast(Real, ping_interval))
        for source, ping_times in ping_times_by_source.items():
            if any(
                right - left < interval_s - 1.0e-9
                for left, right in pairwise(sorted(set(ping_times)))
            ):
                active_ping_cadence_valid = False
                violations.append(
                    "active_ping_cadence_violation:" + ":".join(source)
                )
    else:
        violations.append("active_ping_interval_invalid")
    active_echo_source_valid = (
        active_echo_source_ids.issubset(active_ping_event_ids)
        and active_echo_event_ids.isdisjoint(active_ping_event_ids)
        and not any(
            "active_echo_source_missing" in violation
            or "active_echo_evidence_id_missing" in violation
            for violation in violations
        )
    )
    if not active_echo_source_valid:
        violations.append("active_echo_source_unresolved")

    for region_id, cycles_by_level in entry_cycles_by_region.items():
        required = int(_RUNTIME_POLICY["region_transition_confirm_cycles"])
        if max(cycles_by_level, default=0) < required:
            continue
        cycles = tuple(cycles_by_level.get(level) for level in range(1, required + 1))
        if any(cycle is None for cycle in cycles) or any(
            right <= left
            for left, right in pairwise(
                tuple(cycle for cycle in cycles if cycle is not None)
            )
        ):
            entry_confirmation_progression_valid = False
            violations.append(
                f"runtime_entry_confirmation_progression_invalid:{region_id}"
            )
    if active_ping_count_during_passive:
        violations.append("active_ping_during_passive_tracking")
    missing_scan_regions = sorted(expected_scan_region_ids - scan_regions_with_pings)
    if missing_scan_regions:
        violations.append(
            "runtime_active_scan_region_ping_missing:" + ",".join(missing_scan_regions)
        )
    if boundary_violation_count:
        violations.append("boundary_projection_invalid")
    return {
        "available": True,
        "checked": True,
        "valid": not violations,
        "frame_count": len(runtime_frames),
        "violations": list(dict.fromkeys(violations)),
        "region_side_m": policy_side,
        "target_detection_radius_m": target_radius,
        "uuv_detection_radius_m": uuv_radius,
        "task_group_size": policy_values.get("task_group_size"),
        "max_coverage_gap_area_m2": max_gap,
        "active_ping_count_during_passive": active_ping_count_during_passive,
        "tracking_owner_gap_frames": tracking_owner_gap_frames,
        "max_visible_uuv_count": max_visible_uuv_count,
        "runtime_scan_regions_with_pings": len(scan_regions_with_pings),
        "max_runtime_scan_coverage": max_runtime_scan_coverage,
        "runtime_entry_confirmation_count": max_entry_confirmation_count,
        "execution_revision_monotonic": execution_revision_monotonic,
        "execution_revision_advanced": execution_revision_advanced,
        "max_runtime_group_count": max_runtime_group_count,
        "runtime_replacement_pair_count": len(replacement_pairs),
        "replacement_contract_valid": replacement_contract_valid,
        "active_ping_count": len(active_ping_event_ids),
        "active_echo_count": len(active_echo_event_ids),
        "active_ping_cadence_valid": active_ping_cadence_valid,
        "active_echo_source_valid": active_echo_source_valid,
        "entry_evidence_public": entry_evidence_public,
        "entry_confirmation_progression_valid": (
            entry_confirmation_progression_valid
        ),
        "tracking_owner_observed": tracking_owner_observed,
        "max_tracking_owner_count": max_tracking_owner_count,
        "handoff_atomic_valid": handoff_atomic_valid,
        "canonical_transport_valid": canonical_transport_valid,
        "execution_frame_contract_valid": execution_frame_contract_valid,
        "boundary_checks": boundary_check_count,
        "boundary_violations": boundary_violation_count,
    }
