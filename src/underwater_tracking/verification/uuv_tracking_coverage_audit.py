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
        }

    violations: list[str] = []
    policy_values: dict[str, object] = {}
    active_ping_count_during_passive = 0
    tracking_owner_gap_frames = 0
    max_visible_uuv_count = 0
    boundary_check_count = 0
    boundary_violation_count = 0
    for index, (raw_trace_frame, raw_frame) in enumerate(runtime_entries):
        frame = cast(Mapping[str, object], raw_frame)
        frame_violations = validate_uuv_only_frame(frame)
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
            violations.append(f"frame_{index}:transport_hash_mismatch")

        execution = frame.get("execution")
        if not isinstance(execution, Mapping):
            violations.append(f"frame_{index}:execution_missing")
            continue
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
            if group.get("sensor_mode") == "passive"
            or group.get("lifecycle") in {
                "passive_track",
                "dedicated_track",
                "dedicated_release_pending",
            }
        )
        owner_is_valid = (
            isinstance(owner_id, str)
            and owner is not None
            and owner.get("ownership_status") == "owner"
            and len(_runtime_sequence(owner.get("member_uuv_ids")) or ()) == 3
        )
        if (tracking_mode == "dedicated" or passive_groups) and not owner_is_valid:
            tracking_owner_gap_frames += 1

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
            if _RUNTIME_LIFECYCLE_SENSOR_MODES.get(lifecycle) != sensor_mode:
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
            for value in frame.get("uuvs", ())
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
        for item in runtime_uuvs:
            uuv_id = item.get("uuv_id")
            group_id = item.get("group_instance_id")
            if item.get("physically_exposed") is True and group_id not in group_by_id:
                violations.append(f"frame_{index}:uuv_group_missing:{uuv_id}")
            group = group_by_id.get(group_id)
            if group is None:
                continue
            if item.get("group_lifecycle") != group.get("lifecycle"):
                violations.append(f"frame_{index}:uuv_lifecycle_mismatch:{uuv_id}")
            if item.get("sensor_mode") != group.get("sensor_mode"):
                violations.append(f"frame_{index}:uuv_sensor_mode_mismatch:{uuv_id}")

        frame_time = frame.get("sim_time_s")
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
            if raw_event.get("event_type") != "active_ping":
                continue
            event_time = raw_event.get("sim_time_s")
            payload = raw_event.get("payload")
            emitter_id = payload.get("emitter_id") if isinstance(payload, Mapping) else None
            if passive_member_ids and event_time == frame_time and (
                emitter_id in passive_member_ids or not isinstance(emitter_id, str)
            ):
                active_ping_count_during_passive += 1

        if tracking_mode == "regional" and linked_group_ids and any(
            group_id not in linked_group_ids
            and group.get("lifecycle") not in {"exiting", "disappeared"}
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
    if active_ping_count_during_passive:
        violations.append("active_ping_during_passive_tracking")
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
        "boundary_checks": boundary_check_count,
        "boundary_violations": boundary_violation_count,
    }
