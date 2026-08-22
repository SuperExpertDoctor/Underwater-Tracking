"""Deterministic translation from target intent to physical guidance."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence
from typing import Literal

from underwater_tracking.domain.adversary_models import (
    AdversaryIntent,
    AdversaryIntentDecision,
    AdversaryMissionState,
    TargetLocalContact,
)
from underwater_tracking.domain.adversary_models import AdversaryOperatingBoundary
from underwater_tracking.domain.platforms import MotionLimits, SubmarineMotionLimits
from underwater_tracking.simulation.kinematics import MotionState, wrap_angle

GuidanceSource = Literal["llm", "mission_route", "boundary_avoidance", "safe_hold"]


@dataclass(frozen=True, slots=True)
class TargetGuidanceCommand:
    decision_id: str | None
    intent: AdversaryIntent
    waypoint_xy: tuple[float, float]
    desired_heading_rad: float
    desired_speed_mps: float
    valid_until_s: int
    source: GuidanceSource
    desired_depth_m: float | None = None


@dataclass(frozen=True, slots=True)
class TargetGuidanceResult:
    command: TargetGuidanceCommand
    next_route_index: int


def resolve_target_guidance(
    *,
    decision: AdversaryIntentDecision | None,
    mission: AdversaryMissionState,
    contacts: tuple[TargetLocalContact, ...],
    state: MotionState,
    limits: MotionLimits,
    operating_boundary: AdversaryOperatingBoundary,
    exclusion_regions: tuple[tuple[tuple[float, float], ...], ...],
    sim_time_s: int,
    previous_guidance: TargetGuidanceCommand | None,
    submarine_limits: SubmarineMotionLimits | None = None,
    current_depth_m: float = 0.0,
) -> TargetGuidanceResult:
    """Resolve one immutable input packet into a repeatable guidance command."""
    if sim_time_s < 0:
        raise ValueError("sim_time_s must be non-negative")
    boundary = (
        operating_boundary.min_x,
        operating_boundary.max_x,
        operating_boundary.min_y,
        operating_boundary.max_y,
    )
    active_contacts = tuple(contact for contact in contacts if contact.status == "active")
    intent = decision.intent if decision is not None else (
        "avoid_contact" if active_contacts else mission.current_intent
    )
    decision_id = decision.decision_id if decision is not None else None
    source: GuidanceSource = "llm" if decision is not None else "mission_route"
    next_route_index = mission.current_route_index

    if decision is None and previous_guidance is not None and previous_guidance.valid_until_s > sim_time_s:
        if _guidance_segment_is_safe(state.position_xy, previous_guidance.waypoint_xy, boundary, exclusion_regions):
            command = previous_guidance
            return TargetGuidanceResult(command=command, next_route_index=next_route_index)

    if intent == "hold_position":
        waypoint = state.position_xy
        desired_speed = limits.min_speed_mps
    elif intent == "escape_to_region" and decision is not None:
        escape_region_id = decision.escape_region_id
        if escape_region_id is None or escape_region_id not in mission.escape_regions:
            raise ValueError("escape_region_id is not a configured escape region")
        waypoint = _polygon_centroid(mission.escape_regions[escape_region_id])
        desired_speed = limits.max_speed_mps
    elif intent in {"avoid_contact", "break_contact"}:
        waypoint = (
            _away_from_contacts(state, active_contacts, limits.max_speed_mps)
            if active_contacts
            else _mission_route_waypoint(state.position_xy, mission)[0]
        )
        desired_speed = limits.max_speed_mps if intent == "break_contact" else _cruise_speed(limits)
    else:
        waypoint, next_route_index = _mission_route_waypoint(state.position_xy, mission)
        desired_speed = _cruise_speed(limits)
        intent = "continue_mission"
        decision_id = decision_id if decision is not None else None

    if not _point_is_safe(waypoint, boundary, exclusion_regions):
        waypoint = _safe_boundary_point(state.position_xy, boundary)
        source = "boundary_avoidance"
        desired_speed = min(desired_speed, _cruise_speed(limits))
    elif not _guidance_segment_is_safe(state.position_xy, waypoint, boundary, exclusion_regions):
        waypoint = _safe_boundary_point(state.position_xy, boundary)
        source = "boundary_avoidance"
        desired_speed = min(desired_speed, _cruise_speed(limits))

    heading = state.heading_rad
    dx = waypoint[0] - state.position_xy[0]
    dy = waypoint[1] - state.position_xy[1]
    if math.hypot(dx, dy) > 1e-9:
        heading = wrap_angle(math.atan2(dy, dx))
    if source == "boundary_avoidance" and desired_speed <= limits.min_speed_mps:
        source = "safe_hold"
    desired_depth_m = _desired_depth(
        decision.depth_intent if decision is not None else "maintain_depth",
        current_depth_m,
        submarine_limits,
    )
    command = TargetGuidanceCommand(
        decision_id=decision_id,
        intent=intent,
        waypoint_xy=(float(waypoint[0]), float(waypoint[1])),
        desired_heading_rad=heading,
        desired_speed_mps=float(desired_speed),
        valid_until_s=sim_time_s + 120,
        source=source,
        desired_depth_m=desired_depth_m,
    )
    return TargetGuidanceResult(command=command, next_route_index=next_route_index)


def _desired_depth(
    depth_intent: str,
    current_depth_m: float,
    limits: SubmarineMotionLimits | None,
) -> float | None:
    if limits is None:
        return None
    current = min(limits.max_depth_m, max(limits.min_depth_m, current_depth_m))
    if depth_intent == "go_deeper":
        return min(limits.max_depth_m, current + max(25.0, limits.max_vertical_speed_mps * 30.0))
    if depth_intent == "go_shallower":
        return max(limits.min_depth_m, current - max(25.0, limits.max_vertical_speed_mps * 30.0))
    return current


def _cruise_speed(limits: MotionLimits) -> float:
    return min(limits.max_speed_mps, max(limits.min_speed_mps, limits.max_speed_mps * 0.6))


def _mission_route_waypoint(
    position: tuple[float, float], mission: AdversaryMissionState
) -> tuple[tuple[float, float], int]:
    index = mission.current_route_index
    if index < len(mission.mission_route_xy) - 1:
        waypoint = mission.mission_route_xy[index + 1]
        if math.dist(position, waypoint) <= 50.0:
            index += 1
            waypoint = mission.mission_route_xy[index]
    else:
        waypoint = _polygon_centroid(mission.task_region_polygon_xy)
    return waypoint, index


def _polygon_centroid(polygon: Sequence[tuple[float, float]]) -> tuple[float, float]:
    if not polygon:
        raise ValueError("polygon must not be empty")
    area_twice = 0.0
    centroid_x = 0.0
    centroid_y = 0.0
    for start, end in zip(polygon, (*polygon[1:], polygon[0])):
        cross = start[0] * end[1] - end[0] * start[1]
        area_twice += cross
        centroid_x += (start[0] + end[0]) * cross
        centroid_y += (start[1] + end[1]) * cross
    if abs(area_twice) <= 1e-9:
        return (
            sum(point[0] for point in polygon) / len(polygon),
            sum(point[1] for point in polygon) / len(polygon),
        )
    return (centroid_x / (3.0 * area_twice), centroid_y / (3.0 * area_twice))


def _away_from_contacts(
    state: MotionState,
    contacts: Sequence[TargetLocalContact],
    speed: float,
) -> tuple[float, float]:
    vector_x = 0.0
    vector_y = 0.0
    for contact in contacts:
        absolute_bearing = state.heading_rad + contact.relative_bearing_rad
        weight = (1.0 + _threat_weight(contact.threat_level)) / max(contact.estimated_range_m, 1.0)
        vector_x -= math.cos(absolute_bearing) * weight
        vector_y -= math.sin(absolute_bearing) * weight
    magnitude = math.hypot(vector_x, vector_y)
    if magnitude <= 1e-12:
        return (
            state.position_xy[0] + speed * math.cos(state.heading_rad),
            state.position_xy[1] + speed * math.sin(state.heading_rad),
        )
    return (
        state.position_xy[0] + speed * vector_x / magnitude,
        state.position_xy[1] + speed * vector_y / magnitude,
    )


def _threat_weight(level: str) -> float:
    return {"low": 1.0, "medium": 2.0, "high": 3.0, "critical": 4.0}.get(level, 1.0)


def _point_is_safe(
    point: tuple[float, float],
    bounds: tuple[float, float, float, float],
    exclusions: Sequence[Sequence[tuple[float, float]]],
) -> bool:
    min_x, max_x, min_y, max_y = bounds
    return (
        min_x <= point[0] <= max_x
        and min_y <= point[1] <= max_y
        and not any(_point_in_polygon(point, polygon) for polygon in exclusions)
    )


def _guidance_segment_is_safe(
    start: tuple[float, float],
    end: tuple[float, float],
    bounds: tuple[float, float, float, float],
    exclusions: Sequence[Sequence[tuple[float, float]]],
) -> bool:
    if not _point_is_safe(end, bounds, exclusions):
        return False
    return not any(
        _segment_intersects_polygon(start, end, polygon) for polygon in exclusions
    )


def _safe_boundary_point(
    point: tuple[float, float], bounds: tuple[float, float, float, float]
) -> tuple[float, float]:
    min_x, max_x, min_y, max_y = bounds
    margin = min(50.0, (max_x - min_x) * 0.25, (max_y - min_y) * 0.25)
    return (
        min(max(point[0], min_x + margin), max_x - margin),
        min(max(point[1], min_y + margin), max_y - margin),
    )


def _point_in_polygon(point: tuple[float, float], polygon: Sequence[tuple[float, float]]) -> bool:
    inside = False
    for start, end in zip(polygon, (*polygon[1:], polygon[0])):
        if (start[1] > point[1]) != (end[1] > point[1]):
            crossing_x = (end[0] - start[0]) * (point[1] - start[1]) / (end[1] - start[1]) + start[0]
            if point[0] < crossing_x:
                inside = not inside
    return inside


def _segment_intersects_polygon(
    start: tuple[float, float], end: tuple[float, float], polygon: Sequence[tuple[float, float]]
) -> bool:
    return _point_in_polygon(start, polygon) or _point_in_polygon(end, polygon) or any(
        _segments_intersect(start, end, edge_start, edge_end)
        for edge_start, edge_end in zip(polygon, (*polygon[1:], polygon[0]))
    )


def _segments_intersect(
    first_start: tuple[float, float], first_end: tuple[float, float],
    second_start: tuple[float, float], second_end: tuple[float, float],
) -> bool:
    def orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    first = orientation(first_start, first_end, second_start)
    second = orientation(first_start, first_end, second_end)
    third = orientation(second_start, second_end, first_start)
    fourth = orientation(second_start, second_end, first_end)
    if ((first > 0 > second) or (first < 0 < second)) and ((third > 0 > fourth) or (third < 0 < fourth)):
        return True
    epsilon = 1e-9
    if abs(first) <= epsilon and _on_segment(first_start, second_start, first_end):
        return True
    if abs(second) <= epsilon and _on_segment(first_start, second_end, first_end):
        return True
    if abs(third) <= epsilon and _on_segment(second_start, first_start, second_end):
        return True
    if abs(fourth) <= epsilon and _on_segment(second_start, first_end, second_end):
        return True
    return False


def _on_segment(
    start: tuple[float, float], point: tuple[float, float], end: tuple[float, float]
) -> bool:
    return (
        min(start[0], end[0]) - 1e-9 <= point[0] <= max(start[0], end[0]) + 1e-9
        and min(start[1], end[1]) - 1e-9 <= point[1] <= max(start[1], end[1]) + 1e-9
    )


__all__ = ["TargetGuidanceCommand", "TargetGuidanceResult", "resolve_target_guidance"]
