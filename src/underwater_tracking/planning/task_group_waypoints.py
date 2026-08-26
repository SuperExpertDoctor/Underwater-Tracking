"""Region-constrained waypoint planning for one two-UUV execution group."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin
from types import MappingProxyType

from underwater_tracking.domain.execution_models import ExecutionRegion, TaskGroupAssignment

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class TaskGroupWaypointPlan:
    """Short immutable route projection for one task-group/region pair."""

    task_group_id: str
    region_id: str
    waypoints_by_uuv: Mapping[str, tuple[Point, ...]]
    focus_xy: Point
    predicted_entry_xy: Point | None
    max_step_m: float
    max_turn_delta_rad: float
    min_separation_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "waypoints_by_uuv", MappingProxyType(dict(self.waypoints_by_uuv)))

    @property
    def cache_key(self) -> tuple[str, str]:
        return (self.task_group_id, self.region_id)

    @property
    def first_waypoints(self) -> Mapping[str, Point]:
        return MappingProxyType(
            {
                uuv_id: route[0]
                for uuv_id, route in self.waypoints_by_uuv.items()
                if route
            }
        )


class TaskGroupWaypointHistory:
    """Bounded route history keyed by both task group and region identity."""

    def __init__(self, *, limit: int = 64) -> None:
        if limit < 1:
            raise ValueError("waypoint history limit must be positive")
        self._limit = limit
        self._routes: dict[tuple[str, str], TaskGroupWaypointPlan] = {}

    def get(self, task_group_id: str, region_id: str) -> TaskGroupWaypointPlan | None:
        return self._routes.get((task_group_id, region_id))

    def put(self, plan: TaskGroupWaypointPlan) -> None:
        self._routes[plan.cache_key] = plan
        while len(self._routes) > self._limit:
            self._routes.pop(next(iter(self._routes)))

    def keys(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._routes)

    def clear(self) -> None:
        self._routes.clear()


def plan_task_group_waypoints(
    *,
    task_group: TaskGroupAssignment,
    region: ExecutionRegion,
    uuv_positions: Mapping[str, Point],
    target_position_xy: Point | None = None,
    predicted_entry_xy: Point | None = None,
    target_velocity_xy: Point = (0.0, 0.0),
    uuv_headings: Mapping[str, float] | None = None,
    previous_waypoints: Mapping[str, Sequence[Point]] | None = None,
    max_step_m: float = 900.0,
    max_turn_delta_rad: float = pi / 3.0,
    min_separation_m: float = 300.0,
    horizon_steps: int = 3,
    standoff_m: float = 650.0,
    min_standoff_m: float | None = None,
) -> TaskGroupWaypointPlan:
    """Build a deterministic, bounded route inside ``region``.

    The first group uses the globally known target position when supplied. A
    successor group uses ``predicted_entry_xy``. Future groups use the center
    of their forecast corridor. All route points are projected into the task
    polygon and every committed step is checked against movement, turn, and
    pairwise separation limits.
    """

    if task_group.region_id != region.region_id:
        raise ValueError("task group and execution region IDs do not match")
    if task_group.task_group_id != region.task_group_id:
        raise ValueError("execution region must name its task group")
    if len(region.geometry) < 3:
        raise ValueError("execution region geometry must contain at least three points")
    if max_step_m <= 0.0 or max_turn_delta_rad <= 0.0:
        raise ValueError("waypoint motion limits must be positive")
    if (
        min_separation_m < 0.0
        or horizon_steps < 1
        or standoff_m <= 0.0
        or (min_standoff_m is not None and min_standoff_m < 0.0)
    ):
        raise ValueError("waypoint planning limits are invalid")
    members = tuple(sorted(task_group.member_uuv_ids))
    if any(member not in uuv_positions for member in members):
        raise ValueError("waypoint planning requires every task-group member position")
    headings = uuv_headings or {}
    previous = previous_waypoints or {}
    polygon = tuple((float(point[0]), float(point[1])) for point in region.geometry)
    requested_focus = predicted_entry_xy
    if requested_focus is None and region.slot_index == 1:
        requested_focus = target_position_xy
    focus = _focus_point(polygon, requested_focus)
    velocity_angle = atan2(target_velocity_xy[1], target_velocity_xy[0])
    if hypot(*target_velocity_xy) <= 1e-9:
        velocity_angle = 0.0
    normal_angle = velocity_angle + pi / 2.0
    desired = (
        _polar_point(focus, standoff_m, normal_angle),
        _polar_point(focus, standoff_m, normal_angle + pi),
    )
    minimum_standoff = (
        min_standoff_m if min_standoff_m is not None else min(300.0, standoff_m * 0.5)
    )
    route_points: dict[str, Point] = {}
    for index, member in enumerate(members):
        route_points[member] = _choose_point(
            start=uuv_positions[member],
            preferred=desired[index],
            focus=focus,
            polygon=polygon,
            heading=headings.get(member),
            previous=previous.get(member, ()),
            max_step_m=max_step_m,
            max_turn_delta_rad=max_turn_delta_rad,
            standoff_m=standoff_m,
            min_standoff_m=minimum_standoff,
            phase=index,
        )
    if len(members) == 2 and hypot(
        route_points[members[0]][0] - route_points[members[1]][0],
        route_points[members[0]][1] - route_points[members[1]][1],
    ) < min_separation_m:
        route_points = _repair_pair(
            route_points,
            members,
            uuv_positions,
            focus,
            polygon,
            headings,
            max_step_m,
            max_turn_delta_rad,
            min_separation_m,
            standoff_m,
            minimum_standoff,
        )
    if len(members) > 1 and _minimum_separation(route_points.values()) < min_separation_m:
        raise ValueError("task-group waypoint separation constraint is infeasible")

    routes = {
        member: tuple(route_points[member] for _ in range(horizon_steps))
        for member in members
    }
    _validate_routes(
        routes,
        uuv_positions,
        headings,
        polygon,
        max_step_m,
        max_turn_delta_rad,
        min_separation_m,
    )
    return TaskGroupWaypointPlan(
        task_group_id=task_group.task_group_id,
        region_id=region.region_id,
        waypoints_by_uuv=routes,
        focus_xy=focus,
        predicted_entry_xy=predicted_entry_xy,
        max_step_m=max_step_m,
        max_turn_delta_rad=max_turn_delta_rad,
        min_separation_m=min_separation_m,
    )


def _focus_point(polygon: Sequence[Point], requested: Point | None) -> Point:
    candidate = requested or (
        sum(point[0] for point in polygon) / len(polygon),
        sum(point[1] for point in polygon) / len(polygon),
    )
    return _project_to_polygon(candidate, polygon)


def _choose_point(
    *,
    start: Point,
    preferred: Point,
    focus: Point,
    polygon: Sequence[Point],
    heading: float | None,
    previous: Sequence[Point],
    max_step_m: float,
    max_turn_delta_rad: float,
    standoff_m: float,
    min_standoff_m: float,
    phase: int,
) -> Point:
    prior = previous[-1] if previous else None
    candidates: list[Point] = []
    angles = tuple(
        (atan2(preferred[1] - focus[1], preferred[0] - focus[0]) + phase * pi / 12.0)
        + index * pi / 12.0
        for index in range(24)
    )
    radii = (standoff_m, standoff_m * 0.75, standoff_m * 0.5, standoff_m * 0.25)
    for radius in radii:
        for angle in angles:
            desired = _polar_point(focus, radius, angle)
            candidate = _move_towards(start, desired, max_step_m)
            candidate = _project_to_polygon(candidate, polygon)
            if not _within_step(start, candidate, max_step_m):
                continue
            if heading is not None and not _within_turn(
                start, candidate, heading, max_turn_delta_rad
            ):
                continue
            if hypot(candidate[0] - focus[0], candidate[1] - focus[1]) < min_standoff_m:
                continue
            candidates.append(candidate)
    if not candidates:
        candidate = _project_to_polygon(start, polygon)
        if _within_step(start, candidate, max_step_m) and (
            heading is None or _within_turn(start, candidate, heading, max_turn_delta_rad)
        ):
            return candidate
        raise ValueError("no waypoint satisfies region and motion constraints")

    def score(candidate: Point) -> tuple[float, float, float, float]:
        focus_cost = hypot(candidate[0] - focus[0], candidate[1] - focus[1])
        prior_cost = (
            hypot(candidate[0] - prior[0], candidate[1] - prior[1])
            if prior is not None
            else 0.0
        )
        preferred_cost = hypot(candidate[0] - preferred[0], candidate[1] - preferred[1])
        return (prior_cost, preferred_cost, focus_cost, candidate[0] + candidate[1])

    return min(candidates, key=score)


def _repair_pair(
    points: dict[str, Point],
    members: tuple[str, ...],
    positions: Mapping[str, Point],
    focus: Point,
    polygon: Sequence[Point],
    headings: Mapping[str, float],
    max_step_m: float,
    max_turn_delta_rad: float,
    min_separation_m: float,
    standoff_m: float,
    min_standoff_m: float,
) -> dict[str, Point]:
    first, second = members
    candidates = []
    for phase in range(24):
        angle = phase * pi / 12.0
        left = _choose_point(
            start=positions[first],
            preferred=_polar_point(focus, standoff_m, angle),
            focus=focus,
            polygon=polygon,
            heading=headings.get(first),
            previous=(),
            max_step_m=max_step_m,
            max_turn_delta_rad=max_turn_delta_rad,
            standoff_m=standoff_m,
            min_standoff_m=min_standoff_m,
            phase=0,
        )
        right = _choose_point(
            start=positions[second],
            preferred=_polar_point(focus, standoff_m, angle + pi),
            focus=focus,
            polygon=polygon,
            heading=headings.get(second),
            previous=(),
            max_step_m=max_step_m,
            max_turn_delta_rad=max_turn_delta_rad,
            standoff_m=standoff_m,
            min_standoff_m=min_standoff_m,
            phase=1,
        )
        distance = hypot(left[0] - right[0], left[1] - right[1])
        if distance >= min_separation_m:
            candidates.append((distance, left, right))
    if not candidates:
        return points
    _, left, right = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    return {first: left, second: right}


def _validate_routes(
    routes: Mapping[str, Sequence[Point]],
    starts: Mapping[str, Point],
    headings: Mapping[str, float],
    polygon: Sequence[Point],
    max_step_m: float,
    max_turn_delta_rad: float,
    min_separation_m: float,
) -> None:
    for member, route in routes.items():
        previous = starts[member]
        previous_heading = headings.get(member)
        for point in route:
            if not _point_in_polygon(point, polygon):
                raise ValueError("task-group waypoint lies outside its execution region")
            if not _within_step(previous, point, max_step_m):
                raise ValueError("task-group waypoint exceeds maximum displacement")
            if previous_heading is not None and not _within_turn(
                previous, point, previous_heading, max_turn_delta_rad
            ):
                raise ValueError("task-group waypoint exceeds maximum turn")
            if previous != point:
                previous_heading = atan2(point[1] - previous[1], point[0] - previous[0])
            previous = point
    for left_index, left in enumerate(routes):
        for right in tuple(routes)[left_index + 1 :]:
            for left_point, right_point in zip(routes[left], routes[right]):
                if hypot(
                    left_point[0] - right_point[0], left_point[1] - right_point[1]
                ) < min_separation_m:
                    raise ValueError("task-group route violates minimum separation")


def _polar_point(origin: Point, radius: float, angle: float) -> Point:
    return (origin[0] + radius * cos(angle), origin[1] + radius * sin(angle))


def _move_towards(start: Point, target: Point, max_distance: float) -> Point:
    distance = hypot(target[0] - start[0], target[1] - start[1])
    if distance <= max_distance:
        return target
    ratio = max_distance / distance
    return (
        start[0] + (target[0] - start[0]) * ratio,
        start[1] + (target[1] - start[1]) * ratio,
    )


def _within_step(start: Point, end: Point, maximum: float) -> bool:
    return hypot(end[0] - start[0], end[1] - start[1]) <= maximum + 1e-6


def _within_turn(start: Point, end: Point, heading: float, maximum: float) -> bool:
    if start == end:
        return True
    desired = atan2(end[1] - start[1], end[0] - start[0])
    delta = (desired - heading + pi) % (2.0 * pi) - pi
    return abs(delta) <= maximum + 1e-9


def _minimum_separation(points: Sequence[Point]) -> float:
    values = tuple(points)
    if len(values) < 2:
        return float("inf")
    return min(
        hypot(left[0] - right[0], left[1] - right[1])
        for index, left in enumerate(values)
        for right in values[index + 1 :]
    )


def _point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = point
    inside = False
    for start, end in zip(polygon, (*polygon[1:], polygon[0])):
        if _distance_to_segment(point, start, end) <= 1e-7:
            return True
        if (start[1] > y) != (end[1] > y):
            crossing = (end[0] - start[0]) * (y - start[1]) / (end[1] - start[1]) + start[0]
            if x < crossing:
                inside = not inside
    return inside


def _project_to_polygon(point: Point, polygon: Sequence[Point]) -> Point:
    if _point_in_polygon(point, polygon):
        return point
    return min(
        (
            _closest_point_on_segment(point, start, end)
            for start, end in zip(polygon, (*polygon[1:], polygon[0]))
        ),
        key=lambda candidate: hypot(candidate[0] - point[0], candidate[1] - point[1]),
    )


def _closest_point_on_segment(point: Point, start: Point, end: Point) -> Point:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-12:
        return start
    ratio = max(
        0.0,
        min(
            1.0,
            ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy)
            / length_squared,
        ),
    )
    return (start[0] + ratio * dx, start[1] + ratio * dy)


def _distance_to_segment(point: Point, start: Point, end: Point) -> float:
    closest = _closest_point_on_segment(point, start, end)
    return hypot(point[0] - closest[0], point[1] - closest[1])


__all__ = ["TaskGroupWaypointHistory", "TaskGroupWaypointPlan", "plan_task_group_waypoints"]
