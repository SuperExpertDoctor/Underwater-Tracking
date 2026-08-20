from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import hypot
from typing import Iterable

Point = tuple[float, float]
Bounds = tuple[float, float, float, float]
GridKey = tuple[int, int]

_NEIGHBORS: tuple[GridKey, ...] = ((0, -1), (-1, 0), (1, 0), (0, 1))


@dataclass(frozen=True, slots=True)
class RoutePlan:
    """A complete route whose final point is always the home battle group."""

    points: tuple[Point, ...]
    stop_points: tuple[Point, ...]
    stop_indices: tuple[int, ...]
    distance_m: float

    @property
    def returns_home(self) -> bool:
        return bool(self.points) and self.points[0] == self.points[-1]


class AStarRoutePlanner:
    """Plan deterministic grid routes around forbidden region interiors."""

    def __init__(self, *, grid_size_m: float = 50.0, max_expanded_nodes: int = 200_000) -> None:
        if grid_size_m <= 0.0:
            raise ValueError("grid_size_m must be positive")
        if max_expanded_nodes < 1:
            raise ValueError("max_expanded_nodes must be positive")
        self._grid_size_m = grid_size_m
        self._max_expanded_nodes = max_expanded_nodes

    def plan(
        self,
        start: Point,
        stops: Iterable[Point],
        home: Point,
        forbidden_regions: Iterable[Bounds] = (),
        map_bounds: Bounds = (-10_000.0, 10_000.0, -10_000.0, 10_000.0),
    ) -> RoutePlan | None:
        bounds = _validate_bounds(map_bounds)
        stop_points = tuple(stops)
        forbidden = tuple(_as_bounds(region) for region in forbidden_regions)
        requested = (start, *stop_points, home)
        if any(not _inside_map(point, bounds) for point in requested):
            return None
        if any(_inside_region(point, region) for point in requested for region in forbidden):
            return None

        route_points: list[Point] = [start]
        stop_indices: list[int] = []
        for segment_index, (segment_start, segment_goal) in enumerate(
            zip(requested, requested[1:])
        ):
            start_key = self._to_key(segment_start, bounds)
            goal_key = self._to_key(segment_goal, bounds)
            if self._blocked(start_key, bounds, forbidden) or self._blocked(
                goal_key, bounds, forbidden
            ):
                return None
            segment = self._search(start_key, goal_key, bounds, forbidden)
            if segment is None:
                return None
            grid_points = tuple(self._from_key(key, bounds) for key in segment)
            previous_length = len(route_points)
            route_points.extend(grid_points[1:])
            is_stop = segment_index < len(stop_points)
            if route_points[-1] != segment_goal or (
                is_stop and len(route_points) == previous_length
            ):
                route_points.append(segment_goal)
            if is_stop:
                stop_indices.append(len(route_points) - 1)
        if len(route_points) == 1:
            route_points.append(home)

        points = tuple(route_points)
        if any(_inside_region(point, region) for point in points for region in forbidden):
            return None
        distance = sum(
            hypot(right[0] - left[0], right[1] - left[1])
            for left, right in zip(points, points[1:])
        )
        return RoutePlan(
            points=points,
            stop_points=stop_points,
            stop_indices=tuple(stop_indices),
            distance_m=distance,
        )

    def _search(
        self,
        start: GridKey,
        goal: GridKey,
        bounds: Bounds,
        forbidden: tuple[Bounds, ...],
    ) -> tuple[GridKey, ...] | None:
        frontier: list[tuple[float, float, GridKey]] = []
        heappush(frontier, (0.0, 0.0, start))
        came_from: dict[GridKey, GridKey | None] = {start: None}
        cost_so_far: dict[GridKey, float] = {start: 0.0}
        expanded = 0
        while frontier and expanded < self._max_expanded_nodes:
            _, current_cost, current = heappop(frontier)
            if current == goal:
                return _reconstruct(came_from, current)
            if current_cost > cost_so_far.get(current, float("inf")):
                continue
            expanded += 1
            for delta_x, delta_y in _NEIGHBORS:
                neighbor = (current[0] + delta_x, current[1] + delta_y)
                if self._blocked(neighbor, bounds, forbidden):
                    continue
                new_cost = current_cost + self._grid_size_m
                if new_cost >= cost_so_far.get(neighbor, float("inf")):
                    continue
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current
                priority = new_cost + self._heuristic(neighbor, goal)
                heappush(frontier, (priority, new_cost, neighbor))
        return None

    def _heuristic(self, left: GridKey, right: GridKey) -> float:
        return (abs(left[0] - right[0]) + abs(left[1] - right[1])) * self._grid_size_m

    def _to_key(self, point: Point, bounds: Bounds) -> GridKey:
        return (
            round((point[0] - bounds[0]) / self._grid_size_m),
            round((point[1] - bounds[2]) / self._grid_size_m),
        )

    def _from_key(self, key: GridKey, bounds: Bounds) -> Point:
        return (
            bounds[0] + key[0] * self._grid_size_m,
            bounds[2] + key[1] * self._grid_size_m,
        )

    def _blocked(
        self,
        key: GridKey,
        bounds: Bounds,
        forbidden: tuple[Bounds, ...],
    ) -> bool:
        point = self._from_key(key, bounds)
        return not _inside_map(point, bounds) or any(
            _inside_region(point, region) for region in forbidden
        )


def _reconstruct(
    came_from: dict[GridKey, GridKey | None],
    current: GridKey,
) -> tuple[GridKey, ...]:
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return tuple(path)


def _validate_bounds(bounds: Bounds) -> Bounds:
    if len(bounds) != 4 or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError("map bounds must have increasing x/y limits")
    return bounds


def _inside_map(point: Point, bounds: Bounds) -> bool:
    return bounds[0] <= point[0] <= bounds[1] and bounds[2] <= point[1] <= bounds[3]


def _inside_region(point: Point, region: Bounds) -> bool:
    return region[0] < point[0] < region[1] and region[2] < point[1] < region[3]


def _as_bounds(region: Bounds | object) -> Bounds:
    if isinstance(region, tuple) and len(region) == 4:
        return tuple(float(value) for value in region)  # type: ignore[return-value]
    try:
        return (
            float(getattr(region, "min_x")),
            float(getattr(region, "max_x")),
            float(getattr(region, "min_y")),
            float(getattr(region, "max_y")),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("forbidden region must expose rectangular bounds") from exc
