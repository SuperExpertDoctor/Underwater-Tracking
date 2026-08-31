"""Truth-free multi-UUV formation correction.

This is the narrow integration boundary adapted from
``multi_uuv_tracking_pure_python_20260817``. It consumes the current belief
mean and the planner's waypoints only; it never reads target truth or emits a
complete assignment. Allocation, bounds, separation, and plan verification
remain owned by the existing carrier planner.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FormationCorrection:
    """Corrected routes and operator-useful slot diagnostics."""

    waypoints_by_member: dict[str, tuple[tuple[float, float], ...]]
    slot_points_by_member: dict[str, tuple[float, float]]
    slot_error_m_by_member: dict[str, float]


def formation_slot_point(
    target_position: Sequence[float],
    target_velocity: Sequence[float],
    slot_offset_deg: float,
    radius_m: float,
    horizon_s: float,
    *,
    fallback_heading_rad: float | None = None,
    minimum_target_speed_mps: float = 0.2,
) -> tuple[float, float] | None:
    """Project one standoff slot from the belief's predicted target motion."""
    position = np.asarray(target_position, dtype=float).reshape(-1)[:2]
    velocity = np.asarray(target_velocity, dtype=float).reshape(-1)[:2]
    if position.shape != (2,) or velocity.shape != (2,):
        raise ValueError("target position and velocity must contain two values")
    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise ValueError("target position and velocity must be finite")
    speed = float(np.linalg.norm(velocity))
    if speed >= minimum_target_speed_mps:
        heading = math.atan2(float(velocity[1]), float(velocity[0]))
    elif fallback_heading_rad is not None and math.isfinite(fallback_heading_rad):
        heading = float(fallback_heading_rad)
    else:
        return None
    future = position + velocity * max(0.0, float(horizon_s))
    radial = heading + math.pi + math.radians(float(slot_offset_deg))
    return (
        float(future[0] + float(radius_m) * math.cos(radial)),
        float(future[1] + float(radius_m) * math.sin(radial)),
    )


def correct_waypoints_toward_slot(
    points: Sequence[Sequence[float]],
    slot_point: Sequence[float],
    maximum_endpoint_correction_m: float,
) -> tuple[tuple[float, float], ...]:
    """Bend a route gradually toward a slot with a bounded endpoint change."""
    route = np.asarray(points, dtype=float)
    if route.ndim != 2 or route.shape[0] == 0 or route.shape[1] < 2:
        raise ValueError("points must have shape (N, 2+) with N > 0")
    slot = np.asarray(slot_point, dtype=float).reshape(-1)[:2]
    if slot.shape != (2,) or not np.isfinite(route[:, :2]).all() or not np.isfinite(slot).all():
        raise ValueError("points and slot point must be finite two-dimensional values")
    delta = slot - route[-1, :2]
    distance = float(np.linalg.norm(delta))
    maximum = max(0.0, float(maximum_endpoint_correction_m))
    if maximum == 0.0:
        delta[:] = 0.0
    elif distance > maximum:
        delta *= maximum / distance
    corrected = route[:, :2].copy()
    for index in range(corrected.shape[0]):
        corrected[index] += ((index + 1) / corrected.shape[0]) * delta
    return tuple((float(point[0]), float(point[1])) for point in corrected)


def apply_formation_correction(
    *,
    member_ids: Sequence[str],
    waypoints_by_member: Mapping[str, Sequence[Sequence[float]]],
    target_position: Sequence[float],
    target_velocity: Sequence[float],
    target_heading_rad: float | None,
    radius_m: float,
    horizon_s: float,
    maximum_endpoint_correction_m: float,
    bounds_xy: Sequence[float] | None = None,
) -> FormationCorrection:
    """Apply deterministic slot shaping to a previously generated group route."""
    members = tuple(sorted(member_ids))
    if not members:
        return FormationCorrection({}, {}, {})
    if len(members) == 1:
        offsets: tuple[float, ...] = (0.0,)
    else:
        span = 120.0
        offsets = tuple(-span / 2.0 + span * index / (len(members) - 1) for index in range(len(members)))
    corrected: dict[str, tuple[tuple[float, float], ...]] = {}
    slots: dict[str, tuple[float, float]] = {}
    errors: dict[str, float] = {}
    for member, offset in zip(members, offsets, strict=True):
        route = waypoints_by_member.get(member)
        if not route:
            continue
        slot = formation_slot_point(
            target_position,
            target_velocity,
            offset,
            radius_m,
            horizon_s,
            fallback_heading_rad=target_heading_rad,
        )
        if slot is None:
            corrected[member] = tuple((float(point[0]), float(point[1])) for point in route)
            continue
        route_corrected = correct_waypoints_toward_slot(
            route, slot, maximum_endpoint_correction_m
        )
        if bounds_xy is not None:
            min_x, max_x, min_y, max_y = (float(value) for value in bounds_xy)
            route_corrected = tuple(
                (
                    min(max(point[0], min_x), max_x),
                    min(max(point[1], min_y), max_y),
                )
                for point in route_corrected
            )
        corrected[member] = route_corrected
        slots[member] = slot
        errors[member] = math.hypot(
            route_corrected[-1][0] - slot[0], route_corrected[-1][1] - slot[1]
        )
    return FormationCorrection(corrected, slots, errors)
