from __future__ import annotations

import pytest

from underwater_tracking.simulation.formation_control import (
    apply_formation_correction,
    correct_waypoints_toward_slot,
    formation_slot_point,
)


def test_slot_projection_uses_belief_velocity_without_truth() -> None:
    slot = formation_slot_point((0.0, 0.0), (2.0, 0.0), 0.0, 800.0, 10.0)
    assert slot == pytest.approx((-780.0, 0.0))


def test_waypoint_correction_is_bounded_and_progressive() -> None:
    corrected = correct_waypoints_toward_slot(
        ((0.0, 0.0), (100.0, 0.0)), (1000.0, 0.0), 100.0
    )
    assert corrected == ((50.0, 0.0), (200.0, 0.0))


def test_formation_adapter_clamps_corrected_routes_to_bounds() -> None:
    correction = apply_formation_correction(
        member_ids=("U2", "U1"),
        waypoints_by_member={"U1": ((0.0, 0.0),), "U2": ((0.0, 0.0),)},
        target_position=(0.0, 0.0),
        target_velocity=(1.0, 0.0),
        target_heading_rad=0.0,
        radius_m=800.0,
        horizon_s=10.0,
        maximum_endpoint_correction_m=400.0,
        bounds_xy=(-100.0, 100.0, -100.0, 100.0),
    )
    assert set(correction.waypoints_by_member) == {"U1", "U2"}
    assert all(
        -100.0 <= coordinate <= 100.0
        for route in correction.waypoints_by_member.values()
        for point in route
        for coordinate in point
    )
