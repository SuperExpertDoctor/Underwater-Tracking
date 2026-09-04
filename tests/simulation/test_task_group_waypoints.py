from __future__ import annotations

from math import atan2, hypot, pi

import numpy as np
import pytest

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.execution_models import (
    ExecutionRegion,
    ReserveUUVState,
    TaskGroupAssignment,
)
from underwater_tracking.domain.mission_models import (
    ExecutableMissionPlan,
    RegionMissionState,
    UUVMissionMode,
)
from underwater_tracking.domain.models import DeploymentState
from underwater_tracking.planning.task_group_waypoints import (
    TaskGroupWaypointHistory,
    plan_task_group_waypoints,
)
from underwater_tracking.planning.coverage import (
    coverage_gap_area_m2,
    serpentine_coverage_waypoints_by_uuv,
)
from underwater_tracking.planning.route_safety import transition_separation_is_safe
from underwater_tracking.runtime.mission_controller import MissionController
from underwater_tracking.simulation.engine import SimulationEngine


def _region(slot: int, *, task_group_id: str | None = None) -> ExecutionRegion:
    target_id = "target-1"
    x_min = (slot - 1) * 2_000.0 - 1_000.0
    x_max = x_min + 2_000.0
    return ExecutionRegion(
        region_id=f"{target_id}:task:{slot:02d}",
        target_id=target_id,
        slot_index=slot - 1,
        execution_revision=3,
        prediction_id="pred:target-1:3",
        geometry=(
            (x_min, -1_000.0),
            (x_max, -1_000.0),
            (x_max, 1_000.0),
            (x_min, 1_000.0),
        ),
        centerline_indices=(slot - 1,),
        start_s=(slot - 1) * 450.0,
        end_s=slot * 450.0 + 90.0,
        geometry_revision=1,
        predecessor_region_id=(
            f"{target_id}:task:{slot - 1:02d}" if slot > 1 else None
        ),
        successor_region_id=(
            f"{target_id}:task:{slot + 1:02d}" if slot < 4 else None
        ),
        handoff_start_s=slot * 450.0 if slot < 4 else None,
        handoff_end_s=slot * 450.0 + 90.0 if slot < 4 else None,
        task_group_id=task_group_id or f"{target_id}:task-group:{slot:02d}",
        evidence_ids=(f"evidence:{slot}",),
    )


def _group(slot: int) -> TaskGroupAssignment:
    region_id = f"target-1:task:{slot:02d}"
    return TaskGroupAssignment(
        task_group_id=f"target-1:task-group:{slot:02d}",
        target_id="target-1",
        region_id=region_id,
        execution_revision=3,
        member_uuv_ids=(f"uuv-{slot}a", f"uuv-{slot}b"),
        active_verifier_uuv_id=f"uuv-{slot}a",
        passive_tracker_uuv_id=f"uuv-{slot}b",
        evidence_ids=(f"evidence:{slot}",),
    )


def test_three_serpentine_routes_cover_every_point_of_square() -> None:
    square = (
        (0.0, 0.0),
        (2_000.0, 0.0),
        (2_000.0, 2_000.0),
        (0.0, 2_000.0),
    )

    routes = serpentine_coverage_waypoints_by_uuv(
        square,
        ("U1", "U2", "U3"),
        detection_radius_m=600.0,
    )

    assert set(routes) == {"U1", "U2", "U3"}
    assert coverage_gap_area_m2(square, routes, 600.0) <= 1e-6


def test_waypoint_history_isolated_by_task_group_and_region() -> None:
    history = TaskGroupWaypointHistory(limit=4)
    first = plan_task_group_waypoints(
        task_group=_group(1),
        region=_region(1),
        uuv_positions={"uuv-1a": (-700.0, 0.0), "uuv-1b": (-700.0, 700.0)},
        target_position_xy=(-500.0, 0.0),
        target_velocity_xy=(4.0, 0.0),
    )
    second = plan_task_group_waypoints(
        task_group=_group(2),
        region=_region(2),
        uuv_positions={"uuv-2a": (1_300.0, 0.0), "uuv-2b": (1_300.0, 700.0)},
        predicted_entry_xy=(1_500.0, 0.0),
        target_velocity_xy=(4.0, 0.0),
    )

    history.put(first)
    history.put(second)

    assert history.keys() == (first.cache_key, second.cache_key)
    assert history.get(*first.cache_key) is first
    assert history.get(*second.cache_key) is second
    assert first.cache_key != second.cache_key


def test_current_group_uses_global_target_geometry_and_limits_motion() -> None:
    plan = plan_task_group_waypoints(
        task_group=_group(1),
        region=_region(1),
        uuv_positions={"uuv-1a": (-700.0, 0.0), "uuv-1b": (-700.0, 700.0)},
        target_position_xy=(-500.0, 0.0),
        target_velocity_xy=(4.0, 0.0),
        uuv_headings={"uuv-1a": 0.0, "uuv-1b": -pi / 2.0},
        max_step_m=900.0,
        max_turn_delta_rad=pi,
        min_separation_m=300.0,
    )

    points = plan.first_waypoints
    assert plan.focus_xy == (-500.0, 0.0)
    assert all(-1_000.0 <= point[0] <= 1_000.0 for point in points.values())
    assert all(
        300.0 <= hypot(point[0] + 500.0, point[1]) <= 900.0
        for point in points.values()
    )
    assert hypot(
        points["uuv-1a"][0] - points["uuv-1b"][0],
        points["uuv-1a"][1] - points["uuv-1b"][1],
    ) >= 300.0
    assert all(
        hypot(
            point[0] + 700.0,
            point[1] - (0.0 if member == "uuv-1a" else 700.0),
        )
        <= 900.0 + 1e-6
        for member, point in points.items()
    )


def test_successor_group_uses_predicted_entry_point() -> None:
    plan = plan_task_group_waypoints(
        task_group=_group(2),
        region=_region(2),
        uuv_positions={"uuv-2a": (1_300.0, 0.0), "uuv-2b": (1_300.0, 700.0)},
        predicted_entry_xy=(1_500.0, 0.0),
        target_position_xy=(-500.0, 0.0),
        target_velocity_xy=(4.0, 0.0),
        max_turn_delta_rad=pi,
    )

    assert plan.focus_xy == (1_500.0, 0.0)
    assert all(
        hypot(point[0] - 1_500.0, point[1]) <= 700.0
        for point in plan.first_waypoints.values()
    )


def test_future_group_routes_stay_inside_its_polygon() -> None:
    plan = plan_task_group_waypoints(
        task_group=_group(4),
        region=_region(4),
        uuv_positions={"uuv-4a": (5_300.0, 0.0), "uuv-4b": (5_300.0, 700.0)},
        target_position_xy=(-500.0, 0.0),
        max_turn_delta_rad=pi,
    )

    assert all(
        5_000.0 <= point[0] <= 7_000.0 and -1_000.0 <= point[1] <= 1_000.0
        for route in plan.waypoints_by_uuv.values()
        for point in route
    )


def test_waypoints_respect_turn_limit() -> None:
    plan = plan_task_group_waypoints(
        task_group=_group(1),
        region=_region(1),
        uuv_positions={"uuv-1a": (-700.0, 0.0), "uuv-1b": (-700.0, 700.0)},
        target_position_xy=(-500.0, 0.0),
        uuv_headings={"uuv-1a": 0.0, "uuv-1b": -pi / 2.0},
        max_turn_delta_rad=pi / 2.0,
    )

    for member, route in plan.waypoints_by_uuv.items():
        start = (-700.0, 0.0) if member.endswith("a") else (-700.0, 700.0)
        heading = 0.0 if member.endswith("a") else -pi / 2.0
        first = route[0]
        angle = atan2(first[1] - start[1], first[0] - start[0])
        delta = abs((angle - heading + pi) % (2 * pi) - pi)
        assert delta <= pi / 2.0 + 1e-9


def test_infeasible_minimum_separation_is_reported() -> None:
    with pytest.raises(ValueError, match="separation"):
        plan_task_group_waypoints(
            task_group=_group(1),
            region=_region(1),
            uuv_positions={"uuv-1a": (-500.0, 0.0), "uuv-1b": (-500.0, 0.0)},
            target_position_xy=(-500.0, 0.0),
            standoff_m=1.0,
            min_separation_m=1_000.0,
            max_turn_delta_rad=pi,
        )


def test_task_group_planner_rejects_a_mid_transition_crossing() -> None:
    starts = {"uuv-1a": (0.0, -650.0), "uuv-1b": (0.0, 650.0)}
    plan = plan_task_group_waypoints(
        task_group=_group(1),
        region=_region(1),
        uuv_positions=starts,
        target_position_xy=(0.0, 0.0),
        target_velocity_xy=(1.0, 0.0),
        max_step_m=1_500.0,
        max_turn_delta_rad=pi,
        min_separation_m=300.0,
    )
    left = plan.first_waypoints["uuv-1a"]
    right = plan.first_waypoints["uuv-1b"]

    assert transition_separation_is_safe(
        starts["uuv-1a"],
        left,
        starts["uuv-1b"],
        right,
        min_separation_m=300.0,
    )


def test_hold_spread_keeps_opposite_members_on_their_nearest_safe_side() -> None:
    members = ("uuv-left", "uuv-right")
    positions = np.asarray(
        (
            (-400.0, -1.0e-12),
            (400.0, 1.0e-12),
        ),
        dtype=float,
    )

    commands = SimulationEngine._hold_spread_commands(members, positions)

    assert commands["uuv-left"] == pytest.approx((-900.0, 0.0))
    assert commands["uuv-right"] == pytest.approx((900.0, 0.0))
    assert transition_separation_is_safe(
        (-400.0, -1.0e-12),
        commands["uuv-left"],
        (400.0, 1.0e-12),
        commands["uuv-right"],
        min_separation_m=300.0,
    )


def test_public_boundary_protocol_requires_evidence_before_role_takeover() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=17, mission_controller=controller)
    region = RegionMissionState(
        region_id="target_00:task:01",
        target_id="target_00",
        task_group_id="target_00:task-group:01",
        active_scan_uuv_ids=("uuv_00",),
        passive_track_uuv_ids=("uuv_01",),
        reserve_uuv_ids=("uuv_02",),
        region_polygon=(
            (-2_000.0, -2_000.0),
            (2_000.0, -2_000.0),
            (2_000.0, 2_000.0),
            (-2_000.0, 2_000.0),
        ),
        scan_waypoints=((0.0, 0.0),),
    )
    controller._regions = {region.region_id: region}
    controller._uuv_modes.update(
        {
            "uuv_00": UUVMissionMode.ACTIVE_SCAN,
            "uuv_01": UUVMissionMode.PASSIVE_TRACK,
            "uuv_02": UUVMissionMode.ONBOARD,
        }
    )
    engine._deployment_states["uuv_00"] = DeploymentState.DEPLOYED
    engine._waterborne_uuv_ids.add("uuv_00")
    engine._uuvs["uuv_00"].position_xy = (0.0, 0.0)

    assert engine.begin_boundary_exit("uuv_00", region)
    exit_point = engine._uuvs["uuv_00"].waypoints[0]
    engine._uuvs["uuv_00"].position_xy = exit_point
    assert engine.complete_boundary_exit("uuv_00", sim_time_s=30)
    assert engine._uuv_is_physically_exposed("uuv_00") is False

    assert engine.begin_boundary_entry(
        "uuv_02",
        region,
        role="active_verifier",
        outgoing_uuv_id="uuv_00",
    )
    assert not engine.complete_boundary_replacement(
        "uuv_02",
        outgoing_uuv_id="uuv_00",
        region=region,
    )
    entry_target = engine._uuvs["uuv_02"].waypoints[0]
    engine._uuvs["uuv_02"].position_xy = entry_target
    assert engine.complete_boundary_replacement(
        "uuv_02",
        outgoing_uuv_id="uuv_00",
        region=region,
        observation_ids=("obs:replacement:1",),
    )

    updated = controller.snapshot().regions[0]
    assert updated.active_scan_uuv_ids == ("uuv_02",)
    assert controller.snapshot().uuv_modes["uuv_02"] is UUVMissionMode.ACTIVE_SCAN
    assert engine._sensor_modes["uuv_02"] == "active"


def test_engine_waypoint_projection_uses_region_geometry_and_scoped_history() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=23)
    for uuv_id, position in (
        ("uuv_00", (-8_000.0, -6_500.0)),
        ("uuv_01", (-8_000.0, -5_700.0)),
        ("uuv_02", (-6_000.0, -6_500.0)),
        ("uuv_03", (-6_000.0, -5_700.0)),
    ):
        engine._deployment_states[uuv_id] = DeploymentState.DEPLOYED
        engine._waterborne_uuv_ids.add(uuv_id)
        engine._uuvs[uuv_id].position_xy = position

    first_region = ExecutionRegion(
        region_id="target_00:task:01",
        target_id="target_00",
        slot_index=0,
        execution_revision=1,
        prediction_id="pred:target_00:1",
        geometry=(
            (-9_000.0, -8_000.0),
            (-6_000.0, -8_000.0),
            (-6_000.0, -5_000.0),
            (-9_000.0, -5_000.0),
        ),
        centerline_indices=(0,),
        start_s=0.0,
        end_s=540.0,
        geometry_revision=1,
        task_group_id="target_00:task-group:01",
        evidence_ids=("evidence:global-track",),
    )
    first_group = TaskGroupAssignment(
        task_group_id="target_00:task-group:01",
        target_id="target_00",
        region_id=first_region.region_id,
        execution_revision=1,
        member_uuv_ids=("uuv_00", "uuv_01"),
        active_verifier_uuv_id="uuv_00",
        passive_tracker_uuv_id="uuv_01",
        evidence_ids=("evidence:global-track",),
    )

    first = engine.plan_task_group_waypoints(first_group, first_region, max_turn_delta_rad=pi)
    second = engine.plan_task_group_waypoints(
        first_group,
        first_region,
        target_position_xy=(-7_000.0, -6_000.0),
        max_turn_delta_rad=pi,
    )

    assert first.focus_xy == (-7_500.0, -6_500.0)
    assert second.focus_xy == (-7_000.0, -6_000.0)
    assert engine.task_group_waypoint_cache_keys() == (first.cache_key,)
    assert first.cache_key == ("target_00:task-group:01", "target_00:task:01")


def test_uuv_only_task_group_plan_cannot_start_carrier_service() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=29, mission_controller=controller)
    assignments = []
    for slot in range(1, 5):
        x_min = -9_000.0 + (slot - 1) * 3_000.0
        assignments.append(
            RegionMissionState(
                region_id=f"target_00:task:{slot:02d}",
                target_id="target_00",
                task_group_id=f"target_00:task-group:{slot:02d}",
                region_polygon=(
                    (x_min, -9_000.0),
                    (x_min + 3_000.0, -9_000.0),
                    (x_min + 3_000.0, -3_000.0),
                    (x_min, -3_000.0),
                ),
                scan_waypoints=((x_min + 1_500.0, -6_000.0),),
            )
        )
    groups = tuple(
        TaskGroupAssignment(
            task_group_id=f"target_00:task-group:{slot:02d}",
            target_id="target_00",
            region_id=f"target_00:task:{slot:02d}",
            execution_revision=1,
            member_uuv_ids=(f"uuv_{(slot - 1) * 2:02d}", f"uuv_{(slot - 1) * 2 + 1:02d}"),
            active_verifier_uuv_id=f"uuv_{(slot - 1) * 2:02d}",
            passive_tracker_uuv_id=f"uuv_{(slot - 1) * 2 + 1:02d}",
            evidence_ids=(f"evidence:group:{slot}",),
        )
        for slot in range(1, 5)
    )
    plan = ExecutableMissionPlan(
        revision=1,
        region_assignments=tuple(assignments),
        task_groups=groups,
        reserve_uuvs=tuple(ReserveUUVState(uuv_id=f"uuv_{index:02d}") for index in range(8, 12)),
    )

    assert engine.apply_verified_mission_plan(plan)
    assert engine._mission_stop_ids == {}
    assert not any(
        event.event_type.startswith("carrier_")
        for event in engine.events()
    )
    assert all(
        event.event_type not in {"uuv_deployed", "uuv_recovery_started", "uuv_recovered"}
        for event in engine.events()
    )
    frame = engine.step()
    assert frame["sim_time_s"] == 5
