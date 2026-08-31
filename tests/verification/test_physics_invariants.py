from math import pi

from underwater_tracking.verification.physics_invariants import (
    EntityMotionLimits,
    PhysicsInvariantMonitor,
)


def _limits() -> EntityMotionLimits:
    return EntityMotionLimits(
        max_speed_mps=4.0,
        max_acceleration_mps2=1.0,
        max_deceleration_mps2=1.0,
        max_turn_rate_rad_s=0.5,
    )


def _frame(frame_id: int, sim_time_s: int, x: float, *, lifecycle: str = "deployed") -> dict[str, object]:
    return {
        "frame_id": frame_id,
        "sim_time_s": sim_time_s,
        "entities": [
            {
                "entity_id": "uuv_00",
                "entity_kind": "uuv",
                "position_xy": (x, 0.0),
                "speed_mps": 2.0,
                "heading_rad": 0.0,
                "lifecycle_state": lifecycle,
            }
        ],
    }


def test_unexplained_uuv_position_jump_is_a_teleport() -> None:
    monitor = PhysicsInvariantMonitor({"uuv_00": _limits()})
    monitor.observe(_frame(0, 0, 0.0, lifecycle="onboard"))
    monitor.observe(_frame(1, 5, 5000.0))
    audit = monitor.result("uuv_00")
    assert audit.observed_steps == 1
    assert audit.teleport_count == 1
    assert audit.limit_violation_count > 0


def test_deployment_event_allows_only_small_launch_transition() -> None:
    monitor = PhysicsInvariantMonitor({"uuv_00": _limits()})
    monitor.observe(_frame(0, 0, 0.0, lifecycle="onboard"))
    monitor.observe(
        _frame(1, 5, 20.0),
        events=(
            {
                "event_type": "uuv_deployed",
                "payload": {"uuv_id": "uuv_00"},
            },
        ),
    )
    assert monitor.result("uuv_00").teleport_count == 0


def test_onboard_uuv_parent_transport_uses_deployment_state_and_skips_self_motion_limits() -> None:
    limits = EntityMotionLimits(
        max_speed_mps=4.0,
        max_acceleration_mps2=0.1,
        max_deceleration_mps2=0.1,
        max_turn_rate_rad_s=0.05,
    )
    monitor = PhysicsInvariantMonitor({"uuv_00": limits})
    monitor.observe(
        {
            "frame_id": 0,
            "sim_time_s": 0,
            "entities": [
                {
                    "entity_id": "uuv_00",
                    "entity_kind": "uuv",
                    "position_xy": (0.0, 0.0),
                    "speed_mps": 0.0,
                    "heading_rad": 0.0,
                    "deployment_state": "onboard",
                }
            ],
        }
    )
    monitor.observe(
        {
            "frame_id": 1,
            "sim_time_s": 5,
            "entities": [
                {
                    "entity_id": "uuv_00",
                    "entity_kind": "uuv",
                    "position_xy": (40.0, 0.0),
                    "speed_mps": 0.0,
                    "heading_rad": 1.5707963267948966,
                    "deployment_state": "onboard",
                }
            ],
        }
    )

    audit = monitor.result("uuv_00")
    assert audit.teleport_count == 0
    assert audit.limit_violation_count == 0
    assert audit.max_acceleration_mps2 == 0.0
    assert audit.max_turn_rate_rad_s == 0.0


def test_onboard_uuv_must_match_its_owner_motion_state() -> None:
    monitor = PhysicsInvariantMonitor({"uuv_00": _limits()})
    monitor.observe(
        {
            "frame_id": 0,
            "sim_time_s": 0,
            "entities": [
                {
                    "entity_id": "mother_ship_01",
                    "entity_kind": "mother_ship",
                    "position_xy": (40.0, 20.0),
                    "speed_mps": 5.0,
                    "heading_rad": 0.5,
                },
                {
                    "entity_id": "uuv_00",
                    "entity_kind": "uuv",
                    "position_xy": (0.0, 0.0),
                    "speed_mps": 0.0,
                    "heading_rad": 0.0,
                    "deployment_state": "onboard",
                    "owner_id": "mother_ship_01",
                },
            ],
        }
    )

    audit = monitor.result("uuv_00")
    assert audit.owner_colocation_violation_count == 1
    assert audit.limit_violation_count == 1


def test_route_and_formation_deviation_are_audited() -> None:
    monitor = PhysicsInvariantMonitor({"mother_ship_01": _limits()})
    monitor.observe(
        {
            "frame_id": 0,
            "sim_time_s": 0,
            "entities": [
                {
                    "entity_id": "mother_ship_01",
                    "entity_kind": "mother_ship",
                    "position_xy": (0.0, 0.0),
                    "speed_mps": 2.0,
                    "heading_rad": 0.0,
                    "route_deviation_m": 51.0,
                    "route_tolerance_m": 50.0,
                    "formation_error_m": 101.0,
                    "formation_tolerance_m": 100.0,
                }
            ],
        }
    )

    audit = monitor.result("mother_ship_01")
    assert audit.route_violation_count == 1
    assert audit.formation_violation_count == 1
    assert audit.limit_violation_count == 2


def test_uuv_mileage_and_energy_reserves_are_audited() -> None:
    monitor = PhysicsInvariantMonitor({"uuv_00": _limits()})
    monitor.observe(
        {
            "frame_id": 0,
            "sim_time_s": 0,
            "entities": [
                {
                    "entity_id": "uuv_00",
                    "entity_kind": "uuv",
                    "position_xy": (0.0, 0.0),
                    "speed_mps": 0.0,
                    "heading_rad": 0.0,
                    "deployment_state": "deployed",
                    "mileage_m": 50_001.0,
                    "max_mileage_m": 50_000.0,
                    "energy_fraction": 0.09,
                    "min_energy_fraction": 0.10,
                }
            ],
        }
    )

    audit = monitor.result("uuv_00")
    assert audit.resource_violation_count == 2
    assert audit.limit_violation_count == 2


def test_uuv_deployment_handoff_does_not_charge_carrier_derivatives_to_uuv() -> None:
    limits = EntityMotionLimits(
        max_speed_mps=4.0,
        max_acceleration_mps2=0.1,
        max_deceleration_mps2=0.1,
        max_turn_rate_rad_s=0.05,
    )
    monitor = PhysicsInvariantMonitor({"uuv_00": limits})
    monitor.observe(_frame(0, 0, 0.0, lifecycle="onboard"))
    deployed = _frame(1, 5, 40.0, lifecycle="deployed")
    deployed["entities"][0].update(speed_mps=4.0, heading_rad=pi / 2.0)
    monitor.observe(
        deployed,
        events=(
            {
                "event_type": "uuv_deployed",
                "payload": {"uuv_id": "uuv_00"},
            },
        ),
    )

    audit = monitor.result("uuv_00")
    assert audit.teleport_count == 0
    assert audit.limit_violation_count == 0
    assert audit.max_acceleration_mps2 == 0.0
    assert audit.max_turn_rate_rad_s == 0.0


def test_deployment_event_entity_id_allows_handoff_transition() -> None:
    limits = EntityMotionLimits(
        max_speed_mps=4.0,
        max_acceleration_mps2=0.1,
        max_deceleration_mps2=0.1,
        max_turn_rate_rad_s=0.05,
    )
    monitor = PhysicsInvariantMonitor({"uuv_00": limits})
    monitor.observe(_frame(0, 0, 0.0, lifecycle="onboard"))
    deployed = _frame(1, 5, 40.0, lifecycle="deployed")
    deployed["entities"][0].update(speed_mps=4.0, heading_rad=pi / 2.0)
    monitor.observe(
        deployed,
        events=(
            {
                "event_type": "uuv_deployed",
                "entity_id": "uuv_00",
                "payload": {},
            },
        ),
    )

    audit = monitor.result("uuv_00")
    assert audit.teleport_count == 0
    assert audit.limit_violation_count == 0


def test_boundary_and_depth_limits_are_reported() -> None:
    limits = EntityMotionLimits(
        max_speed_mps=4.0,
        max_acceleration_mps2=1.0,
        max_deceleration_mps2=1.0,
        max_turn_rate_rad_s=0.5,
        min_depth_m=100.0,
        max_depth_m=200.0,
        max_vertical_speed_mps=1.0,
        max_vertical_acceleration_mps2=0.2,
        max_pitch_rad=0.2,
    )
    monitor = PhysicsInvariantMonitor(
        {"target_00": limits},
        bounds_by_entity={"target_00": (0.0, 100.0, 0.0, 100.0)},
    )
    monitor.observe(
        {
            "frame_id": 0,
            "sim_time_s": 0,
            "entities": [
                {
                    "entity_id": "target_00",
                    "entity_kind": "submarine",
                    "position_xy": (50.0, 50.0),
                    "speed_mps": 2.0,
                    "heading_rad": 0.0,
                    "depth_m": 100.0,
                    "vertical_speed_mps": 0.0,
                }
            ],
        }
    )
    monitor.observe(
        {
            "frame_id": 1,
            "sim_time_s": 5,
            "entities": [
                {
                    "entity_id": "target_00",
                    "entity_kind": "submarine",
                    "position_xy": (150.0, 50.0),
                    "speed_mps": 2.0,
                    "heading_rad": 0.0,
                    "depth_m": 300.0,
                    "vertical_speed_mps": 2.0,
                }
            ],
        }
    )
    audit = monitor.result("target_00")
    assert audit.boundary_violation_count == 1
    assert audit.limit_violation_count > 0
