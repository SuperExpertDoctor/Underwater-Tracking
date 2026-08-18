from __future__ import annotations

from underwater_tracking.domain.platforms import (
    CarrierPlatformState,
    CommunicationCapability,
    MotionLimits,
    PlatformCapability,
    PlatformKind,
    PlatformRoster,
    SonarCapability,
    UUVPlatformState,
    USVPlatformState,
)
from underwater_tracking.domain.regional_models import (
    CommunicationRequirement,
    GridSpec,
    RegionCell,
    RegionTask,
    SonarPolicy,
    TargetRegionPlan,
    TimeWindow,
)
from underwater_tracking.planning.regional_validation import validate_regional_plan


def _capability(kind: PlatformKind, *, active: bool = True) -> PlatformCapability:
    return PlatformCapability(
        kind=kind,
        motion=MotionLimits(
            max_speed_mps=10.0,
            max_acceleration_mps2=1.0,
            max_turn_rate_rad_s=1.0,
        ),
        sonar=SonarCapability(
            passive_range_m=2_000.0,
            passive_bearing_variance_rad2=0.1,
            active_source_range_m=1_000.0,
            active_receive_range_m=1_000.0,
            active_range_sigma_m=5.0,
            active_bearing_sigma_rad=0.1,
            active_capable=active,
            ping_cooldown_s=10,
            ping_energy_cost_fraction=0.1,
            clutter_sensitivity=0.1,
            exposure_cost=0.1,
        ),
        communications=CommunicationCapability(
            surface_range_m=2_000.0,
            acoustic_range_m=1_000.0,
        ),
    )


def _roster(
    *,
    uuv_ids: tuple[str, ...] = ("uuv-1", "uuv-2"),
    usv_ids: tuple[str, ...] = ("usv-1",),
    active_uuv: bool = True,
) -> PlatformRoster:
    uuvs = tuple(
        UUVPlatformState(
            platform_id=platform_id,
            platform_index=index,
            position_xy=(0.0, 0.0),
            heading_rad=0.0,
            speed_mps=2.0,
            energy_fraction=0.9,
            deployment_state="deployed",
            capability=_capability(PlatformKind.UUV, active=active_uuv),
            master_connected=True,
        )
        for index, platform_id in enumerate(uuv_ids)
    )
    usvs = tuple(
        USVPlatformState(
            platform_id=platform_id,
            platform_index=index,
            position_xy=(0.0, 0.0),
            heading_rad=0.0,
            speed_mps=2.0,
            energy_fraction=0.9,
            deployment_state="deployed",
            capability=_capability(PlatformKind.USV),
            distance_to_carrier_m=0.0,
        )
        for index, platform_id in enumerate(usv_ids)
    )
    return PlatformRoster(usvs=usvs, uuvs=uuvs)


def _carrier(radius: float = 500.0) -> CarrierPlatformState:
    return CarrierPlatformState(
        carrier_id="carrier-1",
        position_xy=(0.0, 0.0),
        heading_rad=0.0,
        speed_mps=0.0,
        support_radius_m=radius,
        onboard_platform_ids=(),
        deployed_platform_ids=("uuv-1", "uuv-2", "usv-1"),
        returning_platform_ids=(),
    )


def _plan(
    *,
    tasks: tuple[RegionTask, ...] = (),
    cell_count: int = 1,
) -> TargetRegionPlan:
    cells = tuple(
        RegionCell(
            region_id=f"T1:cell:{index}:0",
            target_id="T1",
            grid_x=index,
            grid_y=0,
            min_x=index * 100.0,
            max_x=(index + 1) * 100.0,
            min_y=0.0,
            max_y=100.0,
            center_xy=(index * 100.0 + 50.0, 50.0),
            cell_size_m=100.0,
            first_entry_s=0,
            last_exit_s=100,
            visit_windows=(TimeWindow(start_s=0, end_s=100),),
        )
        for index in range(cell_count)
    )
    if not tasks:
        tasks = tuple(
            RegionTask(
                region_id=cell.region_id,
                target_id="T1",
                active_window=TimeWindow(start_s=0, end_s=100),
                required_uuv_count=2,
                uuv_roles=("passive_tracker", "handoff_reserve"),
                required_usv_count=1,
                usv_role="surface_relay",
                communication=CommunicationRequirement(),
            )
            for cell in cells
        )
    return TargetRegionPlan(
        target_id="T1",
        grid_spec=GridSpec(),
        cell_size_m=100.0,
        cells=cells,
        tasks=tasks,
        prediction_id="prediction-1",
        intent_label="patrol",
        intent_confidence=0.9,
    )


def test_far_usv_is_rejected() -> None:
    task = RegionTask(
        region_id="T1:cell:0:0",
        target_id="T1",
        active_window=TimeWindow(start_s=0, end_s=100),
        required_uuv_count=2,
        uuv_roles=("passive_tracker", "handoff_reserve"),
        required_usv_count=1,
        usv_role="surface_relay",
        assigned_uuv_ids=("uuv-1", "uuv-2"),
        assigned_usv_ids=("usv-1",),
        assignment_status="active",
    )
    far_usv = _roster().model_copy(
        update={
            "usvs": (
                _roster().usvs[0].model_copy(update={"position_xy": (2_000.0, 0.0)}),
            )
        }
    )
    issues = validate_regional_plan(
        _plan(tasks=(task,)),
        far_usv,
        carrier=_carrier(radius=500.0),
    )
    assert "usv_outside_carrier_radius" in issues


def test_active_sonar_requires_capability() -> None:
    task = RegionTask(
        region_id="T1:cell:0:0",
        target_id="T1",
        active_window=TimeWindow(start_s=0, end_s=100),
        required_uuv_count=2,
        uuv_roles=("passive_tracker", "active_verifier"),
        sonar_policy=SonarPolicy(
            passive_required=True,
            active_allowed=True,
            active_mode="probe",
        ),
        assigned_uuv_ids=("uuv-1", "uuv-2"),
        assignment_status="active",
    )
    issues = validate_regional_plan(
        _plan(tasks=(task,)),
        _roster(active_uuv=False),
    )
    assert "active_sonar_not_supported" in issues


def test_double_booking_is_rejected_but_adjacent_relay_overlap_is_allowed() -> None:
    first = _plan().tasks[0].model_copy(
        update={
            "assigned_uuv_ids": ("uuv-1", "uuv-2"),
            "assigned_usv_ids": ("usv-1",),
            "assignment_status": "active",
        }
    )
    second = _plan(cell_count=2).tasks[1].model_copy(
        update={
            "assigned_uuv_ids": ("uuv-1", "uuv-2"),
            "assigned_usv_ids": ("usv-1",),
            "assignment_status": "active",
        }
    )
    issues = validate_regional_plan(
        _plan(tasks=(first, second), cell_count=2),
        _roster(),
    )
    assert "uuv_double_booked" in issues
    assert "usv_double_booked" not in issues
