from __future__ import annotations

from underwater_tracking.domain.platforms import (
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
    GridSpec,
    RegionCell,
    RegionTask,
    TargetRegionPlan,
    TimeWindow,
)
from underwater_tracking.planning.regional_allocation import allocate_regional_tasks


def _capability(kind: PlatformKind) -> PlatformCapability:
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
            active_capable=True,
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


def _roster(uuv_count: int) -> PlatformRoster:
    uuvs = tuple(
        UUVPlatformState(
            platform_id=f"uuv-{index}",
            platform_index=index,
            position_xy=(0.0, 0.0),
            heading_rad=0.0,
            speed_mps=2.0,
            energy_fraction=0.9,
            deployment_state="deployed",
            capability=_capability(PlatformKind.UUV),
            master_connected=True,
        )
        for index in range(uuv_count)
    )
    usv = USVPlatformState(
        platform_id="usv-1",
        platform_index=0,
        position_xy=(0.0, 0.0),
        heading_rad=0.0,
        speed_mps=2.0,
        energy_fraction=0.9,
        deployment_state="deployed",
        capability=_capability(PlatformKind.USV),
        distance_to_carrier_m=0.0,
    )
    return PlatformRoster(usvs=(usv,), uuvs=uuvs)


def _plan() -> TargetRegionPlan:
    cell = RegionCell(
        region_id="T1:cell:0:0",
        target_id="T1",
        grid_x=0,
        grid_y=0,
        min_x=0.0,
        max_x=100.0,
        min_y=0.0,
        max_y=100.0,
        center_xy=(50.0, 50.0),
        cell_size_m=100.0,
        first_entry_s=0,
        last_exit_s=100,
        visit_windows=(TimeWindow(start_s=0, end_s=100),),
    )
    task = RegionTask(
        region_id=cell.region_id,
        target_id="T1",
        active_window=TimeWindow(start_s=0, end_s=100),
        required_uuv_count=2,
        uuv_roles=("passive_tracker", "handoff_reserve"),
        required_usv_count=1,
        usv_role="surface_relay",
        assigned_uuv_ids=("uuv-0", "uuv-1"),
        assigned_usv_ids=("usv-1",),
    )
    return TargetRegionPlan(
        target_id="T1",
        grid_spec=GridSpec(),
        cell_size_m=100.0,
        cells=(cell,),
        tasks=(task,),
        prediction_id="prediction-1",
        intent_label="patrol",
        intent_confidence=0.9,
    )


def test_missing_platform_degrades_region_instead_of_dropping_it() -> None:
    result = allocate_regional_tasks(_plan(), _roster(uuv_count=1))
    task = result.tasks["T1:cell:0:0"]
    assert task.assignment_status == "degraded"
    assert task.assigned_uuv_ids == ("uuv-0",)
    assert task.assigned_usv_ids == ("usv-1",)
    assert task.degraded_reasons


def test_allocation_is_deterministic_and_assigns_roles() -> None:
    first = allocate_regional_tasks(_plan(), _roster(uuv_count=2))
    second = allocate_regional_tasks(_plan(), _roster(uuv_count=2))
    assert first.tasks == second.tasks
    assert first.tasks["T1:cell:0:0"].assignment_status == "active"
    assert first.tasks["T1:cell:0:0"].assigned_uuv_ids == ("uuv-0", "uuv-1")
    assert first.waypoints_by_member["uuv-0"]


def test_allocation_preserves_llm_members_without_filling_advisory_counts() -> None:
    task = _plan().tasks[0].model_copy(
        update={
            "tracking_mode": "uuv_primary_usv_relay",
            "required_uuv_count": 0,
            "required_usv_count": 0,
            "uuv_roles": (),
            "usv_role": "surface_relay",
            "assigned_uuv_ids": ("uuv-0",),
            "assigned_usv_ids": ("usv-1",),
            "assignment_status": "active",
        }
    )
    plan = _plan().model_copy(update={"tasks": (task,)})

    result = allocate_regional_tasks(plan, _roster(uuv_count=2))

    allocated = result.tasks[task.region_id]
    assert allocated.assigned_uuv_ids == ("uuv-0",)
    assert allocated.assigned_usv_ids == ("usv-1",)
    assert "insufficient_uuv" not in result.issues
    assert "insufficient_usv" not in result.issues


def test_empty_llm_membership_is_uncovered_without_automatic_selection() -> None:
    task = _plan().tasks[0].model_copy(
        update={
            "tracking_mode": "heuristic_uuv",
            "required_uuv_count": 0,
            "required_usv_count": 0,
            "uuv_roles": (),
            "usv_role": None,
            "assigned_uuv_ids": (),
            "assigned_usv_ids": (),
            "assignment_status": "planned",
        }
    )

    result = allocate_regional_tasks(
        _plan().model_copy(update={"tasks": (task,)}), _roster(uuv_count=2)
    )

    assert result.tasks[task.region_id].assignment_status == "uncovered"
    assert result.tasks[task.region_id].assigned_uuv_ids == ()
    assert result.tasks[task.region_id].assigned_usv_ids == ()
