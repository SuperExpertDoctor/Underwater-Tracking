from __future__ import annotations

import pytest
from pydantic import ValidationError

from underwater_tracking.domain.mission_adapters import legacy_frame_to_uuv_view
from underwater_tracking.domain.mission_models import (
    CarrierExecutionMode,
    CarrierMissionModel,
    CarrierRouteStatus,
    PredictionGrid,
    PredictionGridCell,
    RegionLifecycle,
    RegionMissionState,
    UUVMissionMode,
    validate_region_transition,
)


def _cell() -> PredictionGridCell:
    return PredictionGridCell(
        target_id="T1",
        revision=4,
        grid_x=2,
        grid_y=-1,
        min_x=-100.0,
        max_x=900.0,
        min_y=-600.0,
        max_y=400.0,
        cell_size_m=1000.0,
        probability=0.8,
        first_entry_s=30,
        last_exit_s=90,
        imm_model_probabilities={"CV": 0.6, "CT": 0.4},
        covariance_summary=(100.0, 100.0, 0.0),
        intent_label="transit",
        intent_confidence=0.8,
    )


def test_uuv_mode_and_region_lifecycle_are_closed_sets() -> None:
    assert UUVMissionMode.ACTIVE_SCAN.value == "ACTIVE_SCAN"
    assert RegionLifecycle.HANDOFF_PENDING.value == "HANDOFF_PENDING"


def test_carrier_execution_and_route_states_include_moving_rendezvous() -> None:
    assert CarrierExecutionMode.FORMATION_FOLLOW.value == "FORMATION_FOLLOW"
    assert CarrierExecutionMode.MISSION_ROUTE.value == "MISSION_ROUTE"
    assert CarrierExecutionMode.RENDEZVOUS_RETURN.value == "RENDEZVOUS_RETURN"
    mission = CarrierMissionModel(
        carrier_id="carrier_01",
        home_battle_group_id="home",
        route_status=CarrierRouteStatus.RENDEZVOUS_BLOCKED,
    )
    assert mission.route_status is CarrierRouteStatus.RENDEZVOUS_BLOCKED


def test_prediction_grid_ids_are_stable_within_revision() -> None:
    cell = _cell()
    grid = PredictionGrid(
        target_id="T1",
        revision=4,
        origin=(0.0, 0.0),
        cell_size_m=1000.0,
        cells=(cell,),
        centerline_region_ids=(cell.region_id,),
    )

    assert cell.region_id == "T1:r4:cell:2:-1"
    assert grid.cell(2, -1) == cell


def test_prediction_grid_rejects_non_square_cell() -> None:
    with pytest.raises(ValidationError, match="square"):
        payload = _cell().model_dump()
        payload["max_y"] = 300.0
        PredictionGridCell(**payload)


def test_region_transitions_allow_only_declared_lifecycle_edges() -> None:
    assert validate_region_transition(RegionLifecycle.PLANNED, RegionLifecycle.CARRIER_DEPLOYING)
    assert validate_region_transition(RegionLifecycle.ACTIVE_SCAN, RegionLifecycle.PASSIVE_TRACK)
    assert not validate_region_transition(RegionLifecycle.RECOVERED, RegionLifecycle.PASSIVE_TRACK)


def test_region_state_rejects_uuv_in_two_active_modes() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        RegionMissionState(
            region_id="T1:r4:cell:2:-1",
            target_id="T1",
            lifecycle=RegionLifecycle.ACTIVE_SCAN,
            active_scan_uuv_ids=("U1",),
            passive_track_uuv_ids=("U1",),
        )


def test_carrier_inventory_counts_are_derived_from_disjoint_sets() -> None:
    mission = CarrierMissionModel(
        carrier_id="carrier_01",
        home_battle_group_id="home",
        route_status="RETURNING_TO_FLEET",
        onboard_uuv_ids=("U1", "U2"),
        ready_uuv_ids=("U3",),
        reserved_uuv_ids=("U4",),
        recoverable_uuv_ids=("U5",),
    )

    assert mission.total_uuv_capacity == 5
    assert mission.ready_uuv_count == 1
    assert mission.reserved_uuv_count == 1


def test_legacy_usv_fields_are_ignored_but_new_view_has_none() -> None:
    view = legacy_frame_to_uuv_view({"uuvs": [], "usvs": [{"usv_id": "USV1"}]})
    assert view["uuvs"] == []
    assert "usvs" not in view
