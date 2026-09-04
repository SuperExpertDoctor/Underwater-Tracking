from __future__ import annotations

import pytest
from pydantic import ValidationError

from underwater_tracking.domain.mission_adapters import legacy_frame_to_uuv_view
from underwater_tracking.domain.mission_models import (
    AcceptedHandoffObservation,
    CarrierExecutionMode,
    CarrierMissionModel,
    CarrierRouteStatus,
    HandoffEvidence,
    MissionSnapshot,
    PredictionGrid,
    PredictionGridCell,
    RegionLifecycle,
    RegionMissionState,
    UUVMissionMode,
    validate_region_transition,
)
from underwater_tracking.domain.execution_models import (
    GroupSensorMode,
    TaskGroupInstance,
    TaskGroupLifecycle,
    TrackingControlState,
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


def _handoff_evidence(**updates: object) -> HandoffEvidence:
    values: dict[str, object] = {
        "predecessor_region_id": "R1",
        "successor_region_id": "R2",
        "plan_revision": 3,
        "observation_cycle_s": 60,
        "required_uuv_ids": ("U4", "U5"),
        "deployed_uuv_ids": ("U4", "U5"),
        "healthy_uuv_ids": ("U4", "U5"),
        "passive_mode_uuv_ids": ("U4", "U5"),
        "accepted_observations": (
            AcceptedHandoffObservation(
                observation_id="obs-u4",
                observer_uuv_id="U4",
                observed_at_s=60,
            ),
            AcceptedHandoffObservation(
                observation_id="obs-u5",
                observer_uuv_id="U5",
                observed_at_s=60,
            ),
        ),
    }
    values.update(updates)
    return HandoffEvidence(**values)


def test_handoff_evidence_rejects_duplicate_and_foreign_observers() -> None:
    with pytest.raises(ValidationError, match="observation IDs"):
        _handoff_evidence(
            accepted_observations=(
                AcceptedHandoffObservation(
                    observation_id="duplicate",
                    observer_uuv_id="U4",
                    observed_at_s=60,
                ),
                AcceptedHandoffObservation(
                    observation_id="duplicate",
                    observer_uuv_id="U5",
                    observed_at_s=60,
                ),
            )
        )

    with pytest.raises(ValidationError, match="required passive"):
        _handoff_evidence(
            accepted_observations=(
                AcceptedHandoffObservation(
                    observation_id="foreign",
                    observer_uuv_id="U9",
                    observed_at_s=60,
                ),
            )
        )


def test_handoff_evidence_requires_current_cycle_for_each_observation() -> None:
    with pytest.raises(ValidationError, match="observation cycle"):
        _handoff_evidence(
            accepted_observations=(
                AcceptedHandoffObservation(
                    observation_id="stale",
                    observer_uuv_id="U4",
                    observed_at_s=30,
                ),
            )
        )


def test_blocked_handoff_evidence_is_never_complete() -> None:
    assert not _handoff_evidence(blocked_reason="successor_unavailable").is_complete(
        group_min_size=2
    )
    assert not _handoff_evidence(
        accepted_observations=(
            AcceptedHandoffObservation(
                observation_id="only-one",
                observer_uuv_id="U4",
                observed_at_s=60,
            ),
        )
    ).is_complete(group_min_size=2)


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


def test_mission_snapshot_mapping_defaults_use_independent_factories() -> None:
    mapping_fields = (
        "pending_region_revisions",
        "uuv_modes",
        "uuv_resources",
        "resource_episode_by_uuv",
        "dedicated_target_by_uuv",
        "carrier_missions",
    )

    for field_name in mapping_fields:
        assert MissionSnapshot.model_fields[field_name].default_factory is dict

    first = MissionSnapshot(scenario_id="S1", sim_time_s=0, plan_revision=0)
    second = MissionSnapshot(scenario_id="S1", sim_time_s=0, plan_revision=0)
    first.uuv_modes["U1"] = UUVMissionMode.PASSIVE_TRACK

    assert second.uuv_modes == {}


def test_mission_snapshot_rejects_non_passive_dedicated_owner() -> None:
    owner = TaskGroupInstance(
        group_instance_id="T1:task:01:deploy:000001",
        target_id="T1",
        region_id="T1:task:01",
        deployment_revision=1,
        member_uuv_ids=("U1", "U2", "U3"),
        lifecycle=TaskGroupLifecycle.ENTERING,
        sensor_mode=GroupSensorMode.ACTIVE,
        ownership_status="owner",
        reason="initial_deployment",
        evidence_ids=("plan:1",),
    )

    with pytest.raises(ValidationError, match="owner"):
        MissionSnapshot(
            scenario_id="S1",
            sim_time_s=0,
            plan_revision=1,
            task_groups=(owner,),
            tracking_control=TrackingControlState(
                mode="dedicated",
                tracking_owner_group_id=owner.group_instance_id,
            ),
        )


def test_mission_snapshot_rejects_candidate_owner_reference() -> None:
    candidate = TaskGroupInstance(
        group_instance_id="T1:task:01:deploy:000001",
        target_id="T1",
        region_id="T1:task:01",
        deployment_revision=1,
        member_uuv_ids=("U1", "U2", "U3"),
        lifecycle=TaskGroupLifecycle.ENTERING,
        sensor_mode=GroupSensorMode.ACTIVE,
        ownership_status="candidate",
        reason="initial_deployment",
        evidence_ids=("plan:1",),
    )

    with pytest.raises(ValidationError, match="current passive owner"):
        MissionSnapshot(
            scenario_id="S1",
            sim_time_s=0,
            plan_revision=1,
            task_groups=(candidate,),
            tracking_control=TrackingControlState(
                tracking_owner_group_id=candidate.group_instance_id,
            ),
        )


def test_mission_snapshot_rejects_orphan_pending_successor() -> None:
    candidate = TaskGroupInstance(
        group_instance_id="T1:task:01:deploy:000001",
        target_id="T1",
        region_id="T1:task:01",
        deployment_revision=1,
        member_uuv_ids=("U1", "U2", "U3"),
        lifecycle=TaskGroupLifecycle.ENTERING,
        sensor_mode=GroupSensorMode.ACTIVE,
        ownership_status="candidate",
        reason="initial_deployment",
        evidence_ids=("plan:1",),
    )

    with pytest.raises(ValidationError, match="requires a tracking owner"):
        MissionSnapshot(
            scenario_id="S1",
            sim_time_s=0,
            plan_revision=1,
            task_groups=(candidate,),
            tracking_control=TrackingControlState(
                pending_successor_group_id=candidate.group_instance_id,
            ),
        )
