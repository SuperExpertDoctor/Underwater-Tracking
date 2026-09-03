from __future__ import annotations

from pydantic import ValidationError
import pytest

from underwater_tracking.domain.execution_models import (
    DeterministicIntentState,
    ExecutionDegradation,
    ExecutionRegion,
    GroupSensorMode,
    GlobalTargetTrackView,
    IMMModelForecast,
    IMMPredictedTrack,
    OperationalExecutionSnapshot,
    ReserveUUVState,
    TaskGroupInstance,
    TaskGroupAssignment,
    TaskGroupLifecycle,
    TrackingControlState,
)


def _snapshot(**updates: object) -> OperationalExecutionSnapshot:
    track = GlobalTargetTrackView(
        target_id="target_00",
        track_revision=7,
        sim_time_s=120,
        position_xy=(10.0, 20.0),
        velocity_xy=(2.0, 1.0),
        heading_rad=0.46,
        acceleration_xy=(0.1, 0.0),
        turn_rate_rad_s=0.01,
        bounded_history=((0.0, 0.0, 0.0), (120.0, 10.0, 20.0)),
        source_event_ids=("target-step-7",),
        freshness_status="fresh",
    )
    branches = tuple(
        IMMModelForecast(
            model_name=name,
            state_mean=(10.0, 20.0, 2.0, 1.0, 0.01),
            state_covariance=tuple(
                tuple(1.0 if row == col else 0.0 for col in range(5))
                for row in range(5)
            ),
            model_probability=probability,
            innovation=(0.1, 0.2),
            likelihood=0.8,
            source_observation_ids=("obs-1",),
        )
        for name, probability in (
            ("CV", 0.5),
            ("CT_LEFT", 0.3),
            ("CT_RIGHT", 0.2),
        )
    )
    prediction = IMMPredictedTrack(
        prediction_id="prediction-7",
        prediction_revision=3,
        target_id="target_00",
        origin_sim_time_s=120,
        times_s=(150.0, 180.0, 210.0, 240.0),
        centerline_xy=((70.0, 50.0), (130.0, 80.0), (190.0, 110.0), (250.0, 140.0)),
        covariance_xy=tuple((4.0, 0.0, 0.0, 4.0) for _ in range(4)),
        corridor_radius_m=(10.0, 11.0, 12.0, 13.0),
        model_branches=branches,
        model_probabilities={"CV": 0.5, "CT_LEFT": 0.3, "CT_RIGHT": 0.2},
        clipping_records=(),
        source_track_revision=7,
        source_observation_ids=("obs-1",),
        prediction_regime="imm",
    )
    intent = DeterministicIntentState(
        target_id="target_00",
        intent_label="transit",
        confidence=0.9,
        intent_revision=4,
        prediction_revision=3,
        rule_version="deterministic-intent-v1",
        features={"mean_speed_mps": 2.2, "curvature": 0.01},
        thresholds={"enter_confidence": 0.7},
        evidence_ids=("target-step-7",),
    )
    regions = tuple(
        ExecutionRegion(
            region_id=f"target_00:task:{index:02d}",
            target_id="target_00",
            slot_index=index,
            execution_revision=9,
            prediction_id="prediction-7",
            geometry=((index * 50.0, 0.0), (index * 50.0 + 40.0, 0.0),
                      (index * 50.0 + 40.0, 40.0), (index * 50.0, 40.0)),
            centerline_indices=(index - 1,),
            start_s=(index - 1) * 450,
            end_s=(index - 1) * 450 + 540,
            geometry_revision=2,
            predecessor_region_id=(
                f"target_00:task:{index - 1:02d}" if index > 1 else None
            ),
            successor_region_id=(
                f"target_00:task:{index + 1:02d}" if index < 4 else None
            ),
            handoff_start_s=(index - 1) * 450 + 450,
            handoff_end_s=(index - 1) * 450 + 540,
            status="active" if index == 1 else "planned",
            task_group_id=f"TG-{index:02d}",
            evidence_ids=(f"region-evidence-{index}",),
        )
        for index in range(1, 5)
    )
    groups = tuple(
        TaskGroupAssignment(
            task_group_id=f"TG-{index:02d}",
            target_id="target_00",
            region_id=f"target_00:task:{index:02d}",
            execution_revision=9,
            member_uuv_ids=(f"uuv_{(index - 1) * 2:02d}", f"uuv_{(index - 1) * 2 + 1:02d}"),
            active_verifier_uuv_id=f"uuv_{(index - 1) * 2:02d}",
            passive_tracker_uuv_id=f"uuv_{(index - 1) * 2 + 1:02d}",
            status="active" if index == 1 else "prepositioning",
            evidence_ids=(f"group-evidence-{index}",),
        )
        for index in range(1, 5)
    )
    values: dict[str, object] = {
        "scenario_id": "S1",
        "target_id": "target_00",
        "execution_revision": 9,
        "source_snapshot_revision": 12,
        "source_sim_time_s": 120,
        "prediction_revision": 3,
        "prediction_id": "prediction-7",
        "intent_revision": 4,
        "expert_request_version": 0,
        "generated_at_s": 120.0,
        "valid_from_s": 120.0,
        "valid_until_s": 1920.0,
        "plan_source": "deterministic",
        "target_track": track,
        "prediction": prediction,
        "intent": intent,
        "regions": regions,
        "task_groups": groups,
        "reserve_uuvs": tuple(
            ReserveUUVState(
                uuv_id=f"uuv_{index:02d}",
                status="reserve",
                priority=index,
                resource_episode=0,
            )
            for index in range(8, 12)
        ),
        "current_region_id": "target_00:task:01",
        "next_region_id": "target_00:task:02",
        "evidence_ids": ("target-step-7", "execution-9"),
        "degradation": ExecutionDegradation(status="nominal"),
    }
    values.update(updates)
    return OperationalExecutionSnapshot(**values)


def test_snapshot_requires_four_stable_regions_and_two_uuv_groups() -> None:
    snapshot = _snapshot()

    assert snapshot.model_config["frozen"] is True
    assert tuple(region.region_id for region in snapshot.regions) == (
        "target_00:task:01",
        "target_00:task:02",
        "target_00:task:03",
        "target_00:task:04",
    )
    assert all(len(group.member_uuv_ids) == 2 for group in snapshot.task_groups)
    assert snapshot.current_region_id in {region.region_id for region in snapshot.regions}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("regions", ()),
        ("regions", _snapshot().regions + (_snapshot().regions[0],)),
        ("current_region_id", "target_00:task:99"),
        ("next_region_id", "target_00:task:99"),
        ("evidence_ids", ()),
    ],
)
def test_snapshot_rejects_invalid_topology_or_empty_evidence(
    field: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        _snapshot(**{field: value})


def test_snapshot_rejects_duplicate_uuv_and_mixed_execution_revision() -> None:
    duplicate_group = _snapshot().task_groups[1].model_copy(
        update={"member_uuv_ids": ("uuv_00", "uuv_03"), "active_verifier_uuv_id": "uuv_02"}
    )
    with pytest.raises(ValidationError, match="UUV"):
        _snapshot(task_groups=(_snapshot().task_groups[0], duplicate_group, *_snapshot().task_groups[2:]))

    mixed_region = _snapshot().regions[0].model_copy(update={"execution_revision": 8})
    with pytest.raises(ValidationError, match="execution_revision"):
        _snapshot(regions=(mixed_region, *_snapshot().regions[1:]))


def test_snapshot_rejects_non_imm_prediction_branch_set_or_mismatched_prediction() -> None:
    fallback = _snapshot(
        prediction=_snapshot().prediction.model_copy(update={"prediction_regime": "bspline"})
    )
    assert fallback.prediction.prediction_regime == "bspline"

    mixed_group = _snapshot().task_groups[0].model_copy(update={"region_id": "target_00:task:02"})
    with pytest.raises(ValidationError, match="region"):
        _snapshot(task_groups=(mixed_group, *_snapshot().task_groups[1:]))


def _instance(
    *,
    slot: int,
    deployment_revision: int = 2,
    lifecycle: TaskGroupLifecycle = TaskGroupLifecycle.ENTERING,
    sensor_mode: GroupSensorMode = GroupSensorMode.ACTIVE,
    ownership_status: str = "candidate",
) -> TaskGroupInstance:
    group_id = f"target_00:task:{slot:02d}:deploy:{deployment_revision:06d}"
    return TaskGroupInstance(
        group_instance_id=group_id,
        target_id="target_00",
        region_id=f"target_00:task:{slot:02d}",
        deployment_revision=deployment_revision,
        member_uuv_ids=(
            f"{group_id}:member:01",
            f"{group_id}:member:02",
            f"{group_id}:member:03",
        ),
        lifecycle=lifecycle,
        sensor_mode=sensor_mode,
        ownership_status=ownership_status,
        reason="initial_deployment",
        evidence_ids=("plan:2",),
    )


def test_execution_region_requires_exact_configured_square() -> None:
    values = _snapshot().regions[0].model_dump()
    values.update(
        {
            "center": (1_000.0, 2_000.0),
            "side_length_m": 2_000.0,
            "geometry": (
                (0.0, 1_000.0),
                (2_000.0, 1_000.0),
                (2_000.0, 3_000.0),
                (0.0, 3_000.0),
            ),
        }
    )
    region = ExecutionRegion.model_validate(values)
    assert region.center == (1_000.0, 2_000.0)
    assert region.side_length_m == 2_000.0

    values["geometry"] = (
        (0.0, 1_000.0),
        (2_000.0, 1_000.0),
        (1_800.0, 3_000.0),
        (0.0, 3_000.0),
    )
    with pytest.raises(ValidationError, match="square"):
        ExecutionRegion.model_validate(values)


def test_execution_region_rejects_bow_tie_corner_order() -> None:
    values = _snapshot().regions[0].model_dump()
    values.update(
        {
            "center": (1_000.0, 2_000.0),
            "side_length_m": 2_000.0,
            "geometry": (
                (0.0, 1_000.0),
                (2_000.0, 3_000.0),
                (2_000.0, 1_000.0),
                (0.0, 3_000.0),
            ),
        }
    )

    with pytest.raises(ValidationError, match="simple"):
        ExecutionRegion.model_validate(values)


def test_execution_region_normalizes_iterable_geometry_without_derived_fields() -> None:
    values = _snapshot().regions[0].model_dump()
    values.pop("center")
    values.pop("side_length_m")
    values["geometry"] = (
        point
        for point in (
            (0.0, 1_000.0),
            (2_000.0, 1_000.0),
            (2_000.0, 3_000.0),
            (0.0, 3_000.0),
        )
    )

    region = ExecutionRegion.model_validate(values)

    assert region.center == (1_000.0, 2_000.0)
    assert region.side_length_m == 2_000.0


def test_execution_region_reports_short_point_as_validation_error() -> None:
    values = _snapshot().regions[0].model_dump()
    values["geometry"] = (
        (0.0, 1_000.0),
        (2_000.0, 1_000.0),
        (2_000.0,),
        (0.0, 3_000.0),
    )

    with pytest.raises(ValidationError):
        ExecutionRegion.model_validate(values)


def test_execution_region_rejects_bow_tie_square_perimeter() -> None:
    values = _snapshot().regions[0].model_dump()
    values["geometry"] = (
        (50.0, 0.0),
        (90.0, 40.0),
        (90.0, 0.0),
        (50.0, 40.0),
    )

    with pytest.raises(ValidationError, match="simple square perimeter"):
        ExecutionRegion.model_validate(values)


def test_execution_region_reports_malformed_geometry_as_validation_error() -> None:
    values = _snapshot().regions[0].model_dump()
    values["geometry"] = (None, (90.0, 0.0), (90.0, 40.0), (50.0, 40.0))

    with pytest.raises(ValidationError):
        ExecutionRegion.model_validate(values)


def test_task_group_instance_requires_exactly_three_unique_members() -> None:
    group = _instance(slot=1)
    assert len(group.member_uuv_ids) == 3

    with pytest.raises(ValidationError, match="exactly three"):
        TaskGroupInstance.model_validate(
            group.model_dump() | {"member_uuv_ids": group.member_uuv_ids[:2]}
        )
    with pytest.raises(ValidationError, match="exactly three"):
        TaskGroupInstance.model_validate(
            group.model_dump()
            | {"member_uuv_ids": (*group.member_uuv_ids, "duplicate-extra")}
        )
    with pytest.raises(ValidationError, match="unique"):
        TaskGroupInstance.model_validate(
            group.model_dump()
            | {"member_uuv_ids": (group.member_uuv_ids[0],) * 3}
        )


def test_task_group_instance_rejects_empty_member_id_value() -> None:
    group = _instance(slot=1)

    with pytest.raises(ValidationError):
        TaskGroupInstance.model_validate(
            group.model_dump()
            | {"member_uuv_ids": ("", group.member_uuv_ids[1], group.member_uuv_ids[2])}
        )


def test_task_group_instance_requires_target_and_region_identity_consistency() -> None:
    group = _instance(slot=1)

    with pytest.raises(ValidationError, match="region"):
        TaskGroupInstance.model_validate(
            group.model_dump() | {"region_id": "other_target:task:01"}
        )


def test_task_group_instance_rejects_empty_member_ids() -> None:
    group = _instance(slot=1)

    with pytest.raises(ValidationError, match="member IDs must not be empty"):
        TaskGroupInstance.model_validate(
            group.model_dump()
            | {"member_uuv_ids": (" ", *group.member_uuv_ids[1:])}
        )


def test_task_group_instance_region_must_belong_to_target() -> None:
    group = _instance(slot=1)

    with pytest.raises(ValidationError, match="region must belong to its target"):
        TaskGroupInstance.model_validate(
            group.model_dump() | {"target_id": "another-target"}
        )


@pytest.mark.parametrize(
    ("lifecycle", "sensor_mode"),
    [
        (TaskGroupLifecycle.ACTIVE_SCAN, GroupSensorMode.PASSIVE),
        (TaskGroupLifecycle.PASSIVE_TRACK, GroupSensorMode.ACTIVE),
        (TaskGroupLifecycle.DEDICATED_TRACK, GroupSensorMode.ACTIVE),
        (TaskGroupLifecycle.DEDICATED_RELEASE_PENDING, GroupSensorMode.OFF),
        (TaskGroupLifecycle.DISAPPEARED, GroupSensorMode.PASSIVE),
    ],
)
def test_task_group_instance_rejects_invalid_lifecycle_sensor_combinations(
    lifecycle: TaskGroupLifecycle, sensor_mode: GroupSensorMode
) -> None:
    group = _instance(slot=1)
    with pytest.raises(ValidationError, match="sensor mode"):
        TaskGroupInstance.model_validate(
            group.model_dump() | {"lifecycle": lifecycle, "sensor_mode": sensor_mode}
        )


def test_snapshot_accepts_parallel_four_slot_replacement() -> None:
    base = _snapshot()
    groups = tuple(
        _instance(
            slot=slot,
            deployment_revision=2 if phase == "entering" else 1,
            lifecycle=(
                TaskGroupLifecycle.PASSIVE_TRACK
                if slot == 1 and phase == "entering"
                else TaskGroupLifecycle.ENTERING
                if phase == "entering"
                else TaskGroupLifecycle.EXITING
            ),
            sensor_mode=(
                GroupSensorMode.PASSIVE
                if slot == 1 and phase == "entering"
                else GroupSensorMode.ACTIVE
            ),
            ownership_status=(
                "owner" if slot == 1 and phase == "entering" else "candidate"
            ),
        )
        for slot in range(1, 5)
        for phase in ("entering", "exiting")
    )
    snapshot = OperationalExecutionSnapshot.model_validate(
        base.model_dump()
        | {
            "task_groups": groups,
            "tracking_control": TrackingControlState(
                mode="regional",
                tracking_owner_group_id=groups[0].group_instance_id,
            ),
        }
    )
    assert len(snapshot.task_groups) == 8


def test_regional_snapshot_requires_at_least_four_runtime_groups() -> None:
    base = _snapshot()
    groups = tuple(_instance(slot=slot) for slot in range(1, 4))

    with pytest.raises(ValidationError, match="regional execution cardinality requires four to eight"):
        OperationalExecutionSnapshot.model_validate(
            base.model_dump()
            | {
                "task_groups": groups,
                "tracking_control": TrackingControlState(mode="regional"),
            }
        )


def test_dedicated_snapshot_rejects_more_than_five_runtime_groups() -> None:
    base = _snapshot()
    groups = (
        _instance(
            slot=1,
            lifecycle=TaskGroupLifecycle.DEDICATED_TRACK,
            sensor_mode=GroupSensorMode.PASSIVE,
            ownership_status="owner",
        ),
        *(
            _instance(slot=((index - 1) % 4) + 1, deployment_revision=index + 2)
            for index in range(1, 6)
        ),
    )

    with pytest.raises(ValidationError, match="dedicated execution cardinality"):
        OperationalExecutionSnapshot.model_validate(
            base.model_dump()
            | {
                "task_groups": groups,
                "tracking_control": TrackingControlState(
                    mode="dedicated",
                    tracking_owner_group_id=groups[0].group_instance_id,
                ),
            }
        )


@pytest.mark.parametrize(
    "lifecycle",
    [
        TaskGroupLifecycle.PASSIVE_TRACK,
        TaskGroupLifecycle.DEDICATED_TRACK,
        TaskGroupLifecycle.DEDICATED_RELEASE_PENDING,
    ],
)
def test_dedicated_snapshot_accepts_current_passive_owner(
    lifecycle: TaskGroupLifecycle,
) -> None:
    base = _snapshot()
    owner = _instance(
        slot=1,
        lifecycle=lifecycle,
        sensor_mode=GroupSensorMode.PASSIVE,
        ownership_status="owner",
    )

    snapshot = OperationalExecutionSnapshot.model_validate(
        base.model_dump()
        | {
            "task_groups": (owner,),
            "tracking_control": TrackingControlState(
                mode="dedicated",
                tracking_owner_group_id=owner.group_instance_id,
            ),
        }
    )

    assert snapshot.task_groups == (owner,)


@pytest.mark.parametrize(
    "owner",
    [
        _instance(slot=1, ownership_status="owner"),
        _instance(
            slot=1,
            lifecycle=TaskGroupLifecycle.PASSIVE_TRACK,
            sensor_mode=GroupSensorMode.PASSIVE,
            ownership_status="candidate",
        ),
    ],
)
def test_dedicated_snapshot_requires_current_passive_owner(
    owner: TaskGroupInstance,
) -> None:
    base = _snapshot()

    with pytest.raises(ValidationError, match="current passive owner"):
        OperationalExecutionSnapshot.model_validate(
            base.model_dump()
            | {
                "task_groups": (owner,),
                "tracking_control": TrackingControlState(
                    mode="dedicated",
                    tracking_owner_group_id=owner.group_instance_id,
                ),
            }
        )


def test_snapshot_rejects_unknown_pending_successor() -> None:
    base = _snapshot()
    groups = tuple(_instance(slot=slot) for slot in range(1, 5))

    with pytest.raises(ValidationError, match="pending successor"):
        OperationalExecutionSnapshot.model_validate(
            base.model_dump()
            | {
                "task_groups": groups,
                "tracking_control": TrackingControlState(
                    pending_successor_group_id="missing-group"
                ),
            }
        )


def test_snapshot_rejects_unknown_or_duplicate_tracking_owner() -> None:
    base = _snapshot()
    groups = tuple(_instance(slot=slot) for slot in range(1, 5))
    with pytest.raises(ValidationError, match="owner"):
        OperationalExecutionSnapshot.model_validate(
            base.model_dump()
            | {
                "task_groups": groups,
                "tracking_control": TrackingControlState(
                    tracking_owner_group_id="missing-group"
                ),
            }
        )

    owned_groups = tuple(
        group.model_copy(update={"ownership_status": "owner"}) for group in groups[:2]
    ) + groups[2:]
    with pytest.raises(ValidationError, match="owner"):
        OperationalExecutionSnapshot.model_validate(
            base.model_dump()
            | {
                "task_groups": owned_groups,
                "tracking_control": TrackingControlState(
                    tracking_owner_group_id=owned_groups[0].group_instance_id
                ),
            }
        )


def test_snapshot_rejects_invalid_regional_tracking_owner() -> None:
    base = _snapshot()
    groups = (
        _instance(
            slot=1,
            lifecycle=TaskGroupLifecycle.ACTIVE_SCAN,
            sensor_mode=GroupSensorMode.ACTIVE,
            ownership_status="owner",
        ),
        *(_instance(slot=slot) for slot in range(2, 5)),
    )

    with pytest.raises(ValidationError, match="owner"):
        OperationalExecutionSnapshot.model_validate(
            base.model_dump()
            | {
                "task_groups": groups,
                "tracking_control": TrackingControlState(
                    tracking_owner_group_id=groups[0].group_instance_id
                ),
            }
        )


@pytest.mark.parametrize(
    ("lifecycle", "sensor_mode"),
    [
        (TaskGroupLifecycle.DEDICATED_TRACK, GroupSensorMode.ACTIVE),
    ],
)
def test_snapshot_rejects_invalid_dedicated_tracking_owner(
    lifecycle: TaskGroupLifecycle, sensor_mode: GroupSensorMode
) -> None:
    base = _snapshot()
    owner = _instance(
        slot=1,
        lifecycle=TaskGroupLifecycle.DEDICATED_TRACK,
        sensor_mode=GroupSensorMode.PASSIVE,
        ownership_status="owner",
    ).model_copy(update={"lifecycle": lifecycle, "sensor_mode": sensor_mode})

    with pytest.raises(ValidationError):
        OperationalExecutionSnapshot.model_validate(
            base.model_dump()
            | {
                "task_groups": (owner,),
                "tracking_control": TrackingControlState(
                    mode="dedicated",
                    tracking_owner_group_id=owner.group_instance_id,
                ),
            }
        )


def test_snapshot_rejects_pending_successor_without_owner() -> None:
    base = _snapshot()
    groups = tuple(
        _instance(
            slot=slot,
            lifecycle=TaskGroupLifecycle.PASSIVE_TRACK
            if slot == 1
            else TaskGroupLifecycle.ENTERING,
            sensor_mode=GroupSensorMode.PASSIVE
            if slot == 1
            else GroupSensorMode.ACTIVE,
            ownership_status="owner" if slot == 1 else "candidate",
        )
        for slot in range(1, 5)
    )

    with pytest.raises(ValidationError, match="successor"):
        OperationalExecutionSnapshot.model_validate(
            base.model_dump()
            | {
                "task_groups": groups,
                "tracking_control": TrackingControlState(
                    tracking_owner_group_id=groups[0].group_instance_id,
                    pending_successor_group_id="missing-successor",
                ),
            }
        )


def test_snapshot_rejects_two_regional_runtime_groups() -> None:
    base = _snapshot()

    with pytest.raises(ValidationError, match="cardinality"):
        OperationalExecutionSnapshot.model_validate(
            base.model_dump()
            | {
                "task_groups": (_instance(slot=1), _instance(slot=2)),
                "tracking_control": TrackingControlState(),
            }
        )


def test_snapshot_rejects_duplicate_regional_slots_without_replacement_pair() -> None:
    base = _snapshot()
    groups = tuple(
        _instance(slot=1, deployment_revision=revision)
        for revision in range(1, 5)
    )

    with pytest.raises(ValidationError, match="region"):
        OperationalExecutionSnapshot.model_validate(
            base.model_dump()
            | {
                "task_groups": groups,
                "tracking_control": TrackingControlState(),
            }
        )


def test_snapshot_accepts_dedicated_steady_owner() -> None:
    base = _snapshot()
    owner = _instance(
        slot=1,
        lifecycle=TaskGroupLifecycle.DEDICATED_TRACK,
        sensor_mode=GroupSensorMode.PASSIVE,
        ownership_status="owner",
    )

    snapshot = OperationalExecutionSnapshot.model_validate(
        base.model_dump()
        | {
            "task_groups": (owner,),
            "tracking_control": TrackingControlState(
                mode="dedicated",
                tracking_owner_group_id=owner.group_instance_id,
            ),
        }
    )

    assert snapshot.task_groups == (owner,)


def test_snapshot_accepts_dedicated_restore_transition() -> None:
    base = _snapshot()
    owner = _instance(
        slot=1,
        deployment_revision=1,
        lifecycle=TaskGroupLifecycle.DEDICATED_RELEASE_PENDING,
        sensor_mode=GroupSensorMode.PASSIVE,
        ownership_status="owner",
    )
    incoming = tuple(
        _instance(
            slot=slot,
            deployment_revision=2,
            sensor_mode=GroupSensorMode.PASSIVE
            if slot == 1
            else GroupSensorMode.ACTIVE,
        )
        for slot in range(1, 5)
    )

    snapshot = OperationalExecutionSnapshot.model_validate(
        base.model_dump()
        | {
            "task_groups": (owner, *incoming),
            "tracking_control": TrackingControlState(
                mode="dedicated",
                tracking_owner_group_id=owner.group_instance_id,
                pending_successor_group_id=incoming[0].group_instance_id,
            ),
        }
    )

    assert len(snapshot.task_groups) == 5


def test_snapshot_rejects_invalid_dedicated_runtime_cardinality() -> None:
    base = _snapshot()
    owner = _instance(
        slot=1,
        lifecycle=TaskGroupLifecycle.DEDICATED_TRACK,
        sensor_mode=GroupSensorMode.PASSIVE,
        ownership_status="owner",
    )
    extra = _instance(slot=2)

    with pytest.raises(ValidationError, match="cardinality"):
        OperationalExecutionSnapshot.model_validate(
            base.model_dump()
            | {
                "task_groups": (owner, extra),
                "tracking_control": TrackingControlState(
                    mode="dedicated",
                    tracking_owner_group_id=owner.group_instance_id,
                ),
            }
        )


def test_snapshot_rejects_nine_runtime_instances() -> None:
    base = _snapshot()
    groups = tuple(
        _instance(slot=((index - 1) % 4) + 1)
        .model_copy(
            update={
                "group_instance_id": f"instance-{index}",
                "member_uuv_ids": (
                    f"instance-{index}:member:01",
                    f"instance-{index}:member:02",
                    f"instance-{index}:member:03",
                ),
            }
        )
        for index in range(1, 10)
    )
    with pytest.raises(ValidationError, match="regional execution cardinality requires four to eight"):
        OperationalExecutionSnapshot.model_validate(
            base.model_dump()
            | {
                "task_groups": groups,
                "tracking_control": TrackingControlState(),
            }
        )
