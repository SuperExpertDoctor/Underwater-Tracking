from __future__ import annotations

from pydantic import ValidationError
import pytest

from underwater_tracking.domain.execution_models import (
    DeterministicIntentState,
    ExecutionDegradation,
    ExecutionRegion,
    GlobalTargetTrackView,
    IMMModelForecast,
    IMMPredictedTrack,
    OperationalExecutionSnapshot,
    ReserveUUVState,
    TaskGroupAssignment,
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
