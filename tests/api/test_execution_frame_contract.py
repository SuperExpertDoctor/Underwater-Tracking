from __future__ import annotations

import json
from hashlib import sha256

import pytest
from pydantic import ValidationError

from underwater_tracking.api.frame_builder import operational_frame_json
from underwater_tracking.api.frame_logger import FrameLogger
from underwater_tracking.api.hub import OperationalHub
from underwater_tracking.api.replay import ReplayService
from underwater_tracking.config.models import TrackingPolicyConfig
from underwater_tracking.domain.execution_models import (
    GroupSensorMode,
    TaskGroupInstance,
    TaskGroupLifecycle,
    TrackingControlState,
)
from tests.domain.test_execution_models import _snapshot as execution_snapshot
from underwater_tracking.api.frame_builder import (
    build_operational_frame,
    operational_frame_payload,
)
from underwater_tracking.api.live import OperationalFramePublisher
from underwater_tracking.domain.agent_models import PredictedTrackRef
from underwater_tracking.domain.models import (
    DeploymentState,
    EventLevel,
    GroupQuality,
    GroupReport,
    RuntimeEvent,
    SituationSnapshot,
    TargetBelief,
    UUVState,
    UUVStatus,
)
from underwater_tracking.domain.prediction_models import AcceptedPrediction, PredictionHealth


def _parallel_replacement_snapshot():
    base = _runtime_execution_snapshot()
    groups = []
    states = []
    for slot in range(1, 5):
        region_id = f"target_00:task:{slot:02d}"
        outgoing_id = f"S1:{region_id}:deploy:000008:outgoing"
        incoming_id = f"S1:{region_id}:deploy:000009:incoming"
        outgoing_members = tuple(
            f"uuv_{(slot - 1) * 6 + member:02d}" for member in range(3)
        )
        incoming_members = tuple(
            f"uuv_{(slot - 1) * 6 + 3 + member:02d}" for member in range(3)
        )
        outgoing = TaskGroupInstance(
            group_instance_id=outgoing_id,
            target_id="target_00",
            region_id=region_id,
            deployment_revision=8,
            member_uuv_ids=outgoing_members,
            lifecycle=TaskGroupLifecycle.EXITING,
            sensor_mode=GroupSensorMode.PASSIVE,
            ownership_status="candidate",
            reason="region_replacement",
            evidence_ids=(f"{outgoing_id}:evidence",),
        )
        incoming = TaskGroupInstance(
            group_instance_id=incoming_id,
            target_id="target_00",
            region_id=region_id,
            deployment_revision=9,
            member_uuv_ids=incoming_members,
            lifecycle=TaskGroupLifecycle.ACTIVE_SCAN,
            sensor_mode=GroupSensorMode.ACTIVE,
            ownership_status="candidate",
            source_group_instance_id=outgoing_id,
            reason="region_replacement",
            evidence_ids=(f"{incoming_id}:evidence",),
        )
        groups.extend((outgoing, incoming))
        for group in (outgoing, incoming):
            for uuv_id in group.member_uuv_ids:
                states.append(
                    UUVState(
                        uuv_id=uuv_id,
                        position_xy=(float(len(states)), 0.0),
                        heading_rad=0.0,
                        speed_mps=1.0,
                        energy_fraction=1.0,
                        remaining_range_m=50_000.0,
                        status=UUVStatus.TRACK,
                        deployment_state=DeploymentState.DEPLOYED,
                        physically_exposed=True,
                        group_id=group.group_instance_id,
                        sensor_mode=group.sensor_mode.value,
                    )
                )
    regions = tuple(
        region.model_copy(
            update={"task_group_id": groups[index * 2 + 1].group_instance_id}
        )
        for index, region in enumerate(base.regions)
    )
    return base.model_copy(
        deep=True,
        update={
            "regions": regions,
            "task_groups": tuple(groups),
            "reserve_uuvs": (),
            "tracking_policy": TrackingPolicyConfig(),
            "tracking_control": TrackingControlState(mode="regional"),
        },
    ), SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=12,
        sim_time_s=120,
        uuvs=tuple(states),
        group_reports=(),
        pending_events=(),
    )


def _runtime_execution_snapshot():
    """Build the strict UUV-only snapshot used by frame transport tests."""
    base = execution_snapshot()
    groups = tuple(
        TaskGroupInstance(
            group_instance_id=f"target_00:task:{index:02d}:deploy:000009",
            target_id="target_00",
            region_id=f"target_00:task:{index:02d}",
            deployment_revision=9,
            member_uuv_ids=tuple(
                f"uuv_{(index - 1) * 3 + member:02d}" for member in range(3)
            ),
            lifecycle=(
                TaskGroupLifecycle.ACTIVE_SCAN
                if index == 1
                else TaskGroupLifecycle.ENTERING
            ),
            sensor_mode=GroupSensorMode.ACTIVE,
            ownership_status="candidate",
            reason="initial_deployment",
            evidence_ids=(f"runtime-group-evidence-{index}",),
        )
        for index in range(1, 5)
    )
    regions = tuple(
        region.model_copy(
            update={
                "center": (1_000.0 + (index - 1) * 2_000.0, 0.0),
                "side_length_m": 2_000.0,
                "geometry": (
                    ((index - 1) * 2_000.0, 1_000.0),
                    (index * 2_000.0, 1_000.0),
                    (index * 2_000.0, -1_000.0),
                    ((index - 1) * 2_000.0, -1_000.0),
                ),
                "task_group_id": groups[index - 1].group_instance_id,
            }
        )
        for index, region in enumerate(base.regions, start=1)
    )
    return base.model_copy(
        deep=True,
        update={
            "regions": regions,
            "task_groups": groups,
            "reserve_uuvs": (),
            "tracking_policy": TrackingPolicyConfig(),
            "tracking_control": TrackingControlState(mode="regional"),
        },
    )


def _replacement_frame():
    execution, situation = _parallel_replacement_snapshot()
    return build_operational_frame(
        situation,
        plan=None,
        ledger_tail=(),
        events=(),
        metrics=(),
        uuv_only=True,
        execution_snapshot=execution,
    )


def _frame():
    snapshot = _runtime_execution_snapshot()
    situation = SituationSnapshot(
        scenario_id=snapshot.scenario_id,
        snapshot_revision=snapshot.source_snapshot_revision,
        sim_time_s=int(snapshot.source_sim_time_s),
        uuvs=(),
        group_reports=(),
        pending_events=(),
    )
    return build_operational_frame(
        situation,
        plan=None,
        ledger_tail=(),
        events=(),
        metrics=(),
        uuv_only=True,
        execution_snapshot=snapshot,
    )


def test_execution_frame_projects_one_authoritative_four_region_snapshot() -> None:
    frame = _frame()

    assert frame.execution is not None
    assert frame.execution.execution_revision == 9
    assert frame.execution.source_snapshot_revision == 12
    assert len(frame.execution.regions) == 4
    assert len(frame.execution.task_groups) == 4
    assert tuple(row.region_id for row in frame.execution.regions) == (
        "target_00:task:01",
        "target_00:task:02",
        "target_00:task:03",
        "target_00:task:04",
    )
    assert frame.execution.regions[0].task_group_id == frame.execution.task_groups[0].group_instance_id
    assert all(len(group.member_uuv_ids) == 3 for group in frame.execution.task_groups)
    assert not hasattr(frame.execution, "reserve_uuv_ids")
    assert frame.execution.degraded is False


def test_frame_projects_real_tracking_policy_and_all_visible_groups() -> None:
    frame = _replacement_frame()

    assert frame.execution is not None
    assert frame.execution.tracking_policy.task_region_side_m == 2_000.0
    assert frame.execution.tracking_policy.target_detection_radius_m == 1_000.0
    assert frame.execution.tracking_policy.uuv_active_detection_radius_m == 600.0
    assert len(frame.execution.task_groups) == 8
    assert len([uuv for uuv in frame.uuvs if uuv.physically_exposed]) == 24
    assert all(len(group.member_uuv_ids) == 3 for group in frame.execution.task_groups)
    assert {group.group_instance_id for group in frame.execution.task_groups} == {
        uuv.group_instance_id for uuv in frame.uuvs
    }
    payload = operational_frame_payload(frame)
    serialized = json.dumps(payload)
    assert "active_verifier_uuv_id" not in serialized
    assert "passive_tracker_uuv_id" not in serialized
    assert "reserve_uuv_ids" not in payload["execution"]


def test_execution_view_rejects_three_groups_in_one_region() -> None:
    frame = _replacement_frame()
    assert frame.execution is not None
    execution_payload = frame.execution.model_dump(mode="python")
    groups = list(execution_payload["task_groups"])
    extra = dict(groups[0])
    extra["group_instance_id"] = "target_00:task:01:deploy:000010:invalid"
    extra["deployment_revision"] = 10
    extra["member_uuv_ids"] = (
        "uuv_invalid_01",
        "uuv_invalid_02",
        "uuv_invalid_03",
    )
    execution_payload["task_groups"] = [*groups[:2], extra, *groups[3:]]

    with pytest.raises(ValidationError, match="region cardinality"):
        type(frame.execution).model_validate(execution_payload)


def test_execution_frame_transport_serializers_have_one_canonical_payload(tmp_path) -> None:
    frame = _replacement_frame()
    expected = sha256(
        json.dumps(
            json.loads(frame.model_dump_json()),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    hub = OperationalHub()
    hub.publish(frame)
    assert hub.serialized_snapshot() is not None
    websocket_payload = json.loads(hub.serialized_snapshot().decode("utf-8"))

    path = tmp_path / "frames.jsonl"
    logger = FrameLogger(path)
    logger.append(frame)
    logger.close()
    replayed = ReplayService(path).last()
    assert replayed is not None

    assert sha256(
        json.dumps(websocket_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest() == expected
    assert sha256(
        json.dumps(
            json.loads(operational_frame_json(replayed)),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest() == expected


def test_execution_frame_rejects_mixed_revision_and_candidate_grid_surface() -> None:
    payload = _frame().model_dump(mode="json")
    payload["execution"]["regions"][1]["execution_revision"] = 8
    with pytest.raises(ValidationError, match="revision"):
        type(_frame()).model_validate(payload)

    frame_payload = operational_frame_payload(_frame())
    assert "prediction_grids" not in frame_payload["execution"]
    assert "candidate_regions" not in frame_payload["execution"]


def test_hub_and_replay_use_the_same_execution_frame_payload(tmp_path) -> None:
    frame = _frame()
    hub = OperationalHub()
    hub.publish(frame)
    path = tmp_path / "frames.jsonl"
    logger = FrameLogger(path)
    logger.append(frame)
    logger.close()

    replayed = ReplayService(path).last()
    assert replayed is not None
    assert hub.snapshot() == frame
    assert replayed == frame
    assert json.loads(frame.model_dump_json()) == operational_frame_payload(frame)


def test_legacy_frame_without_execution_remains_readable() -> None:
    frame = _frame().model_copy(
        update={"execution": None, "execution_consistency": None}
    )

    restored = type(frame).model_validate_json(frame.model_dump_json())

    assert restored.execution is None


def test_live_publisher_reads_the_runtime_execution_snapshot() -> None:
    snapshot = _runtime_execution_snapshot()

    class Runtime:
        def active_plan(self):
            return None

        def get_state(self):
            return {}

        @property
        def current_execution_snapshot(self):
            return snapshot

    class Ledger:
        def list_decisions(self, *args: object, **kwargs: object):
            del args, kwargs
            return []

        def list_directives(self, *args: object, **kwargs: object):
            del args, kwargs
            return []

    class Events:
        def list_events(self, *args: object, **kwargs: object):
            del args, kwargs
            return []

    situation = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=12,
        sim_time_s=120,
        uuvs=(),
        group_reports=(),
        pending_events=(),
    )
    frame = OperationalFramePublisher(
        runtime=Runtime(),
        ledger=Ledger(),
        events=Events(),
        hub=OperationalHub(),
    ).publish(situation)

    assert frame.uuv_only is True
    assert frame.execution is not None
    assert frame.execution.execution_revision == snapshot.execution_revision


def test_live_publisher_drops_stale_accepted_prediction_during_execution_rollover() -> None:
    snapshot = _runtime_execution_snapshot()
    stale = PredictedTrackRef(
        prediction_id="prediction-newer",
        target_id=snapshot.target_id,
        sim_time_s=int(snapshot.prediction.origin_sim_time_s),
        horizon_s=60.0,
        sample_step_s=30.0,
        times_s=(150.0, 180.0),
        points_xy=((70.0, 50.0), (130.0, 80.0)),
        corridor_radius_m=(10.0, 11.0),
    )

    class Runtime:
        def active_plan(self):
            return None

        def get_state(self):
            return {
                "accepted_predictions": {
                    snapshot.target_id: AcceptedPrediction(
                        prediction=stale,
                        health=PredictionHealth(
                            status="valid",
                            regime="imm",
                            source_track_age_s=0.0,
                            clipped_point_fraction=0.0,
                            maximum_radius_m=11.0,
                            raw_prediction_id=stale.prediction_id,
                        ),
                    )
                }
            }

        @property
        def current_execution_snapshot(self):
            return snapshot

    class Ledger:
        def list_decisions(self, *args: object, **kwargs: object):
            del args, kwargs
            return []

        def list_directives(self, *args: object, **kwargs: object):
            del args, kwargs
            return []

    class Events:
        def list_events(self, *args: object, **kwargs: object):
            del args, kwargs
            return []

    situation = SituationSnapshot(
        scenario_id=snapshot.scenario_id,
        snapshot_revision=snapshot.source_snapshot_revision,
        sim_time_s=int(snapshot.source_sim_time_s),
        uuvs=(),
        group_reports=(
            GroupReport(
                group_id="G1",
                target_id=snapshot.target_id,
                sim_time_s=int(snapshot.source_sim_time_s),
                member_ids=(),
                belief=TargetBelief(
                    target_id=snapshot.target_id,
                    sim_time_s=int(snapshot.source_sim_time_s),
                    track_revision=snapshot.target_track.track_revision,
                    last_observed_at_s=100, valid_until_s=1920, source_observation_ids=("obs:target_00",),
                    mean=(10.0, 20.0),
                    covariance=((1.0, 0.0), (0.0, 1.0)),
                    model_probabilities={"cv": 1.0},
                    fim_condition=1.0,
                ),
                quality=GroupQuality(
                    instant=0.9,
                    window_mean=0.9,
                    ewma=0.9,
                    components={"fim": 0.9},
                ),
                plan_revision=1,
            ),
        ),
        pending_events=(),
    )
    frame = OperationalFramePublisher(
        runtime=Runtime(),
        ledger=Ledger(),
        events=Events(),
        hub=OperationalHub(),
    ).publish(situation)

    estimate = frame.target_estimates[0]
    assert estimate.prediction is not None
    assert estimate.prediction.prediction_id == snapshot.prediction_id


def test_live_publisher_bounds_operator_thinking_event_references() -> None:
    events = tuple(
        RuntimeEvent(
            event_id=f"event-{index:02d}",
            scenario_id="S1",
            sim_time_s=120,
            event_type="plan_update",
            level=EventLevel.TACTICAL,
        )
        for index in range(40)
    )

    class Runtime:
        def active_plan(self):
            return None

        def get_state(self):
            return {}

    class Ledger:
        def list_decisions(self, *args: object, **kwargs: object):
            del args, kwargs
            return []

        def list_directives(self, *args: object, **kwargs: object):
            del args, kwargs
            return []

    class EventPort:
        def list_events(self, *args: object, **kwargs: object):
            del args, kwargs
            return []

    situation = SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=12,
        sim_time_s=120,
        uuvs=(),
        group_reports=(),
        pending_events=events,
    )
    frame = OperationalFramePublisher(
        runtime=Runtime(),
        ledger=Ledger(),
        events=EventPort(),
        hub=OperationalHub(),
    ).publish(situation)

    assert len(frame.llm_thinking_source_event_ids) == 32
    assert frame.llm_thinking_source_event_ids == tuple(
        event.event_id for event in events[-32:]
    )
