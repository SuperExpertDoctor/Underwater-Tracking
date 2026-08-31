from __future__ import annotations

import json
from threading import Event
from time import monotonic

from underwater_tracking.domain.mission_models import (
    CarrierMissionModel,
    CarrierRouteStatus,
    RegionLifecycle,
    RegionMissionState,
    UUVResourceState,
    UUVMissionMode,
)
from underwater_tracking.domain.models import (
    AdversaryOperationalSummary,
    CarrierState,
    CarrierStatus,
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
from underwater_tracking.runtime.mission_controller import MissionSnapshot
from underwater_tracking.memory.situation_summary import (
    PeriodicSituationSummary,
    PeriodicSituationSummaryWriter,
    build_periodic_situation_summary,
)


def _situation(
    *,
    sim_time_s: int,
    quality: float = 0.8,
    intent: str = "silent_transit",
    uuv2_status: UUVStatus = UUVStatus.TRACKING,
) -> SituationSnapshot:
    uuv2_deployment = (
        DeploymentState.FAILED if uuv2_status is UUVStatus.FAILED else DeploymentState.DEPLOYED
    )
    carrier_uuv_ids = ("U2",) if uuv2_status is not UUVStatus.FAILED else ()
    uuvs = (
        UUVState(
            uuv_id="U1",
            position_xy=(0.0, 0.0),
            heading_rad=0.0,
            speed_mps=0.0,
            energy_fraction=0.9,
            status=UUVStatus.AVAILABLE,
            deployment_state=DeploymentState.ONBOARD,
        ),
        UUVState(
            uuv_id="U2",
            position_xy=(100.0, 0.0),
            heading_rad=0.0,
            speed_mps=1.0,
            energy_fraction=0.5,
            status=uuv2_status,
            deployment_state=uuv2_deployment,
        ),
    )
    return SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=sim_time_s // 30,
        sim_time_s=sim_time_s,
        uuvs=uuvs,
        carrier=CarrierState(
            carrier_id="carrier-01",
            position_xy=(0.0, 0.0),
            heading_rad=0.0,
            speed_mps=1.0,
            status=(CarrierStatus.DEPLOYING if carrier_uuv_ids else CarrierStatus.TRANSIT),
            onboard_uuv_ids=("U1",),
            deployed_uuv_ids=carrier_uuv_ids,
        ),
        group_reports=(
            GroupReport(
                group_id="G1",
                target_id="T1",
                sim_time_s=sim_time_s,
                member_ids=("U2",),
                belief=TargetBelief(
                    target_id="T1",
                    sim_time_s=sim_time_s,
                    mean=(1000.0, 2000.0),
                    covariance=((100.0, 0.0), (0.0, 100.0)),
                    model_probabilities={intent: 0.75, "patrol": 0.25},
                ),
                quality=GroupQuality(
                    instant=quality,
                    window_mean=quality,
                    ewma=quality,
                    components={"coverage": quality},
                ),
                plan_revision=3,
            ),
        ),
        pending_events=(),
        adversary_summaries=(
            AdversaryOperationalSummary(
                target_id="T1",
                sim_time_s=sim_time_s,
                detection_range_m=1200.0,
                intent=intent,
                confidence=0.75,
            ),
        ),
    )


def _mission(
    *,
    lifecycle: RegionLifecycle = RegionLifecycle.ACTIVE_SCAN,
    mode: UUVMissionMode = UUVMissionMode.ACTIVE_SCAN,
    uuv2_healthy: bool = True,
    uuv2_deployment: str = "deployed",
) -> MissionSnapshot:
    return MissionSnapshot(
        scenario_id="S1",
        sim_time_s=600,
        plan_revision=3,
        regions=(
            RegionMissionState(
                region_id="R1",
                target_id="T1",
                lifecycle=lifecycle,
                active_scan_uuv_ids=("U2",),
                coverage=0.8,
                tracking_quality=0.8,
                plan_revision=3,
            ),
        ),
        uuv_modes={"U1": UUVMissionMode.ONBOARD, "U2": mode},
        uuv_resources={
            "U1": UUVResourceState(
                uuv_id="U1",
                carrier_id="carrier-01",
                mileage_m=100.0,
                energy_fraction=0.9,
                deployment_state="onboard",
                resource_episode=1,
            ),
            "U2": UUVResourceState(
                uuv_id="U2",
                carrier_id="carrier-01",
                mileage_m=200.0,
                energy_fraction=0.5,
                healthy=uuv2_healthy,
                deployment_state=uuv2_deployment,
                resource_episode=1,
            ),
        },
        carrier_missions={
            "carrier-01": CarrierMissionModel(
                carrier_id="carrier-01",
                home_battle_group_id="BG1",
                route_status=CarrierRouteStatus.TO_DEPLOY,
                onboard_uuv_ids=("U1",),
                ready_uuv_ids=("U2",),
            ),
        },
    )


def _events(prediction_revision: int = 0) -> tuple[RuntimeEvent, ...]:
    return tuple(
        RuntimeEvent(
            event_id=f"event-{index:03d}",
            scenario_id="S1",
            sim_time_s=600,
            event_type="prediction_revision" if index == 0 else "state_changed",
            entity_id="T1",
            level=EventLevel.INFORMATIONAL,
            payload=(
                {"target_id": "T1", "prediction_revision": prediction_revision}
                if index == 0
                else {}
            ),
        )
        for index in range(70)
    )


def _assert_public_payload(value: object) -> None:
    forbidden = ("truth", "private", "chain_of_thought", "gate_distance")
    if isinstance(value, dict):
        for key, child in value.items():
            assert not any(fragment in str(key).casefold() for fragment in forbidden)
            _assert_public_payload(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _assert_public_payload(child)
    elif isinstance(value, str):
        assert not any(fragment in value.casefold() for fragment in forbidden)


def test_summary_is_deterministic_bounded_and_truth_safe() -> None:
    summary, event = build_periodic_situation_summary(
        _situation(sim_time_s=600),
        _mission(),
        _events(),
        None,
    )

    assert isinstance(summary, PeriodicSituationSummary)
    assert event.event_id == "periodic_situation_summary:S1:600"
    assert summary.changes_since_previous == ()
    assert summary.source_event_ids == tuple(f"event-{index:03d}" for index in range(64))
    assert summary.region_states[0].lifecycle == "ACTIVE_SCAN"
    assert summary.carrier_states[0].route_status == "TO_DEPLOY"
    assert summary.uuv_counts.total == 2
    assert summary.uuv_counts.onboard == 1
    assert summary.uuv_counts.deployed == 1
    assert summary.uuv_counts.healthy == 2
    assert summary.target_estimates[0].quality_score == 0.8
    assert summary.target_estimates[0].intent == "silent_transit"
    assert summary.target_estimates[0].prediction_revision == 0
    assert len(event.payload["summary"]) <= 4000
    _assert_public_payload(event.payload)
    assert len(json.dumps(event.payload, sort_keys=True)) < 24_000


def test_summary_changes_cover_lifecycle_mode_health_quality_intent_and_prediction() -> None:
    previous, _ = build_periodic_situation_summary(_situation(sim_time_s=600), _mission(), (), None)
    current, _ = build_periodic_situation_summary(
        _situation(sim_time_s=1200, quality=0.4, intent="evade", uuv2_status=UUVStatus.FAILED),
        _mission(
            lifecycle=RegionLifecycle.PASSIVE_TRACK,
            mode=UUVMissionMode.PASSIVE_TRACK,
            uuv2_healthy=False,
            uuv2_deployment="failed",
        ),
        _events(prediction_revision=2),
        previous,
    )

    change_types = {change.change_type for change in current.changes_since_previous}
    assert {
        "region_lifecycle",
        "uuv_mode",
        "uuv_deployment",
        "uuv_health",
        "target_quality",
        "target_intent",
        "target_prediction_revision",
    } <= change_types


def test_summary_tracks_plan_and_uuv_assignment_changes_but_ignores_jitter() -> None:
    previous, _ = build_periodic_situation_summary(
        _situation(sim_time_s=600, quality=0.80), _mission(), (), None
    )
    previous_mission = _mission()
    changed_region = previous_mission.regions[0].model_copy(
        update={
            "active_scan_uuv_ids": (),
            "passive_track_uuv_ids": ("U2",),
            "coverage": 0.79,
            "tracking_quality": 0.79,
            "plan_revision": 4,
        }
    )
    changed_resources = previous_mission.uuv_resources["U2"].model_copy(update={"mileage_m": 301.0})
    current_mission = previous_mission.model_copy(
        update={
            "plan_revision": 4,
            "regions": (changed_region,),
            "uuv_resources": {
                **previous_mission.uuv_resources,
                "U2": changed_resources,
            },
        }
    )
    current, event = build_periodic_situation_summary(
        _situation(sim_time_s=1200, quality=0.79),
        current_mission,
        (),
        previous,
    )

    change_types = {change.change_type for change in current.changes_since_previous}
    assert {"plan_revision", "region_uuv_assignment"} <= change_types
    assert "region_coverage" not in change_types
    assert "region_tracking_quality" not in change_types
    assert "target_quality" not in change_types
    assert "uuv_mileage" not in change_types
    assert event.payload["memory_eligible"] is True
    assert "region_uuv_assignment" in event.payload["summary"]
    assert "U2" in event.payload["summary"]


def test_summary_does_not_mark_an_unchanged_followup_as_memory_eligible() -> None:
    previous, _ = build_periodic_situation_summary(_situation(sim_time_s=600), _mission(), (), None)

    current, event = build_periodic_situation_summary(
        _situation(sim_time_s=1200), _mission(), (), previous
    )

    assert current.changes_since_previous == ()
    assert event.payload["memory_eligible"] is False


def test_summary_uses_observed_belief_instead_of_private_adversary_intent() -> None:
    situation = _situation(sim_time_s=600, intent="silent_transit")
    private_summary = situation.adversary_summaries[0].model_copy(
        update={"intent": "evade", "confidence": 0.99}
    )
    situation = situation.model_copy(update={"adversary_summaries": (private_summary,)})

    summary, _ = build_periodic_situation_summary(situation, _mission(), (), None)

    assert summary.target_estimates[0].intent == "silent_transit"
    assert summary.target_estimates[0].intent_confidence == 0.75


def test_summary_preserves_last_public_estimate_when_a_report_is_temporarily_missing() -> None:
    previous, _ = build_periodic_situation_summary(_situation(sim_time_s=600), _mission(), (), None)
    missing_report = _situation(sim_time_s=1200).model_copy(
        update={"group_reports": (), "adversary_summaries": ()}
    )

    current, event = build_periodic_situation_summary(missing_report, _mission(), (), previous)

    assert current.target_estimates == previous.target_estimates
    assert current.changes_since_previous == ()
    assert event.payload["memory_eligible"] is False


def _summary_event(sim_time_s: int = 600) -> RuntimeEvent:
    _, event = build_periodic_situation_summary(
        _situation(sim_time_s=sim_time_s),
        _mission(),
        (),
        None,
    )
    return event


def test_summary_writer_submit_is_nonblocking_while_repository_is_busy(tmp_path) -> None:
    started = Event()
    release = Event()
    persisted: list[str] = []

    class BlockingRepository:
        def append_if_absent(self, **kwargs: object) -> int:
            started.set()
            assert release.wait(1.0)
            persisted.append(str(kwargs["event_id"]))
            return len(persisted)

        def close(self) -> None:
            return None

    repository = BlockingRepository()
    writer = PeriodicSituationSummaryWriter(
        tmp_path / "summary.db",
        repository_factory=lambda _path: repository,
    )
    writer.start()
    assert writer.submit(_summary_event()) is True
    assert started.wait(1.0)

    began = monotonic()
    assert writer.submit(_summary_event(1200)) is True
    assert writer.submit(_summary_event(1800)) is True
    assert monotonic() - began < 0.1
    assert writer.stop(timeout=0.01) is False

    release.set()
    assert writer.stop(timeout=1.0) is True
    assert persisted == [
        "periodic_situation_summary:S1:600",
        "periodic_situation_summary:S1:1200",
        "periodic_situation_summary:S1:1800",
    ]


def test_summary_writer_retries_the_same_event_id_after_persistence_failure(tmp_path) -> None:
    attempts = 0
    persisted = Event()
    persisted_ids: list[str] = []

    class RetryRepository:
        def append_if_absent(self, **kwargs: object) -> int:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("temporary sqlite failure")
            persisted_ids.append(str(kwargs["event_id"]))
            persisted.set()
            return 1

        def close(self) -> None:
            return None

    writer = PeriodicSituationSummaryWriter(
        tmp_path / "summary.db",
        repository_factory=lambda _path: RetryRepository(),
    )
    writer.start()
    assert writer.submit(_summary_event()) is True
    assert persisted.wait(1.0)
    assert writer.stop(timeout=1.0) is True

    metrics = writer.metrics
    assert attempts >= 2
    assert persisted_ids == ["periodic_situation_summary:S1:600"]
    assert metrics.persisted_count == 1
    assert metrics.failed_count >= 1
    assert metrics.queue_backlog == 0


def test_summary_writer_rejects_newest_event_when_bounded_queue_is_full(tmp_path) -> None:
    persisted_ids: list[str] = []

    class RecordingRepository:
        def append_if_absent(self, **kwargs: object) -> int:
            persisted_ids.append(str(kwargs["event_id"]))
            return len(persisted_ids)

        def close(self) -> None:
            return None

    writer = PeriodicSituationSummaryWriter(
        tmp_path / "summary.db",
        queue_limit=2,
        repository_factory=lambda _path: RecordingRepository(),
    )
    assert writer.submit(_summary_event(600)) is True
    assert writer.submit(_summary_event(1200)) is True
    assert writer.submit(_summary_event(1800)) is False
    assert writer.metrics.queue_backlog == 2
    assert writer.metrics.overflow_count == 1

    writer.start()
    assert writer.stop(timeout=1.0) is True
    assert persisted_ids == [
        "periodic_situation_summary:S1:600",
        "periodic_situation_summary:S1:1200",
    ]
