"""Adaptive tracking inputs exposed by the deterministic engine."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from types import SimpleNamespace

import pytest

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.config.models import AgentConfig, RuntimeRetentionConfig
from underwater_tracking.domain.agent_models import PlanCommand, TrackingPlan
from underwater_tracking.domain.adversary_models import AdversaryIntentDecision
from underwater_tracking.domain.models import (
    EventLevel,
    IntelligenceReport,
    IntelligenceSource,
    OperationalScheme,
    SurveillanceCapability,
)
from underwater_tracking.domain.mission_models import RegionLifecycle, UUVMissionMode
from underwater_tracking.simulation.engine import SimulationEngine
from underwater_tracking.simulation.target import HiddenIntent, TargetEntity
from underwater_tracking.runtime.mission_controller import MissionController
from tests.domain.test_execution_models import _snapshot as execution_snapshot
from tests.conftest import CONFIG_PATH


BOUNDARY_EVENT_TYPES = {
    "target_boundary_recovery_started",
    "target_boundary_turn_started",
    "target_boundary_recovery_completed",
    "target_navigation_recovery_failed",
}


def test_uuv_execution_snapshot_uses_region_windows_for_task_groups(tmp_path) -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(
        config,
        seed=config.scenario.seed,
        output_dir=tmp_path,
        mission_controller=controller,
    )
    snapshot = execution_snapshot().model_copy(
        update={"scenario_id": config.scenario.scenario_id}
    )
    engine._clock.sim_time_s = int(snapshot.valid_from_s)

    assert engine.apply_verified_execution_snapshot(snapshot) is True
    assert engine._mission_plan is not None
    assert engine._mission_plan.batches == ()
    assert engine._mission_time_windows() == {
        region.region_id: (int(region.start_s), int(region.end_s))
        for region in snapshot.regions
    }


def test_uuv_execution_snapshot_rejects_scenario_mismatch(tmp_path) -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id="S1")
    engine = SimulationEngine(
        config,
        seed=config.scenario.seed,
        output_dir=tmp_path,
        mission_controller=controller,
    )
    snapshot = execution_snapshot().model_copy(update={"scenario_id": "S1"})
    engine._clock.sim_time_s = int(snapshot.valid_from_s)

    assert engine.apply_verified_execution_snapshot(snapshot) is False
    assert engine._mission_plan is None
    assert engine._last_mission_plan_failure_reason == "execution_snapshot_scenario_mismatch"


def test_uuv_execution_snapshot_initializes_carrier_recovery_metadata(tmp_path) -> None:
    from underwater_tracking.cli import _mission_controller_for

    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = _mission_controller_for(config)
    assert controller is not None
    engine = SimulationEngine(
        config,
        seed=config.scenario.seed,
        output_dir=tmp_path,
        mission_controller=controller,
    )
    snapshot = execution_snapshot().model_copy(
        update={"scenario_id": config.scenario.scenario_id}
    )
    engine._clock.sim_time_s = int(snapshot.valid_from_s)

    assert engine.apply_verified_execution_snapshot(snapshot) is True

    mission = controller.snapshot()
    assert set(mission.carrier_missions) == {
        "carrier_01",
        "carrier_02",
        "carrier_03",
        "carrier_04",
    }
    owners = {
        uuv.platform_id: uuv.home_carrier_id
        for uuv in config.environment.uuvs  # type: ignore[union-attr]
    }
    assert all(
        uuv_id in mission.carrier_missions[carrier_id].ready_uuv_ids
        for uuv_id, carrier_id in owners.items()
    )


def test_execution_snapshot_does_not_partially_recover_non_first_runtime_group(
    tmp_path,
) -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id="S1")
    engine = SimulationEngine(
        config,
        seed=config.scenario.seed,
        output_dir=tmp_path,
        mission_controller=controller,
    )
    base = execution_snapshot()
    engine._clock.sim_time_s = int(base.valid_from_s)
    controller.advance(int(base.valid_from_s), {})
    assert controller.apply_execution_snapshot(base) is True

    region = base.regions[1]
    controller._regions[region.region_id] = controller._regions[
        region.region_id
    ].model_copy(
        update={
            "lifecycle": RegionLifecycle.CARRIER_RECOVERY,
            "handoff_from": region.predecessor_region_id,
        }
    )
    recovered_uuv_id = base.task_groups[1].member_uuv_ids[0]
    controller._recovered_uuv_ids_by_region[region.region_id] = {recovered_uuv_id}
    controller._uuv_modes[recovered_uuv_id] = UUVMissionMode.ONBOARD

    refreshed = base.model_copy(
        deep=True,
        update={
            "scenario_id": base.scenario_id,
            "execution_revision": base.execution_revision + 1,
            "base_execution_revision": base.execution_revision,
            "regions": tuple(
                item.model_copy(
                    update={
                        "execution_revision": base.execution_revision + 1,
                        "status": "handoff_completed"
                        if item.region_id == region.region_id
                        else item.status,
                    }
                )
                for item in base.regions
            ),
            "task_groups": tuple(
                group.model_copy(
                    update={"execution_revision": base.execution_revision + 1}
                )
                for group in base.task_groups
            ),
        },
    )
    engine._clock.sim_time_s = int(refreshed.valid_from_s)

    assert controller.apply_execution_snapshot(refreshed) is True
    assert controller._recovered_uuv_ids_by_region[region.region_id] == {
        recovered_uuv_id
    }
    assert all(
        controller.snapshot().uuv_modes[uuv_id] is UUVMissionMode.ACTIVE_SCAN
        for uuv_id in base.task_groups[1].member_uuv_ids
    )


@pytest.mark.parametrize(
    ("sim_time_s", "valid_from_s", "valid_until_s", "reason"),
    (
        (0, 120, 570, "execution_snapshot_not_yet_valid"),
        (901, 0, 450, "execution_snapshot_expired"),
    ),
)
def test_uuv_execution_snapshot_rejects_non_executable_freshness(
    tmp_path,
    sim_time_s: int,
    valid_from_s: int,
    valid_until_s: int,
    reason: str,
) -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(
        config,
        seed=config.scenario.seed,
        output_dir=tmp_path,
        mission_controller=controller,
    )
    engine._clock.sim_time_s = sim_time_s
    snapshot = execution_snapshot().model_copy(
        update={
            "scenario_id": config.scenario.scenario_id,
            "valid_from_s": valid_from_s,
            "valid_until_s": valid_until_s,
        }
    )

    assert engine.apply_verified_execution_snapshot(snapshot) is False
    assert engine._mission_plan is None
    assert engine._last_mission_plan_failure_reason == reason


def test_uuv_execution_snapshot_replaces_rolling_region_windows(tmp_path) -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(
        config,
        seed=config.scenario.seed,
        output_dir=tmp_path,
        mission_controller=controller,
    )
    first = execution_snapshot().model_copy(
        update={"scenario_id": config.scenario.scenario_id}
    )
    engine._clock.sim_time_s = int(first.valid_from_s)
    assert engine.apply_verified_execution_snapshot(first) is True

    second = first.model_copy(
        deep=True,
        update={
            "execution_revision": first.execution_revision + 1,
            "base_execution_revision": first.execution_revision,
            "valid_from_s": 2_000.0,
            "valid_until_s": 3_800.0,
            "regions": tuple(
                region.model_copy(
                    update={
                        "execution_revision": first.execution_revision + 1,
                        "start_s": region.start_s + 2_000.0,
                        "end_s": region.end_s + 2_000.0,
                        "handoff_start_s": (
                            region.handoff_start_s + 2_000.0
                            if region.handoff_start_s is not None
                            else None
                        ),
                        "handoff_end_s": (
                            region.handoff_end_s + 2_000.0
                            if region.handoff_end_s is not None
                            else None
                        ),
                    }
                )
                for region in first.regions
            ),
            "task_groups": tuple(
                group.model_copy(
                    update={"execution_revision": first.execution_revision + 1}
                )
                for group in first.task_groups
            ),
        },
    )
    engine._clock.sim_time_s = int(second.valid_from_s)

    assert engine.apply_verified_execution_snapshot(second) is True
    assert engine._mission_time_windows() == {
        region.region_id: (int(region.start_s), int(region.end_s))
        for region in second.regions
    }


def test_uuv_exit_prediction_uses_public_estimate_outside_region(tmp_path) -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=config.scenario.seed, output_dir=tmp_path)
    engine._mission_plan = SimpleNamespace(batches=())
    engine._mission_execution_windows = {
        "R1": (0.0, 540.0),
        "R2": (450.0, 990.0),
    }
    engine._latest_reports["T1"] = SimpleNamespace(
        belief=SimpleNamespace(mean=(20.0, 20.0))
    )
    predecessor = SimpleNamespace(
        region_id="R1",
        target_id="T1",
        lifecycle=RegionLifecycle.ACTIVE_SCAN,
        handoff_to="R2",
        region_polygon=((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)),
    )
    successor = SimpleNamespace(
        region_id="R2",
        target_id="T1",
        lifecycle=RegionLifecycle.PASSIVE_TRACK,
        handoff_to="R3",
        region_polygon=((20.0, 20.0), (30.0, 20.0), (30.0, 30.0), (20.0, 30.0)),
    )

    assert (
        engine._mission_exit_prediction(
            600,
            SimpleNamespace(regions=(predecessor, successor)),
        )
        == "R1"
    )


def _boundary_event_target(*, timeout_s: float = 300.0) -> TargetEntity:
    return TargetEntity(
        target_id="target_00",
        position_xy=(9_700.0, 0.0),
        velocity_xy=(12.0, 0.0),
        intent=HiddenIntent.TRANSIT,
        bounds_xy=(-10_000.0, 10_000.0, -10_000.0, 10_000.0),
        max_acceleration_mps2=0.5,
        max_deceleration_mps2=0.8,
        max_turn_rate_rad_s=0.05,
        boundary_recovery_timeout_s=timeout_s,
    )


def test_engine_emits_ordered_non_duplicated_boundary_recovery_events(tmp_path) -> None:
    base = load_app_config(CONFIG_PATH)
    config = base.model_copy(
        update={"timing": base.timing.model_copy(update={"physics_step_s": 1})}
    )
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    target = _boundary_event_target()
    engine._targets["target_00"] = target
    engine._target_intents["target_00"] = target.intent

    for _ in range(240):
        engine.step()
        if target.navigation_state == "NORMAL" and any(
            event.event_type == "target_boundary_recovery_started"
            for event in engine.events()
        ):
            break

    events = [
        event for event in engine.events() if event.event_type in BOUNDARY_EVENT_TYPES
    ]
    assert [event.event_type for event in events] == [
        "target_boundary_recovery_started",
        "target_boundary_turn_started",
        "target_boundary_recovery_completed",
    ]
    assert len({event.event_id for event in events}) == len(events)
    for event in events:
        assert event.entity_id == "target_00"
        assert {
            "target_id",
            "old_state",
            "new_state",
            "position_xy",
            "guard_distance_m",
            "state_age_s",
        } <= event.payload.keys()
        assert event.payload["target_id"] == "target_00"
        assert "error_reason" not in event.payload


def test_engine_emits_navigation_recovery_failure_once_with_reason(tmp_path) -> None:
    base = load_app_config(CONFIG_PATH)
    config = base.model_copy(
        update={"timing": base.timing.model_copy(update={"physics_step_s": 1})}
    )
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    target = _boundary_event_target(timeout_s=2.0)
    engine._targets["target_00"] = target
    engine._target_intents["target_00"] = target.intent

    for _ in range(2):
        engine.step()

    failures = [
        event
        for event in engine.events()
        if event.event_type == "target_navigation_recovery_failed"
    ]
    assert len(failures) == 1
    assert failures[0].payload["old_state"] == "BOUNDARY_DECELERATING"
    assert failures[0].payload["new_state"] == "FAILED"
    assert failures[0].payload["error_reason"] == "boundary_recovery_timeout"
    assert failures[0].payload["state_age_s"] == pytest.approx(2.0)


def test_internal_engine_without_output_directory_does_not_create_a_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    engine = SimulationEngine(load_app_config(CONFIG_PATH), seed=7)
    try:
        engine.step()
        assert engine.logger.count == 1
        assert not (tmp_path / "outputs").exists()
    finally:
        engine.logger.close()


def test_engine_publishes_active_inputs_and_per_uuv_capability(tmp_path) -> None:
    """Queued operational inputs appear only in the next carrier snapshot."""
    base = load_app_config(CONFIG_PATH)
    capability = SurveillanceCapability(
        passive_range_m=5_000.0,
        active_range_m=2_000.0,
        bearing_variance_rad2=0.02,
        active_sonar_available=True,
        max_speed_mps=3.0,
        max_turn_rate_rad_s=0.04,
    )
    config = base.model_copy(
        update={
            "tracking": base.tracking.model_copy(
                update={"uuv_capabilities": {"uuv_00": capability}}
            )
        }
    )
    snapshots = []
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path, carrier=snapshots.append)
    scheme = OperationalScheme(
        scheme_id="night-watch",
        version=2,
        target_priorities={"target_00": 1.0},
        minimum_quality={"target_00": 0.8},
        valid_from_s=0,
        valid_until_s=120,
    )
    current = IntelligenceReport(
        report_id="intel-current",
        source=IntelligenceSource.SONAR,
        target_id="target_00",
        confidence=0.8,
        issued_at_s=0,
        valid_until_s=60,
        assessment={"intent": "evade"},
    )
    future = IntelligenceReport(
        report_id="intel-future",
        source=IntelligenceSource.SIGINT,
        target_id="target_00",
        confidence=0.7,
        issued_at_s=60,
        valid_until_s=120,
        assessment={"activity": "intermittent"},
    )

    engine.set_operational_scheme(scheme)
    engine.submit_intelligence(current)
    engine.submit_intelligence(future)
    for _ in range(config.timing.observation_step_s // config.timing.physics_step_s):
        engine.step()

    snapshot = snapshots[-1]
    assert snapshot.operational_scheme == scheme
    assert snapshot.intelligence_reports == (current,)
    assert {
        event.event_type for event in snapshot.pending_events
    } >= {"operational_scheme_updated", "intelligence_report_received"}
    uuv = next(state for state in snapshot.uuvs if state.uuv_id == "uuv_00")
    assert uuv.capability == capability
    target = next(contact for contact in snapshot.contacts if contact.contact_id == "target_00")
    assert target.bearing_rays
    observations = {ray.uuv_id: ray for ray in target.bearing_rays}
    if "uuv_00" in observations:
        assert observations["uuv_00"].variance_rad2 == capability.bearing_variance_rad2


def test_long_run_belief_history_uses_configured_retention(tmp_path) -> None:
    base = load_app_config(CONFIG_PATH)
    config = base.model_copy(
        update={
            "agent": AgentConfig(
                retention=RuntimeRetentionConfig(belief_history_limit=3)
            )
        }
    )
    engine = SimulationEngine(
        config,
        seed=7,
        output_dir=tmp_path,
        carrier=lambda _snapshot: None,
    )

    for sim_time_s in range(0, 300, 30):
        report = engine._latest_reports["target_00"]
        engine._latest_reports["target_00"] = report.model_copy(update={"belief": report.belief.model_copy(update={
            "sim_time_s": sim_time_s, "last_observed_at_s": sim_time_s,
            "accepted_observation_ids_this_cycle": (f"retention-fixture:{sim_time_s}",),
        })})
        engine._record_belief_history(sim_time_s)

    assert engine.belief_history("target_00") == (
        (210, engine.belief_history("target_00")[-3][1], engine.belief_history("target_00")[-3][2]),
        (240, engine.belief_history("target_00")[-2][1], engine.belief_history("target_00")[-2][2]),
        (270, engine.belief_history("target_00")[-1][1], engine.belief_history("target_00")[-1][2]),
    )


def test_long_platform_core_run_keeps_group_checkpoint_storage_bounded(tmp_path) -> None:
    config = load_app_config("configs/scenario/segmented_single_target.yaml")
    engine = SimulationEngine(config, seed=42, output_dir=tmp_path)
    samples: list[tuple[int, int, int]] = []
    timing_samples: list[float] = []
    sample_started = perf_counter()

    for step in range(1, 241):
        engine.step()
        if step % 30 == 0:
            saver = engine._manager._checkpointer
            checkpoint_count = sum(
                len(checkpoints)
                for namespaces in saver.storage.values()
                for checkpoints in namespaces.values()
            )
            samples.append((checkpoint_count, len(saver.writes), len(saver.blobs)))
            now = perf_counter()
            timing_samples.append(now - sample_started)
            sample_started = now

    retention = (
        config.agent.retention
        if config.agent is not None
        else RuntimeRetentionConfig()
    )
    assert samples
    assert len(timing_samples) == len(samples) == 8
    assert all(
        checkpoints <= retention.group_checkpoint_limit
        for checkpoints, _, _ in samples
    )
    assert max(writes for _, writes, _ in samples) <= retention.group_checkpoint_limit
    first_blob_count = samples[0][2]
    assert all(
        blobs <= first_blob_count + retention.group_checkpoint_limit
        for _, _, blobs in samples
    )
    assert engine._manager.list_groups() == ("target_00",)
    assert engine.publication_situation().group_reports


def test_fallback_capability_uses_configured_active_sonar_range(tmp_path) -> None:
    base = load_app_config(CONFIG_PATH)
    config = base.model_copy(
        update={
            "tracking": base.tracking.model_copy(
                update={"sensor_active_range_m": 500.0, "uuv_capabilities": None}
            )
        }
    )

    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)

    assert engine._uuvs["uuv_00"].capability.active_range_m == 500.0


def test_legacy_default_frame_remains_backward_compatible(tmp_path):
    engine = SimulationEngine(load_app_config(CONFIG_PATH), seed=42, output_dir=tmp_path)

    frame = engine.step()

    assert frame["platform_core"] is False
    assert frame["usvs"] == []
    assert frame["communication_links"] == []
    assert len(frame["uuvs"]) == 12


def test_verification_audit_redacts_submarine_depth_truth(tmp_path):
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(
        config,
        seed=config.scenario.seed,
        output_dir=tmp_path,
        verification_audit=True,
    )

    payload = engine.verification_audit()
    target_audit = next(item for item in payload["audits"] if item["entity_id"] == "target_00")
    target_limits = payload["limits"]["target_00"]

    for field in (
        "min_depth_m",
        "max_depth_m",
        "max_vertical_speed_mps",
        "max_vertical_acceleration_mps2",
        "max_pitch_rad",
    ):
        assert target_audit[field] is None
        assert target_limits[field] is None


def test_uuv_only_production_frames_omit_legacy_usv_projection(tmp_path):
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=config.scenario.seed, output_dir=tmp_path)

    frame = engine.step()
    raw_lines = (tmp_path / "frames.jsonl").read_text(encoding="utf-8").splitlines()

    assert frame["uuvs"]
    assert "usvs" not in frame
    assert len(raw_lines) == 1
    assert "usvs" not in json.loads(raw_lines[0])


def test_uuv_only_rejects_legacy_plan_and_command_execution(tmp_path) -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=config.scenario.seed, output_dir=tmp_path)

    legacy_plan = TrackingPlan(
        plan_id="legacy-plan",
        scenario_id=config.scenario.scenario_id,
        revision=1,
        base_snapshot_revision=0,
    )
    legacy_command = PlanCommand(
        command_id="legacy-command",
        plan_id="legacy-plan",
        plan_revision=1,
        scenario_id=config.scenario.scenario_id,
        group_id="G-target",
        target_id="target_00",
        sim_time_s=0,
        member_ids=("uuv_00",),
    )

    with pytest.raises(ValueError, match="legacy.*UUV-only"):
        engine.apply_tracking_plan(legacy_plan)
    with pytest.raises(ValueError, match="legacy.*UUV-only"):
        engine.apply_plan_command(legacy_command)


def test_uuv_only_return_route_discards_completed_service_stops(tmp_path) -> None:
    """A moving rendezvous tail must not reinterpret old mission indices."""
    from underwater_tracking.cli import _mission_controller_for

    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(
        config,
        seed=config.scenario.seed,
        output_dir=tmp_path,
        mission_controller=_mission_controller_for(config),
    )
    carrier = engine._carrier_entities["carrier_02"]
    start = carrier.position_xy
    old_stop = (start[0] + 500.0, start[1])
    old_endpoint = (start[0] + 200.0, start[1] - 900.0)
    carrier.set_mission_route(
        (start, old_stop, old_endpoint),
        stop_windows={1: (0, 100)},
        rendezvous_xy=old_endpoint,
    )
    # Simulate the external recovery handshake having released the old stop.
    carrier._mission_route_index = 2
    engine._mission_stop_ids["carrier_02"] = ("recover:target_00:cell:0:0",)
    engine._mission_stop_indices["carrier_02"] = (1,)
    engine._mission_stop_windows["carrier_02"] = {1: (0, 100)}

    assert engine._begin_carrier_rendezvous_return("carrier_02", 0)
    assert engine._mission_stop_ids["carrier_02"] == ()
    assert engine._mission_stop_indices["carrier_02"] == ()
    assert engine._mission_stop_windows["carrier_02"] == {}
    assert engine._carrier_committed_service_stops("carrier_02", 0) == ()


def test_uuv_only_plan_closes_adversary_response_chain(tmp_path) -> None:
    """A post-maneuver verified plan closes the blue-response audit."""
    from underwater_tracking.cli import _mission_controller_for
    from underwater_tracking.domain.adversary_models import AdversaryIntentDecision
    from underwater_tracking.domain.mission_models import (
        CarrierMissionModel,
        ExecutableMissionPlan,
        RegionMissionState,
    )

    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = _mission_controller_for(config)
    engine = SimulationEngine(
        config,
        seed=config.scenario.seed,
        output_dir=tmp_path,
        mission_controller=controller,
    )
    engine._clock.sim_time_s = 90
    decision = AdversaryIntentDecision(
        decision_id="target_00:decision:90",
        target_id="target_00",
        intent="avoid_contact",
        confidence=0.9,
        rationale="active sonar exposure requires a bounded course change",
        trigger_event_ids=(),
    )
    engine.apply_adversary_intent(decision)
    engine.step()
    plan = ExecutableMissionPlan(
        revision=1,
        region_assignments=(
            RegionMissionState(
                region_id="target_00:response",
                target_id="target_00",
                active_scan_uuv_ids=("uuv_00",),
                passive_track_uuv_ids=("uuv_01",),
            ),
        ),
        carrier_missions={
            carrier_id: CarrierMissionModel(
                carrier_id=carrier_id,
                home_battle_group_id=f"{carrier_id}:battle-group",
            )
            for carrier_id in engine._carrier_entities
        },
        resource_episode_by_uuv={uuv_id: 0 for uuv_id in engine._uuvs},
    )

    assert engine.apply_verified_mission_plan(plan)
    assert engine._completed_maneuver_response_chains
    response = engine._completed_maneuver_response_chains[-1]
    assert response["decision_id"] == decision.decision_id
    assert response["plan_version"] == 1
    assert any(event.event_id.endswith(":blue_response") for event in engine.events())


def test_uuv_only_does_not_spawn_or_observe_injected_usvs(tmp_path) -> None:
    uuv_only = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    mixed = load_app_config("configs/scenario/segmented_single_target.yaml")
    assert uuv_only.environment is not None and mixed.environment is not None
    environment = uuv_only.environment.model_copy(
        update={"usvs": (mixed.environment.usvs[0],)}
    )
    config = uuv_only.model_copy(update={"environment": environment})

    engine = SimulationEngine(config, seed=config.scenario.seed, output_dir=tmp_path)
    frame = engine.step()

    assert engine._usvs == {}
    assert engine.platform_snapshot().roster.usvs == ()
    assert "usvs" not in frame


def test_adversary_decision_evidence_keeps_exact_provider_call_id(tmp_path) -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=config.scenario.seed, output_dir=tmp_path)
    decision = AdversaryIntentDecision(
        decision_id="target_00:provider-decision:0",
        target_id="target_00",
        intent="avoid_contact",
        confidence=0.9,
        rationale="Local target-side contact evidence requires avoidance.",
    )

    engine.apply_adversary_decision(decision, provider_call_id="LLM-17")

    evidence = engine.verification_evidence()
    recorded = next(
        item
        for item in evidence["adversary_decisions"]
        if item["decision_id"] == decision.decision_id
    )
    assert recorded["provider_call_id"] == "LLM-17"


def test_verification_evidence_waits_for_the_engine_state_lock() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=7, verification_audit=True)
    finished = Event()
    errors: list[BaseException] = []

    def read_evidence() -> None:
        try:
            engine.verification_evidence()
        except BaseException as exc:  # pragma: no cover - assertion below reports it
            errors.append(exc)
        finally:
            finished.set()

    engine._state_lock.acquire()
    reader = Thread(target=read_evidence)
    reader.start()
    try:
        assert not finished.wait(0.05)
    finally:
        engine._state_lock.release()
    assert finished.wait(1.0)
    reader.join(timeout=1.0)
    assert errors == []


def test_uuv_only_execution_rejects_stale_low_energy_resource(tmp_path) -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    from underwater_tracking.domain.mission_models import (
        CarrierMissionModel,
        ExecutableMissionPlan,
        RegionMissionState,
        UUVMissionBatch,
    )
    from underwater_tracking.runtime.mission_controller import MissionController

    assert config.environment is not None
    home = config.environment.carrier.position_xy
    plan = ExecutableMissionPlan(
        revision=1,
        uuv_batches_by_carrier={
            "carrier_01": (
                UUVMissionBatch(
                    carrier_id="carrier_01",
                    candidate_id="R1",
                    uuv_ids=("uuv_00",),
                    active_scan_uuv_ids=("uuv_00",),
                    deployment_point=(home[0] + 10.0, home[1]),
                    recovery_point=(home[0] + 20.0, home[1]),
                    entry_s=0,
                    exit_s=100,
                ),
            )
        },
        region_assignments=(
            RegionMissionState(
                region_id="R1",
                target_id="target_00",
                active_scan_uuv_ids=("uuv_00",),
            ),
        ),
        carrier_missions={
            "carrier_01": CarrierMissionModel(
                carrier_id="carrier_01",
                home_battle_group_id="carrier_battle_group_01",
                ready_uuv_ids=("uuv_00",),
            )
        },
    )
    controller = MissionController(scenario_id=config.scenario.scenario_id)
    engine = SimulationEngine(config, seed=config.scenario.seed, output_dir=tmp_path, mission_controller=controller)
    engine._uuvs["uuv_00"].energy_fraction = 0.05

    assert engine.apply_verified_mission_plan(plan) is False
    assert controller.snapshot().plan_revision == 0


def test_public_belief_changes_require_two_confirmed_observations(tmp_path) -> None:
    config = load_app_config(CONFIG_PATH)
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    baseline = engine._latest_reports["target_00"]
    baseline = baseline.model_copy(
        update={
            "belief": baseline.belief.model_copy(
                update={"source_observation_ids": ("obs-1",)}
            )
        }
    )
    engine._latest_reports["target_00"] = baseline
    engine._emit_belief_change_events(0)
    estimate_event = next(
        event
        for event in engine._events
        if event.event_type == "target_estimate_updated"
    )
    assert estimate_event.level is EventLevel.TACTICAL
    assert estimate_event.payload["plan_impact"] is True
    engine._latest_reports["target_00"] = baseline.model_copy(
        update={
            "belief": baseline.belief.model_copy(
                update={
                    "model_probabilities": {"left_turn": 0.9, "cv": 0.1},
                    "source_observation_ids": ("obs-2",),
                }
            )
        }
    )

    engine._emit_belief_change_events(30)

    refresh_event = next(
        event
        for event in engine._events
        if event.event_type == "target_estimate_updated" and event.sim_time_s == 30
    )
    assert refresh_event.level is EventLevel.TACTICAL
    assert refresh_event.payload["plan_impact"] is False

    assert not {
        event.event_type
        for event in engine._events
        if event.event_type in {"imm_motion_mode_changed", "imm_confidence_shifted"}
    }

    engine._emit_belief_change_events(60)

    events = {
        event.event_type: event
        for event in engine._events
        if event.event_type in {"imm_motion_mode_changed", "imm_confidence_shifted"}
    }
    assert set(events) == {"imm_motion_mode_changed", "imm_confidence_shifted"}
    assert events["imm_motion_mode_changed"].payload["source"] == "public_imm_belief"
    assert events["imm_motion_mode_changed"].payload["motion_model"] == "left_turn"
    assert "target_intent_changed" not in {
        event.event_type for event in engine._events
    }


def test_target_intent_is_visible_as_uncertain_adversary_state_before_llm_decision(
    tmp_path,
) -> None:
    config = load_app_config(
        Path(__file__).resolve().parents[2]
        / "configs/scenario/segmented_single_target.yaml"
    )
    engine = SimulationEngine(config, seed=42, output_dir=tmp_path)
    engine._targets["target_00"].intent = HiddenIntent.EVADE

    engine.step()

    snapshot = engine._build_situation(engine._clock.sim_time_s)
    summary = snapshot.adversary_summaries[0]
    assert summary.intent == "evade"
    assert summary.maneuver == "decoy_evasion"
    assert summary.decision_status == "inconclusive"
    assert any(
        event.event_type == "target_maneuver"
        for event in snapshot.pending_events
    )


@pytest.mark.parametrize(
    "legacy_fields",
    [
        {"usv_ids": ("usv_00",)},
        {"usv_actions": {"usv_00": "relay"}},
    ],
)
def test_plan_command_rejects_legacy_usv_fields(legacy_fields: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="usv"):
        PlanCommand(
            command_id="legacy-usv-command",
            plan_id="legacy-usv-plan",
            plan_revision=1,
            scenario_id="scenario-1",
            group_id="G-relay",
            region_id="target_00:cell:0:0",
            target_id="target_00",
            sim_time_s=0,
            **legacy_fields,
        )


def test_failed_explicit_tick_restores_belief_gate_state(tmp_path) -> None:
    base = load_app_config(
        Path(__file__).resolve().parents[2] / "configs/scenario/segmented_single_target.yaml"
    )
    config = base.model_copy(
        update={"timing": base.timing.model_copy(update={"physics_step_s": 30})}
    )
    engine: SimulationEngine

    def mutate_then_fail(_: object) -> None:
        engine._belief_intent_candidates["target_00"] = ("evade", 9)
        engine._belief_confidence_candidates["target_00"] = (0.3, 9)
        engine._belief_confidence_latches.add("target_00")
        raise RuntimeError("carrier failed after replan update")

    engine = SimulationEngine(config, seed=7, output_dir=tmp_path, carrier=mutate_then_fail)
    with pytest.raises(RuntimeError, match="carrier failed after replan update"):
        engine.step()

    assert engine._belief_intent_candidates == {}
    assert engine._belief_confidence_candidates == {}
    assert engine._belief_confidence_latches == set()


def test_adversary_maneuver_records_regional_response_latency(tmp_path) -> None:
    from underwater_tracking.domain.adversary_models import AdversaryEscapeDecision

    engine = SimulationEngine(load_app_config(CONFIG_PATH), seed=7, output_dir=tmp_path)
    engine.apply_adversary_decision(
        AdversaryEscapeDecision(
            target_id="target_00",
            maneuver="course_change",
            intent="evade",
            waypoint=(100.0, 200.0),
            segment="target-owned-current",
            speed=4.0,
            heading=0.2,
            decoy_action="none",
            decoy_count=0,
            confidence=0.8,
            rationale="Change course after target-side detection evidence.",
            communications_discipline="silent",
        )
    )
    engine.step()
    command = PlanCommand(
        command_id="blue-response",
        plan_id="plan-blue-response",
        plan_revision=3,
        scenario_id=engine._scenario_id,
        group_id="G-target_00",
        region_id="target_00:cell:0:0",
        target_id="target_00",
        sim_time_s=engine._clock.sim_time_s,
        member_ids=("uuv_00", "uuv_01"),
        actions={"uuv_00": "track", "uuv_01": "track"},
    )

    engine.apply_plan_command(command)

    phases = {
        event.payload.get("phase")
        for event in (*engine._events, *engine._pending_runtime_events)
        if event.entity_id == "target_00"
    }
    assert {
        "target_maneuver",
        "prediction_revision",
        "regional_task_revision",
        "effect_change",
        "blue_response",
    } <= phases
    response = next(
        event for event in engine._pending_runtime_events if event.payload.get("phase") == "blue_response"
    )
    assert response.payload["latency_s"] >= 0


def test_only_new_regional_or_relay_plan_closes_a_maneuver_response_chain(tmp_path) -> None:
    from underwater_tracking.domain.adversary_models import AdversaryEscapeDecision

    engine = SimulationEngine(load_app_config(CONFIG_PATH), seed=7, output_dir=tmp_path)
    unrelated = PlanCommand(
        command_id="unrelated",
        plan_id="unrelated",
        plan_revision=1,
        scenario_id=engine._scenario_id,
        group_id="G-target_00",
        target_id="target_00",
        sim_time_s=0,
        member_ids=("uuv_00", "uuv_01"),
        actions={"uuv_00": "track", "uuv_01": "track"},
    )
    engine.apply_plan_command(unrelated)
    engine.apply_adversary_decision(
        AdversaryEscapeDecision(
            target_id="target_00",
            maneuver="course_change",
            intent="evade",
            waypoint=(100.0, 200.0),
            segment="target-owned-current",
            speed=4.0,
            heading=0.2,
            decoy_action="none",
            decoy_count=0,
            confidence=0.8,
            rationale="Change course after target-side detection evidence.",
            communications_discipline="silent",
        )
    )
    engine.step()
    with pytest.raises(ValueError, match="stale plan revision"):
        engine.apply_plan_command(
            unrelated.model_copy(
                update={
                    "command_id": "stale-regional-response",
                    "region_id": "target_00:cell:0:0",
                }
            )
        )
    assert "target_00" in engine._maneuver_response_chains

    engine.apply_plan_command(
        unrelated.model_copy(
            update={
                "command_id": "regional-response",
                "plan_id": "regional-response",
                "plan_revision": 2,
                "region_id": "target_00:cell:0:0",
            }
        )
    )
    assert "target_00" not in engine._maneuver_response_chains
    response = next(
        event
        for event in engine._pending_runtime_events
        if event.payload.get("phase") == "blue_response"
    )
    assert {"chain_id", "decision_id", "prediction_revision", "plan_revision", "latency_s"} <= set(
        response.payload
    )


def test_only_regional_uuv_tracking_commands_close_a_maneuver_response_chain(tmp_path) -> None:
    from underwater_tracking.domain.adversary_models import AdversaryEscapeDecision

    config = load_app_config(CONFIG_PATH)
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    engine.apply_adversary_decision(
        AdversaryEscapeDecision(
            target_id="target_00",
            maneuver="course_change",
            intent="evade",
            waypoint=(100.0, 200.0),
            segment="target-owned-current",
            speed=4.0,
            heading=0.2,
            decoy_action="none",
            decoy_count=0,
            confidence=0.8,
            rationale="Change course after target-side detection evidence.",
            communications_discipline="silent",
        )
    )
    engine.step()

    commands = (
        PlanCommand(
            command_id="unrelated-uuv-track",
            plan_id="unrelated-uuv-track",
            plan_revision=1,
            scenario_id=config.scenario.scenario_id,
            group_id="G-target_00",
            target_id="target_00",
            sim_time_s=0,
            member_ids=("uuv_00", "uuv_01"),
            actions={"uuv_00": "track", "uuv_01": "track"},
        ),
        PlanCommand(
            command_id="unrelated-uuv-noop",
            plan_id="unrelated-uuv-noop",
            plan_revision=2,
            scenario_id=config.scenario.scenario_id,
            group_id="G-target_00",
            target_id="target_00",
            sim_time_s=0,
        ),
    )
    for command in commands:
        engine.apply_plan_command(command)

    assert "target_00" in engine._maneuver_response_chains
    assert not {
        "regional_task_revision",
        "effect_change",
        "blue_response",
    } & {event.payload.get("phase") for event in engine._pending_runtime_events}

    engine.apply_plan_command(
        PlanCommand(
            command_id="regional-uuv-track",
            plan_id="regional-uuv-track",
            plan_revision=3,
            scenario_id=config.scenario.scenario_id,
            group_id="G-target_00",
            region_id="target_00:cell:0:0",
            target_id="target_00",
            sim_time_s=0,
            member_ids=("uuv_00", "uuv_01"),
            actions={"uuv_00": "track", "uuv_01": "track"},
        )
    )

    assert "target_00" not in engine._maneuver_response_chains
