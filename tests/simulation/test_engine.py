"""Adaptive tracking inputs exposed by the deterministic engine."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import pytest

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.config.models import AgentConfig, RuntimeRetentionConfig
from underwater_tracking.domain.agent_models import PlanCommand, TrackingPlan, Waypoint
from underwater_tracking.domain.models import (
    IntelligenceReport,
    IntelligenceSource,
    OperationalScheme,
    SurveillanceCapability,
)
from underwater_tracking.simulation.target import HiddenIntent
from underwater_tracking.simulation.engine import SimulationEngine
from tests.conftest import CONFIG_PATH


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


def test_public_belief_changes_emit_strategic_intent_and_confidence_events(tmp_path) -> None:
    config = load_app_config(CONFIG_PATH)
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    baseline = engine._latest_reports["target_00"]
    engine._emit_belief_change_events(0)
    engine._latest_reports["target_00"] = baseline.model_copy(
        update={
            "belief": baseline.belief.model_copy(
                update={
                    "model_probabilities": {"evade": 0.9, "transit": 0.1}
                }
            )
        }
    )

    engine._emit_belief_change_events(30)

    events = {
        event.event_type: event
        for event in engine._events
        if event.event_type in {"target_intent_changed", "imm_confidence_shifted"}
    }
    assert set(events) == {"target_intent_changed", "imm_confidence_shifted"}
    assert events["target_intent_changed"].payload["source"] == "public_imm_belief"


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


def test_usv_only_relay_command_changes_execution_state(tmp_path) -> None:
    config_path = Path(__file__).resolve().parents[2] / "configs/scenario/segmented_single_target.yaml"
    config = load_app_config(config_path)
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    usv_id = next(iter(engine._usvs))
    start = engine._usvs[usv_id].motion.position_xy

    engine.apply_plan_command(
        PlanCommand(
            command_id="relay-only",
            plan_id="plan-relay-only",
            plan_revision=2,
            scenario_id=config.scenario.scenario_id,
            group_id="G-relay",
            region_id="target_00:cell:0:0",
            target_id="target_00",
            sim_time_s=0,
            usv_ids=(usv_id,),
            usv_actions={usv_id: "relay"},
            waypoints_by_member={usv_id: (Waypoint(x=start[0] + 400.0, y=start[1]),)},
        )
    )

    record = engine._usv_execution_records[usv_id]
    assert record["action"] == "relay"
    assert record["target_id"] == "target_00"
    engine.step()
    assert engine._usvs[usv_id].motion.position_xy[0] > start[0]


def test_plan_revision_rejects_stale_usv_command_before_execution_writes(tmp_path) -> None:
    config = load_app_config(
        Path(__file__).resolve().parents[2] / "configs/scenario/segmented_single_target.yaml"
    )
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    usv_id = next(iter(engine._usvs))
    start = engine._usvs[usv_id].motion.position_xy
    current = PlanCommand(
        command_id="relay-revision-4",
        plan_id="plan-revision-4",
        plan_revision=4,
        scenario_id=config.scenario.scenario_id,
        group_id="G-relay",
        region_id="target_00:cell:0:0",
        target_id="target_00",
        sim_time_s=0,
        usv_ids=(usv_id,),
        usv_actions={usv_id: "relay"},
        waypoints_by_member={usv_id: (Waypoint(x=start[0] + 400.0, y=start[1]),)},
    )
    engine.apply_plan_command(current)
    record = dict(engine._usv_execution_records[usv_id])
    waypoints = list(engine._usv_waypoints[usv_id])
    assert engine._applied_plan_revisions[(config.scenario.scenario_id, "target_00")] == 4

    with pytest.raises(ValueError, match="stale plan revision"):
        engine.apply_plan_command(
            current.model_copy(
                update={
                    "command_id": "hold-revision-3",
                    "plan_id": "plan-revision-3",
                    "plan_revision": 3,
                    "usv_actions": {usv_id: "hold"},
                    "waypoints_by_member": {},
                }
            )
        )

    assert engine._usv_execution_records[usv_id] == record
    assert engine._usv_waypoints[usv_id] == waypoints


def test_usv_hold_stays_at_zero_speed_until_a_new_action_clears_it(tmp_path) -> None:
    config = load_app_config(
        Path(__file__).resolve().parents[2] / "configs/scenario/segmented_single_target.yaml"
    )
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    usv_id = next(iter(engine._usvs))
    start = engine._usvs[usv_id].motion.position_xy
    base = dict(
        scenario_id=config.scenario.scenario_id,
        group_id="G-relay",
        region_id="target_00:cell:0:0",
        target_id="target_00",
        sim_time_s=0,
        usv_ids=(usv_id,),
    )
    engine.apply_plan_command(
        PlanCommand(
            command_id="relay-moving",
            plan_id="relay-moving",
            plan_revision=2,
            usv_actions={usv_id: "relay"},
            waypoints_by_member={usv_id: (Waypoint(x=start[0] + 400.0, y=start[1]),)},
            **base,
        )
    )
    engine.step()
    engine.apply_plan_command(
        PlanCommand(
            command_id="hold",
            plan_id="hold",
            plan_revision=3,
            usv_actions={usv_id: "hold"},
            **base,
        )
    )
    held_position = engine._usvs[usv_id].motion.position_xy
    engine.step()
    engine.step()

    held = engine._usvs[usv_id].motion
    assert held.position_xy == held_position
    assert held.speed_mps == 0.0

    engine.apply_plan_command(
        PlanCommand(
            command_id="return",
            plan_id="return",
            plan_revision=4,
            usv_actions={usv_id: "return"},
            **base,
        )
    )
    assert usv_id not in engine._usv_hold_ids
    for revision, action in ((5, "track"), (6, "relay")):
        engine.apply_plan_command(
            PlanCommand(
                command_id=f"{action}-{revision}",
                plan_id=f"{action}-{revision}",
                plan_revision=revision,
                usv_actions={usv_id: action},
                **base,
            )
        )
        assert usv_id not in engine._usv_hold_ids


def test_fast_regional_replan_events_require_two_observations_and_recover(tmp_path) -> None:
    config = load_app_config(
        Path(__file__).resolve().parents[2] / "configs/scenario/segmented_single_target.yaml"
    )
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    target_id = "target_00"
    usv_id = next(iter(engine._usvs))
    engine._latest_reports[target_id] = engine._latest_reports[target_id].model_copy(
        update={"quality": engine._latest_reports[target_id].quality.model_copy(update={"ewma": 0.0})}
    )
    engine._usv_execution_records[usv_id] = {
        "target_id": target_id,
        "region_id": "target_00:cell:0:0",
        "action": "relay",
    }
    engine._connectivity = engine._connectivity.__class__(links=())
    engine._update_fast_regional_replan_events(10)
    assert not engine._pending_runtime_events
    engine._update_fast_regional_replan_events(20)

    event_types = [event.event_type for event in engine._pending_runtime_events]
    assert event_types.count("regional_feedback_received") == 1
    assert event_types.count("communication_link_lost") == 1
    snapshot_event_types = {
        event.event_type for event in engine._build_situation(20).pending_events
    }
    assert {"regional_feedback_received", "communication_link_lost"} <= snapshot_event_types
    engine._update_fast_regional_replan_events(30)
    assert [event.event_type for event in engine._pending_runtime_events] == event_types

    engine._latest_reports[target_id] = engine._latest_reports[target_id].model_copy(
        update={"quality": engine._latest_reports[target_id].quality.model_copy(update={"ewma": 1.0})}
    )
    engine._rebuild_connectivity()
    engine._update_fast_regional_replan_events(40)
    assert target_id not in engine._regional_quality_latches
    assert usv_id not in engine._relay_failure_latches


def test_failed_explicit_tick_restores_fast_replan_counters(tmp_path) -> None:
    base = load_app_config(
        Path(__file__).resolve().parents[2] / "configs/scenario/segmented_single_target.yaml"
    )
    config = base.model_copy(
        update={"timing": base.timing.model_copy(update={"physics_step_s": 30})}
    )
    engine: SimulationEngine

    def mutate_then_fail(_: object) -> None:
        engine._regional_quality_streaks["target_00"] = 9
        engine._regional_quality_latches.add("target_00")
        engine._relay_failure_streaks["usv_00"] = 9
        engine._relay_failure_latches.add("usv_00")
        raise RuntimeError("carrier failed after replan update")

    engine = SimulationEngine(config, seed=7, output_dir=tmp_path, carrier=mutate_then_fail)
    with pytest.raises(RuntimeError, match="carrier failed after replan update"):
        engine.step()

    assert engine._regional_quality_streaks == {}
    assert engine._regional_quality_latches == set()
    assert engine._relay_failure_streaks == {}
    assert engine._relay_failure_latches == set()


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
        for event in engine._pending_runtime_events
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


def test_only_regional_tracking_or_relay_commands_close_a_maneuver_response_chain(tmp_path) -> None:
    from underwater_tracking.domain.adversary_models import AdversaryEscapeDecision

    config = load_app_config(
        Path(__file__).resolve().parents[2] / "configs/scenario/segmented_single_target.yaml"
    )
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    usv_id = next(iter(engine._usvs))
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
            command_id="unrelated-usv-hold",
            plan_id="unrelated-usv-hold",
            plan_revision=2,
            scenario_id=config.scenario.scenario_id,
            group_id="G-target_00",
            region_id="target_00:cell:0:0",
            target_id="target_00",
            sim_time_s=0,
            usv_ids=(usv_id,),
            usv_actions={usv_id: "hold"},
        ),
        PlanCommand(
            command_id="unrelated-usv-return",
            plan_id="unrelated-usv-return",
            plan_revision=3,
            scenario_id=config.scenario.scenario_id,
            group_id="G-target_00",
            region_id="target_00:cell:0:0",
            target_id="target_00",
            sim_time_s=0,
            usv_ids=(usv_id,),
            usv_actions={usv_id: "return"},
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
            command_id="regional-usv-relay",
            plan_id="regional-usv-relay",
            plan_revision=4,
            scenario_id=config.scenario.scenario_id,
            group_id="G-target_00",
            region_id="target_00:cell:0:0",
            target_id="target_00",
            sim_time_s=0,
            usv_ids=(usv_id,),
            usv_actions={usv_id: "relay"},
        )
    )

    assert "target_00" not in engine._maneuver_response_chains
