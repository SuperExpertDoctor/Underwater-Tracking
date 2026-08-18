"""Adaptive tracking inputs exposed by the deterministic engine."""

from __future__ import annotations

from pathlib import Path

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.agent_models import PlanCommand, Waypoint
from underwater_tracking.domain.models import (
    IntelligenceReport,
    IntelligenceSource,
    OperationalScheme,
    SurveillanceCapability,
)
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
    for _ in range(3):
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
