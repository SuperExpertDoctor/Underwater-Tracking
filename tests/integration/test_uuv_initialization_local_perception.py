from __future__ import annotations

import json
from pathlib import Path

from underwater_tracking.agent.nodes.adversary import AdversaryDecisionGate
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.models import DeploymentState
from underwater_tracking.memory.source_reader import MemorySourceReader
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.persistence.memory import LongTermMemoryRepository
from underwater_tracking.persistence.plans import PlanRepository
from underwater_tracking.simulation.engine import SimulationEngine
from underwater_tracking.simulation.adversary_sensing import (
    ExposedPlatform,
    update_local_platform_detections,
)
from tests.integration.test_uuv_only_mission_acceptance import (
    assert_uuv_only_acceptance,
    run_uuv_only_acceptance,
)
from tests.integration.test_uuv_only_production_acceptance import (
    FixedSeedUUVLLM,
)
from underwater_tracking.cli import _AgentLoop, _mission_controller_for


def _assert_truth_safe(value: object) -> None:
    forbidden = ("truth", "private", "chain_of_thought", "gate_distance")
    if isinstance(value, dict):
        for key, child in value.items():
            assert not any(fragment in str(key).casefold() for fragment in forbidden)
            _assert_truth_safe(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _assert_truth_safe(child)
    elif isinstance(value, str):
        assert not any(fragment in value.casefold() for fragment in forbidden)


def test_real_uuv_default_timeline_local_perception_and_periodic_memory(
    tmp_path: Path,
) -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    assert config.scenario.uuv_only is True
    assert config.environment is not None
    assert len((config.environment.carrier, *config.environment.carriers)) == 4
    assert sum(carrier.role == "mother_ship" for carrier in config.environment.carriers) == 3

    loop = _AgentLoop(
        config,
        database_path=tmp_path / "agent.db",
        llm={"master": FixedSeedUUVLLM()},
        run_id="uuv-initialization-local-perception",
        steps=430,
        seed=20260820,
    )
    controller = _mission_controller_for(config)
    assert controller is not None
    engine = SimulationEngine(
        config,
        seed=20260820,
        output_dir=tmp_path / "frames",
        carrier=loop.on_situation,
        mission_controller=controller,
    )
    initial = engine.publication_situation()
    assert len(initial.carriers) == 4
    assert len(initial.uuvs) == 12
    assert all(uuv.deployment_state is DeploymentState.ONBOARD for uuv in initial.uuvs)
    assert not hasattr(initial, "usvs")
    assert {carrier.role for carrier in initial.carriers} == {"carrier", "mother_ship"}

    frames: list[dict[str, object]] = []
    try:
        loop.attach(engine)
        for _ in range(430):
            frames.append(engine.step())
    finally:
        assert loop.close() is True
        engine.logger.close()

    events = engine.events()
    event_types = [event.event_type for event in events]
    deployment_events = [event for event in events if event.event_type == "uuv_deployed"]
    recovery_request_events = [
        event for event in events if event.event_type == "uuv_recovery_requested"
    ]
    recovered_events = [event for event in events if event.event_type == "uuv_recovered"]
    returned_events = [
        event
        for event in events
        if event.event_type == "carrier_returned_to_fleet"
        and event.entity_id == "carrier_02"
    ]
    assert deployment_events
    assert recovery_request_events
    assert recovered_events
    assert len(returned_events) <= 1
    assert event_types.index("carrier_dispatch_completed") < event_types.index("uuv_deployed")
    assert event_types.index("uuv_deployed") < event_types.index("uuv_recovery_requested")
    assert event_types.index("uuv_recovery_requested") < event_types.index("uuv_recovered")
    if returned_events:
        assert event_types.index("uuv_recovered") < event_types.index(
            "carrier_returned_to_fleet"
        )

    first_deploy_s = min(event.sim_time_s for event in deployment_events)
    assert first_deploy_s == 500
    if returned_events:
        assert returned_events[0].sim_time_s >= max(
            event.sim_time_s for event in recovered_events
        )
    assert all(
        "usv" not in json.dumps(frame, sort_keys=True).casefold()
        for frame in frames
    )

    waterborne_by_time = {
        frame["sim_time_s"]: {
            uuv["platform_id"]
            for uuv in frame["uuvs"]
            if uuv["deployment_state"] in {"deployed", "returning", "failed"}
        }
        for frame in frames
    }
    assert all(
        not visible_ids
        for sim_time_s, visible_ids in waterborne_by_time.items()
        if sim_time_s < first_deploy_s
    )
    deployed_ids = {event.entity_id for event in deployment_events}
    first_post_deploy_s = min(
        sim_time_s for sim_time_s in waterborne_by_time if sim_time_s >= first_deploy_s
    )
    assert deployed_ids <= waterborne_by_time[first_post_deploy_s]

    mission = engine.mission_snapshot()
    assert mission is not None
    routes = {
        uuv_id: tuple(route)
        for region in mission.regions
        for uuv_id, route in region.scan_waypoints_by_uuv.items()
        if uuv_id in deployed_ids and route
    }
    assert len(routes) >= 2
    assert len(set(routes.values())) >= 1
    assert all(len(route) >= 2 for route in routes.values())

    timeline_trace = run_uuv_only_acceptance(20260820)
    assert_uuv_only_acceptance(timeline_trace)
    assert "handoff_completed" in timeline_trace.event_types
    assert "PASSIVE_TRACK" in dict(timeline_trace.lifecycle_trace)["handoff-r1-r2"]

    event_repository = EventRepository(tmp_path / "agent.db")
    memory_repository = LongTermMemoryRepository(tmp_path / "agent.db")
    plan_repository = PlanRepository(tmp_path / "agent.db")
    try:
        summaries = event_repository.list_events(
            scenario_id=config.scenario.scenario_id,
            event_type="periodic_situation_summary",
        )
        assert summaries
        assert [event.sim_time_s for event in summaries] == sorted(
            {event.sim_time_s for event in summaries}
        )
        for event in summaries:
            _assert_truth_safe(event.payload)
            assert event.event_id == (
                f"periodic_situation_summary:{config.scenario.scenario_id}:"
                f"{event.sim_time_s}"
            )
        duplicate = event_repository.append_if_absent(
            event_id=summaries[0].event_id,
            event_type=summaries[0].event_type,
            scenario_id=summaries[0].scenario_id,
            sim_time_s=summaries[0].sim_time_s,
            target_id=summaries[0].target_id,
            severity=summaries[0].severity,
            payload=summaries[0].payload,
        )
        assert duplicate is None

        source_reader = MemorySourceReader(
            memory_repository,
            event_repository=event_repository,
            batch_limit=100,
        )
        sources = []
        for _ in range(32):
            page = source_reader.read_new("acceptance", config.scenario.scenario_id)
            if not page:
                break
            sources.extend(page)
            memory_repository.advance_source_cursor(
                "acceptance",
                config.scenario.scenario_id,
                "runtime_event",
                page[-1].cursor,
            )
            if any(
                source.source_event_ids
                and source.source_event_ids[0].startswith("periodic_situation_summary:")
                for source in page
            ):
                break
        summary_sources = [
            source
            for source in sources
            if source.source_type == "runtime_event"
            and source.source_event_ids
            and source.source_event_ids[0].startswith("periodic_situation_summary:")
        ]
        assert summary_sources
        assert all(source.text.startswith("time=") for source in summary_sources)

        plan_rows = plan_repository._conn.execute(
            "SELECT payload FROM plans WHERE scenario_id = ?",
            (config.scenario.scenario_id,),
        ).fetchall()
        assert all("periodic_situation_summary" not in row["payload"] for row in plan_rows)
    finally:
        plan_repository.close()
        memory_repository.close()
        event_repository.close()


def test_real_engine_local_perception_keeps_target_evidence_local_and_gated() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=7)
    target = engine._targets["target_00"]
    carrier = engine._carrier_entities["carrier_01"]
    target.position_xy = (0.0, 0.0)
    for carrier_id, other_carrier in engine._carrier_entities.items():
        if carrier_id != "carrier_01":
            other_carrier.position_xy = (6000.0, 0.0)

    carrier.position_xy = (1201.0, 0.0)
    engine._update_target_detection_events(0)
    assert engine.build_adversary_inputs(engine._build_situation(0)) == ()

    carrier.position_xy = (1199.0, 0.0)
    engine._update_target_detection_events(30)
    context = engine.build_adversary_inputs(engine._build_situation(30))[0]
    assert {threat.platform_id for threat in context.platform_threats} == {"carrier_01"}
    assert all("blue" not in observation.observation_id for observation in context.observations)
    assert all(
        "position_xy" not in threat.model_dump(mode="json")
        and "true_distance" not in threat.model_dump(mode="json")
        for threat in context.platform_threats
    )

    gate = AdversaryDecisionGate(cooldown_s=60)
    assert gate.should_request(context) is True
    gate.record_decision(context)
    stable_context = context.model_copy(update={"sim_time_s": 90})
    assert gate.should_request(stable_context) is False

    local_detection = update_local_platform_detections(
        target_id="target_00",
        target_position_xy=(0.0, 0.0),
        target_heading_rad=0.0,
        detection_range_m=1200.0,
        release_margin_m=100.0,
        candidates=(
            ExposedPlatform(
                platform_id="uuv_00",
                platform_kind="uuv",
                position_xy=(1300.0, 0.0),
                sensor_mode="active",
                relay_available=True,
            ),
        ),
        previous_ids=frozenset({"uuv_00"}),
        sim_time_s=30,
        seed=7,
    )
    assert local_detection.lost_platform_ids == frozenset()
    assert local_detection.audible_active_emitter_ids == frozenset()
