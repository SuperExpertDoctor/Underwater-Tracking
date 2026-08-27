from __future__ import annotations

from pathlib import Path

from underwater_tracking.cli import _AgentLoop, _mission_controller_for, _step_with_llm_retries
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.simulation.engine import SimulationEngine
from tests.integration.test_uuv_only_production_acceptance import FixedSeedUUVLLM


CONFIG_PATH = "configs/scenario/uuv_only_single_target.yaml"
SEED = 20260820


def _has_event(frame: object, event_type: str) -> bool:
    return any(event.event_type == event_type for event in frame.events)  # type: ignore[attr-defined]


def test_default_entry_starts_uuvs_from_region_boundary_without_carrier_staging(
    tmp_path: Path,
) -> None:
    config = load_app_config(CONFIG_PATH)
    assert config.environment is not None
    controller = _mission_controller_for(config)
    assert controller is not None
    llm = FixedSeedUUVLLM()
    loop = _AgentLoop(
        config,
        database_path=tmp_path / "agent.db",
        llm={"master": llm},
        run_id="truthful-bootstrap-deployment",
        steps=40,
        seed=SEED,
    )
    engine = SimulationEngine(
        config,
        seed=SEED,
        output_dir=tmp_path / "frames",
        carrier=loop.on_situation,
        mission_controller=controller,
    )
    frames: list[object] = []
    try:
        loop.attach(engine)
        baseline = loop.install_deterministic_baseline(engine.publication_situation())
        assert baseline is not None
        authoritative = loop.runtime.active_mission_plan()
        assert authoritative is not None
        assert len(authoritative.task_groups) == 4
        assert len(authoritative.reserve_uuvs) == 4
        audit_baseline = loop.runtime.active_plan()
        assert audit_baseline is not None
        assert audit_baseline.revision == baseline.revision == 1
        assert audit_baseline.concept == "hold_current"
        assert audit_baseline.regional_plans
        assert loop.plans.get_active(config.scenario.scenario_id) == audit_baseline
        assert audit_baseline.evidence_ids
        assert all(
            loop.events.get(evidence_id) is not None
            for evidence_id in audit_baseline.evidence_ids
        )
        baseline_frame = loop.hub.snapshot()
        assert baseline_frame is not None
        frames.append(baseline_frame)
        entry_frame = None
        entry_positions: dict[str, tuple[float, float]] = {}
        for _ in range(40):
            assert _step_with_llm_retries(engine, loop, config) is True
            frame = loop.hub.snapshot()
            assert frame is not None
            frames.append(frame)
            deployed_positions = {
                uuv.uuv_id: (uuv.position.x, uuv.position.y)
                for uuv in frame.uuvs
                if uuv.deployment_state == "deployed"
            }
            if entry_frame is None and deployed_positions:
                entry_frame = frame
                entry_positions = deployed_positions
                continue
            if any(
                uuv.uuv_id in entry_positions
                and (uuv.position.x, uuv.position.y) != entry_positions[uuv.uuv_id]
                for uuv in frame.uuvs
            ):
                break
    finally:
        assert loop.close() is True
        engine.logger.close()

    assert "target_priors" not in baseline_frame.model_dump()
    assert len(baseline_frame.target_estimates) == 1
    assert baseline_frame.target_estimates[0].classification == "submarine"
    assert baseline_frame.plan_version >= 1
    assert len(baseline_frame.uuv_resources) == 12
    assert entry_frame is not None
    assert entry_positions
    assert baseline.batches
    assert all(
        batch.deployment_point is None and batch.recovery_point is None
        for batch in baseline.batches
    )
    latest = frames[-1]
    assert latest.frame_id > baseline_frame.frame_id  # type: ignore[attr-defined]
    assert any(
        uuv.uuv_id in entry_positions
        and (uuv.position.x, uuv.position.y) != entry_positions[uuv.uuv_id]
        for uuv in latest.uuvs  # type: ignore[attr-defined]
    )
    assert any(_has_event(frame, "uuv_boundary_entry_started") for frame in frames)
    assert not any(
        _has_event(frame, event_type)
        for frame in frames
        for event_type in ("uuv_deployed", "uuv_recovery_started", "uuv_recovered")
    )
    assert "usv" not in latest.model_dump_json().casefold()  # type: ignore[attr-defined]
    assert latest.carrier is None  # type: ignore[attr-defined]
    assert latest.carriers == ()  # type: ignore[attr-defined]
    assert latest.carrier_missions == ()  # type: ignore[attr-defined]

    current_situation = engine.publication_situation()
    slave_contexts = engine.build_slave_contexts(current_situation)
    execution_members = {
        member
        for group in current_situation.execution_groups
        for member in group.member_ids
    }
    assert all(
        {platform.platform_id for platform in context.platforms} <= execution_members
        for context in slave_contexts
    )

    first = SimulationEngine(config, seed=SEED)
    second = SimulationEngine(config, seed=SEED)
    assert first.publication_situation().model_dump_json() == second.publication_situation().model_dump_json()
