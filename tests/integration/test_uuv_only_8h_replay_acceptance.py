"""Full-duration deterministic production and replay acceptance.

The test is opt-in because it advances 5,760 physical steps and exercises the
complete agent-coupled production loop. It uses the same deterministic
structured provider as the shorter production acceptance; it does not inject
business frames or replace the backend replay path.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from underwater_tracking.api.replay import ReplayService
from underwater_tracking.cli import _AgentLoop
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.runtime.mission_controller import MissionController
from underwater_tracking.simulation.engine import SimulationEngine
from tests.integration.test_uuv_only_production_acceptance import (
    FixedSeedUUVLLM,
    _co_locate_test_carriers,
)


pytestmark = pytest.mark.long_running


@pytest.mark.skipif(
    os.environ.get("UNDERWATER_TRACKING_RUN_8H") != "1",
    reason="set UNDERWATER_TRACKING_RUN_8H=1 to run the full 8-hour acceptance",
)
def test_fixed_seed_uuv_only_runs_full_8h_and_replays_all_frames(tmp_path: Path) -> None:
    config = _co_locate_test_carriers(
        load_app_config("configs/scenario/uuv_only_single_target.yaml")
    )
    assert config.timing.physics_step_s == 5
    duration_s = config.scenario.duration_s
    steps = duration_s // config.timing.physics_step_s
    assert duration_s == 28_800
    assert steps == 5_760

    llm = FixedSeedUUVLLM()
    loop = _AgentLoop(
        config,
        database_path=tmp_path / "agent.db",
        llm={"master": llm},
        run_id="fixed-seed-uuv-8h",
        steps=steps,
        seed=20260820,
    )
    controller = MissionController(
        scenario_id=config.scenario.scenario_id,
        region_entry_probability_threshold=config.scenario.region_entry_probability_threshold,
        region_transition_confirm_cycles=config.scenario.region_transition_confirm_cycles,
    )
    engine = SimulationEngine(
        config,
        seed=20260820,
        output_dir=tmp_path / "frames",
        carrier=loop.on_situation,
        mission_controller=controller,
    )
    loop.attach(engine)
    try:
        for _ in range(steps):
            engine.step()
            # Exercise the same canonical operational-frame publisher used by
            # the live command center before replaying its persisted log.
            loop.publish_latest()
        final_plan = loop.runtime.active_mission_plan()
        engine_event_types = {event.event_type for event in engine.events()}
        llm_call_count = len(llm.calls)
    finally:
        loop.close()
        engine.logger.close()

    replay = ReplayService(tmp_path / "operational_frames.jsonl")
    frames = replay.range(start_s=0, end_s=duration_s, limit=None)
    assert len(frames) == steps == replay.count(0, duration_s)
    assert frames[0].sim_time_s == config.timing.physics_step_s
    assert frames[-1].sim_time_s == duration_s
    assert all(
        left.sim_time_s < right.sim_time_s
        for left, right in zip(frames, frames[1:])
    )

    for frame in frames:
        assert all(-math.pi <= uuv.heading_rad < math.pi for uuv in frame.uuvs)
    assert all(
        "usv" not in json.dumps(frame.model_dump(mode="json"), sort_keys=True).casefold()
        for frame in frames
    )

    event_repository = EventRepository(tmp_path / "agent.db")
    try:
        ledger_events = event_repository.list_events(
            scenario_id=config.scenario.scenario_id
        )
        ledger_event_types = {event.event_type for event in ledger_events}
        summaries = event_repository.list_events(
            scenario_id=config.scenario.scenario_id,
            event_type="periodic_situation_summary",
        )
        assert len(summaries) >= duration_s // config.timing.progress_report_s
        assert len({event.event_id for event in summaries}) == len(summaries)
        assert all(
            "usv" not in json.dumps(event.payload, sort_keys=True).casefold()
            for event in summaries
        )
        assert all(
            event.event_id
            == f"periodic_situation_summary:{config.scenario.scenario_id}:{event.sim_time_s}"
            for event in summaries
        )
    finally:
        event_repository.close()

    assert final_plan is not None
    assert final_plan.revision > 1
    assert llm_call_count > 0
    assert {"uuv_deployed", "uuv_recovery_requested"}.issubset(engine_event_types)
    assert "uuv_recovered" in engine_event_types
    assert "initialization" in ledger_event_types
