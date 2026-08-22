"""Fixed-seed acceptance of the real UUV-only production entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from underwater_tracking.cli import _AgentLoop, _mission_controller_for
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.agent_models import IntentHypothesis, StrategyProposal
from underwater_tracking.domain.regional_models import (
    UUVRegionalPolicyDecision,
    UUVRegionalStrategyDecisionSet,
)
from underwater_tracking.simulation.engine import SimulationEngine


class FixedSeedUUVLLM:
    """Deterministic structured provider implementing the production LLM port."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None, int]] = []

    def set_simulation_time(self, sim_time_s: int) -> None:
        del sim_time_s

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: type[Any],
        *,
        prompt_version: str = "",
    ) -> Any:
        del prompt_version
        self.calls.append(
            (
                operation,
                int(payload["sim_time_s"]),
                len(payload.get("candidate_regions", ())),
            )
        )
        evidence_ids = tuple(str(value) for value in payload.get("evidence_ids", ()))
        if not evidence_ids:
            evidence_ids = (
                f"estimate:{payload['scenario_id']}:{payload.get('target_id', 'scenario')}:{payload['sim_time_s']}",
            )

        if response_model is IntentHypothesis:
            return IntentHypothesis(
                label="transit",
                confidence=0.8,
                evidence_ids=evidence_ids[:1],
                model_id="fixed-seed-uuv-llm",
                prompt_version="fixed-seed-v1",
            )
        if response_model is UUVRegionalStrategyDecisionSet:
            return UUVRegionalStrategyDecisionSet(
                policies=tuple(
                    UUVRegionalPolicyDecision(
                        candidate_id=str(candidate["candidate_id"]),
                        coverage_mode="required",
                        tracking_mode="passive_track",
                        priority=1.0,
                        required_quality=0.5,
                        active_scan_uuv_count=1,
                        passive_track_uuv_count=1,
                        reserve_uuv_count=0,
                        optional_uuv_count=0,
                        # Resource selection remains deterministic and
                        # capability-aware in MissionOptimizer.
                        assigned_uuv_ids=(),
                        rationale="fixed-seed candidate policy",
                        evidence_ids=evidence_ids[:1],
                    )
                    for candidate in payload["candidate_regions"]
                )
            )
        if response_model is StrategyProposal:
            target_ids = tuple(str(target) for target in payload.get("target_ids", ()))
            return StrategyProposal(
                concept="balanced",
                target_priorities={target_id: 1.0 for target_id in target_ids},
                required_quality={target_id: 0.5 for target_id in target_ids},
                reinforcement_policy={},
                releasable_soft_constraints=tuple(
                    str(value)
                    for value in payload.get("allowed_relaxations", ())[:1]
                ),
                rationale="fixed-seed semantic repair",
                evidence_ids=evidence_ids[:1],
            )
        raise AssertionError(f"unexpected response model: {response_model!r}")

    def close(self) -> None:
        pass


def test_fixed_seed_uuv_only_production_loop_replans_through_carrier_fleet(
    tmp_path: Path,
) -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    config = _co_locate_test_carriers(config)
    llm = FixedSeedUUVLLM()
    loop = _AgentLoop(
        config,
        database_path=tmp_path / "agent.db",
        llm={"master": llm},
        run_id="fixed-seed-uuv-production",
        steps=24,
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
    loop.attach(engine)
    try:
        for _ in range(18):
            engine.step()

        first_plan = loop.runtime.active_mission_plan()
        assert first_plan is not None
        assert engine._mission_plan is not None
        assert engine._mission_plan.revision == first_plan.revision
        assert set(first_plan.uuv_batches_by_carrier)
        assert set(first_plan.uuv_batches_by_carrier) <= {
            "carrier_02",
            "carrier_03",
            "carrier_04",
        }
        regional_calls = [call for call in llm.calls if call[0] == "regional_strategy"]
        assert regional_calls
        assert all(0 < call[2] <= 4 for call in regional_calls)
        assert all(call[2] == 4 for call in regional_calls[:-1])
        assert sum(call[2] for call in regional_calls) >= 16
        assert all(
            batch.uuv_ids
            for batches in first_plan.uuv_batches_by_carrier.values()
            for batch in batches
        )

        loop.runtime.submit_event(
            event_type="uuv_range_exhausted",
            entity_id="uuv_00",
            sim_time_s=engine._clock.sim_time_s,
            payload={"remaining_range_m": 0.0},
        )
        for _ in range(6):
            engine.step()

        second_plan = loop.runtime.active_mission_plan()
        assert second_plan is not None
        assert second_plan.revision > first_plan.revision
        assert engine._mission_plan is not None
        assert engine._mission_plan.revision == second_plan.revision
        assert controller.snapshot().plan_revision == second_plan.revision
        assert loop.events.list_events(
            scenario_id=config.scenario.scenario_id,
            event_type="uuv_range_exhausted",
        )
        assert not loop.paused
        assert loop.carrier_error_count == 0
        frame = engine.step()
        assert "usvs" not in frame
    finally:
        loop.close()


def _co_locate_test_carriers(config: Any) -> Any:
    """Use a reachable fixed-seed logistics geometry for the production trace."""
    assert config.environment is not None
    positions = {
        f"carrier_{index:02d}": (-250.0 + 100.0 * (index - 1), -50.0)
        for index in range(1, 5)
    }
    carriers = []
    for carrier in (config.environment.carrier, *config.environment.carriers):
        position = positions[carrier.platform_id]
        carriers.append(
            carrier.model_copy(
                update={
                    "position_xy": position,
                    "speed_mps": 20.0,
                    "patrol_route_xy": (
                        position,
                        (position[0] + 100.0, position[1]),
                        (position[0] + 100.0, position[1] + 100.0),
                        (position[0], position[1] + 100.0),
                    ),
                }
            )
        )
    primary = carriers[0]
    environment = config.environment.model_copy(
        update={"carrier": primary, "carriers": tuple(carriers)}
    )
    return config.model_copy(update={"environment": environment})
