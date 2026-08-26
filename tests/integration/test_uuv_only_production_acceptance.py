"""Fixed-seed acceptance of the real UUV-only production entrypoint."""

from __future__ import annotations

from math import ceil, floor
from pathlib import Path
from time import monotonic, sleep
from typing import Any

from underwater_tracking.cli import _AgentLoop, _mission_controller_for
from underwater_tracking.agent.llm import LLMContentError
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.agent_models import IntentHypothesis, StrategyProposal
from underwater_tracking.domain.regional_models import (
    TaskRegionProposal,
    TaskRegionProposalSet,
    UUVRegionalPolicyDecision,
    UUVRegionalStrategyDecisionSet,
)
from underwater_tracking.simulation.engine import SimulationEngine
from underwater_tracking.runtime.run_controller import RunController


class FixedSeedUUVLLM:
    """Deterministic structured provider implementing the production LLM port."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int | None, int, int | None]] = []

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
        batch = payload.get("candidate_batch")
        batch_index = int(batch["index"]) if isinstance(batch, dict) else None
        self.calls.append(
            (
                operation,
                int(payload["sim_time_s"]),
                len(payload.get("candidate_regions", ())),
                batch_index,
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
        if response_model is TaskRegionProposalSet:
            prediction = payload["prediction"]
            assert isinstance(prediction, dict)
            points = prediction["points_xy"]
            assert isinstance(points, list) and points
            coordinates = [tuple(point) for point in points]
            coordinate_system = payload["coordinate_system"]
            assert isinstance(coordinate_system, dict)
            origin = coordinate_system["origin_xy"]
            assert isinstance(origin, list) and len(origin) == 2
            map_bounds = coordinate_system["map_bounds_xy"]
            assert isinstance(map_bounds, list) and len(map_bounds) == 4
            xs = tuple(float(point[0]) for point in coordinates)
            ys = tuple(float(point[1]) for point in coordinates)
            horizontal = max(xs) - min(xs) >= max(ys) - min(ys)
            axis_values = xs if horizontal else ys
            cross_values = ys if horizontal else xs
            axis_min = float(map_bounds[0] if horizontal else map_bounds[2])
            axis_max = float(map_bounds[1] if horizontal else map_bounds[3])
            cross_min_bound = float(map_bounds[2] if horizontal else map_bounds[0])
            cross_max_bound = float(map_bounds[3] if horizontal else map_bounds[1])
            chain_extent_m = 9_000.0
            # Place the first forecast point inside R1 after grid alignment.
            # Flooring here can leave that point just beyond R1's upper edge.
            base = ceil((min(axis_values) - 3_000.0) / 1_000.0) * 1_000.0
            base = min(max(base, axis_min), axis_max - chain_extent_m)
            cross = floor(
                ((sum(cross_values) / len(cross_values)) - 1_500.0) / 1_000.0
            ) * 1_000.0
            cross = min(max(cross, cross_min_bound), cross_max_bound - 3_000.0)
            lower_lefts = tuple(
                (base + index * 2_000.0, cross)
                if horizontal
                else (cross, base + index * 2_000.0)
                for index in range(4)
            )
            return TaskRegionProposalSet(
                regions=tuple(
                    TaskRegionProposal(
                        lower_left_xy=lower_left,
                        upper_right_xy=(
                            lower_left[0] + 3_000.0,
                            lower_left[1] + 3_000.0,
                        ),
                        rationale=f"fixed-seed forecast segment {index}",
                    )
                    for index, lower_left in enumerate(lower_lefts, start=1)
                )
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


class InvalidRegionUUVLLM(FixedSeedUUVLLM):
    """Provider that fails only while generating task-region geometry."""

    def __init__(self) -> None:
        super().__init__()
        self.region_failure_count = 0

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: type[Any],
        *,
        prompt_version: str = "",
    ) -> Any:
        if response_model is TaskRegionProposalSet:
            self.region_failure_count += 1
            raise LLMContentError("invalid task-region response")
        return super().invoke_structured(
            operation,
            payload,
            response_model,
            prompt_version=prompt_version,
        )


def test_fixed_seed_uuv_only_production_loop_replans_through_region_boundaries(
    tmp_path: Path,
) -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
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

        first_mission_plan = loop.runtime.active_mission_plan()
        first_plan = loop.runtime.active_plan()
        runtime_state = loop.runtime.get_state()
        assert first_mission_plan is not None, {
            "carrier_errors": loop.carrier_error_details,
            "llm_calls": llm.calls,
            "errors": runtime_state.get("errors"),
            "commit_status": runtime_state.get("commit_status"),
            "epoch_commit_result": runtime_state.get("epoch_commit_result"),
            "runtime_state_keys": sorted(runtime_state),
        }
        assert first_plan is not None
        assert engine._mission_plan is not None
        assert engine._mission_plan.revision == first_mission_plan.revision
        assert first_mission_plan.batches
        assert all(
            batch.deployment_point is None and batch.recovery_point is None
            for batch in first_mission_plan.batches
        )
        assert all(
            not mission.route_xy
            for mission in engine._mission_plan.carrier_missions.values()
        )
        regional_calls = sorted(
            (call for call in llm.calls if call[0] == "regional_strategy"),
            key=lambda call: call[3] if call[3] is not None else -1,
        )
        assert regional_calls
        assert all(0 < call[2] <= 2 for call in regional_calls)
        regional_plan = first_plan.regional_plans["target_00"]
        assert len(regional_plan.task_regions) == 4
        assert all(cell.cell_size_m == 1_000.0 for cell in regional_plan.cells)
        assert all(
            batch.uuv_ids
            for batches in first_mission_plan.uuv_batches_by_carrier.values()
            for batch in batches
        )

        loop.runtime.submit_event(
            event_type="uuv_range_exhausted",
            entity_id="uuv_00",
            sim_time_s=engine._clock.sim_time_s,
            payload={"remaining_range_m": 0.0},
        )
        for _ in range(18):
            engine.step()
            candidate_plan = loop.runtime.active_mission_plan()
            if (
                candidate_plan is not None
                and candidate_plan.revision > first_mission_plan.revision
            ):
                break

        second_mission_plan = loop.runtime.active_mission_plan()
        second_plan = loop.runtime.active_plan()
        assert second_mission_plan is not None
        assert second_plan is not None
        assert second_mission_plan.revision > first_mission_plan.revision, {
            "carrier_errors": loop.carrier_error_details,
            "llm_calls": llm.calls,
            "runtime_errors": loop.runtime.get_state().get("errors"),
        }
        assert engine._mission_plan is not None
        assert engine._mission_plan.revision == second_mission_plan.revision
        assert controller.snapshot().plan_revision == second_plan.revision
        assert not loop.paused
        assert loop.carrier_error_count == 0, loop.carrier_error_details
        frame = engine.step()
        assert "usvs" not in frame
    finally:
        loop.close()


def test_invalid_llm_regions_preserve_moving_deterministic_baseline(
    tmp_path: Path,
) -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    llm = InvalidRegionUUVLLM()
    run_controller = RunController(
        config,
        output_root=tmp_path / "outputs",
        llm={"master": llm},
        steps=0,
        bootstrap_planning=True,
    )
    run_controller.start_run(1, seed=20260820)
    try:
        initial = run_controller.hub.snapshot()
        assert initial is not None
        initial_positions = {
            uuv.uuv_id: (uuv.position.x, uuv.position.y)
            for uuv in initial.uuvs
            if uuv.deployment_state == "deployed"
        }
        assert initial_positions
        deadline = monotonic() + 10.0
        while llm.region_failure_count == 0 and monotonic() < deadline:
            sleep(0.05)
        bundle = run_controller._bundle
        assert bundle is not None
        state = bundle.loop.runtime.get_state()
        assert llm.region_failure_count >= 1, {
            "calls": llm.calls,
            "errors": state.get("errors", ()),
            "commit_status": state.get("commit_status"),
        }
        assert bundle.engine._mission_plan is not None
        assert bundle.engine._mission_plan.revision == 1
        moving = False
        deadline = monotonic() + 5.0
        while not moving and monotonic() < deadline:
            sleep(0.05)
            latest = run_controller.hub.snapshot()
            moving = latest is not None and any(
                uuv.deployment_state == "deployed"
                and uuv.uuv_id in initial_positions
                and (uuv.position.x, uuv.position.y) != initial_positions[uuv.uuv_id]
                for uuv in latest.uuvs
            )
        assert moving
        deadline = monotonic() + 5.0
        while not bundle.loop.paused and monotonic() < deadline:
            sleep(0.05)
        assert bundle.loop.paused
        assert "invalid task-region response" in str(bundle.loop.llm_pause_reason)
    finally:
        run_controller.close()
