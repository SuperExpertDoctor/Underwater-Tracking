"""Cross-layer deterministic health transitions for the live UUV pipeline."""

from __future__ import annotations

from collections.abc import Iterator
import sqlite3
from pathlib import Path
from threading import Event
from time import monotonic

import pytest

from tests.integration.test_uuv_only_production_acceptance import FixedSeedUUVLLM
from underwater_tracking.cli import _AgentLoop, _mission_controller_for
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.agent_models import PredictedTrackRef
from underwater_tracking.domain.execution_models import OperationalExecutionSnapshot
from underwater_tracking.prediction.port import (
    _default_imm_forecaster,
    make_snapshot_predictor,
)
from underwater_tracking.runtime.execution_coordinator import ExecutionCoordinator
from underwater_tracking.simulation.engine import SimulationEngine


CHECKPOINTS_S = (600, 1_800, 3_600, 7_200, 14_400, 21_600, 28_800)
MAP_BOUNDS = (-12_000.0, 12_000.0, -12_000.0, 12_000.0)


class LiveTrackingHarness:
    """The same config/factory path used by the production CLI entrypoint."""

    def __init__(self, tmp_path: Path, *, seed: int) -> None:
        config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
        self.config = config
        self.loop = _AgentLoop(
            config,
            database_path=tmp_path / "agent.db",
            llm={"master": FixedSeedUUVLLM()},
            run_id=f"live-health-{seed}",
            steps=max(CHECKPOINTS_S) // config.timing.physics_step_s,
            seed=seed,
        )
        controller = _mission_controller_for(config)
        if controller is None:
            raise RuntimeError("uuv-only scenario must provide a mission controller")
        self.engine = SimulationEngine(
            config,
            seed=seed,
            output_dir=tmp_path / "frames",
            carrier=self.loop.on_situation,
            mission_controller=controller,
        )
        try:
            self.loop.attach(self.engine)
            if self.loop.install_deterministic_baseline(
                self.engine.publication_situation()
            ) is None:
                raise RuntimeError("deterministic baseline was not installed")
        except BaseException:
            self.loop.close(timeout_s=30.0)
            raise

    def close(self) -> None:
        assert self.loop.close(timeout_s=30.0)
        assert self.loop.shutdown_report().completed
        for owner in (
            self.loop.plans,
            self.loop.events,
            self.loop.ledger,
            self.loop._epoch_repository,
            self.loop.runtime._checkpointer,
            self.loop.runtime._payload_store,
        ):
            connection = (
                getattr(owner, "connection", None)
                or getattr(owner, "conn", None)
                or getattr(owner, "_conn", None)
            )
            if connection is not None:
                with pytest.raises(sqlite3.ProgrammingError):
                    connection.execute("SELECT 1")

    def frames_at(self, checkpoints: tuple[int, ...]) -> Iterator[tuple[int, object]]:
        for checkpoint in checkpoints:
            while self.engine._clock.sim_time_s < checkpoint:
                self.engine.step()
            self.loop.publish_latest()
            frame = self.loop.hub.snapshot()
            if frame is None:
                raise AssertionError(f"no published frame at {checkpoint}s")
            yield checkpoint, frame


def _assert_points_inside_map(points: object) -> None:
    min_x, max_x, min_y, max_y = MAP_BOUNDS
    assert points
    for point in points:
        assert min_x <= point.x <= max_x
        assert min_y <= point.y <= max_y


def _assert_frame_health_and_geometry(frame: object) -> None:
    assert frame.map_bounds.min_x == MAP_BOUNDS[0]
    assert frame.map_bounds.max_x == MAP_BOUNDS[1]
    assert frame.map_bounds.min_y == MAP_BOUNDS[2]
    assert frame.map_bounds.max_y == MAP_BOUNDS[3]
    assert frame.execution_consistency is not None
    assert frame.execution_consistency.valid
    execution = frame.execution
    assert execution is not None
    assert execution.health_status in {"current", "degraded"}
    assert len(execution.regions) == 4
    assert len(execution.task_groups) == 4
    assert all(len(group.member_uuv_ids) == 2 for group in execution.task_groups)
    assert len(
        {
            uuv_id
            for group in execution.task_groups
            for uuv_id in group.member_uuv_ids
        }
    ) == 8
    estimate = next(item for item in frame.target_estimates if item.target_id == "target_00")
    prediction = estimate.prediction
    assert prediction is not None
    assert prediction.health.status in {"valid", "degraded"}
    _assert_points_inside_map(prediction.centerline_xy)
    for region in execution.regions:
        _assert_points_inside_map(region.geometry)


def _revision_candidate(
    snapshot: OperationalExecutionSnapshot,
    *,
    revision: int,
    base_revision: int,
    plan_source: str,
) -> OperationalExecutionSnapshot:
    return snapshot.model_copy(
        deep=True,
        update={
            "execution_revision": revision,
            "base_execution_revision": base_revision,
            "plan_source": plan_source,
            "regions": tuple(
                region.model_copy(update={"execution_revision": revision})
                for region in snapshot.regions
            ),
            "task_groups": tuple(
                group.model_copy(update={"execution_revision": revision})
                for group in snapshot.task_groups
            ),
        },
    )


def test_tracking_pipeline_remains_bounded_and_executable_for_eight_hours(
    tmp_path: Path,
) -> None:
    harness = LiveTrackingHarness(tmp_path, seed=20260828)
    try:
        observed = tuple(harness.frames_at(CHECKPOINTS_S))
        assert [sim_time_s for sim_time_s, _ in observed] == list(CHECKPOINTS_S)
        for sim_time_s, frame in observed:
            assert frame.sim_time_s >= sim_time_s
            assert harness.loop.carrier_error_count == 0, harness.loop.carrier_error_details
            _assert_frame_health_and_geometry(frame)
    finally:
        harness.close()


def test_invalid_imm_cycle_degrades_then_recovers_through_real_predictor(
    tmp_path: Path,
) -> None:
    harness = LiveTrackingHarness(tmp_path, seed=20260828)
    try:
        ((_, frame),) = tuple(harness.frames_at((300,)))
        situation = harness.engine.publication_situation()
        dependencies = harness.loop.runtime._dependencies
        invalid_once = True

        def invalid_imm(context: object) -> PredictedTrackRef | None:
            nonlocal invalid_once
            candidate = _default_imm_forecaster(context)
            if candidate is None or not invalid_once:
                return candidate
            invalid_once = False
            return candidate.model_copy(update={"imm_covariance_xy": ()})

        predictor = make_snapshot_predictor(
            belief_history=dependencies.belief_history,
            horizon_s=harness.config.timing.prediction_horizon_s,
            sample_step_s=harness.config.timing.observation_step_s,
            max_speed_mps=14.0,
            max_turn_rate_rad_s=3.141592653589793 / 300.0,
            health_config=harness.config.tracking.prediction_health,
            imm_forecaster=invalid_imm,
        )

        first = predictor(situation, "target_00")
        second = predictor(situation, "target_00")
        assert first.health.status == "degraded"
        assert first.prediction is not None
        assert any("imm_" in reason for reason in first.health.reason_codes)
        assert second.health.status == "valid", second.health
        assert second.prediction is not None
        assert second.prediction.prediction_regime == "imm"
        assert frame.execution is not None
        assert frame.execution.execution_revision >= 1
    finally:
        harness.close()


def test_delayed_optimizer_rejects_stale_baseline_after_newer_commit(
    tmp_path: Path,
) -> None:
    harness = LiveTrackingHarness(tmp_path, seed=20260828)
    try:
        ((_, _),) = tuple(harness.frames_at((600,)))
        live_coordinator = harness.loop._execution_coordinator
        assert isinstance(live_coordinator, ExecutionCoordinator)
        current = live_coordinator.current
        assert current is not None
        coordinator = ExecutionCoordinator(snapshot=current)
        baseline = _revision_candidate(
            current,
            revision=current.execution_revision + 1,
            base_revision=current.execution_revision,
            plan_source="deterministic",
        )
        started = Event()
        release = Event()

        def delayed_optimizer(
            value: OperationalExecutionSnapshot,
        ) -> OperationalExecutionSnapshot:
            started.set()
            assert release.wait(timeout=5)
            return _revision_candidate(
                value,
                revision=value.execution_revision + 1,
                base_revision=value.execution_revision,
                plan_source="llm_optimized",
            )

        result = coordinator.commit_baseline_then_optimize(
            baseline,
            optimizer=delayed_optimizer,
        )
        assert result.committed
        assert started.wait(timeout=1)
        newer = _revision_candidate(
            baseline,
            revision=baseline.execution_revision + 1,
            base_revision=baseline.execution_revision,
            plan_source="deterministic",
        )
        assert coordinator.commit(newer).committed
        release.set()
        deadline = monotonic() + 2.0
        while monotonic() < deadline and coordinator.execution_revision < newer.execution_revision:
            release.wait(timeout=0.01)
        assert coordinator.current == newer
        assert coordinator.execution_revision > baseline.execution_revision
    finally:
        harness.close()
