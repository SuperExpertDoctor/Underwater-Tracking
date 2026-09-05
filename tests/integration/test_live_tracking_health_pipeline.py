"""Cross-layer deterministic health transitions for the live UUV pipeline."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
import sqlite3
from pathlib import Path
from threading import Event
from time import monotonic
from types import SimpleNamespace

import pytest

from tests.integration.test_uuv_only_production_acceptance import FixedSeedUUVLLM
from underwater_tracking import cli as cli_module
from underwater_tracking.agent.graphs.central import _build_live_regional_generation
from underwater_tracking.agent.nodes import regions as regions_node
from underwater_tracking.cli import _AgentLoop, _mission_controller_for
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.agent_models import IntentHypothesis, PredictedTrackRef
from underwater_tracking.domain.execution_models import OperationalExecutionSnapshot
from underwater_tracking.domain.planning_epoch_models import (
    PlanningEpoch,
    PlanningEpochCapture,
)
from underwater_tracking.domain.prediction_models import AcceptedPrediction, PredictionHealth
from underwater_tracking.domain.regional_models import GridSpec
from underwater_tracking.prediction.port import (
    ForecastContext,
    make_snapshot_predictor,
)
from underwater_tracking.prediction.health import effective_radius_limit_m
from underwater_tracking.planning.dynamic_regions import DynamicRegionChain
from underwater_tracking.planning.region_baseline import build_four_region_baseline
from underwater_tracking.planning.regions import generate_target_region_plan
from underwater_tracking.runtime.execution_coordinator import ExecutionCoordinator
from underwater_tracking.runtime.mission_controller import (
    execution_snapshot_to_mission_plan,
)
from underwater_tracking.simulation.engine import SimulationEngine


CHECKPOINTS_S = (600, 1_800, 3_600, 7_200, 14_400, 21_600, 28_800)
MAP_BOUNDS = (-12_000.0, 12_000.0, -12_000.0, 12_000.0)


def _accelerated_config():
    base = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    timing = base.timing.model_copy(
        update={
            "physics_step_s": 30,
            "observation_step_s": 30,
            "group_report_s": 300,
        }
    )
    return base.model_copy(update={"timing": timing})


class LiveTrackingHarness:
    """The same config/factory path used by the production CLI entrypoint."""

    def __init__(self, tmp_path: Path, *, seed: int) -> None:
        config = _accelerated_config()
        assert config.timing.physics_step_s == 30
        assert config.timing.observation_step_s == 30
        assert config.timing.group_report_s == 300
        assert config.environment is not None
        assert config.environment.map_bounds_xy == MAP_BOUNDS
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
        publisher = self.loop._publisher
        logger = getattr(publisher, "_logger", None)
        assert self.loop.close(timeout_s=30.0)
        assert self.loop.shutdown_report().completed
        handle = getattr(logger, "_handle", None)
        if handle is not None:
            assert handle.closed
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
            assert self.engine._clock.sim_time_s == checkpoint
            self.loop.publish_latest()
            frame = self.loop.hub.snapshot()
            if frame is None:
                raise AssertionError(f"no published frame at {checkpoint}s")
            assert frame.sim_time_s == checkpoint
            yield checkpoint, frame


def _assert_points_inside_map(points: object) -> None:
    min_x, max_x, min_y, max_y = MAP_BOUNDS
    assert points
    for point in points:
        assert min_x <= point.x <= max_x
        assert min_y <= point.y <= max_y


def _assert_frame_health_and_geometry(
    frame: object,
    *,
    harness: LiveTrackingHarness | None = None,
    health_status_at_frame: str | None = None,
    executable_at_frame: bool | None = None,
    current_execution_at_frame: OperationalExecutionSnapshot | None = None,
) -> None:
    assert frame.map_bounds.min_x == MAP_BOUNDS[0]
    assert frame.map_bounds.max_x == MAP_BOUNDS[1]
    assert frame.map_bounds.min_y == MAP_BOUNDS[2]
    assert frame.map_bounds.max_y == MAP_BOUNDS[3]
    assert frame.execution_consistency is not None
    assert frame.execution_consistency.valid
    execution = frame.execution
    assert execution is not None
    assert execution.health_status in {"current", "degraded", "expired"}
    if health_status_at_frame == "expired":
        assert execution.health_status == "expired"
        current = current_execution_at_frame
        if current is None and harness is not None:
            current = harness.loop._execution_coordinator.current
        assert current is not None
        assert current.execution_revision == execution.execution_revision
        assert current.valid_until_s == execution.valid_until_s
        assert execution.valid_until_s < frame.sim_time_s
        expiry_reasons = {
            "execution_snapshot_expired",
            "execution_target_track_hard_stale",
        }
        assert execution.health_reasons
        assert sum(
            reason in expiry_reasons for reason in execution.health_reasons
        ) == 1
        assert execution.evidence_ids
        if executable_at_frame is not None:
            assert not executable_at_frame
        elif harness is not None:
            assert not harness.loop._execution_coordinator.is_executable(
                sim_time_s=float(frame.sim_time_s),
                hard_stale_s=900.0,
            )
    else:
        assert health_status_at_frame in {None, "current", "degraded"}
        assert execution.health_status in {"current", "degraded"}
        if executable_at_frame is not None:
            assert executable_at_frame
        elif harness is not None:
            assert harness.loop._execution_coordinator.is_executable(
                sim_time_s=float(frame.sim_time_s),
                hard_stale_s=900.0,
            )
    assert len(execution.regions) == 4
    assert len(execution.task_groups) == 4
    assert all(len(group.member_uuv_ids) == 3 for group in execution.task_groups)
    assert len(
        {
            uuv_id
            for group in execution.task_groups
            for uuv_id in group.member_uuv_ids
        }
    ) == 12
    estimate = next(item for item in frame.target_estimates if item.target_id == "target_00")
    prediction = estimate.prediction
    # Long-running publication must fail closed, not keep a stale line alive.
    if prediction is None:
        assert (
            frame.sim_time_s >= execution.valid_until_s
            or estimate.estimate_health["status"] in {"expired", "unavailable"}
        )
        assert estimate.world_model is not None
        assert estimate.world_model.data_status in {"expired", "unavailable"}
        assert not estimate.world_model.events
    else:
        assert prediction.health.status in {"valid", "degraded"}
        _assert_points_inside_map(prediction.centerline_xy)
        if prediction.valid_until_s is not None:
            assert frame.sim_time_s < prediction.valid_until_s
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
        observed: list[
            tuple[
                int,
                object,
                str,
                bool,
                OperationalExecutionSnapshot | None,
            ]
        ] = []
        for sim_time_s, frame in harness.frames_at(CHECKPOINTS_S):
            coordinator = harness.loop._execution_coordinator
            health = coordinator.execution_health(
                sim_time_s=float(sim_time_s),
                hard_stale_s=900.0,
            )
            observed.append(
                (
                    sim_time_s,
                    frame,
                    health.status,
                    coordinator.is_executable(
                        sim_time_s=float(sim_time_s),
                        hard_stale_s=900.0,
                    ),
                    coordinator.current,
                )
            )
            active_audit = harness.loop.plans.get_active(
                harness.config.scenario.scenario_id
            )
            if active_audit is not None and coordinator.current is not None:
                assert (
                    active_audit.revision
                    == coordinator.current.execution_revision
                )
        assert [sim_time_s for sim_time_s, *_ in observed] == list(CHECKPOINTS_S)
        for (
            sim_time_s,
            frame,
            health_status,
            executable,
            current_execution,
        ) in observed:
            assert frame.sim_time_s >= sim_time_s
            assert harness.loop.carrier_error_count == 0, harness.loop.carrier_error_details
            _assert_frame_health_and_geometry(
                frame,
                health_status_at_frame=health_status,
                executable_at_frame=executable,
                current_execution_at_frame=current_execution,
            )
    finally:
        harness.close()


def test_live_region_refresh_reprojects_prior_chain_after_partition_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prediction = PredictedTrackRef(
        prediction_id="prediction:T1:prior",
        target_id="T1",
        sim_time_s=1_000,
        horizon_s=1_800.0,
        sample_step_s=100.0,
        times_s=tuple(1_000.0 + index * 100.0 for index in range(19)),
        points_xy=tuple((1_000.0 + index * 200.0, 2_000.0) for index in range(19)),
        corridor_radius_m=(100.0,) * 19,
        source_belief_history_ids=("belief:T1",),
        prediction_regime="imm",
    )
    intent = IntentHypothesis(
        label="transit",
        confidence=0.8,
        evidence_ids=("belief:T1",),
        model_id="test",
        prompt_version="test-v1",
    )
    accepted = AcceptedPrediction(
        prediction=prediction,
        health=PredictionHealth(
            status="valid",
            regime="imm",
            source_track_age_s=0.0,
            clipped_point_fraction=0.0,
            maximum_radius_m=100.0,
            raw_prediction_id=prediction.prediction_id,
        ),
    )
    baseline = build_four_region_baseline(
        accepted,
        target_id="T1",
        execution_revision=4,
        origin_sim_time_s=1_000.0,
        map_bounds_xy=(0.0, 8_000.0, 0.0, 6_000.0),
    )
    prior_chain = DynamicRegionChain(
        target_id="T1",
        prediction_id=prediction.prediction_id,
        execution_revision=4,
        geometry_revision=baseline.regions[0].geometry_revision,
        regions=baseline.regions,
    )
    current_prediction = prediction.model_copy(
        update={"prediction_id": "prediction:T1:current", "sim_time_s": 1_100.0}
    )
    current_accepted = accepted.model_copy(
        update={
            "prediction": current_prediction,
            "health": accepted.health.model_copy(
                update={"raw_prediction_id": current_prediction.prediction_id}
            ),
        }
    )
    prior_plan = generate_target_region_plan(
        prediction,
        intent,
        (0.0, 8_000.0, 0.0, 6_000.0),
        GridSpec(),
    )

    original_build = regions_node.build_four_region_baseline
    calls: list[bool] = []

    def fail_current_partition(accepted_prediction, **kwargs):
        has_prediction = accepted_prediction.prediction is not None
        calls.append(has_prediction)
        if has_prediction:
            raise ValueError("map bounds cannot retain a legal four-region partition")
        return original_build(accepted_prediction, **kwargs)

    monkeypatch.setattr(
        regions_node,
        "build_four_region_baseline",
        fail_current_partition,
    )

    class NoCoordinateLLM:
        def invoke_structured(self, operation, *_args, **_kwargs):
            raise AssertionError(f"unexpected LLM operation: {operation}")

    snapshot = SimpleNamespace(scenario_id="S1", sim_time_s=1_100, active_plan=None)
    dependencies = SimpleNamespace(
        optimizer=SimpleNamespace(
            bounds=(0.0, 8_000.0, 0.0, 6_000.0),
            quality_warning=0.0,
        ),
        grid_spec=GridSpec(),
        llm=NoCoordinateLLM(),
        model_id="test",
        execution_strategy_node=None,
    )
    node = _build_live_regional_generation(dependencies, lambda _: snapshot)
    result = node(
        {
            "snapshot_ref": "S1:snapshot:partition-recovery",
            "intent_hypotheses": {"T1": intent},
            "predictions": {"T1": current_prediction},
            "accepted_predictions": {"T1": current_accepted},
            "dynamic_region_chains": {"T1": prior_chain},
            "regional_plans": {"T1": prior_plan},
            "execution_revision": 5,
        }
    )

    chain = result["dynamic_region_chains"]["T1"]
    assert calls == [True, False]
    assert result["region_generation_modes"] == {"T1": "reprojected_previous"}
    assert result["region_generation_reason_codes"]["T1"] == (
        "current_prediction_partition_unavailable",
        "reprojected_previous_regions",
    )
    assert chain.prediction_id == prediction.prediction_id
    assert chain.execution_revision == 5
    assert tuple(region.geometry for region in chain.regions) == tuple(
        region.geometry for region in prior_chain.regions
    )
    assert all(
        region.prediction_id == prediction.prediction_id for region in chain.regions
    )
    assert result["regional_plans"]["T1"].prediction_id == prediction.prediction_id
    assert current_accepted.health.status == "valid"


def test_uuv_only_ensure_reprojects_current_execution_after_partition_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = LiveTrackingHarness(tmp_path, seed=20260828)
    try:
        tuple(harness.frames_at((600,)))
        healthy_frame = harness.loop.hub.snapshot()
        assert healthy_frame is not None
        assert healthy_frame.execution is not None
        assert healthy_frame.execution_consistency is not None
        invalid_payload = healthy_frame.model_dump(mode="python")
        failed_prediction_id = "prediction:target_00:failed-mismatch"
        invalid_payload["execution"]["health_status"] = "failed"
        invalid_payload["execution"]["health_reasons"] = [
            "execution_region_identity_unbound"
        ]
        invalid_payload["execution"]["prediction_id"] = failed_prediction_id
        for region in invalid_payload["execution"]["regions"]:
            region["prediction_id"] = failed_prediction_id
        invalid_payload["execution_consistency"].update(
            {
                "valid": False,
                "errors": ["execution_region_identity_unbound"],
            }
        )
        with pytest.raises(ValueError, match="prediction ID must match"):
            type(healthy_frame).model_validate(invalid_payload)

        current = harness.loop._execution_coordinator.current
        assert current is not None
        state = harness.loop.runtime.get_state()
        accepted = state["accepted_predictions"]["target_00"]
        assert accepted.prediction is not None
        assert len(current.regions) == 4
        assert all(
            all(
                MAP_BOUNDS[0] <= point[0] <= MAP_BOUNDS[1]
                and MAP_BOUNDS[2] <= point[1] <= MAP_BOUNDS[3]
                for point in region.geometry
            )
            for region in current.regions
        )

        original_build = cli_module.build_four_region_baseline
        calls: list[bool] = []

        def fail_current_partition(accepted_prediction, **kwargs):
            has_prediction = accepted_prediction.prediction is not None
            calls.append(has_prediction)
            if has_prediction:
                raise ValueError(
                    "map bounds cannot retain a legal four-region partition"
                )
            return original_build(accepted_prediction, **kwargs)

        monkeypatch.setattr(
            cli_module,
            "build_four_region_baseline",
            fail_current_partition,
        )

        unbound_prediction = accepted.prediction.model_copy(
            update={"prediction_id": "prediction:target_00:unbound"}
        )
        unbound_accepted = accepted.model_copy(
            update={
                "prediction": unbound_prediction,
                "health": accepted.health.model_copy(
                    update={"raw_prediction_id": unbound_prediction.prediction_id}
                ),
            }
        )
        unbound_state = dict(state)
        unbound_state["accepted_predictions"] = {"target_00": unbound_accepted}
        refreshed = harness.loop._ensure_uuv_only_execution_snapshot(
            harness.engine.publication_situation(),
            prediction_state=unbound_state,
        )

        assert refreshed is None
        assert calls == [True]
        assert harness.loop._execution_coordinator.current == current
        health = harness.loop._execution_coordinator.execution_health(
            sim_time_s=600.0,
            hard_stale_s=900.0,
        )
        assert health.status == "failed"
        assert health.reason_codes == ("execution_region_identity_unbound",)
        assert all(
            region.prediction_id == current.prediction_id
            for region in current.regions
        )
        assert harness.loop.carrier_error_count == 0
    finally:
        harness.close()


def test_region_identity_failure_publishes_failed_invalid_frame_through_agent_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = LiveTrackingHarness(tmp_path, seed=20260828)
    try:
        tuple(harness.frames_at((600,)))
        current = harness.loop._execution_coordinator.current
        assert current is not None
        dependencies = harness.loop.runtime._dependencies
        original_predictor = dependencies.predictor

        def unbound_predictor(situation, target_id):
            accepted = original_predictor(situation, target_id)
            if target_id != "target_00" or accepted.prediction is None:
                return accepted
            prediction = accepted.prediction.model_copy(
                update={"prediction_id": "prediction:target_00:unbound"}
            )
            return accepted.model_copy(
                update={
                    "prediction": prediction,
                    "health": accepted.health.model_copy(
                        update={"raw_prediction_id": prediction.prediction_id}
                    ),
                }
            )

        harness.loop.runtime._dependencies = replace(
            dependencies,
            predictor=unbound_predictor,
        )
        original_build = cli_module.build_four_region_baseline

        def fail_current_partition(accepted_prediction, **kwargs):
            if accepted_prediction.prediction is not None:
                raise ValueError(
                    "map bounds cannot retain a legal four-region partition"
                )
            return original_build(accepted_prediction, **kwargs)

        monkeypatch.setattr(
            cli_module,
            "build_four_region_baseline",
            fail_current_partition,
        )

        while harness.engine._clock.sim_time_s < 1_050:
            harness.engine.step()
        harness.loop.publish_latest()
        frame = harness.loop.hub.snapshot()
        assert frame is not None
        assert frame.execution is not None
        assert frame.execution.health_status == "failed"
        assert "execution_region_identity_unbound" in frame.execution.health_reasons
        assert frame.execution_consistency is not None
        assert not frame.execution_consistency.valid
        assert "execution_region_identity_unbound" in frame.execution_consistency.errors
        assert not harness.loop._execution_coordinator.is_executable(
            sim_time_s=1_050.0,
            hard_stale_s=900.0,
        )
    finally:
        harness.close()


def test_boundary_recovery_keeps_public_prediction_legal_at_map_edge() -> None:
    report = SimpleNamespace(
        target_id="target_00",
        belief=SimpleNamespace(
            sim_time_s=4_530,
            mean=(-12_030.0, 6_326.0, -14.0, 0.0),
            covariance=((25.0, 0.0), (0.0, 25.0)),
            source_observation_ids=("obs-boundary",),
            model_probabilities={"cv": 1.0},
        ),
    )
    snapshot = SimpleNamespace(
        scenario_id="uuv-only-boundary",
        snapshot_revision=157,
        sim_time_s=4_530,
        group_reports=(report,),
        map_bounds_xy=MAP_BOUNDS,
    )
    predictor = make_snapshot_predictor(
        belief_history=lambda _snapshot, _target_id: (
            (4_470, -11_190.0, 6_326.0),
            (4_500, -11_610.0, 6_326.0),
        ),
        horizon_s=300.0,
        sample_step_s=30.0,
        max_speed_mps=14.0,
        max_turn_rate_rad_s=3.141592653589793 / 300.0,
        health_config=_accelerated_config().tracking.prediction_health,
    )

    accepted = predictor(snapshot, "target_00")

    assert accepted.health.status == "degraded", accepted.health
    assert accepted.health.regime == "boundary_recovery"
    prediction = accepted.prediction
    assert prediction is not None, accepted.health
    assert "boundary_recovery_point_out_of_bounds" not in accepted.health.reason_codes
    assert prediction.fallback_reason == "map-projected public-track boundary recovery"
    assert len(prediction.times_s) == len(prediction.points_xy)
    assert len(prediction.points_xy) == len(prediction.corridor_radius_m)
    assert len(prediction.points_xy) == len(prediction.point_confidence)
    _assert_points_inside_map(
        tuple(SimpleNamespace(x=x, y=y) for x, y in prediction.points_xy)
    )


def test_boundary_recovery_remains_accepted_when_public_uncertainty_exceeds_cap() -> None:
    health_config = _accelerated_config().tracking.prediction_health
    report = SimpleNamespace(
        target_id="target_00",
        belief=SimpleNamespace(
            sim_time_s=4_530,
            mean=(-12_030.0, 6_326.0, -14.0, 0.0),
            covariance=((100_000_000.0, 0.0), (0.0, 100_000_000.0)),
            source_observation_ids=("obs-boundary-high-uncertainty",),
            model_probabilities={"cv": 1.0},
        ),
    )
    snapshot = SimpleNamespace(
        scenario_id="uuv-only-boundary",
        snapshot_revision=158,
        sim_time_s=4_530,
        group_reports=(report,),
        map_bounds_xy=MAP_BOUNDS,
    )
    predictor = make_snapshot_predictor(
        belief_history=lambda _snapshot, _target_id: (
            (4_470, -11_190.0, 6_326.0),
            (4_500, -11_610.0, 6_326.0),
        ),
        horizon_s=300.0,
        sample_step_s=30.0,
        max_speed_mps=14.0,
        max_turn_rate_rad_s=3.141592653589793 / 300.0,
        health_config=health_config,
    )

    accepted = predictor(snapshot, "target_00")

    assert accepted.health.status == "degraded", accepted.health
    assert accepted.health.regime == "boundary_recovery"
    prediction = accepted.prediction
    assert prediction is not None, accepted.health
    assert prediction.prediction_regime == "boundary_recovery"
    assert max(prediction.corridor_radius_m) <= effective_radius_limit_m(
        MAP_BOUNDS,
        health_config,
    )
    assert len(prediction.times_s) == len(prediction.points_xy)
    assert len(prediction.points_xy) == len(prediction.corridor_radius_m)
    assert len(prediction.points_xy) == len(prediction.point_confidence)
    _assert_points_inside_map(
        tuple(SimpleNamespace(x=x, y=y) for x, y in prediction.points_xy)
    )


def test_uuv_only_epoch_commit_does_not_advance_controller_without_execution_apply(
    tmp_path: Path,
) -> None:
    harness = LiveTrackingHarness(tmp_path, seed=20260828)
    try:
        situation = harness.engine.publication_situation()
        controller = harness.engine._mission_controller
        assert controller is not None
        mission_before = controller.snapshot()
        current = harness.loop._execution_coordinator.current
        assert current is not None
        audit = harness.loop.plans.get_active(situation.scenario_id)
        assert audit is not None
        audit = audit.model_copy(
            update={
                "plan_id": (
                    f"{situation.scenario_id}:task9:controller-boundary"
                )
            }
        )
        epoch = PlanningEpoch(
            epoch_id=f"{situation.scenario_id}:task9:controller-boundary",
            scenario_id=situation.scenario_id,
            base_physics_revision=situation.snapshot_revision,
            base_sim_time_s=situation.sim_time_s,
            observation_batch_id=f"task9:{situation.snapshot_revision}",
            resource_manifest_hash="task9-controller-boundary",
            active_plan_version=mission_before.plan_revision,
        )
        harness.loop._epoch_repository.create(
            PlanningEpochCapture(
                epoch=epoch,
                situation=situation,
                mission=mission_before,
            )
        )

        result = harness.loop._epoch_commit_port.commit(
            epoch=epoch,
            audit_projection=audit,
            executable_plan=execution_snapshot_to_mission_plan(current),
        )

        assert result.status == "committed", result.failure_message
        assert result.executable_plan is not None
        assert result.executable_plan.revision == mission_before.plan_revision + 1
        assert controller.snapshot().plan_revision == mission_before.plan_revision
        assert harness.loop._execution_coordinator.execution_revision == current.execution_revision
    finally:
        harness.close()


def test_uuv_only_apply_rejection_preserves_engine_revision_reason(
    tmp_path: Path,
) -> None:
    harness = LiveTrackingHarness(tmp_path, seed=20260828)
    try:
        current = harness.loop._execution_coordinator.current
        controller = harness.engine._mission_controller
        assert current is not None
        assert controller is not None
        stale_plan = execution_snapshot_to_mission_plan(current)
        newer_plan = stale_plan.model_copy(
            update={"revision": stale_plan.revision + 1}
        )
        assert controller.apply_verified_plan(newer_plan)

        assert harness.engine.apply_verified_mission_plan(stale_plan) is False
        reason = harness.engine._last_mission_plan_failure_reason
        assert (
            reason
            == "mission_controller_revision_conflict:candidate=1:controller=2"
        )
        with pytest.raises(
            RuntimeError,
            match="engine_apply_rejected:mission_controller_revision_conflict",
        ):
            harness.loop._apply_execution_snapshot_or_raise(current)
    finally:
        harness.close()


def test_uuv_only_execution_track_projects_out_of_bounds_public_report(
    tmp_path: Path,
) -> None:
    harness = LiveTrackingHarness(tmp_path, seed=20260828)
    try:
        ((_, _),) = tuple(harness.frames_at((300,)))
        situation = harness.engine.publication_situation()
        prior = situation.target_search_priors[0]
        outbound_prior = prior.model_copy(
            update={"center_xy": (-12_030.0, 6_326.0)}
        )
        outbound_situation = situation.model_copy(
            update={"target_search_priors": (outbound_prior,)}
        )
        prediction_state = harness.loop.runtime.get_state()

        snapshot = harness.loop._ensure_uuv_only_execution_snapshot(
            outbound_situation,
            prediction_state=prediction_state,
        )

        assert snapshot is not None
        assert snapshot.target_track.position_xy == (-12_000.0, 6_326.0)
        _assert_points_inside_map(
            tuple(
                SimpleNamespace(
                    x=sample.position_xy[0],
                    y=sample.position_xy[1],
                )
                for sample in snapshot.target_track.bounded_history
            )
        )
    finally:
        harness.close()


def test_graph_prediction_refresh_keeps_executable_degraded_prediction(
    tmp_path: Path,
) -> None:
    harness = LiveTrackingHarness(tmp_path, seed=20260828)
    try:
        tuple(harness.frames_at((300,)))
        state = harness.loop.runtime.get_state()
        predictions = state.get("predictions") or {}
        accepted_predictions = state.get("accepted_predictions") or {}
        assert predictions
        accepted = accepted_predictions["target_00"]
        assert accepted.prediction == predictions["target_00"]
        assert accepted.health.status == "degraded"

        # A graph checkpoint can carry the public prediction channel without
        # carrying the live accepted channel after a refresh boundary.
        harness.loop.runtime._capture_graph_prediction_state(
            {
                "predictions": predictions,
                "prediction_snapshot_revision": state[
                    "prediction_snapshot_revision"
                ],
            }
        )

        live_state = harness.loop.runtime._live_prediction_fragment()
        recovered = live_state["accepted_predictions"]["target_00"]
        assert recovered.prediction == live_state["predictions"]["target_00"]
        assert recovered.health.status == "degraded"
        snapshot = harness.loop._ensure_uuv_only_execution_snapshot(
            harness.engine.publication_situation(),
            prediction_state=live_state,
        )
        assert snapshot is not None
    finally:
        harness.close()


def test_uuv_only_source_gap_does_not_renew_old_execution_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = LiveTrackingHarness(tmp_path, seed=20260828)
    try:
        tuple(harness.frames_at((600,)))
        current = harness.loop._execution_coordinator.current
        assert current is not None
        current_window = (current.valid_from_s, current.valid_until_s)
        source_missing = harness.engine.publication_situation().model_copy(
            update={
                "sim_time_s": 20_000,
                "snapshot_revision": 20_000,
                "group_reports": (),
                "target_search_priors": (),
            }
        )
        monkeypatch.setattr(
            harness.engine,
            "publication_situation",
            lambda: source_missing,
        )
        harness.loop.on_situation(source_missing)
        harness.loop.publish_latest()

        assert harness.loop._execution_coordinator.current == current
        assert (
            harness.loop._execution_coordinator.current.valid_from_s,
            harness.loop._execution_coordinator.current.valid_until_s,
        ) == current_window
        health = harness.loop._execution_coordinator.execution_health(
            sim_time_s=20_000.0,
            hard_stale_s=900.0,
        )
        assert health.status == "expired"
        assert health.reason_codes == ("execution_target_track_hard_stale",)
        assert not harness.loop._execution_coordinator.is_executable(
            sim_time_s=20_000.0,
            hard_stale_s=900.0,
        )
        frame = harness.loop.hub.snapshot()
        assert frame is not None
        assert frame.sim_time_s == 20_000
        assert frame.execution is not None
        assert frame.execution.health_status == "expired"
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

        def bounded_imm(context: ForecastContext) -> PredictedTrackRef:
            prior = next(
                item
                for item in context.snapshot.target_search_priors
                if item.target_id == context.target_id
            )
            position = tuple(float(value) for value in prior.center_xy)
            steps = max(1, int(context.horizon_s // context.sample_step_s))
            return PredictedTrackRef(
                prediction_id=context.prediction_id,
                target_id=context.target_id,
                sim_time_s=int(context.snapshot.sim_time_s),
                horizon_s=context.horizon_s,
                sample_step_s=context.sample_step_s,
                times_s=tuple(
                    context.snapshot.sim_time_s + (index + 1) * context.sample_step_s
                    for index in range(steps)
                ),
                points_xy=tuple(position for _ in range(steps)),
                corridor_radius_m=tuple(10.0 for _ in range(steps)),
                imm_model_probabilities={"CV": 1.0},
                imm_covariance_xy=tuple(
                    (1.0, 0.0, 0.0, 1.0) for _ in range(steps)
                ),
            )

        def invalid_imm(context: ForecastContext) -> PredictedTrackRef | None:
            nonlocal invalid_once
            candidate = bounded_imm(context)
            if invalid_once:
                invalid_once = False
                return candidate.model_copy(update={"imm_covariance_xy": ()})
            return candidate

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


def test_uuv_only_agent_loop_publishes_authoritative_prior_frame(
    tmp_path: Path,
) -> None:
    harness = LiveTrackingHarness(tmp_path, seed=20260828)
    try:
        tuple(harness.frames_at((300,)))
        frame = harness.loop.hub.snapshot()
        assert frame is not None
        estimate = next(
            item
            for item in frame.target_estimates
            if item.target_id == "target_00"
        )
        assert estimate.prediction is not None
        assert estimate.prediction.health.status == "degraded"
        assert estimate.prediction.health.reason_codes == (
            "public_target_search_envelope",
            "estimate_provenance_missing",
        )
        assert estimate.prediction.health.source_track_age_s is None
        assert estimate.quality.quality_score is None
        assert estimate.prediction.health.regime == "short_history"
        assert frame.execution_consistency is not None
        assert frame.execution_consistency.valid
        assert harness.loop.carrier_error_count == 0, (
            harness.loop.carrier_error_details
        )
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
