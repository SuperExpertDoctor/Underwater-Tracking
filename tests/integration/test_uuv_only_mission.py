from underwater_tracking.domain.mission_models import (
    CarrierMissionModel,
    ExecutableMissionPlan,
    RegionMissionState,
    UUVMissionBatch,
)
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.runtime.mission_controller import MissionController
from underwater_tracking.simulation.engine import SimulationEngine


def test_uuv_only_controller_trace_is_deterministic_and_contains_no_usv_data() -> None:
    plan = ExecutableMissionPlan(
        revision=1,
        uuv_batches_by_carrier={
            "carrier_01": (
                UUVMissionBatch(
                    carrier_id="carrier_01",
                    candidate_id="T1:r1",
                    uuv_ids=("U1", "U2"),
                    active_scan_uuv_ids=("U1",),
                    passive_track_uuv_ids=("U2",),
                    deployment_point=(0.0, 100.0),
                    recovery_point=(100.0, 100.0),
                    entry_s=0,
                    exit_s=100,
                ),
            )
        },
        region_assignments=(
            RegionMissionState(
                region_id="T1:r1",
                target_id="T1",
                active_scan_uuv_ids=("U1",),
                passive_track_uuv_ids=("U2",),
            ),
        ),
        carrier_missions={
            "carrier_01": CarrierMissionModel(
                carrier_id="carrier_01",
                home_battle_group_id="home",
                ready_uuv_ids=("U1", "U2"),
            )
        },
    )

    def trace() -> tuple[dict[str, object], ...]:
        controller = MissionController(
            scenario_id="S1",
            region_transition_confirm_cycles=2,
        )
        controller.apply_verified_plan(plan)
        controller.advance(0, {"deployed_uuv_ids": {"T1:r1": ("U1", "U2")}})
        controller.advance(1, {"entry_probability": {"T1:r1": 0.8}})
        controller.advance(2, {"entry_probability": {"T1:r1": 0.8}})
        snapshot = controller.snapshot()
        return tuple(
            {
                "region": region.model_dump(mode="json"),
                "uuv_modes": {
                    uuv_id: mode.value for uuv_id, mode in sorted(snapshot.uuv_modes.items())
                },
                "events": [event.model_dump(mode="json") for event in snapshot.events],
            }
            for region in snapshot.regions
        )

    first = trace()
    second = trace()
    assert first == second
    serialized = str(first)
    assert "usv" not in serialized.lower()
    assert "U1" in serialized and "U2" in serialized


def test_engine_advances_uuv_controller_at_observation_boundary(tmp_path) -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    controller = MissionController(
        scenario_id=config.scenario.scenario_id,
        region_entry_probability_threshold=(
            config.scenario.region_entry_probability_threshold
        ),
        region_transition_confirm_cycles=(
            config.scenario.region_transition_confirm_cycles
        ),
    )
    engine = SimulationEngine(
        config,
        seed=config.scenario.seed,
        output_dir=tmp_path,
        mission_controller=controller,
    )

    for _ in range(
        config.timing.observation_step_s // config.timing.physics_step_s
    ):
        frame = engine.step()

    snapshot = engine.mission_snapshot()
    assert snapshot is not None
    assert snapshot.sim_time_s == config.timing.observation_step_s
    assert snapshot.scenario_id == config.scenario.scenario_id
    assert frame["uuvs"]
    assert frame["usvs"] == []
