from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.models import BearingObservation
from underwater_tracking.simulation.engine import SimulationEngine


def test_verification_evidence_projects_public_observation_observer(tmp_path) -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=config.scenario.seed, output_dir=tmp_path)
    engine._target_rays["target_00"] = (
        BearingObservation(
            observation_id="passive:uuv_00:target_00:5",
            scenario_id=config.scenario.scenario_id,
            sim_time_s=5,
            uuv_id="uuv_00",
            target_id="target_00",
            azimuth_rad=0.2,
            variance_rad2=0.01,
            detection_confidence=0.95,
        ),
    )

    evidence = engine.verification_evidence()

    assert evidence["public_observations"] == (
        {
            "observation_id": "passive:uuv_00:target_00:5",
            "target_id": "target_00",
            "sim_time_s": 5,
            "observer_id": "uuv_00",
        },
    )


def test_verification_evidence_keeps_historic_public_observations(tmp_path) -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=config.scenario.seed, output_dir=tmp_path)
    observations = tuple(
        BearingObservation(
            observation_id=f"passive:uuv_00:target_00:{sim_time_s}",
            scenario_id=config.scenario.scenario_id,
            sim_time_s=sim_time_s,
            uuv_id="uuv_00",
            target_id="target_00",
            azimuth_rad=0.2,
            variance_rad2=0.01,
            detection_confidence=0.95,
        )
        for sim_time_s in (5, 10)
    )

    engine._target_rays["target_00"] = (observations[-1],)
    engine._record_public_observations("target_00", observations)

    evidence = engine.verification_evidence()

    assert tuple(
        item["observation_id"] for item in evidence["public_observations"]
    ) == tuple(observation.observation_id for observation in observations)
