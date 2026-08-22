from __future__ import annotations

import pytest

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.models import DeploymentState
from underwater_tracking.domain.observations import PassiveSonarObservation
from underwater_tracking.simulation.engine import SimulationEngine


CONFIG_PATH = "configs/scenario/uuv_only_single_target.yaml"


def test_uuv_only_initialization_has_prior_but_no_estimate_or_execution_group() -> None:
    engine = SimulationEngine(load_app_config(CONFIG_PATH), seed=7)

    snapshot = engine.publication_situation()

    assert snapshot.group_reports == ()
    assert snapshot.execution_groups == ()
    assert tuple(prior.prior_id for prior in snapshot.target_search_priors) == (
        "intel-target-00-initial",
    )
    assert engine._assignments == {}
    assert engine._latest_reports == {}
    assert engine.build_slave_contexts(snapshot) == ()
    adversary_inputs = engine.build_adversary_inputs(snapshot)
    assert len(adversary_inputs) == 1
    assert adversary_inputs[0].local_contacts == ()
    assert adversary_inputs[0].platform_threats == ()


def test_execution_group_requires_physical_exposure_and_does_not_create_belief() -> None:
    engine = SimulationEngine(load_app_config(CONFIG_PATH), seed=7)

    with pytest.raises(ValueError, match="not physically exposed"):
        engine.activate_execution_group(
            target_id="target_00",
            region_id="region-00",
            member_ids=("uuv_00", "uuv_01"),
        )

    engine.request_uuv_deployment("uuv_00", reason="test")
    engine.request_uuv_deployment("uuv_01", reason="test")
    group = engine.activate_execution_group(
        target_id="target_00",
        region_id="region-00",
        member_ids=("uuv_00", "uuv_01"),
    )

    assert group.member_ids == ("uuv_00", "uuv_01")
    assert group.mode == "active_scan"
    assert engine._latest_reports == {}
    assert engine.publication_situation().group_reports == ()
    assert engine.publication_situation().execution_groups == (group,)

    with pytest.raises(ValueError, match="already belongs"):
        engine.activate_execution_group(
            target_id="target_00",
            region_id="region-01",
            member_ids=("uuv_00", "uuv_01"),
        )


def test_target_prior_expires_once_at_its_validity_boundary() -> None:
    engine = SimulationEngine(load_app_config(CONFIG_PATH), seed=7)
    engine._clock.sim_time_s = 1800

    snapshot = engine.publication_situation()
    expired = [event for event in engine.events() if event.event_type == "target_prior_expired"]

    assert snapshot.target_search_priors == ()
    assert len(expired) == 1
    assert expired[0].entity_id == "target_00"
    assert expired[0].payload["prior_id"] == "intel-target-00-initial"
    engine.publication_situation()
    assert len([event for event in engine.events() if event.event_type == "target_prior_expired"]) == 1


def test_failed_uuv_cannot_join_execution_group() -> None:
    engine = SimulationEngine(load_app_config(CONFIG_PATH), seed=7)
    engine.request_uuv_deployment("uuv_00", reason="test")
    engine.request_uuv_deployment("uuv_01", reason="test")
    engine.fail_uuv("uuv_01")

    assert engine._deployment_states["uuv_01"] is DeploymentState.FAILED
    with pytest.raises(ValueError, match="not deployable"):
        engine.activate_execution_group(
            target_id="target_00",
            region_id="region-00",
            member_ids=("uuv_00", "uuv_01"),
        )


def test_real_fused_bearings_create_the_first_tracking_report() -> None:
    engine = SimulationEngine(load_app_config(CONFIG_PATH), seed=7)
    engine.request_uuv_deployment("uuv_00", reason="test")
    engine.request_uuv_deployment("uuv_01", reason="test")
    group = engine.activate_execution_group(
        target_id="target_00",
        region_id="region-00",
        member_ids=("uuv_00", "uuv_01"),
    )

    observations = tuple(
        PassiveSonarObservation(
            observation_id=f"passive:{uuv_id}:target_00:30",
            scenario_id=engine._scenario_id,
            sim_time_s=30,
            observer_id=uuv_id,
            target_id="target_00",
            azimuth_rad=0.25 if uuv_id == "uuv_00" else 0.35,
            variance_rad2=0.01,
            detection_confidence=0.9,
            snr_db=8.0,
        )
        for uuv_id in group.member_ids
    )
    engine._fuse_execution_group_observations(30, observations)

    report = engine._latest_reports["target_00"]
    assert report.belief.source_observation_ids == tuple(
        observation.observation_id for observation in observations
    )
    assert engine.publication_situation().group_reports
