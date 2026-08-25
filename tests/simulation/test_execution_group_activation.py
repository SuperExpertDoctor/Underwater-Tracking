from __future__ import annotations

import pytest

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.models import DeploymentState
from underwater_tracking.domain.observations import PassiveSonarObservation
from underwater_tracking.simulation.engine import SimulationEngine


CONFIG_PATH = "configs/scenario/uuv_only_single_target.yaml"


def test_uuv_only_initialization_has_known_submarine_but_no_execution_group() -> None:
    engine = SimulationEngine(load_app_config(CONFIG_PATH), seed=7)

    snapshot = engine.publication_situation()

    assert snapshot.group_reports == ()
    assert snapshot.execution_groups == ()
    assert snapshot.target_search_priors == ()
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


def test_known_submarine_does_not_expire_at_the_former_prior_boundary() -> None:
    engine = SimulationEngine(load_app_config(CONFIG_PATH), seed=7)
    engine._clock.sim_time_s = 1800

    snapshot = engine.publication_situation()
    assert snapshot.target_search_priors == ()
    contact = next(item for item in snapshot.contacts if item.contact_id == "target_00")
    assert contact.classification.value == "submarine"
    assert contact.estimated_position_xy is not None


def test_known_submarine_contact_tracks_the_global_target_each_physics_step() -> None:
    engine = SimulationEngine(load_app_config(CONFIG_PATH), seed=7)
    initial_position = next(
        item
        for item in engine.publication_situation().contacts
        if item.contact_id == "target_00"
    ).estimated_position_xy

    engine.step()

    contact = next(
        item
        for item in engine.publication_situation().contacts
        if item.contact_id == "target_00"
    )
    assert contact.estimated_position_xy == engine._targets["target_00"].position_xy
    assert contact.estimated_position_xy != initial_position


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


def test_reused_target_filter_publishes_the_current_execution_group_members() -> None:
    engine = SimulationEngine(load_app_config(CONFIG_PATH), seed=7)
    for uuv_id in ("uuv_00", "uuv_01", "uuv_02", "uuv_03"):
        engine.request_uuv_deployment(uuv_id, reason="test")

    first = engine.activate_execution_group(
        target_id="target_00",
        region_id="region-00",
        member_ids=("uuv_00", "uuv_01"),
    )
    first_observations = tuple(
        PassiveSonarObservation(
            observation_id=f"first:{uuv_id}",
            scenario_id=engine._scenario_id,
            sim_time_s=30,
            observer_id=uuv_id,
            target_id="target_00",
            azimuth_rad=0.25 if uuv_id == "uuv_00" else 0.35,
            variance_rad2=0.01,
            detection_confidence=0.9,
            snr_db=8.0,
        )
        for uuv_id in first.member_ids
    )
    engine._fuse_execution_group_observations(30, first_observations)

    second = engine.activate_execution_group(
        target_id="target_00",
        region_id="region-01",
        member_ids=("uuv_02", "uuv_03"),
    )
    second_observations = (
        first_observations[0].model_copy(
            update={
                "observation_id": "second:uuv_02",
                "observer_id": "uuv_02",
                "sim_time_s": 60,
            }
        ),
    )
    engine._fuse_execution_group_observations(
        60,
        (
            *first_observations,
            *second_observations,
        ),
    )

    report = engine._latest_reports["target_00"]
    assert report.group_id == second.group_id
    assert report.member_ids == second.member_ids
    assert report.belief.source_observation_ids == tuple(
        observation.observation_id for observation in second_observations
    )
