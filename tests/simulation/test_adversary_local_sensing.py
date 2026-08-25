from __future__ import annotations

from math import hypot

from math import isfinite

import pytest

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.observations import PassiveSonarObservation
from underwater_tracking.domain.models import DeploymentState
from underwater_tracking.simulation.adversary_sensing import (
    ExposedPlatform,
    TargetContactMemory,
    update_local_platform_detections,
)
from underwater_tracking.simulation.engine import SimulationEngine


def _platform(
    platform_id: str = "uuv_00",
    *,
    position_xy: tuple[float, float] = (1200.0, 0.0),
    platform_kind: str = "uuv",
    sensor_mode: str = "passive",
) -> ExposedPlatform:
    return ExposedPlatform(
        platform_id=platform_id,
        platform_kind=platform_kind,  # type: ignore[arg-type]
        position_xy=position_xy,
        sensor_mode=sensor_mode,  # type: ignore[arg-type]
        relay_available=True,
    )


def _sense(
    candidates: tuple[ExposedPlatform, ...],
    *,
    previous_ids: frozenset[str] = frozenset(),
    sim_time_s: int = 0,
    seed: int = 7,
):
    return update_local_platform_detections(
        target_id="target_00",
        target_position_xy=(0.0, 0.0),
        target_heading_rad=0.0,
        detection_range_m=1200.0,
        release_margin_m=100.0,
        candidates=candidates,
        previous_ids=previous_ids,
        sim_time_s=sim_time_s,
        seed=seed,
    )


def test_local_sensor_acquires_only_inside_configured_range() -> None:
    acquired = _sense((_platform(position_xy=(1200.0, 0.0)),))
    outside = _sense((_platform(position_xy=(1201.0, 0.0)),))

    assert acquired.acquired_platform_ids == frozenset({"uuv_00"})
    assert len(acquired.detections) == 1
    assert outside.detections == ()
    assert outside.acquired_platform_ids == frozenset()


def test_local_sensor_range_is_circular_and_independent_of_submarine_heading() -> None:
    candidates = (
        _platform("east", position_xy=(1200.0, 0.0)),
        _platform("north", position_xy=(0.0, 1200.0)),
        _platform("west", position_xy=(-1200.0, 0.0)),
        _platform("south", position_xy=(0.0, -1200.0)),
    )

    result = _sense(candidates)

    assert {detection.platform_id for detection in result.detections} == {
        "east",
        "north",
        "west",
        "south",
    }


def test_local_sensor_uses_three_dimensional_range_when_depth_is_present() -> None:
    candidate = _platform(position_xy=(1100.0, 0.0))
    deep_candidate = ExposedPlatform(
        platform_id=candidate.platform_id,
        platform_kind=candidate.platform_kind,
        position_xy=candidate.position_xy,
        sensor_mode=candidate.sensor_mode,
        relay_available=candidate.relay_available,
        depth_m=700.0,
    )
    result = update_local_platform_detections(
        target_id="target_00",
        target_position_xy=(0.0, 0.0),
        target_depth_m=700.0,
        target_heading_rad=0.0,
        detection_range_m=1200.0,
        release_margin_m=100.0,
        candidates=(deep_candidate,),
        previous_ids=frozenset(),
        sim_time_s=0,
        seed=7,
    )
    assert result.acquired_platform_ids == frozenset({"uuv_00"})


def test_detection_hysteresis_retains_until_release_margin_then_loses() -> None:
    retained = _sense(
        (_platform(position_xy=(1250.0, 0.0)),),
        previous_ids=frozenset({"uuv_00"}),
    )
    boundary = _sense(
        (_platform(position_xy=(1300.0, 0.0)),),
        previous_ids=frozenset({"uuv_00"}),
    )
    lost = _sense(
        (_platform(position_xy=(1301.0, 0.0)),),
        previous_ids=frozenset({"uuv_00"}),
    )

    assert {item.platform_id for item in retained.detections} == {"uuv_00"}
    assert retained.acquired_platform_ids == frozenset()
    assert retained.lost_platform_ids == frozenset()
    assert {item.platform_id for item in boundary.detections} == {"uuv_00"}
    assert lost.detections == ()
    assert lost.lost_platform_ids == frozenset({"uuv_00"})


def test_local_estimates_are_noisy_but_deterministic_and_do_not_expose_coordinates() -> None:
    first = _sense((_platform(position_xy=(800.0, 600.0)),), seed=19)
    second = _sense((_platform(position_xy=(800.0, 600.0)),), seed=19)

    assert first == second
    detection = first.detections[0]
    assert isfinite(detection.estimated_range_m)
    assert isfinite(detection.relative_bearing_rad)
    payload = detection.model_dump(mode="json")
    assert "position_xy" not in payload
    assert "true_distance" not in payload


def test_active_emitter_audibility_uses_strict_sensor_range_not_hysteresis() -> None:
    retained = _sense(
        (_platform(position_xy=(1250.0, 0.0), sensor_mode="active"),),
        previous_ids=frozenset({"uuv_00"}),
    )
    audible = _sense(
        (_platform(position_xy=(1199.0, 0.0), sensor_mode="active"),),
        previous_ids=frozenset({"uuv_00"}),
    )

    assert {item.platform_id for item in retained.detections} == {"uuv_00"}
    assert retained.audible_active_emitter_ids == frozenset()
    assert audible.audible_active_emitter_ids == frozenset({"uuv_00"})


def test_contact_memory_emits_episode_transitions_and_expires_lost_context() -> None:
    memory = TargetContactMemory("target_00", ttl_s=120)
    acquired = _sense((_platform(position_xy=(800.0, 0.0)),), sim_time_s=0)
    stable = _sense((_platform(position_xy=(810.0, 0.0)),), sim_time_s=30)
    lost = _sense(
        (_platform(position_xy=(1301.0, 0.0)),),
        previous_ids=frozenset({"uuv_00"}),
        sim_time_s=60,
    )

    assert [item.event_type for item in memory.update(acquired, 0)] == [
        "target_detection_acquired"
    ]
    assert memory.update(stable, 30) == ()
    assert [item.event_type for item in memory.update(lost, 60)] == [
        "target_detection_lost"
    ]
    assert memory.active(100) == ()
    assert tuple(contact.status for contact in memory.context(100)) == ("lost",)
    assert memory.context(181) == ()


def test_contact_memory_buckets_range_and_threat_changes_and_new_emitter() -> None:
    memory = TargetContactMemory("target_00")
    first = _sense((_platform(position_xy=(700.0, 0.0)),), sim_time_s=0)
    same_bucket = _sense((_platform(position_xy=(720.0, 0.0)),), sim_time_s=30)
    next_bucket = _sense((_platform(position_xy=(1000.0, 0.0)),), sim_time_s=60)
    active_inside_release = _sense(
        (_platform(position_xy=(1100.0, 0.0), sensor_mode="active"),),
        sim_time_s=90,
        previous_ids=frozenset({"uuv_00"}),
    )

    memory.update(first, 0)
    assert memory.update(same_bucket, 30) == ()
    assert "target_contact_range_changed" in {
        item.event_type for item in memory.update(next_bucket, 60)
    }
    assert "target_active_emitter_acquired" in {
        item.event_type for item in memory.update(active_inside_release, 90)
    }


def test_target_exposure_includes_surface_group_and_only_waterborne_uuvs() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=7)

    initial = {item.platform_id: item for item in engine._target_exposed_platforms()}
    assert set(initial) == {"carrier_01", "carrier_02", "carrier_03", "carrier_04"}
    assert {item.platform_kind for item in initial.values()} == {
        "carrier",
        "mother_ship",
    }

    engine.request_uuv_deployment("uuv_00")
    deployed = {item.platform_id: item for item in engine._target_exposed_platforms()}
    assert "uuv_00" in deployed

    engine.fail_uuv("uuv_00")
    failed_waterborne = {item.platform_id for item in engine._target_exposed_platforms()}
    assert "uuv_00" in failed_waterborne

    engine._deployment_states["uuv_00"] = DeploymentState.ONBOARD
    engine._waterborne_uuv_ids.discard("uuv_00")
    recovered = {item.platform_id for item in engine._target_exposed_platforms()}
    assert "uuv_00" not in recovered


def test_initial_carrier_group_places_the_flagship_at_triangle_center() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=7)

    flagship = engine._carrier_entities["carrier_01"].position_xy
    escorts = tuple(
        engine._carrier_entities[carrier_id].position_xy
        for carrier_id in ("carrier_02", "carrier_03", "carrier_04")
    )
    centroid = (
        sum(position[0] for position in escorts) / len(escorts),
        sum(position[1] for position in escorts) / len(escorts),
    )
    radii = tuple(hypot(position[0] - flagship[0], position[1] - flagship[1]) for position in escorts)

    assert centroid == pytest.approx(flagship)
    assert radii == pytest.approx((radii[0], radii[0], radii[0]))


def test_target_local_sensing_rejects_usv_exposure_kind() -> None:
    with pytest.raises(ValueError, match="target platform kind"):
        _sense((_platform(platform_kind="usv"),))


def test_engine_emits_one_acquire_loss_pair_per_detection_episode() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=7)
    target = engine._targets["target_00"]
    carrier = engine._carrier_entities["carrier_01"]
    detection_range_m = target.detection_range_m
    target.position_xy = (0.0, 0.0)
    for carrier_id, other_carrier in engine._carrier_entities.items():
        if carrier_id != "carrier_01":
            other_carrier.position_xy = (6000.0, 0.0)

    carrier.position_xy = (detection_range_m - 1.0, 0.0)
    engine._update_target_detection_events(0)
    carrier.position_xy = (detection_range_m + 101.0, 0.0)
    engine._update_target_detection_events(30)
    carrier.position_xy = (detection_range_m - 1.0, 0.0)
    engine._update_target_detection_events(60)

    transitions = [
        event
        for event in engine._events
        if event.entity_id == "target_00"
        and event.event_type in {"target_detection_acquired", "target_detection_lost"}
    ]
    assert [event.event_id for event in transitions] == [
        "target_detection_acquired:target_00:carrier_01:e1",
        "target_detection_lost:target_00:carrier_01:e1",
        "target_detection_acquired:target_00:carrier_01:e2",
    ]


def test_target_input_uses_local_estimates_and_ignores_blue_observations() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=7)
    target = engine._targets["target_00"]
    carrier = engine._carrier_entities["carrier_01"]
    detection_range_m = target.detection_range_m
    target.position_xy = (0.0, 0.0)
    for carrier_id, other_carrier in engine._carrier_entities.items():
        if carrier_id != "carrier_01":
            other_carrier.position_xy = (6000.0, 0.0)

    carrier.position_xy = (detection_range_m + 1.0, 0.0)
    engine._update_target_detection_events(0)
    outside = engine._build_situation(0).model_copy(
        update={
            "platform_observations": (
                PassiveSonarObservation(
                    observation_id="blue-bearing-1",
                    scenario_id=config.scenario.scenario_id,
                    sim_time_s=0,
                    observer_id="uuv_00",
                    target_id="target_00",
                    azimuth_rad=0.2,
                    variance_rad2=0.1,
                    detection_confidence=0.9,
                    snr_db=8.0,
                ),
            )
        }
    )
    outside_contexts = engine.build_adversary_inputs(outside)
    assert len(outside_contexts) == 1
    assert outside_contexts[0].platform_threats == ()
    assert any(
        trigger.event_type == "target_mission_initialized"
        for trigger in outside_contexts[0].trigger_events
    )

    carrier.position_xy = (detection_range_m - 1.0, 0.0)
    engine._update_target_detection_events(30)
    local_situation = engine._build_situation(30)
    contexts = engine.build_adversary_inputs(local_situation)

    assert len(contexts) == 1
    context = contexts[0]
    assert {threat.platform_id for threat in context.platform_threats} == {"carrier_01"}
    assert all(
        observation.observation_id != "blue-bearing-1"
        for observation in context.observations
    )
    assert context.platform_threats[0].estimated_range_m != detection_range_m - 1.0
    assert all(
        "position_xy" not in threat.model_dump(mode="json")
        and "true_distance" not in threat.model_dump(mode="json")
        for threat in context.platform_threats
    )


def test_engine_retains_platform_threat_but_drops_out_of_range_active_ping() -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=7)
    target = engine._targets["target_00"]
    detection_range_m = target.detection_range_m
    target.position_xy = (0.0, 0.0)
    for carrier in engine._carrier_entities.values():
        carrier.position_xy = (6000.0, 0.0)
    uuv = engine._uuvs["uuv_00"]
    uuv.position_xy = (detection_range_m + 50.0, 0.0)
    engine._deployment_states["uuv_00"] = DeploymentState.DEPLOYED
    engine._waterborne_uuv_ids.add("uuv_00")
    engine.set_sensor_mode("uuv_00", "active", ping_contact_id="target_00")
    engine._target_detected_platform_ids["target_00"] = ("uuv_00",)

    engine._update_target_detection_events(0)
    retained_context = engine.build_adversary_inputs(engine._build_situation(0))[0]
    assert {threat.platform_id for threat in retained_context.platform_threats} == {
        "uuv_00"
    }
    assert retained_context.communications_acoustic_exposure.active_emitter_exposure == 0.0
    assert all(
        observation.kind != "active_sonar"
        for observation in retained_context.observations
    )

    uuv.position_xy = (detection_range_m - 1.0, 0.0)
    engine._update_target_detection_events(30)
    audible_context = engine.build_adversary_inputs(engine._build_situation(30))[0]
    assert audible_context.communications_acoustic_exposure.active_emitter_exposure == 1.0
    assert any(
        observation.kind == "active_sonar"
        for observation in audible_context.observations
    )
