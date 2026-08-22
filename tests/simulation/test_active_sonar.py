# tests/simulation/test_active_sonar.py
"""Active-sonar probe model and decoy entities (spec 5.1/11.1 amendment, R5).

The engine emits decoys with the same passive bearing observations as
submarines; classification comes exclusively from active pings. All tests
construct custom configs (decoy behavior is off by default), are
deterministic under the fixed seed, and never touch truth-boundary gates.
"""

from math import hypot

import pytest

from underwater_tracking.agent.nodes.event_monitor import EventMonitor
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.models import DeploymentState, EventLevel
from underwater_tracking.simulation.engine import SimulationEngine
from underwater_tracking.simulation.target import HiddenIntent, TRANSITION_PROBABILITIES
from tests.conftest import CONFIG_PATH


def _decoy_config(**overrides: object) -> object:
    base = load_app_config(CONFIG_PATH)
    tracking = base.tracking.model_copy(update=overrides)
    scenario = base.scenario.model_copy(update={"initial_decoy_count": 1})
    return base.model_copy(update={"tracking": tracking, "scenario": scenario})


def _run(config: object, steps: int, *, tmp_path, sink=None) -> list[dict[str, object]]:
    engine = SimulationEngine(
        config,
        seed=7,
        output_dir=tmp_path,
        evaluation_sink=sink.append if sink is not None else None,
    )
    frames: list[dict[str, object]] = []
    for _ in range(steps):
        frames.append(engine.step())
    return frames


def test_active_sonar_event_types_classify():
    monitor = EventMonitor()
    assert monitor.classify("active_ping") is EventLevel.INFORMATIONAL
    assert monitor.classify("contact_classified") is EventLevel.INFORMATIONAL


def test_decoy_spawns_unverified_contact(tmp_path):
    config = _decoy_config()
    frames = _run(config, 1, tmp_path=tmp_path)
    contacts = {c["contact_id"]: c for c in frames[0]["contacts"]}
    assert "decoy_00" in contacts
    assert contacts["decoy_00"]["classification"] == "unverified"
    assert set(frames[0].keys()) <= {"carrier", "contacts", "reservations", "run_id", "scenario_id",
                                     "sim_time_s", "step_index", "uuvs", "group_reports",
                                     "tracks", "quality", "assignments", "events",
                                     "waypoint_commands", "platform_core", "usvs",
                                     "communication_links", "sonar_observations",
                                     "execution_groups", "target_search_priors"}


def test_decoy_is_passively_indistinguishable_from_a_submarine(tmp_path):
    config = _decoy_config()
    frames = _run(config, 1, tmp_path=tmp_path)
    contacts = {c["contact_id"]: c for c in frames[0]["contacts"]}
    rays = contacts["decoy_00"]["bearing_rays"]
    assert 0 < len(rays) <= 12  # probabilistic detection, never geometry-only
    assert all(not ray["is_false_alarm"] for ray in rays)


def test_unverified_ping_request_does_not_expose_contact_position(tmp_path):
    config = _decoy_config()
    frames = _run(config, 6, tmp_path=tmp_path)
    requests = [
        event
        for frame in frames
        for event in frame["events"]
        if event["event_type"] == "active_ping"
        and event["entity_id"] == "decoy_00"
        and "emitter_id" not in event["payload"]
    ]
    assert requests
    assert all("position_xy" not in event["payload"] for event in requests)


def test_truth_reports_decoys(tmp_path):
    config = _decoy_config()
    truths: list[dict[str, object]] = []
    _run(config, 1, tmp_path=tmp_path, sink=truths)
    assert truths
    decoys = truths[-1]["decoys"]
    assert [d["decoy_id"] for d in decoys] == ["decoy_00"]


def test_decoy_drift_speed_is_configured(tmp_path):
    config = _decoy_config()
    truths: list[dict[str, object]] = []
    _run(config, 3, tmp_path=tmp_path, sink=truths)
    # One physics step moves the decoy EXACTLY by speed * step (post-noise)
    # heading; a multi-step chord would be bent short by the heading walk.
    first = truths[0]["decoys"][0]["position_xy"]
    last = truths[1]["decoys"][0]["position_xy"]
    delta = hypot(last[0] - first[0], last[1] - first[1])
    assert delta == pytest.approx(0.5 * config.timing.physics_step_s, abs=1e-9)


def test_unheard_ping_still_consumes_active_sonar_energy(tmp_path):
    config = _decoy_config(sensor_ping_heard_probability=0.0)
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    baseline = SimulationEngine(config, seed=7, output_dir=tmp_path / "baseline")
    engine.set_sensor_mode("uuv_00", "active", ping_contact_id="decoy_00")

    frame = engine.step()
    baseline_frame = baseline.step()
    ping_energy = config.tracking.sensor_ping_energy_cost
    actual = next(item["energy_fraction"] for item in frame["uuvs"] if item["uuv_id"] == "uuv_00")
    without_ping = next(
        item["energy_fraction"] for item in baseline_frame["uuvs"] if item["uuv_id"] == "uuv_00"
    )

    assert actual == pytest.approx(without_ping - ping_energy, abs=1e-9)
    assert not any(event["event_type"] == "contact_classified" for event in frame["events"])


def test_active_ping_classifies_and_drains_energy(tmp_path):
    config = _decoy_config(
        sensor_ping_heard_probability=1.0,
        sensor_active_classify_decoy_prob=0.0,  # decoys ALWAYS classify submarine
    )
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    engine.set_sensor_mode("uuv_00", "active", ping_contact_id="decoy_00")
    frame = engine.step()
    contacts = {c["contact_id"]: c for c in frame["contacts"]}
    assert contacts["decoy_00"]["classification"] == "submarine"
    assert contacts["decoy_00"]["estimated_position_xy"] is not None
    uuvs = {u["uuv_id"]: u for u in frame["uuvs"]}
    # The ping drains exactly 2e-4; uuv_00 also burns motion energy in the
    # same step (40 m at 2e-6/m + 10 s at 1e-7/s = 8.1e-5), so the total
    # sits just below 1 - 2e-4.
    assert uuvs["uuv_00"]["energy_fraction"] == pytest.approx(1.0 - 2e-4, abs=1e-4)
    assert any(e["event_type"] == "contact_classified" for e in frame["events"])
    assert any(e["event_type"] == "active_ping" for e in frame["events"])


def test_heard_ping_queues_bounded_evasive_sprint(tmp_path, monkeypatch):
    config = _decoy_config(
        sensor_ping_heard_probability=1.0,
        sensor_active_classify_submarine_prob=1.0,
    )
    truths: list[dict[str, object]] = []
    engine = SimulationEngine(
        config, seed=7, output_dir=tmp_path, evaluation_sink=truths.append
    )
    monkeypatch.setitem(
        TRANSITION_PROBABILITIES,
        HiddenIntent.EVADE,
        {HiddenIntent.EVADE: 1.0},
    )
    engine.set_sensor_mode("uuv_00", "active", ping_contact_id="target_00")
    engine.step()
    ping_target = truths[-1]["targets"][0]
    ping_vx, ping_vy = ping_target["velocity_xy"]
    ping_speed = hypot(ping_vx, ping_vy)
    assert ping_target["intent_label"] == "evade"
    assert ping_speed < config.tracking.submarine_sprint_speed_mps

    target_entity = engine._targets["target_00"]
    engine.step()
    next_target = truths[-1]["targets"][0]
    next_vx, next_vy = next_target["velocity_xy"]
    next_speed = hypot(next_vx, next_vy)
    max_speed_delta = (
        target_entity.max_acceleration_mps2 * config.timing.physics_step_s
    )

    assert next_speed > ping_speed
    assert next_speed - ping_speed <= max_speed_delta + 1e-9
    assert next_speed <= config.tracking.submarine_sprint_speed_mps + 1e-9


@pytest.mark.parametrize("deployment_state", [
    DeploymentState.RETURNING,
    DeploymentState.ONBOARD,
    DeploymentState.FAILED,
])
def test_active_sonar_does_not_ping_or_classify_non_deployed_uuvs(
    tmp_path, deployment_state: DeploymentState
):
    config = _decoy_config(
        sensor_ping_heard_probability=1.0,
        sensor_active_classify_decoy_prob=1.0,
    )
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    uuv_id = "uuv_00"
    if deployment_state is DeploymentState.RETURNING:
        engine.request_uuv_recovery(uuv_id)
    elif deployment_state is DeploymentState.ONBOARD:
        engine.request_uuv_recovery(uuv_id)
        engine._uuvs[uuv_id].position_xy = (-2950.0, -3000.0)
        engine.step()
    else:
        engine.fail_uuv(uuv_id)

    engine.set_sensor_mode(uuv_id, "active", ping_contact_id="decoy_00")
    frame = engine.step()

    executed_events = [
        event
        for event in frame["events"]
        if event["event_type"] in {"active_ping", "contact_classified"}
        and event["payload"].get("uuv_id") == uuv_id
    ]
    contacts = {contact["contact_id"]: contact for contact in frame["contacts"]}
    assert executed_events == []
    assert contacts["decoy_00"]["classification"] == "unverified"


def test_drop_contact_removes_the_decoy(tmp_path):
    config = _decoy_config()
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    engine.step()
    engine.drop_contact("decoy_00")
    frame = engine.step()
    assert "decoy_00" not in {c["contact_id"] for c in frame["contacts"]}


def test_promote_contact_creates_target_and_group(tmp_path):
    config = _decoy_config(
        sensor_ping_heard_probability=1.0,
        sensor_active_classify_decoy_prob=0.0,  # ALWAYS submarine
    )
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    engine.set_sensor_mode("uuv_00", "active", ping_contact_id="decoy_00")
    engine.step()  # the commanded ping classifies decoy_00 as submarine
    engine.promote_contact("decoy_00")
    frame = engine.step()  # the group exists from promotion; the frame carries it
    reports = {r["target_id"]: r for r in frame["group_reports"]}
    assert "decoy_00" in reports
    assert 2 <= len(reports["decoy_00"]["member_ids"]) <= 4


def test_promoted_legacy_target_uses_configured_motion_limits(tmp_path):
    config = _decoy_config(
        submarine_sprint_speed_mps=19.0,
        submarine_turn_rate_rad_s=0.17,
    )
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    engine._contact_state["decoy_00"]["position_xy"] = (100.0, 200.0)

    engine.promote_contact("decoy_00")

    target = engine._targets["decoy_00"]
    assert target.max_speed_mps == config.tracking.submarine_sprint_speed_mps
    assert target.max_turn_rate_rad_s == config.tracking.submarine_turn_rate_rad_s


def test_reserved_uuv_is_skipped_from_decoy_observation(tmp_path):
    config = _decoy_config()
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    engine.set_reservations({"target_00": ("uuv_00",)})
    frame = engine.step()
    contacts = {c["contact_id"]: c for c in frame["contacts"]}
    rays = contacts["decoy_00"]["bearing_rays"]
    assert 0 < len(rays) <= 11
    assert all(not ray["is_false_alarm"] for ray in rays)
    assert "uuv_00" not in {r["uuv_id"] for r in rays}
