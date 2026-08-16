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
from underwater_tracking.domain.models import EventLevel
from underwater_tracking.simulation.engine import SimulationEngine
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
                                     "waypoint_commands"}


def test_decoy_is_passively_indistinguishable_from_a_submarine(tmp_path):
    config = _decoy_config()
    frames = _run(config, 1, tmp_path=tmp_path)
    contacts = {c["contact_id"]: c for c in frames[0]["contacts"]}
    assert len(contacts["decoy_00"]["bearing_rays"]) == 12  # every observer


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
    # One 10 s step moves the decoy EXACTLY 5 m along its (post-noise)
    # heading; a multi-step chord would be bent short by the heading walk.
    first = truths[0]["decoys"][0]["position_xy"]
    last = truths[1]["decoys"][0]["position_xy"]
    delta = hypot(last[0] - first[0], last[1] - first[1])
    assert delta == pytest.approx(0.5 * 10.0, abs=1e-9)  # 0.5 m/s * one step


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


def test_heard_ping_triggers_evasive_sprint(tmp_path):
    config = _decoy_config(
        sensor_ping_heard_probability=1.0,
        sensor_active_classify_submarine_prob=1.0,
    )
    truths: list[dict[str, object]] = []
    engine = SimulationEngine(
        config, seed=7, output_dir=tmp_path, evaluation_sink=truths.append
    )
    engine.set_sensor_mode("uuv_00", "active", ping_contact_id="target_00")
    engine.step()
    target = truths[-1]["targets"][0]
    vx, vy = target["velocity_xy"]
    assert hypot(vx, vy) == pytest.approx(14.0, abs=1e-6)  # EVADE sprint
    assert target["intent_label"] == "evade"


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


def test_reserved_uuv_is_skipped_from_decoy_observation(tmp_path):
    config = _decoy_config()
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path)
    engine.set_reservations({"target_00": ("uuv_00",)})
    frame = engine.step()
    contacts = {c["contact_id"]: c for c in frame["contacts"]}
    rays = contacts["decoy_00"]["bearing_rays"]
    assert len(rays) == 11
    assert "uuv_00" not in {r["uuv_id"] for r in rays}
