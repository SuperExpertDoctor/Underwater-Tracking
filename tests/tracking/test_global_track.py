from __future__ import annotations

from math import atan2, isclose

import pytest

from underwater_tracking.config.loader import load_app_config
from underwater_tracking.simulation.engine import SimulationEngine
from underwater_tracking.tracking.global_track import GlobalTrackStore


def test_store_derives_executed_motion_features_and_bounds_history() -> None:
    store = GlobalTrackStore(history_limit=3)

    store.observe("target_00", 0, (0.0, 0.0), velocity_xy=(1.0, 0.0), source_event_ids=("e0",))
    store.observe("target_00", 10, (10.0, 0.0), velocity_xy=(1.0, 0.0), source_event_ids=("e1",))
    store.observe("target_00", 20, (20.0, 10.0), velocity_xy=(1.0, 1.0), source_event_ids=("e2",))
    store.observe("target_00", 30, (30.0, 20.0), velocity_xy=(1.0, 1.0), source_event_ids=("e3",))

    track = store.snapshot("target_00")

    assert track.position_xy == (30.0, 20.0)
    assert track.velocity_xy == (1.0, 1.0)
    assert track.heading_rad == pytest.approx(atan2(1.0, 1.0))
    assert track.acceleration_xy == pytest.approx((0.0, 0.0))
    assert track.turn_rate_rad_s == pytest.approx(0.0)
    assert tuple(sample.sim_time_s for sample in track.bounded_history) == (10.0, 20.0, 30.0)
    assert track.source_event_ids == ("e3",)


def test_store_replaces_equal_timestamp_and_rejects_reverse_samples() -> None:
    store = GlobalTrackStore(history_limit=8)
    store.observe("target_00", 10, (1.0, 2.0), velocity_xy=(1.0, 0.0))
    store.observe("target_00", 20, (2.0, 2.0), velocity_xy=(1.0, 0.0))
    store.observe("target_00", 20, (2.5, 2.0), velocity_xy=(1.0, 0.0), source_event_ids=("replacement",))

    assert len(store.history("target_00")) == 2
    assert store.snapshot("target_00").position_xy == (2.5, 2.0)
    assert store.snapshot("target_00").source_event_ids == ("replacement",)
    with pytest.raises(ValueError, match="older"):
        store.observe("target_00", 19, (2.4, 2.0), velocity_xy=(1.0, 0.0))


def test_store_checkpoint_restore_is_deterministic() -> None:
    store = GlobalTrackStore(history_limit=8)
    store.observe("target_00", 0, (0.0, 0.0), velocity_xy=(1.0, 0.0))
    store.observe("target_00", 10, (10.0, 0.0), velocity_xy=(1.0, 0.0))
    checkpoint = store.checkpoint()
    expected = store.snapshot("target_00").model_dump(mode="json")

    store.observe("target_00", 20, (20.0, 2.0), velocity_xy=(1.0, 0.2))
    store.restore(checkpoint)

    assert store.snapshot("target_00").model_dump(mode="json") == expected


def test_uuv_only_engine_exposes_global_track_but_legacy_history_remains_compatible(tmp_path) -> None:
    config = load_app_config("configs/scenario/uuv_only_single_target.yaml")
    engine = SimulationEngine(config, seed=config.scenario.seed, output_dir=tmp_path)

    for _ in range(6):
        engine.step()

    track = engine.global_target_track("target_00")
    assert track is not None
    assert track.track_revision >= 2
    assert track.sim_time_s == engine._clock.sim_time_s
    assert track.position_xy == engine._targets["target_00"].position_xy
    assert len(track.bounded_history) >= 2
    assert engine.global_target_history("target_00")[-1][0] == engine._clock.sim_time_s
    assert isclose(track.velocity_xy[0], engine._targets["target_00"].velocity_xy[0])
