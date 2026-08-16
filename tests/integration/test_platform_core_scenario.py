from array import array
from collections import deque
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from math import atan2, cos, hypot, pi, sin
import random
import re
from typing import Any, cast
from uuid import UUID

import numpy as np
import pytest

from underwater_tracking.agent.llm import HTTPStructuredLLM
from underwater_tracking.cli import _AgentLoop, _create_public_run_dir
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.config.models import AppConfig
from underwater_tracking.persistence.frame_log import FrameLogger
from underwater_tracking.simulation.clock import SimulationClock
from underwater_tracking.simulation import engine as engine_module
from underwater_tracking.simulation.engine import _ExplicitPlatformCoreCheckpoint, SimulationEngine
from underwater_tracking.simulation.kinematics import (
    MotionCommand,
    MotionState,
    advance_motion,
    wrap_angle,
)


SCENARIO = Path("configs/scenario/segmented_single_target.yaml")


class _ManualPayloadCopy:
    """A deliberately incomplete deepcopy implementation used for rollback coverage."""

    def __init__(self, payload: list[str]) -> None:
        self.payload = payload

    def __deepcopy__(self, memo: dict[int, object]) -> "_ManualPayloadCopy":
        del memo
        return type(self)(list(self.payload))


class _DeferredRollbackSlot:
    """Slot state whose failed-tick field is intentionally unset at checkpoint time."""

    __slots__ = ("payload", "failed_tick")

    payload: list[str]
    failed_tick: list[str]

    def __init__(self, payload: list[str]) -> None:
        self.payload = payload


class _ReverseCopyDict(dict[str, object]):
    """A mapping whose deepcopy intentionally reverses its insertion order."""

    def __deepcopy__(self, memo: dict[int, object]) -> "_ReverseCopyDict":
        copied = type(self)()
        memo[id(self)] = copied
        for key, value in reversed(tuple(self.items())):
            copied[deepcopy(key, memo)] = deepcopy(value, memo)
        return copied


class _UnsafeSetMember:
    """A set member whose equality must never run during checkpoint association."""

    comparisons = 0

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        del other
        type(self).comparisons += 1
        raise AssertionError("rollback checkpoint invoked user equality")

    def __deepcopy__(self, memo: dict[int, object]) -> "_UnsafeSetMember":
        del memo
        return type(self)()


class _UnmappedSet(set[_UnsafeSetMember]):
    """A set copier that creates members without registering their identities."""

    def __deepcopy__(self, memo: dict[int, object]) -> "_UnmappedSet":
        copied = type(self)()
        memo[id(self)] = copied
        for member in self:
            copied.add(type(member)())
        return copied


class _AttributedDeque(deque[object]):
    """A deque subclass with state outside its sequence contents."""

    payload: list[str]
    removed: list[str]
    added: list[str]


class _AttributedBytearray(bytearray):
    """A bytearray subclass with state outside its byte contents."""

    payload: list[str]
    removed: list[str]
    added: list[str]


def _capture_explicit_runtime_checkpoint(
    engine: SimulationEngine, monkeypatch: pytest.MonkeyPatch
) -> list[_ExplicitPlatformCoreCheckpoint]:
    captured: list[_ExplicitPlatformCoreCheckpoint] = []
    checkpoint = engine._checkpoint_explicit_platform_core

    def capture() -> _ExplicitPlatformCoreCheckpoint:
        result = checkpoint()
        captured.append(result)
        return result

    monkeypatch.setattr(engine, "_checkpoint_explicit_platform_core", capture)
    return captured


def _stable_int(text: str) -> int:
    return int.from_bytes(sha256(text.encode("utf-8")).digest()[:8], "big")


def _public_quality_reconstruction(
    run_id: str,
    observers: dict[str, dict[str, object]],
    observations: list[dict[str, object]],
) -> tuple[float, float] | None:
    """Reproduce the pre-fix public run-id attack when a seed is exposed."""
    match = re.fullmatch(r"run-(\d+)-[0-9a-f]+", run_id)
    if match is None:
        return None
    seed = int(match.group(1))
    recovered: list[tuple[tuple[float, float], float]] = []
    for observation in observations[:3]:
        observer_id = str(observation["observer_id"])
        target_id = str(observation["target_id"])
        observer = observers[observer_id]
        capability = observer["capability"]
        assert isinstance(capability, dict)
        sonar = capability["sonar"]
        assert isinstance(sonar, dict)
        quality_rng_key = f"quality:platform:{target_id}:{observer_id}"
        quality_rng = random.Random(seed ^ _stable_int(quality_rng_key))
        quality_noise = quality_rng.gauss(0.0, 0.075)
        confidence = float(observation["detection_confidence"])
        range_fraction = (0.9 + quality_noise - confidence) / 0.7
        position_xy = observer["position_xy"]
        assert isinstance(position_xy, tuple)
        recovered.append((position_xy, float(sonar["passive_range_m"]) * range_fraction))

    (x0, y0), r0 = recovered[0]
    (x1, y1), r1 = recovered[1]
    (x2, y2), r2 = recovered[2]
    a = 2.0 * (x1 - x0)
    b = 2.0 * (y1 - y0)
    c = 2.0 * (x2 - x0)
    d = 2.0 * (y2 - y0)
    e = r0**2 - r1**2 + x1**2 + y1**2 - x0**2 - y0**2
    f = r0**2 - r2**2 + x2**2 + y2**2 - x0**2 - y0**2
    determinant = a * d - b * c
    assert determinant != 0.0
    return ((e * d - b * f) / determinant, (a * f - e * c) / determinant)


def _normalized_frame(frame: dict[str, object]) -> dict[str, object]:
    return {**frame, "run_id": "RUN"}


def _validated_platform_core_config(
    *,
    carrier_updates: dict[str, object],
    motion_updates: dict[str, object] | None = None,
    deployment_state: str | None = None,
) -> AppConfig:
    data = load_app_config(SCENARIO).model_dump()
    data["environment"]["carrier"].update(carrier_updates)
    if deployment_state is not None:
        for usv in data["environment"]["usvs"]:
            usv["deployment_state"] = deployment_state
    if motion_updates is not None:
        data["platforms"]["motion_profiles"]["usv_standard"].update(motion_updates)
    return AppConfig.model_validate(data)


def _small_support_fast_carrier_config() -> AppConfig:
    return _validated_platform_core_config(
        carrier_updates={"speed_mps": 100.0, "support_radius_m": 650.0},
        deployment_state="onboard",
    )


def test_explicit_platform_core_world_spawns_from_yaml(tmp_path: Path) -> None:
    engine = SimulationEngine(load_app_config(SCENARIO), seed=42, output_dir=tmp_path)

    snapshot = engine.platform_snapshot()

    assert snapshot.scenario_id == "segmented-single-target"
    assert snapshot.carrier.carrier_id == "carrier_01"
    assert [usv.platform_id for usv in snapshot.roster.usvs] == [
        "usv_00", "usv_01", "usv_02", "usv_03"
    ]
    assert [uuv.platform_id for uuv in snapshot.roster.uuvs] == [
        f"uuv_{index:02d}" for index in range(12)
    ]
    assert snapshot.carrier.onboard_platform_ids == tuple(
        f"uuv_{index:02d}" for index in range(12)
    )


def test_explicit_initial_snapshot_normalizes_onboard_uuvs_only(tmp_path: Path) -> None:
    data = load_app_config(SCENARIO).model_dump()
    data["environment"]["uuvs"][0].update(
        {"position_xy": [-7000.0, -7000.0], "heading_rad": 1.2}
    )
    data["environment"]["uuvs"][1].update(
        {
            "deployment_state": "deployed",
            "position_xy": [-6000.0, -6000.0],
            "heading_rad": 0.7,
        }
    )
    config = AppConfig.model_validate(data)
    engine = SimulationEngine(config, seed=42, output_dir=tmp_path)

    snapshot = engine.platform_snapshot()
    states = {state.platform_id: state for state in snapshot.roster.uuvs}

    onboard = states["uuv_00"]
    assert onboard.deployment_state == "onboard"
    assert onboard.position_xy == snapshot.carrier.position_xy
    assert onboard.heading_rad == snapshot.carrier.heading_rad
    assert onboard.speed_mps == 0.0
    deployed = states["uuv_01"]
    assert deployed.deployment_state == "deployed"
    assert deployed.position_xy == (-6000.0, -6000.0)
    assert deployed.heading_rad == pytest.approx(0.7)


def test_explicit_frame_exposes_usvs_and_distance_links(tmp_path: Path) -> None:
    engine = SimulationEngine(load_app_config(SCENARIO), seed=42, output_dir=tmp_path)

    frame = engine.step()
    for _ in range(2):
        frame = engine.step()

    assert frame["platform_core"] is True
    assert len(frame["usvs"]) == 4
    assert frame["uuvs"][0]["deployment_state"] == "onboard"
    assert any(link["medium"] == "surface" for link in frame["communication_links"])
    assert frame["sonar_observations"]


def test_public_platform_frame_hides_seed_and_blocks_exact_quality_reconstruction(
    tmp_path: Path,
) -> None:
    seed = 42
    first_engine = SimulationEngine(
        load_app_config(SCENARIO), seed=seed, output_dir=tmp_path / "first"
    )
    second_engine = SimulationEngine(
        load_app_config(SCENARIO), seed=seed, output_dir=tmp_path / "second"
    )
    for _ in range(3):
        first_frame = first_engine.step()
        second_frame = second_engine.step()

    run_id = str(first_frame["run_id"])
    assert re.fullmatch(r"run-[0-9a-f]{32}", run_id)
    assert UUID(hex=run_id.removeprefix("run-")).hex == run_id.removeprefix("run-")
    assert _normalized_frame(first_frame) == _normalized_frame(second_frame)
    assert first_frame["run_id"] != second_frame["run_id"]
    assert "seed" not in first_frame

    public_observers = {
        str(platform["platform_id"]): platform
        for platform in [*first_frame["usvs"], *first_frame["uuvs"]]
    }
    public_observations = [
        observation
        for observation in first_frame["sonar_observations"]
        if observation["target_id"] == "target_00"
    ]
    assert len(public_observations) >= 3

    truth_xy = first_engine._targets["target_00"].position_xy
    known_seed_reconstruction = _public_quality_reconstruction(
        f"run-{seed}-deadbeef", public_observers, public_observations
    )
    assert known_seed_reconstruction is not None
    assert hypot(
        known_seed_reconstruction[0] - truth_xy[0],
        known_seed_reconstruction[1] - truth_xy[1],
    ) < 1e-6
    assert _public_quality_reconstruction(run_id, public_observers, public_observations) is None


def test_agent_public_paths_and_manifest_hide_constructor_seed(tmp_path: Path) -> None:
    seed = 2_718_281_828
    output_root = tmp_path / "outputs"
    run_dir = _create_public_run_dir("run", output_root=output_root)
    serve_dir = _create_public_run_dir("serve", output_root=output_root)

    for prefix, directory in (("run", run_dir), ("serve", serve_dir)):
        assert directory.parent == output_root
        assert re.fullmatch(rf"{prefix}-[0-9a-f]{{32}}", directory.name)
        assert str(seed) not in directory.name

    loop = _AgentLoop(
        load_app_config(SCENARIO),
        database_path=run_dir / "agent.db",
        llm=cast(HTTPStructuredLLM, object()),
        run_id=run_dir.name,
        steps=3,
        seed=seed,
    )
    try:
        loop.write_manifest(run_dir)
    finally:
        loop.close()

    manifest_text = (run_dir / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["run_id"] == run_dir.name
    assert "seed" not in manifest
    assert str(seed) not in manifest_text


def test_explicit_uuv_energy_uses_yaml_motion_profile(tmp_path: Path) -> None:
    data = load_app_config(SCENARIO).model_dump()
    data["environment"]["uuvs"][0]["deployment_state"] = "deployed"
    profile = data["platforms"]["motion_profiles"]["uuv_standard"]
    profile["transit_energy_per_m"] = 0.001
    profile["hotel_energy_per_s"] = 0.002
    config = AppConfig.model_validate(data)
    engine = SimulationEngine(config, seed=42, output_dir=tmp_path)
    uuv = engine._uuvs["uuv_00"]
    uuv.set_waypoints([(uuv.position_xy[0] + 1_000.0, uuv.position_xy[1])])
    before_position = uuv.position_xy
    before_energy = uuv.energy_fraction

    engine.step()

    displacement_m = hypot(
        uuv.position_xy[0] - before_position[0],
        uuv.position_xy[1] - before_position[1],
    )
    assert uuv.energy_fraction == pytest.approx(
        before_energy
        - displacement_m * profile["transit_energy_per_m"]
        - config.timing.physics_step_s * profile["hotel_energy_per_s"]
    )


def test_explicit_onboard_uuvs_match_carrier_in_frame_and_snapshot(tmp_path: Path) -> None:
    engine = SimulationEngine(load_app_config(SCENARIO), seed=42, output_dir=tmp_path)
    engine._uuvs["uuv_00"].heading_rad = 1.2
    engine._uuvs["uuv_00"].speed_mps = 3.0

    frame = engine.step()
    snapshot = engine.platform_snapshot()
    frame_uuv = next(state for state in frame["uuvs"] if state["platform_id"] == "uuv_00")
    snapshot_uuv = next(state for state in snapshot.roster.uuvs if state.platform_id == "uuv_00")

    for state in (frame_uuv, snapshot_uuv.model_dump()):
        assert state["deployment_state"] == "onboard"
        assert state["position_xy"] == snapshot.carrier.position_xy
        assert state["heading_rad"] == snapshot.carrier.heading_rad
        assert state["speed_mps"] == 0.0


def test_explicit_recovered_uuv_matches_carrier_heading_and_speed(tmp_path: Path) -> None:
    engine = SimulationEngine(load_app_config(SCENARIO), seed=42, output_dir=tmp_path)
    uuv_id = "uuv_00"
    engine.request_uuv_deployment(uuv_id)
    engine._uuvs[uuv_id].heading_rad = 1.2
    engine._uuvs[uuv_id].speed_mps = 3.0
    engine.request_uuv_recovery(uuv_id)

    frame = engine.step()
    snapshot = engine.platform_snapshot()
    frame_uuv = next(state for state in frame["uuvs"] if state["platform_id"] == uuv_id)
    snapshot_uuv = next(state for state in snapshot.roster.uuvs if state.platform_id == uuv_id)

    for state in (frame_uuv, snapshot_uuv.model_dump()):
        assert state["deployment_state"] == "onboard"
        assert state["position_xy"] == snapshot.carrier.position_xy
        assert state["heading_rad"] == snapshot.carrier.heading_rad
        assert state["speed_mps"] == 0.0


def test_explicit_step_restores_runtime_and_log_after_sink_failure(tmp_path: Path) -> None:
    base = load_app_config(SCENARIO)
    config = base.model_copy(
        update={"timing": base.timing.model_copy(update={"physics_step_s": 30})}
    )
    should_fail = True

    def fail_once(_: dict[str, object]) -> None:
        nonlocal should_fail
        if should_fail:
            should_fail = False
            raise RuntimeError("sink failed")

    engine = SimulationEngine(
        config,
        seed=42,
        output_dir=tmp_path / "failing",
        evaluation_sink=fail_once,
    )
    reference = SimulationEngine(config, seed=42, output_dir=tmp_path / "reference")
    before_carrier_entity = engine._carrier_entity
    before_usvs = engine._usvs
    before_usv = engine._usvs["usv_00"]
    before_uuvs = engine._uuvs
    before_uuv = engine._uuvs["uuv_00"]
    before_targets = engine._targets
    before_snapshot = engine.platform_snapshot()
    before_target = engine._targets["target_00"]
    before_target_state = (
        before_target.position_xy,
        before_target.velocity_xy,
        before_target.intent,
        before_target._desired_heading_rad,
        before_target._desired_speed_mps,
    )
    before_master_rng = engine._master_rng.getstate()
    before_entity_rngs = {key: rng.getstate() for key, rng in engine._entity_rngs.items()}
    before_observer_rngs = {key: rng.getstate() for key, rng in engine._observer_rngs.items()}
    before_quality_rngs = {key: rng.getstate() for key, rng in engine._quality_rngs.items()}
    before_log = engine.logger.path.read_bytes()

    with pytest.raises(RuntimeError, match="sink failed"):
        engine.step()

    target = engine._targets["target_00"]
    assert engine._carrier_entity is before_carrier_entity
    assert engine._usvs is before_usvs
    assert engine._usvs["usv_00"] is before_usv
    assert engine._uuvs is before_uuvs
    assert engine._uuvs["uuv_00"] is before_uuv
    assert engine._targets is before_targets
    assert target is before_target
    assert engine.platform_snapshot() == before_snapshot
    assert (
        target.position_xy,
        target.velocity_xy,
        target.intent,
        target._desired_heading_rad,
        target._desired_speed_mps,
    ) == before_target_state
    assert engine._master_rng.getstate() == before_master_rng
    assert {key: rng.getstate() for key, rng in engine._entity_rngs.items()} == before_entity_rngs
    assert {key: rng.getstate() for key, rng in engine._observer_rngs.items()} == before_observer_rngs
    assert {key: rng.getstate() for key, rng in engine._quality_rngs.items()} == before_quality_rngs
    assert engine._clock.sim_time_s == 0
    assert engine._step_index == 0
    assert engine.platform_snapshot().communication_links == before_snapshot.communication_links
    assert engine.logger.count == 0
    assert engine.logger.path.read_bytes() == before_log

    recovered_frame = engine.step()
    reference_frame = reference.step()
    assert {key: value for key, value in recovered_frame.items() if key != "run_id"} == {
        key: value for key, value in reference_frame.items() if key != "run_id"
    }


def test_explicit_rollback_reinserts_removed_runtime_children_and_preserves_aliases(
    tmp_path: Path,
) -> None:
    engine: SimulationEngine
    target_id = "target_00"
    uuv_id = "uuv_00"

    def mutate_then_fail(_: dict[str, object]) -> None:
        removed_target = engine._targets.pop(target_id)
        engine._targets["sink-added-target"] = removed_target
        engine._uuv_platform_capabilities.pop(uuv_id)
        engine._uuv_motion_limits.pop(uuv_id)
        removed_nested = engine._contact_state.pop("rollback-nested")
        nested_list = cast(list[str], removed_nested["values"])
        nested_list.append("mutated")
        removed_nested["sink-added"] = True
        engine._contact_state["sink-added-nested"] = {"values": ["added"]}
        raise RuntimeError("sink mutated runtime")

    engine = SimulationEngine(
        load_app_config(SCENARIO),
        seed=42,
        output_dir=tmp_path,
        evaluation_sink=mutate_then_fail,
    )
    before_target = engine._targets[target_id]
    before_target_state = (
        before_target.position_xy,
        before_target.velocity_xy,
        before_target.intent,
    )
    before_capability = engine._uuv_platform_capabilities[uuv_id]
    before_motion = engine._uuv_motion_limits[uuv_id]
    assert before_capability.motion is before_motion
    before_nested: dict[str, object] = {"values": ["original"]}
    before_nested_list = cast(list[str], before_nested["values"])
    engine._contact_state["rollback-nested"] = before_nested

    with pytest.raises(RuntimeError, match="sink mutated runtime"):
        engine.step()

    assert engine._targets[target_id] is before_target
    assert (
        before_target.position_xy,
        before_target.velocity_xy,
        before_target.intent,
    ) == before_target_state
    assert "sink-added-target" not in engine._targets
    assert engine._uuv_platform_capabilities[uuv_id] is before_capability
    assert engine._uuv_motion_limits[uuv_id] is before_motion
    assert engine._uuv_platform_capabilities[uuv_id].motion is before_motion
    assert engine._contact_state["rollback-nested"] is before_nested
    assert engine._contact_state["rollback-nested"]["values"] is before_nested_list
    assert before_nested == {"values": ["original"]}
    assert "sink-added-nested" not in engine._contact_state


def test_explicit_rollback_restores_custom_deepcopy_payload_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine: SimulationEngine

    def mutate_then_fail(_: dict[str, object]) -> None:
        holder.payload.append("failed-tick")
        raise RuntimeError("sink mutated custom deepcopy payload")

    engine = SimulationEngine(
        load_app_config(SCENARIO),
        seed=42,
        output_dir=tmp_path,
        evaluation_sink=mutate_then_fail,
    )
    payload = ["checkpoint"]
    holder = _ManualPayloadCopy(payload)
    engine._contact_state["rollback-custom-copy"] = {"holder": holder, "payload": payload}
    checkpoints = _capture_explicit_runtime_checkpoint(engine, monkeypatch)

    with pytest.raises(RuntimeError, match="sink mutated custom deepcopy payload"):
        engine.step()

    assert len(checkpoints) == 1
    checkpoint_payload = cast(
        _ManualPayloadCopy,
        checkpoints[0].runtime.snapshot["_contact_state"]["rollback-custom-copy"]["holder"],
    ).payload
    restored = engine._contact_state["rollback-custom-copy"]
    restored_holder = cast(_ManualPayloadCopy, restored["holder"])
    assert checkpoint_payload is not payload
    assert restored_holder is holder
    assert restored_holder.payload is payload
    assert restored["payload"] is payload
    assert payload == ["checkpoint"]
    assert restored_holder.payload is not checkpoint_payload
    assert restored["payload"] is not checkpoint_payload


def test_explicit_rollback_restores_nested_deque_children_and_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine: SimulationEngine

    def mutate_then_fail(_: dict[str, object]) -> None:
        queue.popleft()
        first.append("mutated")
        second.append("mutated")
        queue.append(["added"])
        raise RuntimeError("sink mutated deque")

    engine = SimulationEngine(
        load_app_config(SCENARIO),
        seed=42,
        output_dir=tmp_path,
        evaluation_sink=mutate_then_fail,
    )
    first = ["first"]
    second = ["second"]
    queue: deque[list[str]] = deque([first, second])
    engine._contact_state["rollback-deque"] = {"queue": queue}
    checkpoints = _capture_explicit_runtime_checkpoint(engine, monkeypatch)

    with pytest.raises(RuntimeError, match="sink mutated deque"):
        engine.step()

    assert len(checkpoints) == 1
    checkpoint_queue = cast(
        deque[list[str]],
        checkpoints[0].runtime.snapshot["_contact_state"]["rollback-deque"]["queue"],
    )
    assert checkpoint_queue is not queue
    assert queue is engine._contact_state["rollback-deque"]["queue"]
    assert list(queue) == [first, second]
    assert queue[0] is first
    assert queue[1] is second
    assert first == ["first"]
    assert second == ["second"]
    assert queue[0] is not checkpoint_queue[0]
    assert queue[1] is not checkpoint_queue[1]


def test_explicit_rollback_restores_deque_and_bytearray_subclass_attributes(
    tmp_path: Path,
) -> None:
    engine: SimulationEngine

    def mutate_then_fail(_: dict[str, object]) -> None:
        queue.popleft()
        queue.append("added")
        queue_payload.append("mutated")
        del queue.removed
        queue.added = ["added"]
        buffer[:] = b"mutated"
        buffer_payload.append("mutated")
        del buffer.removed
        buffer.added = ["added"]
        raise RuntimeError("sink mutated container subclasses")

    engine = SimulationEngine(
        load_app_config(SCENARIO),
        seed=42,
        output_dir=tmp_path,
        evaluation_sink=mutate_then_fail,
    )
    queue = _AttributedDeque(["first", "second"])
    queue_payload = ["queue"]
    queue.removed = ["queue-removed"]
    queue.payload = queue_payload
    queue_removed = queue.removed
    buffer = _AttributedBytearray(b"checkpoint")
    buffer_payload = ["buffer"]
    buffer.removed = ["buffer-removed"]
    buffer.payload = buffer_payload
    buffer_removed = buffer.removed
    engine._contact_state["container-subclasses"] = {"buffer": buffer, "queue": queue}

    with pytest.raises(RuntimeError, match="sink mutated container subclasses"):
        engine.step()

    restored = engine._contact_state["container-subclasses"]
    assert restored["queue"] is queue
    assert list(queue) == ["first", "second"]
    assert queue.payload is queue_payload
    assert queue_payload == ["queue"]
    assert queue.removed is queue_removed
    assert queue.removed == ["queue-removed"]
    assert not hasattr(queue, "added")
    assert restored["buffer"] is buffer
    assert bytes(buffer) == b"checkpoint"
    assert buffer.payload is buffer_payload
    assert buffer_payload == ["buffer"]
    assert buffer.removed is buffer_removed
    assert buffer.removed == ["buffer-removed"]
    assert not hasattr(buffer, "added")


def test_explicit_rollback_restores_ndarray_metadata_before_later_sections(tmp_path: Path) -> None:
    engine: SimulationEngine

    def mutate_then_fail(_: dict[str, object]) -> None:
        values.dtype = np.dtype(np.float64)  # type: ignore[misc]
        values.shape = (1, 2)
        values[...] = 0
        values.flags.writeable = False
        raise RuntimeError("sink mutated array metadata")

    engine = SimulationEngine(
        load_app_config(SCENARIO),
        seed=42,
        output_dir=tmp_path,
        evaluation_sink=mutate_then_fail,
    )
    values: np.ndarray[Any, Any] = np.array([1, 2, 3, 4], dtype=np.int32)
    before_values = values.copy()
    before_log = engine.logger.path.read_bytes()
    engine._contact_state["array-metadata"] = {"values": values}

    with pytest.raises(RuntimeError, match="sink mutated array metadata"):
        engine.step()

    assert engine._contact_state["array-metadata"]["values"] is values
    assert values.dtype == before_values.dtype
    assert values.shape == before_values.shape
    assert values.flags.writeable
    assert np.array_equal(values, before_values)
    assert engine._step_index == 0
    assert engine._clock.sim_time_s == 0
    assert engine.logger.count == 0
    assert engine.logger.path.read_bytes() == before_log


def test_explicit_rollback_restores_ndarray_after_dtype_and_shape_shorten(
    tmp_path: Path,
) -> None:
    engine: SimulationEngine

    def mutate_then_fail(_: dict[str, object]) -> None:
        values.dtype = np.dtype(np.uint8)  # type: ignore[misc]
        values.resize((3,), refcheck=False)
        values[...] = 0
        raise RuntimeError("sink shortened array buffer")

    engine = SimulationEngine(
        load_app_config(SCENARIO),
        seed=42,
        output_dir=tmp_path,
        evaluation_sink=mutate_then_fail,
    )
    values: np.ndarray[Any, Any] = np.array([1, 2, 3, 4], dtype=np.int32)
    before_values = values.copy()
    before_flags = (
        values.flags.writeable,
        values.flags.aligned,
        values.flags.c_contiguous,
        values.flags.f_contiguous,
        values.flags.owndata,
        values.flags.writebackifcopy,
    )
    engine._contact_state["shortened-array"] = {"values": values}

    with pytest.raises(RuntimeError, match="sink shortened array buffer"):
        engine.step()

    assert engine._contact_state["shortened-array"]["values"] is values
    assert values.dtype == before_values.dtype
    assert values.shape == before_values.shape
    assert np.array_equal(values, before_values)
    assert (
        values.flags.writeable,
        values.flags.aligned,
        values.flags.c_contiguous,
        values.flags.f_contiguous,
        values.flags.owndata,
        values.flags.writebackifcopy,
    ) == before_flags


def test_explicit_rollback_restores_fortran_ndarray_strides_before_values(tmp_path: Path) -> None:
    engine: SimulationEngine

    def mutate_then_fail(_: dict[str, object]) -> None:
        values.resize((4,), refcheck=False)
        values[...] = 0
        raise RuntimeError("sink resized Fortran array")

    engine = SimulationEngine(
        load_app_config(SCENARIO),
        seed=42,
        output_dir=tmp_path,
        evaluation_sink=mutate_then_fail,
    )
    values = np.asfortranarray(np.array([[1, 2], [3, 4]], dtype=np.int32))
    before_values = values.copy(order="F")
    before_strides = values.strides
    before_flags = (
        values.flags.writeable,
        values.flags.aligned,
        values.flags.c_contiguous,
        values.flags.f_contiguous,
        values.flags.owndata,
        values.flags.writebackifcopy,
    )
    engine._contact_state["fortran-array"] = {"values": values}

    with pytest.raises(RuntimeError, match="sink resized Fortran array"):
        engine.step()

    assert engine._contact_state["fortran-array"]["values"] is values
    assert values.dtype == before_values.dtype
    assert values.shape == before_values.shape
    assert values.strides == before_strides
    assert np.array_equal(values, before_values)
    assert (
        values.flags.writeable,
        values.flags.aligned,
        values.flags.c_contiguous,
        values.flags.f_contiguous,
        values.flags.owndata,
        values.flags.writebackifcopy,
    ) == before_flags


def test_explicit_rollback_restores_object_ndarray_aliases(tmp_path: Path) -> None:
    engine: SimulationEngine

    def mutate_then_fail(_: dict[str, object]) -> None:
        shared_payload.append("mutated")
        values[0] = ["replacement"]
        values[1] = "replacement"
        raise RuntimeError("sink replaced object array values")

    engine = SimulationEngine(
        load_app_config(SCENARIO),
        seed=42,
        output_dir=tmp_path,
        evaluation_sink=mutate_then_fail,
    )
    shared_payload = ["checkpoint"]
    values = np.empty(2, dtype=object)
    values[0] = shared_payload
    values[1] = shared_payload
    engine._contact_state["object-array"] = {"values": values, "alias": shared_payload}

    with pytest.raises(RuntimeError, match="sink replaced object array values"):
        engine.step()

    restored = engine._contact_state["object-array"]
    assert restored["values"] is values
    assert values[0] is shared_payload
    assert values[1] is shared_payload
    assert restored["alias"] is shared_payload
    assert shared_payload == ["checkpoint"]


def test_explicit_checkpoint_rejects_non_owning_ndarray_before_tick(tmp_path: Path) -> None:
    sink_calls: list[None] = []

    def record_sink(_: dict[str, object]) -> None:
        sink_calls.append(None)

    engine = SimulationEngine(
        load_app_config(SCENARIO),
        seed=42,
        output_dir=tmp_path,
        evaluation_sink=record_sink,
    )
    values = np.arange(6, dtype=np.int32).reshape((3, 2))[:, 1]
    assert not values.flags.owndata
    engine._contact_state["array-view"] = {"values": values}

    with pytest.raises(RuntimeError, match="non-owning ndarray"):
        engine.step()

    assert sink_calls == []
    assert engine._step_index == 0
    assert engine._clock.sim_time_s == 0
    assert engine.logger.count == 0


def test_explicit_checkpoint_rejects_non_contiguous_owning_ndarray_before_tick(
    tmp_path: Path,
) -> None:
    sink_calls: list[None] = []

    def record_sink(_: dict[str, object]) -> None:
        sink_calls.append(None)

    engine = SimulationEngine(
        load_app_config(SCENARIO),
        seed=42,
        output_dir=tmp_path,
        evaluation_sink=record_sink,
    )
    values = np.arange(9, dtype=np.int32).reshape((3, 3)).copy()
    values.strides = (8, 4)
    assert values.flags.owndata
    assert not values.flags.c_contiguous
    assert not values.flags.f_contiguous
    engine._contact_state["non-contiguous-owning-array"] = {"values": values}

    with pytest.raises(RuntimeError, match="non-C/non-F owning ndarray"):
        engine.step()

    assert sink_calls == []
    assert engine._step_index == 0
    assert engine._clock.sim_time_s == 0
    assert engine.logger.count == 0


def test_explicit_checkpoint_rejects_singleton_fortran_ndarray_strides_before_tick(
    tmp_path: Path,
) -> None:
    sink_calls: list[None] = []

    def record_sink(_: dict[str, object]) -> None:
        sink_calls.append(None)

    engine = SimulationEngine(
        load_app_config(SCENARIO),
        seed=42,
        output_dir=tmp_path,
        evaluation_sink=record_sink,
    )
    values = np.empty((2, 1), dtype=np.int32, order="F")
    values[...] = [[1], [2]]
    assert values.flags.owndata
    assert values.flags.c_contiguous
    assert values.flags.f_contiguous
    assert values.strides == (4, 8)
    engine._contact_state["singleton-fortran-array"] = {"values": values}

    with pytest.raises(RuntimeError, match="cannot restore owning ndarray strides"):
        engine.step()

    assert sink_calls == []
    assert engine._step_index == 0
    assert engine._clock.sim_time_s == 0
    assert engine.logger.count == 0


def test_explicit_checkpoint_rejects_ndarray_subclass_before_tick(tmp_path: Path) -> None:
    class _RollbackUnsafeArray(np.ndarray[Any, Any]):
        pass

    sink_calls: list[None] = []

    def record_sink(_: dict[str, object]) -> None:
        sink_calls.append(None)

    engine = SimulationEngine(
        load_app_config(SCENARIO),
        seed=42,
        output_dir=tmp_path,
        evaluation_sink=record_sink,
    )
    values = np.ndarray.__new__(_RollbackUnsafeArray, shape=(2,), dtype=np.int32)
    values[...] = [1, 2]
    assert values.flags.owndata
    assert values.flags.c_contiguous
    engine._contact_state["ndarray-subclass"] = {"values": values}
    before_log = engine.logger.path.read_bytes()

    with pytest.raises(RuntimeError, match="ndarray subclasses"):
        engine.step()

    assert sink_calls == []
    assert engine._step_index == 0
    assert engine._clock.sim_time_s == 0
    assert engine.logger.count == 0
    assert engine.logger.path.read_bytes() == before_log


def test_explicit_checkpoint_rejects_self_referential_object_ndarray_before_tick(
    tmp_path: Path,
) -> None:
    engine = SimulationEngine(load_app_config(SCENARIO), seed=42, output_dir=tmp_path)
    values = np.empty(1, dtype=object)
    values[0] = values
    engine._contact_state["self-referential-array"] = {"values": values}

    with pytest.raises(RuntimeError, match="self-referential object ndarray"):
        engine.step()

    assert engine._step_index == 0
    assert engine._clock.sim_time_s == 0
    assert engine.logger.count == 0


def test_explicit_rollback_continues_after_array_restore_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine: SimulationEngine

    def mutate_then_fail(_: dict[str, object]) -> None:
        engine._master_rng.random()
        engine._clock.step_s = 99
        values.flags.writeable = False
        raise RuntimeError("sink triggered array restore failure")

    def fail_array_restore(*_: object) -> None:
        raise RuntimeError("array restore failed")

    engine = SimulationEngine(
        load_app_config(SCENARIO),
        seed=42,
        output_dir=tmp_path,
        evaluation_sink=mutate_then_fail,
    )
    values = np.array([1, 2, 3, 4], dtype=np.int32)
    clock = engine._clock
    clock_state = (clock.step_s, clock.sim_time_s)
    master_rng = engine._master_rng
    master_state = master_rng.getstate()
    logger = engine.logger
    before_log = logger.path.read_bytes()
    engine._contact_state["array-restore-failure"] = {"values": values}
    monkeypatch.setattr(engine_module, "_restore_explicit_array", fail_array_restore)

    with pytest.raises(RuntimeError, match="sink triggered array restore failure") as exc_info:
        engine.step()

    assert isinstance(exc_info.value.__context__, ExceptionGroup)
    assert engine._clock is clock
    assert (clock.step_s, clock.sim_time_s) == clock_state
    assert engine._master_rng is master_rng
    assert master_rng.getstate() == master_state
    assert engine.logger is logger
    assert logger.count == 0
    assert logger.path.read_bytes() == before_log


def test_explicit_rollback_restores_original_clock_and_logger_objects(tmp_path: Path) -> None:
    engine: SimulationEngine

    def replace_then_fail(_: dict[str, object]) -> None:
        engine._clock = SimulationClock(step_s=99, sim_time_s=123)
        engine.logger = FrameLogger(tmp_path / "replacement-log")
        raise RuntimeError("sink replaced clock and logger")

    engine = SimulationEngine(
        load_app_config(SCENARIO),
        seed=42,
        output_dir=tmp_path / "original-log",
        evaluation_sink=replace_then_fail,
    )
    clock = engine._clock
    logger = engine.logger
    before_log = logger.path.read_bytes()

    with pytest.raises(RuntimeError, match="sink replaced clock and logger"):
        engine.step()

    assert engine._clock is clock
    assert engine._clock.sim_time_s == 0
    assert engine.logger is logger
    assert logger.count == 0
    assert logger.path.read_bytes() == before_log


def test_explicit_rollback_restores_mutated_clock_state_in_place(tmp_path: Path) -> None:
    engine: SimulationEngine

    def mutate_then_fail(_: dict[str, object]) -> None:
        engine._clock.step_s = 99
        raise RuntimeError("sink mutated clock step")

    engine = SimulationEngine(
        load_app_config(SCENARIO),
        seed=42,
        output_dir=tmp_path,
        evaluation_sink=mutate_then_fail,
    )
    clock = engine._clock
    before_state = (clock.step_s, clock.sim_time_s)

    with pytest.raises(RuntimeError, match="sink mutated clock step"):
        engine.step()

    assert engine._clock is clock
    assert (clock.step_s, clock.sim_time_s) == before_state


def test_explicit_checkpoint_rejects_nested_random_generator_before_tick(tmp_path: Path) -> None:
    sink_calls: list[None] = []

    def record_sink(_: dict[str, object]) -> None:
        sink_calls.append(None)

    engine = SimulationEngine(
        load_app_config(SCENARIO),
        seed=42,
        output_dir=tmp_path,
        evaluation_sink=record_sink,
    )
    engine._contact_state["nested-rng"] = {"rng": random.Random(7)}

    with pytest.raises(RuntimeError, match=r"random\.Random"):
        engine.step()

    assert sink_calls == []
    assert engine._step_index == 0
    assert engine._clock.sim_time_s == 0
    assert engine.logger.count == 0


def test_explicit_rollback_removes_slot_populated_during_failed_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine: SimulationEngine

    def mutate_then_fail(_: dict[str, object]) -> None:
        slot_state.payload.append("mutated")
        slot_state.failed_tick = ["added"]
        raise RuntimeError("sink populated slot")

    engine = SimulationEngine(
        load_app_config(SCENARIO),
        seed=42,
        output_dir=tmp_path,
        evaluation_sink=mutate_then_fail,
    )
    payload = ["checkpoint"]
    slot_state = _DeferredRollbackSlot(payload)
    engine._contact_state["rollback-slot"] = {"state": slot_state}
    checkpoints = _capture_explicit_runtime_checkpoint(engine, monkeypatch)

    with pytest.raises(RuntimeError, match="sink populated slot"):
        engine.step()

    assert len(checkpoints) == 1
    checkpoint_payload = cast(
        _DeferredRollbackSlot,
        checkpoints[0].runtime.snapshot["_contact_state"]["rollback-slot"]["state"],
    ).payload
    restored = cast(_DeferredRollbackSlot, engine._contact_state["rollback-slot"]["state"])
    assert checkpoint_payload is not payload
    assert restored is slot_state
    assert restored.payload is payload
    assert payload == ["checkpoint"]
    assert restored.payload is not checkpoint_payload
    assert not hasattr(restored, "failed_tick")


def test_explicit_rollback_restores_replaced_uuv_waypoint_list_after_sink_failure(
    tmp_path: Path,
) -> None:
    engine: SimulationEngine
    uuv_id = "uuv_00"

    def replace_waypoints_then_fail(_: dict[str, object]) -> None:
        engine._uuvs[uuv_id].set_waypoints([(900.0, 900.0)])
        engine._uuvs[uuv_id].waypoints.append((901.0, 901.0))
        raise RuntimeError("sink replaced waypoints")

    engine = SimulationEngine(
        load_app_config(SCENARIO),
        seed=42,
        output_dir=tmp_path,
        evaluation_sink=replace_waypoints_then_fail,
    )
    engine.request_uuv_deployment(uuv_id)
    before_uuv = engine._uuvs[uuv_id]
    before_uuv.set_waypoints([(1_000.0, 200.0)])
    before_waypoints = before_uuv.waypoints
    before_waypoint_values = list(before_waypoints)

    with pytest.raises(RuntimeError, match="sink replaced waypoints"):
        engine.step()

    assert engine._uuvs[uuv_id] is before_uuv
    assert engine._uuvs[uuv_id].waypoints is before_waypoints
    assert before_waypoints == before_waypoint_values


def test_explicit_checkpoint_rejects_unsupported_array_before_tick(tmp_path: Path) -> None:
    sink_calls: list[None] = []

    def record_sink(_: dict[str, object]) -> None:
        sink_calls.append(None)

    engine = SimulationEngine(
        load_app_config(SCENARIO),
        seed=42,
        output_dir=tmp_path,
        evaluation_sink=record_sink,
    )
    before_log = engine.logger.path.read_bytes()
    engine._contact_state["unsupported-array"] = {"values": array("i", [1, 2])}

    with pytest.raises(RuntimeError, match=r"unsupported mutable runtime node type array\.array"):
        engine.step()

    assert sink_calls == []
    assert engine._step_index == 0
    assert engine._clock.sim_time_s == 0
    assert engine.logger.count == 0
    assert engine.logger.path.read_bytes() == before_log


def test_explicit_rollback_matches_reverse_deepcopy_dictionary_keys(tmp_path: Path) -> None:
    engine: SimulationEngine
    first = ["first"]
    second = ["second"]
    mapping = _ReverseCopyDict({"first": first, "second": second})

    def mutate_then_fail(_: dict[str, object]) -> None:
        first.append("failed-tick")
        second.append("failed-tick")
        raise RuntimeError("sink mutated reverse-copy mapping")

    engine = SimulationEngine(
        load_app_config(SCENARIO),
        seed=42,
        output_dir=tmp_path,
        evaluation_sink=mutate_then_fail,
    )
    engine._contact_state["reverse-copy"] = {"mapping": mapping}

    with pytest.raises(RuntimeError, match="sink mutated reverse-copy mapping"):
        engine.step()

    restored = engine._contact_state["reverse-copy"]
    assert restored["mapping"] is mapping
    assert mapping["first"] is first
    assert mapping["second"] is second
    assert first == ["first"]
    assert second == ["second"]


def test_explicit_checkpoint_rejects_unmapped_set_member_without_equality(tmp_path: Path) -> None:
    engine = SimulationEngine(load_app_config(SCENARIO), seed=42, output_dir=tmp_path)
    member = _UnsafeSetMember()
    _UnsafeSetMember.comparisons = 0
    engine._contact_state["unsafe-set"] = {"members": _UnmappedSet({member})}

    with pytest.raises(RuntimeError, match="cannot safely associate unordered set member"):
        engine.step()

    assert _UnsafeSetMember.comparisons == 0
    assert engine._step_index == 0
    assert engine._clock.sim_time_s == 0


def test_explicit_rollback_restores_rng_container_and_object_identities(tmp_path: Path) -> None:
    engine: SimulationEngine

    def replace_rngs_then_fail(_: dict[str, object]) -> None:
        master_rng.random()
        entity_primary.random()
        observer_primary.random()
        quality_primary.random()

        entity_rngs["entity-primary"] = random.Random(401)
        entity_rngs.pop("entity-removed")
        entity_rngs["sink-added"] = random.Random(402)
        observer_rngs["observer-primary"] = random.Random(403)
        observer_rngs.pop("observer-removed")
        observer_rngs["sink-added"] = random.Random(404)
        quality_rngs["quality-primary"] = random.Random(405)
        quality_rngs.pop("quality-removed")
        quality_rngs["sink-added"] = random.Random(406)

        engine._master_rng = random.Random(407)
        engine._entity_rngs = {"replacement": random.Random(408)}
        engine._observer_rngs = {"replacement": random.Random(409)}
        engine._quality_rngs = {"replacement": random.Random(410)}
        raise RuntimeError("sink replaced RNGs")

    engine = SimulationEngine(
        load_app_config(SCENARIO),
        seed=42,
        output_dir=tmp_path,
        evaluation_sink=replace_rngs_then_fail,
    )
    master_rng = engine._master_rng
    entity_rngs = engine._entity_rngs
    observer_rngs = engine._observer_rngs
    quality_rngs = engine._quality_rngs
    entity_primary = random.Random(101)
    observer_primary = random.Random(102)
    quality_primary = random.Random(103)
    entity_rngs.update(
        {
            "entity-primary": entity_primary,
            "entity-alias": entity_primary,
            "entity-removed": random.Random(104),
        }
    )
    observer_rngs.update(
        {
            "observer-primary": observer_primary,
            "observer-removed": random.Random(105),
        }
    )
    quality_rngs.update(
        {
            "quality-primary": quality_primary,
            "quality-alias": quality_primary,
            "quality-removed": random.Random(106),
        }
    )
    expected_entity_rngs = dict(entity_rngs)
    expected_observer_rngs = dict(observer_rngs)
    expected_quality_rngs = dict(quality_rngs)
    master_state = master_rng.getstate()
    entity_states = {rng_id: rng.getstate() for rng_id, rng in entity_rngs.items()}
    observer_states = {rng_id: rng.getstate() for rng_id, rng in observer_rngs.items()}
    quality_states = {rng_id: rng.getstate() for rng_id, rng in quality_rngs.items()}

    with pytest.raises(RuntimeError, match="sink replaced RNGs"):
        engine.step()

    assert engine._master_rng is master_rng
    assert engine._master_rng.getstate() == master_state
    assert engine._entity_rngs is entity_rngs
    assert engine._observer_rngs is observer_rngs
    assert engine._quality_rngs is quality_rngs
    assert engine._entity_rngs == expected_entity_rngs
    assert engine._observer_rngs == expected_observer_rngs
    assert engine._quality_rngs == expected_quality_rngs
    assert all(engine._entity_rngs[rng_id] is rng for rng_id, rng in expected_entity_rngs.items())
    assert all(
        engine._observer_rngs[rng_id] is rng for rng_id, rng in expected_observer_rngs.items()
    )
    assert all(
        engine._quality_rngs[rng_id] is rng for rng_id, rng in expected_quality_rngs.items()
    )
    assert {rng_id: rng.getstate() for rng_id, rng in entity_rngs.items()} == entity_states
    assert {rng_id: rng.getstate() for rng_id, rng in observer_rngs.items()} == observer_states
    assert {rng_id: rng.getstate() for rng_id, rng in quality_rngs.items()} == quality_states


def test_explicit_step_keeps_write_error_primary_when_restore_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = load_app_config(SCENARIO)
    config = base.model_copy(
        update={"timing": base.timing.model_copy(update={"physics_step_s": 30})}
    )
    engine = SimulationEngine(config, seed=42, output_dir=tmp_path)

    def fail_write(_: dict[str, object]) -> None:
        raise RuntimeError("write failed")

    def fail_restore(_: object) -> None:
        raise RuntimeError("restore failed")

    monkeypatch.setattr(engine.logger, "write", fail_write)
    monkeypatch.setattr(engine.logger, "restore", fail_restore)

    with pytest.raises(RuntimeError, match="write failed") as exc_info:
        engine.step()

    rollback_error = exc_info.value.__context__
    assert isinstance(rollback_error, ExceptionGroup)
    assert len(rollback_error.exceptions) == 1
    logger_error = rollback_error.exceptions[0]
    assert isinstance(logger_error, RuntimeError)
    assert str(logger_error) == "explicit runtime rollback failed to restore logger position"
    assert isinstance(logger_error.__cause__, RuntimeError)
    assert str(logger_error.__cause__) == "restore failed"


def test_explicit_step_propagates_checkpoint_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    engine = SimulationEngine(load_app_config(SCENARIO), seed=42, output_dir=tmp_path)

    def fail_checkpoint() -> object:
        raise RuntimeError("checkpoint failed")

    monkeypatch.setattr(engine.logger, "checkpoint", fail_checkpoint)

    with pytest.raises(RuntimeError, match="checkpoint failed"):
        engine.step()
    assert engine._clock.sim_time_s == 0
    assert engine._step_index == 0


def test_platform_snapshot_never_contains_target_truth(tmp_path: Path) -> None:
    engine = SimulationEngine(load_app_config(SCENARIO), seed=42, output_dir=tmp_path)

    payload = engine.platform_snapshot().model_dump()
    frame = engine.step()

    snapshot_rendered = repr(payload).lower()
    frame_rendered = repr(frame).lower()
    assert "target_00" not in snapshot_rendered
    assert "truth" not in snapshot_rendered
    assert "true_position" not in snapshot_rendered
    assert "true_position" not in frame_rendered
    assert "target_truth" not in frame_rendered
    assert "ground_truth" not in frame_rendered


def test_usvs_remain_inside_carrier_support_radius_during_smoke_run(
    tmp_path: Path,
) -> None:
    engine = SimulationEngine(load_app_config(SCENARIO), seed=42, output_dir=tmp_path)

    for _ in range(12):
        engine.step()
        snapshot = engine.platform_snapshot()
        assert all(
            usv.distance_to_carrier_m <= snapshot.carrier.support_radius_m
            for usv in snapshot.roster.usvs
        )


def test_onboard_usvs_follow_fast_carrier_without_transit_energy(
    tmp_path: Path,
) -> None:
    config = _small_support_fast_carrier_config()
    engine = SimulationEngine(config, seed=42, output_dir=tmp_path)
    assert config.platforms is not None
    hotel_energy_per_s = config.platforms.motion_profiles["usv_standard"].hotel_energy_per_s
    physics_step_s = config.timing.physics_step_s

    for tick in range(1, 5):
        engine.step()
        snapshot = engine.platform_snapshot()
        for usv in snapshot.roster.usvs:
            assert usv.deployment_state == "onboard"
            assert usv.position_xy == snapshot.carrier.position_xy
            assert usv.heading_rad == snapshot.carrier.heading_rad
            assert usv.speed_mps == 0.0
            assert usv.distance_to_carrier_m == 0.0
            assert usv.energy_fraction == pytest.approx(
                1.0 - tick * physics_step_s * hotel_energy_per_s
            )


def test_deployed_usv_boundary_uses_limited_motion_and_energy(
    tmp_path: Path,
) -> None:
    config = _validated_platform_core_config(
        carrier_updates={
            "speed_mps": 0.5,
            "support_radius_m": 650.0,
            "patrol_route_xy": [
                [-8000.0, -8000.0],
                [-10000.0, -8000.0],
                [-10000.0, -6000.0],
            ],
        },
        motion_updates={"max_speed_mps": 0.5, "max_acceleration_mps2": 0.02},
        deployment_state="deployed",
    )
    engine = SimulationEngine(config, seed=42, output_dir=tmp_path)
    carrier_xy = engine._carrier_entity.position_xy
    engine._usvs["usv_00"].motion = MotionState(
        position_xy=(carrier_xy[0] + 650.0, carrier_xy[1]),
        heading_rad=pi,
        speed_mps=0.3,
    )
    before = engine.platform_snapshot().roster.usvs[0]

    engine.step()

    after = next(
        usv for usv in engine.platform_snapshot().roster.usvs if usv.platform_id == "usv_00"
    )
    actual_displacement_m = hypot(
        after.position_xy[0] - before.position_xy[0],
        after.position_xy[1] - before.position_xy[1],
    )
    actual_heading_rad = atan2(
        after.position_xy[1] - before.position_xy[1],
        after.position_xy[0] - before.position_xy[0],
    )
    assert after.distance_to_carrier_m <= 650.0 + 1e-9
    assert after.distance_to_carrier_m == pytest.approx(650.0, abs=1e-6)
    assert actual_displacement_m <= 0.5 * config.timing.physics_step_s + 1e-9
    assert after.speed_mps == pytest.approx(
        actual_displacement_m / config.timing.physics_step_s
    )
    assert after.speed_mps <= 0.5 + 1e-9
    assert wrap_angle(after.heading_rad - actual_heading_rad) == pytest.approx(0.0)
    assert config.platforms is not None
    motion = config.platforms.motion_profiles["usv_standard"]
    assert (
        abs(after.speed_mps - before.speed_mps)
        <= motion.max_acceleration_mps2 * config.timing.physics_step_s + 1e-9
    )
    assert abs(wrap_angle(after.heading_rad - before.heading_rad)) <= (
        motion.max_turn_rate_rad_s * config.timing.physics_step_s + 1e-9
    )
    assert after.energy_fraction == pytest.approx(
        before.energy_fraction
        - actual_displacement_m * motion.transit_energy_per_m
        - config.timing.physics_step_s * motion.hotel_energy_per_s
    )


def test_deployed_usv_boundary_uses_actual_displacement_heading(tmp_path: Path) -> None:
    config = _validated_platform_core_config(
        carrier_updates={"speed_mps": 0.0, "support_radius_m": 650.0},
        motion_updates={"max_turn_rate_rad_s": 0.1},
        deployment_state="deployed",
    )
    engine = SimulationEngine(config, seed=42, output_dir=tmp_path)
    usv = engine._usvs["usv_00"]
    carrier_xy = engine._carrier_entity.position_xy
    previous_motion = MotionState(
        position_xy=(carrier_xy[0] + 649.0, carrier_xy[1]),
        heading_rad=1.4,
        speed_mps=0.1,
    )
    candidate_motion = advance_motion(
        previous_motion,
        MotionCommand(
            desired_heading_rad=1.0,
            desired_speed_mps=usv.limits.max_speed_mps,
        ),
        usv.limits,
        config.timing.physics_step_s,
    )
    candidate_distance_to_carrier = hypot(
        candidate_motion.position_xy[0] - carrier_xy[0],
        candidate_motion.position_xy[1] - carrier_xy[1],
    )
    assert candidate_distance_to_carrier > engine._carrier_entity.support_radius_m
    usv.motion = candidate_motion

    engine._constrain_usv_to_carrier_support(
        usv,
        previous_motion=previous_motion,
        previous_energy_fraction=1.0,
        dt_s=config.timing.physics_step_s,
    )

    actual_heading_rad = atan2(
        usv.motion.position_xy[1] - previous_motion.position_xy[1],
        usv.motion.position_xy[0] - previous_motion.position_xy[0],
    )
    assert usv.motion.position_xy != candidate_motion.position_xy
    assert abs(wrap_angle(actual_heading_rad - candidate_motion.heading_rad)) > 1e-3
    assert actual_heading_rad == pytest.approx(1.54, abs=0.05)
    assert usv.motion.heading_rad == pytest.approx(actual_heading_rad)
    assert abs(wrap_angle(usv.motion.heading_rad - previous_motion.heading_rad)) <= (
        usv.limits.max_turn_rate_rad_s * config.timing.physics_step_s + 1e-9
    )


def test_deployed_usv_boundary_rejects_actual_heading_beyond_turn_limit(
    tmp_path: Path,
) -> None:
    config = _validated_platform_core_config(
        carrier_updates={"speed_mps": 0.0, "support_radius_m": 650.0},
        motion_updates={"max_turn_rate_rad_s": 0.1},
        deployment_state="deployed",
    )
    engine = SimulationEngine(config, seed=42, output_dir=tmp_path)
    usv = engine._usvs["usv_00"]
    carrier_xy = engine._carrier_entity.position_xy
    previous_motion = MotionState(
        position_xy=(carrier_xy[0] + 649.0, carrier_xy[1]),
        heading_rad=0.0,
        speed_mps=6.5,
    )
    correction_angle_rad = 0.1
    usv.motion = MotionState(
        position_xy=(
            carrier_xy[0] + 660.0 * cos(correction_angle_rad),
            carrier_xy[1] + 660.0 * sin(correction_angle_rad),
        ),
        heading_rad=correction_angle_rad,
        speed_mps=6.5,
    )

    with pytest.raises(RuntimeError, match="turn-rate limit"):
        engine._constrain_usv_to_carrier_support(
            usv,
            previous_motion=previous_motion,
            previous_energy_fraction=1.0,
            dt_s=config.timing.physics_step_s,
        )

    assert usv.motion == previous_motion
    assert usv.energy_fraction == 1.0


def test_deployed_usv_support_constraint_rejects_infeasible_motion(
    tmp_path: Path,
) -> None:
    config = _validated_platform_core_config(
        carrier_updates={"speed_mps": 100.0, "support_radius_m": 650.0},
        deployment_state="deployed",
    )
    engine = SimulationEngine(config, seed=42, output_dir=tmp_path)

    with pytest.raises(
        RuntimeError, match="carrier support constraint is infeasible for USV 'usv_01'"
    ):
        engine.step()


def test_infeasible_usv_tick_restores_platform_core_state(tmp_path: Path) -> None:
    config = _validated_platform_core_config(
        carrier_updates={"speed_mps": 100.0, "support_radius_m": 650.0},
        deployment_state="deployed",
    )
    engine = SimulationEngine(config, seed=42, output_dir=tmp_path)
    before_carrier = (
        engine._carrier_entity.position_xy,
        engine._carrier_entity.heading_rad,
        engine._carrier_entity._next_corner_index,
    )
    before_usvs = {
        usv_id: (usv.motion, usv.energy_fraction, usv.command)
        for usv_id, usv in engine._usvs.items()
    }
    before_sim_time_s = engine._clock.sim_time_s
    before_step_index = engine._step_index
    before_events = list(engine._events)
    before_pending_events = list(engine._pending_runtime_events)

    with pytest.raises(RuntimeError, match="carrier support constraint is infeasible"):
        engine.step()

    assert (
        engine._carrier_entity.position_xy,
        engine._carrier_entity.heading_rad,
        engine._carrier_entity._next_corner_index,
    ) == before_carrier
    assert {
        usv_id: (usv.motion, usv.energy_fraction, usv.command)
        for usv_id, usv in engine._usvs.items()
    } == before_usvs
    assert engine._clock.sim_time_s == before_sim_time_s
    assert engine._step_index == before_step_index
    assert engine._events == before_events
    assert engine._pending_runtime_events == before_pending_events


def test_explicit_snapshot_uses_configured_carrier_heading(tmp_path: Path) -> None:
    config = load_app_config(SCENARIO)
    assert config.environment is not None
    configured_heading = 0.73
    carrier = config.environment.carrier.model_copy(
        update={"heading_rad": configured_heading}
    )
    environment = config.environment.model_copy(update={"carrier": carrier})
    config = config.model_copy(update={"environment": environment})

    engine = SimulationEngine(config, seed=42, output_dir=tmp_path)

    assert engine.platform_snapshot().carrier.heading_rad == pytest.approx(configured_heading)
