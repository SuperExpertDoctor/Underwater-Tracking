from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from underwater_tracking.agent.llm import LLMError
from underwater_tracking.cli import (
    _AgentLoop,
    _LIVE_LLM_REQUEST_TIMEOUT_S,
    _build_llm,
    _step_with_llm_retries,
)
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.agent_models import Segment, SegmentPlan, TrackingPlan
from underwater_tracking.domain.adversary_models import AdversaryEscapeDecision
from underwater_tracking.domain.slave_models import SlaveSonarDecision
from underwater_tracking.simulation.engine import SimulationEngine


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/scenario/segmented_single_target.yaml"


class RecordingRoleLLM:
    def __init__(self, *, fail: Exception | None = None) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.fail = fail

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: type[Any],
        *,
        prompt_version: str = "",
    ) -> Any:
        del prompt_version
        self.calls.append((operation, payload))
        if self.fail is not None:
            raise self.fail
        if response_model is SlaveSonarDecision:
            platforms = payload["platform_capabilities"]
            assert isinstance(platforms, list)
            receiver = next(
                item["platform_id"]
                for item in platforms
                if item["available"] and item["passive_capable"]
            )
            assert isinstance(receiver, str)
            belief = payload["belief_derived_quality"]
            assert isinstance(belief, dict)
            rotation = payload["rotation_and_future_segments"]
            assert isinstance(rotation, dict)
            return SlaveSonarDecision(
                mode="passive",
                emitter=None,
                receiver_ids=(receiver,),
                target_id=str(payload["target_id"]),
                group_id=str(payload["group_id"]),
                handoff_segment=str(rotation["current_segment_id"]),
                rationale="Keep passive coverage continuous while the relay is available.",
                confidence=0.8,
                expected_information_gain=0.2,
                energy_cost_fraction=0.0,
                exposure_cost=0.0,
                cooldown_s=0,
            )
        if response_model is AdversaryEscapeDecision:
            belief = payload["belief"]
            limits = payload["kinematic_limits"]
            assert isinstance(belief, dict)
            assert isinstance(limits, dict)
            return AdversaryEscapeDecision(
                target_id=str(payload["target_id"]),
                maneuver="hold_course",
                intent="hold_course",
                waypoint=tuple(belief["estimated_position_xy"]),
                segment="target-owned-current",
                speed=min(1.0, float(limits["max_speed_mps"])),
                heading=float(belief["estimated_heading"]),
                decoy_action="none",
                decoy_count=0,
                confidence=0.8,
                rationale="Hold course while the observed threat picture remains uncertain.",
                communications_discipline="silent",
            )
        raise AssertionError(f"unexpected response model {response_model!r}")


def _loop(tmp_path: Path, clients: dict[str, RecordingRoleLLM]) -> tuple[_AgentLoop, SimulationEngine]:
    config = load_app_config(CONFIG_PATH)
    loop = _AgentLoop(
        config,
        database_path=tmp_path / "agent.db",
        llm=clients,
        run_id="runtime-test",
        steps=3,
        seed=7,
    )
    engine = SimulationEngine(
        config,
        seed=7,
        output_dir=tmp_path / "frames",
        carrier=loop.on_situation,
    )
    loop.attach(engine)
    return loop, engine


def test_explicit_cycle_calls_slave_and_adversary_without_truth_payload(tmp_path: Path) -> None:
    clients = {role: RecordingRoleLLM() for role in ("master", "slave", "adversary")}
    loop, engine = _loop(tmp_path, clients)
    try:
        assert loop.scenario_id == "segmented-single-target"
        frame = engine.step()
        engine.step()
        frame = engine.step()
        assert frame["sim_time_s"] == 30
        assert loop.situation is not None
        assert loop.situation.map_bounds_xy == (-12000.0, 12000.0, -12000.0, 12000.0)
        assert clients["slave"].calls
        assert all(
            operation == "slave_sonar_decision"
            for operation, _ in clients["slave"].calls
        )
        assert [operation for operation, _ in clients["adversary"].calls] == [
            "adversary_escape"
        ]
        adversary_payload = clients["adversary"].calls[0][1]
        assert "position_xy" not in adversary_payload
        assert "target_truth" not in adversary_payload
        for event in frame["events"]:
            if event["event_type"] == "active_ping":
                assert "position_xy" not in event["payload"]
        assert loop.paused is False
        assert engine._adversary_decision_history["target_00"]
    finally:
        loop.close()


def test_stable_observation_does_not_repeat_adversary_llm_before_cooldown(
    tmp_path: Path,
) -> None:
    clients = {role: RecordingRoleLLM() for role in ("master", "slave", "adversary")}
    loop, engine = _loop(tmp_path, clients)
    try:
        for _ in range(6):
            engine.step()

        assert engine._clock.sim_time_s == 60
        assert [operation for operation, _ in clients["adversary"].calls] == [
            "adversary_escape"
        ]
    finally:
        loop.close()


def test_llm_outage_keeps_physics_and_operational_frames_advancing(
    tmp_path: Path,
) -> None:
    failure = LLMError("slave provider unavailable")
    clients = {
        "master": RecordingRoleLLM(),
        "slave": RecordingRoleLLM(fail=failure),
        "adversary": RecordingRoleLLM(),
    }
    loop, engine = _loop(tmp_path, clients)
    try:
        config = load_app_config(CONFIG_PATH)
        for _ in range(6):
            assert _step_with_llm_retries(engine, loop, config) is True

        assert engine._clock.sim_time_s == 60
        assert engine._step_index == 6
        assert loop.paused is False
        assert loop.reconnectable is True
        assert loop.runtime.llm_paused is False
        assert loop.runtime.llm_pause_reason is None
        assert clients["slave"].calls
        assert all(
            operation == "slave_sonar_decision"
            for operation, _ in clients["slave"].calls
        )
        assert clients["adversary"].calls
        assert all(
            operation == "adversary_escape"
            for operation, _ in clients["adversary"].calls
        )
        assert engine._adversary_decision_history["target_00"]
        raw_frames = (tmp_path / "frames" / "frames.jsonl").read_text().splitlines()
        operational_frames = (
            tmp_path / "operational_frames.jsonl"
        ).read_text().splitlines()
        assert len(raw_frames) == 6
        # A bootstrap frame is published before a slow provider call and the
        # completed frame is published after it. Both are truthful states for
        # the same physical tick; replay consumers must use monotonic frame
        # ids and simulation time rather than assume one line per tick.
        assert len(operational_frames) >= len(raw_frames)
        operational_payloads = [json.loads(line) for line in operational_frames]
        sim_times = [frame["sim_time_s"] for frame in operational_payloads]
        assert set(sim_times) == {10, 20, 30, 40, 50, 60}
        assert sim_times[-1] == 60
        assert [frame["frame_id"] for frame in operational_payloads] == sorted(
            frame["frame_id"] for frame in operational_payloads
        )
    finally:
        loop.close()


def test_slave_outage_does_not_short_circuit_adversary_brain(
    tmp_path: Path,
) -> None:
    clients = {
        "master": RecordingRoleLLM(),
        "slave": RecordingRoleLLM(fail=LLMError("slave provider unavailable")),
        "adversary": RecordingRoleLLM(),
    }
    loop, engine = _loop(tmp_path, clients)
    try:
        for _ in range(3):
            engine.step()

        assert [operation for operation, _ in clients["adversary"].calls] == [
            "adversary_escape"
        ]
        assert engine._adversary_decision_history["target_00"]
        assert loop.paused is False
    finally:
        loop.close()


def test_outer_retry_does_not_repeat_the_same_engine_cycle(tmp_path: Path) -> None:
    clients = {role: RecordingRoleLLM() for role in ("master", "slave", "adversary")}
    loop, engine = _loop(tmp_path, clients)
    original_step = engine.step
    attempts = 0

    def fail_once() -> dict[str, object]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise LLMError("temporary provider outage")
        return original_step()

    engine.step = fail_once  # type: ignore[method-assign]
    try:
        assert _step_with_llm_retries(engine, loop, load_app_config(CONFIG_PATH)) is False
        assert attempts == 1
        assert engine._clock.sim_time_s == 0
        assert loop.paused is True
    finally:
        loop.close()


def test_live_llm_clients_use_one_short_transport_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_app_config(CONFIG_PATH)
    assert config.llm is not None
    monkeypatch.setenv(config.llm.api_key_env, "test-live-key")

    clients = _build_llm(config)
    try:
        assert set(clients) == {"master", "slave", "adversary"}
        for client in clients.values():
            assert client._max_attempts == 1
            assert client._client.timeout.read == _LIVE_LLM_REQUEST_TIMEOUT_S
    finally:
        for client in clients.values():
            client.close()


def test_committed_segment_plan_reaches_slave_as_spatial_handoff_context(
    tmp_path: Path,
) -> None:
    config = load_app_config(CONFIG_PATH)
    engine = SimulationEngine(config, seed=7, output_dir=tmp_path / "frames")
    plan = TrackingPlan(
        plan_id="segmented-plan",
        scenario_id=config.scenario.scenario_id,
        revision=1,
        base_snapshot_revision=0,
        member_ids_by_target={},
        segment_plan=SegmentPlan(
            segments=(
                Segment(
                    index=0,
                    start_s=0,
                    end_s=60,
                    group_id="G-target_00:surface-cell",
                    intercept_xy=(-4000.0, -4000.0),
                ),
                Segment(
                    index=1,
                    start_s=60,
                    end_s=120,
                    group_id="G-relay-next",
                    intercept_xy=(-2500.0, -2500.0),
                ),
            )
        ),
    )
    engine.apply_tracking_plan(plan)

    context = engine.build_slave_contexts(engine._build_situation(30))[0]

    assert context.current_segment_id == "plan:0:0-60"
    assert [segment.owner_group_id for segment in context.handoff_segments] == [
        "G-target_00:surface-cell",
        "G-relay-next",
    ]
    assert context.handoff_segments[1].intercept_xy == (-2500.0, -2500.0)
    assert context.handoff_segments[1].start_s == 60
