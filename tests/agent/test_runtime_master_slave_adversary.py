from __future__ import annotations

import json
from pathlib import Path
from threading import Event
import time
from typing import Any

import pytest

from underwater_tracking.agent.llm import LLMConfigError, LLMError, UnavailableStructuredLLM
from underwater_tracking.cli import (
    _AgentLoop,
    _build_llm,
    _step_with_llm_retries,
)
from underwater_tracking.config.loader import load_app_config
from underwater_tracking.domain.agent_models import Segment, SegmentPlan, TrackingPlan
from underwater_tracking.domain.adversary_models import (
    AdversaryEscapeDecision,
    AdversaryIntentDecision,
)
from underwater_tracking.domain.slave_models import SlaveSonarDecision
from underwater_tracking.memory.embeddings import SentenceTransformerEmbeddingProvider
from underwater_tracking.memory.retriever import MemoryRetriever
from underwater_tracking.memory.worker import MemoryWorker
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
        if response_model is AdversaryIntentDecision:
            return AdversaryIntentDecision(
                decision_id=f"intent-{payload['target_id']}-{len(self.calls)}",
                target_id=str(payload["target_id"]),
                intent="continue_mission",
                confidence=0.8,
                rationale="Continue the configured mission route while local evidence is stable.",
            )
        raise AssertionError(f"unexpected response model {response_model!r}")


class BlockingRoleLLM(RecordingRoleLLM):
    def __init__(self, started: Event, release: Event) -> None:
        super().__init__()
        self._started = started
        self._release = release

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: type[Any],
        *,
        prompt_version: str = "",
    ) -> Any:
        self._started.set()
        if not self._release.wait(timeout=5.0):
            raise TimeoutError("blocking test provider was not released")
        return super().invoke_structured(
            operation,
            payload,
            response_model,
            prompt_version=prompt_version,
        )


def _loop(
    tmp_path: Path,
    clients: dict[str, RecordingRoleLLM],
    *,
    background_carrier: bool = False,
) -> tuple[_AgentLoop, SimulationEngine]:
    config = load_app_config(CONFIG_PATH)
    loop = _AgentLoop(
        config,
        database_path=tmp_path / "agent.db",
        llm=clients,
        run_id="runtime-test",
        steps=3,
        seed=7,
        background_carrier=background_carrier,
    )
    engine = SimulationEngine(
        config,
        seed=7,
        output_dir=tmp_path / "frames",
        carrier=loop.on_situation,
    )
    loop.attach(engine)
    return loop, engine


def test_blocked_background_provider_does_not_stop_physics(
    tmp_path: Path,
) -> None:
    provider_started = Event()
    release_provider = Event()
    clients = {
        "master": RecordingRoleLLM(),
        "slave": RecordingRoleLLM(),
        "adversary": BlockingRoleLLM(provider_started, release_provider),
    }
    loop, engine = _loop(tmp_path, clients, background_carrier=True)
    try:
        for _ in range(6):
            engine.step()
        assert provider_started.wait(timeout=2.0)

        for _ in range(6):
            engine.step()

        assert engine._clock.sim_time_s == 60
        assert engine._step_index == 12

        release_provider.set()
        deadline = time.monotonic() + 5.0
        while loop._background_thread is not None and time.monotonic() < deadline:
            loop.apply_background_cycle()
            time.sleep(0.01)
        assert loop._background_thread is None
    finally:
        release_provider.set()
        loop.close()


def test_agent_loop_without_memory_credentials_is_explicitly_degraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UNDERWATER_TRACKING_API_KEY", "")
    clients = {role: RecordingRoleLLM() for role in ("master", "slave", "adversary")}
    loop, engine = _loop(tmp_path, clients)
    try:
        assert loop._deps().memory_service is not None
        assert loop._memory_worker is None
        assert loop._memory_degraded_reason is not None
        outcome = loop._memory_service.accept_turn(  # type: ignore[union-attr]
            {
                "user_id": "operator",
                "conversation_id": "conversation-1",
                "scenario_id": loop.scenario_id,
                "message_id": "message-1",
                "text": "persist this",
            },
            {"message_id": "message-2", "role": "assistant", "text": "saved"},
        )
        assert outcome["status"] == "degraded"
        assert outcome["degraded_reason"] == loop._memory_degraded_reason
        assert isinstance(outcome["stream_cursor"], int)
        assert loop._memory_short_term.get_short_term(  # type: ignore[attr-defined]
            "operator", "conversation-1", loop.scenario_id
        ) is not None
    finally:
        del engine
        loop.close()


def test_agent_loop_uses_real_memory_provider_chain_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UNDERWATER_TRACKING_API_KEY", "test-memory-key")
    clients = {role: RecordingRoleLLM() for role in ("master", "slave", "adversary")}
    loop, engine = _loop(tmp_path, clients)
    worker = loop._memory_worker
    try:
        assert isinstance(loop._memory_service._retriever, MemoryRetriever)  # type: ignore[attr-defined]
        assert isinstance(loop._memory_embedding_provider, SentenceTransformerEmbeddingProvider)
        assert isinstance(loop._memory_worker, MemoryWorker)
        assert loop._memory_worker.is_running
        assert loop._memory_degraded_reason is None
        assert worker is not None
        assert worker._embedding_provider is not loop._memory_embedding_provider
        assert isinstance(worker._embedding_provider, SentenceTransformerEmbeddingProvider)
        assert worker._embedding_provider._ledger is loop._memory_worker_ledger
        assert worker._reasoner._llm is not loop.llm
        assert worker._reasoner._llm._ledger is loop._memory_worker_ledger
        assert loop._memory_worker_ledger is not loop.ledger
    finally:
        del engine
        loop.close()
        assert worker is not None
        assert worker.is_running is False


def test_local_memory_provider_does_not_require_embedding_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UNDERWATER_TRACKING_API_KEY", "test-chat-key")
    monkeypatch.delenv("UNUSED_REMOTE_EMBEDDING_KEY", raising=False)
    config = load_app_config(CONFIG_PATH)
    assert config.memory is not None
    config = config.model_copy(
        update={
            "memory": config.memory.model_copy(
                update={"embedding_api_key_env": "UNUSED_REMOTE_EMBEDDING_KEY"}
            )
        }
    )
    clients = {role: RecordingRoleLLM() for role in ("master", "slave", "adversary")}
    loop = _AgentLoop(
        config,
        database_path=tmp_path / "agent.db",
        llm=clients,
        run_id="local-memory-no-embedding-key",
        steps=1,
        seed=7,
    )
    try:
        assert isinstance(loop._memory_embedding_provider, SentenceTransformerEmbeddingProvider)
        assert loop._memory_degraded_reason is None
    finally:
        loop.close()


def test_agent_loop_without_chat_credentials_is_constructible_and_degraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UNDERWATER_TRACKING_API_KEY", "")
    config = load_app_config(CONFIG_PATH)

    loop = _AgentLoop(
        config,
        database_path=tmp_path / "agent.db",
        llm=None,
        run_id="missing-chat-credentials",
        steps=1,
        seed=7,
    )
    try:
        assert loop.paused is True
        assert loop.reconnectable is False
        assert loop.llm_pause_reason is not None
        assert loop._memory_service.degraded_reason is not None
    finally:
        loop.close()


def test_legacy_flat_llm_without_chat_credentials_uses_degraded_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UNDERWATER_TRACKING_API_KEY", "")
    config = load_app_config(CONFIG_PATH)
    assert config.llm is not None
    legacy_config = config.model_copy(
        update={
            "llm": config.llm.model_copy(update={"api_key": None, "roles": None}),
        }
    )

    clients = _build_llm(legacy_config)

    assert set(clients) == {"master", "slave", "adversary"}
    assert all(isinstance(client, UnavailableStructuredLLM) for client in clients.values())
    with pytest.raises(LLMConfigError, match="chat credentials"):
        clients["master"].invoke_structured(
            "strategy", {}, TrackingPlan
        )
    for client in clients.values():
        client.close()


def test_agent_loop_accepts_legacy_flat_llm_without_chat_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UNDERWATER_TRACKING_API_KEY", "")
    config = load_app_config(CONFIG_PATH)
    assert config.llm is not None
    legacy_config = config.model_copy(
        update={
            "llm": config.llm.model_copy(update={"api_key": None, "roles": None}),
        }
    )

    loop = _AgentLoop(
        legacy_config,
        database_path=tmp_path / "agent.db",
        llm=None,
        run_id="legacy-flat-no-chat",
        steps=1,
        seed=7,
    )
    try:
        assert loop.paused is True
        assert loop.reconnectable is False
        assert loop.llm_pause_reason is not None
        assert "chat" in loop.llm_pause_reason
        assert loop._memory_service.degraded_reason is not None
        outcome = loop._memory_service.accept_turn(  # type: ignore[union-attr]
            {
                "user_id": "operator",
                "conversation_id": "legacy-conversation",
                "scenario_id": loop.scenario_id,
                "message_id": "legacy-message-1",
                "text": "保存原始会话，不生成摘要",
            },
            None,
        )
        assert outcome["status"] == "degraded"
        assert isinstance(outcome["work_id"], str)
        assert isinstance(outcome["stream_cursor"], int)
        assert loop._memory_short_term.get_short_term(  # type: ignore[attr-defined]
            "operator", "legacy-conversation", loop.scenario_id
        ) is not None
        assert loop._memory_long_term.get_work(outcome["work_id"]) is not None  # type: ignore[arg-type]
    finally:
        loop.close()


def test_explicit_cycle_calls_slave_and_adversary_without_truth_payload(tmp_path: Path) -> None:
    clients = {role: RecordingRoleLLM() for role in ("master", "slave", "adversary")}
    loop, engine = _loop(tmp_path, clients)
    try:
        assert loop._clock.step_s == loop._config.timing.observation_step_s
        assert loop.scenario_id == "segmented-single-target"
        for _ in range(6):
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
            "adversary_mission_decision"
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
        # Keep the target-side contact stable so this test exercises the
        # cooldown gate rather than the intentional contact-loss trigger.
        engine._targets["target_00"].detection_range_m = 10_000.0
        for _ in range(12):
            engine.step()

        assert engine._clock.sim_time_s == 60
        assert [operation for operation, _ in clients["adversary"].calls] == [
            "adversary_mission_decision"
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
        for _ in range(12):
            assert _step_with_llm_retries(engine, loop, config) is True

        assert engine._clock.sim_time_s == 60
        assert engine._step_index == 12
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
            operation == "adversary_mission_decision"
            for operation, _ in clients["adversary"].calls
        )
        assert engine._adversary_decision_history["target_00"]
        raw_frames = (tmp_path / "frames" / "frames.jsonl").read_text().splitlines()
        operational_frames = (
            tmp_path / "operational_frames.jsonl"
        ).read_text().splitlines()
        assert len(raw_frames) == 12
        # A bootstrap frame is published before a slow provider call and the
        # completed frame is published after it. Both are truthful states for
        # the same physical tick; replay consumers must use monotonic frame
        # ids and simulation time rather than assume one line per tick.
        assert len(operational_frames) >= len(raw_frames)
        operational_payloads = [json.loads(line) for line in operational_frames]
        sim_times = [frame["sim_time_s"] for frame in operational_payloads]
        assert set(sim_times) == {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60}
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
        for _ in range(6):
            engine.step()

        assert [operation for operation, _ in clients["adversary"].calls] == [
            "adversary_mission_decision"
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


def test_live_llm_clients_use_role_transport_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_app_config(CONFIG_PATH)
    assert config.llm is not None
    monkeypatch.setenv(config.llm.api_key_env, "test-live-key")

    clients = _build_llm(config)
    try:
        assert set(clients) == {"master", "slave", "adversary"}
        for role, client in clients.items():
            role_config = config.llm.for_role(role)
            assert client._max_attempts == role_config.max_retries
            assert client._client.timeout.read == role_config.request_timeout_s
    finally:
        for client in clients.values():
            client.close()


def test_llm_reconnect_enters_terminal_state_after_configured_attempts(
    tmp_path: Path,
) -> None:
    clients = {role: RecordingRoleLLM() for role in ("master", "slave", "adversary")}
    loop, _ = _loop(tmp_path, clients)
    try:
        config = load_app_config(CONFIG_PATH)
        max_attempts = min(
            role.max_retries for role in config.llm.roles.values()
        ) + 1
        for _ in range(max_attempts):
            loop.mark_llm_paused(LLMError("provider unavailable"))
            assert loop.reconnectable is True
        loop.mark_llm_paused(LLMError("provider unavailable"))
        assert loop.reconnectable is False
        assert loop._waiting_for_llm_reconnect() is True
        assert "bounded LLM reconnect attempts exhausted" in loop.llm_pause_reason
    finally:
        loop.close()


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
