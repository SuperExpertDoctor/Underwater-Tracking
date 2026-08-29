from __future__ import annotations

from pathlib import Path
from threading import Event
import time
from typing import Any

import pytest

from underwater_tracking.agent.llm import LLMConfigError, LLMContentError, LLMError
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


class BlockingFailureRoleLLM(BlockingRoleLLM):
    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: type[Any],
        *,
        prompt_version: str = "",
    ) -> Any:
        del operation, payload, response_model, prompt_version
        self._started.set()
        if not self._release.wait(timeout=5.0):
            raise TimeoutError("blocking test provider was not released")
        raise LLMError("adversary provider unavailable")


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


def test_blocked_background_provider_failure_stops_physics_and_raises(
    tmp_path: Path,
) -> None:
    provider_started = Event()
    release_provider = Event()
    clients = {
        "master": RecordingRoleLLM(),
        "slave": RecordingRoleLLM(),
        "adversary": BlockingFailureRoleLLM(provider_started, release_provider),
    }
    loop, engine = _loop(tmp_path, clients, background_carrier=True)
    try:
        for _ in range(6):
            engine.step()
        assert provider_started.wait(timeout=2.0)

        release_provider.set()
        deadline = time.monotonic() + 5.0
        raised = False
        while time.monotonic() < deadline:
            try:
                engine.step()
            except LLMError as exc:
                assert str(exc) == "adversary provider unavailable"
                raised = True
                break
            time.sleep(0.01)

        assert raised
        failed_step = engine._step_index
        with pytest.raises(LLMError, match="adversary provider unavailable"):
            engine.step()
        assert engine._step_index == failed_step
        assert loop.paused is True
        assert loop.reconnectable is False
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
        assert loop._memory_embedding_provider is not None
        assert loop._memory_worker_embedding_provider is not None
        assert loop._memory_embedding_provider._model_path == loop._memory_worker_embedding_provider._model_path
    finally:
        del engine
        loop.close()
        assert worker is not None
        assert worker.is_running is False


def test_strict_agent_loop_does_not_degrade_when_embedding_snapshot_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UNDERWATER_TRACKING_API_KEY", "test-chat-key")
    config = load_app_config(CONFIG_PATH)
    assert config.memory is not None
    invalid_config = config.model_copy(
        update={
            "memory": config.memory.model_copy(
                update={"embedding_model_path": str(tmp_path / "missing-snapshot")}
            )
        }
    )
    clients = {role: RecordingRoleLLM() for role in ("master", "slave", "adversary")}

    with pytest.raises(LLMConfigError, match="embedding_model_path"):
        _AgentLoop(
            invalid_config,
            database_path=tmp_path / "strict-agent.db",
            llm=clients,
            run_id="strict-invalid-embedding",
            steps=1,
            seed=7,
            llm_execution_required=True,
        )


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


def test_legacy_flat_llm_without_chat_credentials_is_rejected(
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

    with pytest.raises(LLMConfigError, match="role-specific chat configuration"):
        _build_llm(legacy_config)


def test_agent_loop_rejects_legacy_flat_llm_without_chat_credentials(
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

    with pytest.raises(LLMConfigError, match="role-specific chat configuration"):
        _AgentLoop(
            legacy_config,
            database_path=tmp_path / "agent.db",
            llm=None,
            run_id="legacy-flat-no-chat",
            steps=1,
            seed=7,
        )
    return

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


def test_llm_outage_stops_physics_and_raises(
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
        with pytest.raises(LLMError, match="slave provider unavailable"):
            for _ in range(12):
                _step_with_llm_retries(engine, loop, config)

        assert engine._clock.sim_time_s < 60
        assert engine._step_index < 12
        assert loop.paused is True
        assert loop.reconnectable is False
        assert loop.runtime.llm_paused is True
        assert loop.runtime.llm_reconnectable is False
    finally:
        loop.close()


def test_slave_outage_short_circuits_the_entire_algorithm(
    tmp_path: Path,
) -> None:
    clients = {
        "master": RecordingRoleLLM(),
        "slave": RecordingRoleLLM(fail=LLMError("slave provider unavailable")),
        "adversary": RecordingRoleLLM(),
    }
    loop, engine = _loop(tmp_path, clients)
    try:
        with pytest.raises(LLMError, match="slave provider unavailable"):
            for _ in range(6):
                engine.step()

        assert engine._adversary_decision_history.get("target_00", []) == []
        assert loop.paused is True
        assert loop.reconnectable is False
    finally:
        loop.close()


def test_strict_mode_escalates_non_llm_local_brain_errors(
    tmp_path: Path,
) -> None:
    clients = {
        "master": RecordingRoleLLM(),
        "slave": RecordingRoleLLM(fail=ValueError("invalid slave decision")),
        "adversary": RecordingRoleLLM(),
    }
    loop, engine = _loop(tmp_path, clients)
    loop._llm_execution_required = True
    try:
        with pytest.raises(LLMContentError, match="slave LLM decision"):
            for _ in range(6):
                engine.step()

        assert loop.paused is True
        assert loop.reconnectable is False
        assert isinstance(loop._fatal_llm_error, LLMContentError)
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
        with pytest.raises(LLMError, match="temporary provider outage"):
            _step_with_llm_retries(engine, loop, load_app_config(CONFIG_PATH))
        assert attempts == 1
        assert engine._clock.sim_time_s == 0
        assert loop.paused is True
        assert loop.reconnectable is False
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


def test_llm_failure_is_terminal_and_not_reconnectable(
    tmp_path: Path,
) -> None:
    clients = {role: RecordingRoleLLM() for role in ("master", "slave", "adversary")}
    loop, _ = _loop(tmp_path, clients)
    try:
        error = LLMError("provider unavailable")
        with pytest.raises(LLMError, match="provider unavailable"):
            loop.raise_llm_failure(error)
        assert loop._fatal_llm_error is error
        assert loop.reconnectable is False
        assert loop._waiting_for_llm_reconnect() is True
        assert loop.llm_pause_reason == "provider unavailable"
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
