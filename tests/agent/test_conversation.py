"""Conversation routing tests for the unified expert assistant."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from underwater_tracking.agent.llm import StructuredLLM
from underwater_tracking.agent.nodes.conversation import (
    ConversationContext,
    process_conversation_message,
)
from underwater_tracking.agent.nodes.questions import QuestionAnswer
from underwater_tracking.domain.agent_models import ExpertDirective
from underwater_tracking.domain.conversation_models import (
    ConversationClassification,
    ConversationMessage,
    ConversationTurnResult,
)
from underwater_tracking.domain.models import SituationSnapshot
from underwater_tracking.persistence.events import EventRepository
from underwater_tracking.persistence.ledger import DecisionLedger


class RecordingLLM(StructuredLLM[Any]):
    def __init__(self, classification: ConversationClassification) -> None:
        self.classification = classification
        self.calls: list[str] = []

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: type[Any],
        *,
        prompt_version: str = "",
    ) -> Any:
        del payload, prompt_version
        self.calls.append(operation)
        if response_model is ConversationClassification:
            return self.classification
        if response_model is QuestionAnswer:
            return QuestionAnswer(answer="当前证据不足以支持进一步判断。")
        raise AssertionError(f"unexpected response model {response_model!r}")


@dataclass
class ConversationRig:
    context: ConversationContext
    llm: RecordingLLM
    events: EventRepository

    def close(self) -> None:
        self.events.close()
        self.context.ledger.close()


def make_rig(
    tmp_path: Path,
    classification: ConversationClassification,
) -> ConversationRig:
    database_path = tmp_path / "conversation.db"
    events = EventRepository(database_path)
    ledger = DecisionLedger(database_path)
    llm = RecordingLLM(classification)
    situation = SituationSnapshot.model_construct(
        scenario_id="S1",
        snapshot_revision=3,
        sim_time_s=900,
        uuvs=(),
        group_reports=(),
        pending_events=(),
    )
    return ConversationRig(
        context=ConversationContext(
            scenario_id="S1",
            situation=situation,
            active_plan=None,
            ledger=ledger,
            events=events,
            llm=llm,
        ),
        llm=llm,
        events=events,
    )


def message(text: str, *, expected_plan_version: int = 0) -> ConversationMessage:
    return ConversationMessage(
        conversation_id="conversation-1",
        message_id="message-1",
        role="expert",
        text=text,
        expected_plan_version=expected_plan_version,
        target_scope=("T1",),
        region_scope=("region_1",),
    )


def feedback_proposal() -> ExpertDirective:
    return ExpertDirective(
        directive_id="classifier-proposal",
        raw_text="",
        target_scope=(),
        directive_type="feedback",
        feedback_region_ids=("region_1",),
        feedback_text="增加 region_1 的接力余量",
        confidence=0.95,
        status="preview",
    )


def classification(
    kind: str,
    *,
    proposal: ExpertDirective | None = None,
    evidence_ids: tuple[str, ...] = (),
) -> ConversationClassification:
    return ConversationClassification(
        classification=kind,
        confidence=0.95,
        target_scope=(),
        region_scope=("region_1",),
        evidence_ids=evidence_ids,
        proposal=proposal,
        expected_plan_version=0,
        clarification_question="请说明需要调整的目标或区域。" if kind == "clarification" else None,
    )


def test_plan_revision_only_returns_a_preview_without_auto_apply(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, classification("plan_revision", proposal=feedback_proposal()))
    try:
        result = process_conversation_message(message("增加 region_1 的接力余量"), rig.context)

        assert result.classification.classification == "plan_revision"
        assert result.proposal is not None
        assert result.proposal.status == "preview"
        assert result.applied is False
        assert rig.events.list_events(scenario_id="S1", event_type="directive_applied") == []
        assert rig.llm.calls == ["conversation_classification"]
    finally:
        rig.close()


def test_evidence_query_is_read_only_and_emits_no_event(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, classification("evidence_query"))
    try:
        result = process_conversation_message(message("为什么保持当前方案？"), rig.context)

        assert result.classification.classification == "evidence_query"
        assert result.proposal is None
        assert len(result.messages) == 2
        assert result.messages[-1].role == "assistant"
        assert rig.events.list_events(scenario_id="S1") == []
        assert rig.llm.calls == ["conversation_classification", "question"]
    finally:
        rig.close()


def test_mixed_returns_independent_preview_and_evidence_without_applying(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, classification("mixed", proposal=feedback_proposal()))
    try:
        result = process_conversation_message(message("为什么保持方案，并增加 region_1 接力？"), rig.context)

        assert result.classification.classification == "mixed"
        assert result.proposal is not None
        assert result.proposal.status == "preview"
        assert [item.role for item in result.messages] == ["expert", "assistant", "assistant"]
        assert rig.events.list_events(scenario_id="S1") == []
        assert rig.llm.calls == ["conversation_classification", "question"]
    finally:
        rig.close()


def test_clarification_returns_one_follow_up_message(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, classification("clarification"))
    try:
        result = process_conversation_message(message("请调整一下"), rig.context)

        assert result.proposal is None
        assert len(result.messages) == 2
        assert result.messages[-1].role == "assistant"
        assert result.messages[-1].text == "请说明需要调整的目标或区域。"
        assert rig.llm.calls == ["conversation_classification"]
    finally:
        rig.close()


def test_conversation_rejects_unknown_evidence_and_stale_plan_version(tmp_path: Path) -> None:
    rig = make_rig(tmp_path, classification("evidence_query", evidence_ids=("ghost",)))
    try:
        with pytest.raises(ValueError, match="unknown evidence"):
            process_conversation_message(message("为什么？"), rig.context)
        with pytest.raises(ValueError, match="plan version"):
            process_conversation_message(message("为什么？", expected_plan_version=1), rig.context)
    finally:
        rig.close()


def test_conversation_http_routes_keep_preview_and_apply_explicit() -> None:
    from fastapi.testclient import TestClient

    from underwater_tracking.api.app import create_app
    from underwater_tracking.api.hub import OperationalHub

    class FakeRuntime:
        def __init__(self) -> None:
            self.received: list[ConversationMessage] = []
            self.applied: list[tuple[str, str, int]] = []
            self.last_result: ConversationTurnResult | None = None

        def active_plan(self) -> None:
            return None

        def ask(self, raw_text: str, counterfactual: dict[str, object] | None = None) -> QuestionAnswer:
            del raw_text, counterfactual
            return QuestionAnswer(answer="只读回答")

        def preview_directive(self, raw_text: str) -> ExpertDirective:
            del raw_text
            return feedback_proposal()

        def apply_directive(self, directive_id: str) -> ExpertDirective:
            del directive_id
            return feedback_proposal().model_copy(update={"status": "applied"})

        def preview_assignment(self, *, uuv_ids: tuple[str, ...], target_id: str) -> ExpertDirective:
            del uuv_ids, target_id
            return feedback_proposal()

        def submit_sensor_mode(self, **kwargs: object) -> None:
            del kwargs

        def conversation_message(self, item: ConversationMessage) -> Any:
            self.received.append(item)
            result = process_conversation_message
            del result
            self.last_result = ConversationTurnResult(
                conversation_id=item.conversation_id,
                turn_id="conversation-1:turn:1",
                classification=classification("plan_revision", proposal=feedback_proposal()),
                messages=(
                    item,
                    item.model_copy(
                        update={
                            "message_id": "assistant-1",
                            "role": "assistant",
                            "text": "已生成方案预览，请确认后应用。",
                            "proposal": feedback_proposal(),
                            "turn_id": "conversation-1:turn:1",
                        }
                    ),
                ),
                proposal=feedback_proposal(),
                expected_plan_version=0,
            )
            return self.last_result

        def apply_conversation(self, conversation_id: str, turn_id: str, expected_plan_version: int) -> Any:
            self.applied.append((conversation_id, turn_id, expected_plan_version))
            assert self.last_result is not None
            return self.last_result.model_copy(update={"applied": True})

    class EmptyReplay:
        def range(self, start_s: float = 0.0, end_s: float | None = None) -> list[Any]:
            del start_s, end_s
            return []

    runtime = FakeRuntime()
    app = create_app(runtime=runtime, replay=EmptyReplay(), hub=OperationalHub())
    client = TestClient(app)

    response = client.post(
        "/api/conversation/messages",
        json={
            "conversation_id": "conversation-1",
            "text": "增加 region_1 的接力余量",
            "expected_plan_version": 0,
            "target_scope": ["T1"],
            "region_scope": ["region_1"],
        },
    )
    assert response.status_code == 200
    assert response.json()["proposal"]["status"] == "preview"
    assert runtime.received[0].role == "expert"

    applied = client.post(
        "/api/conversation/conversation-1/apply",
        json={"turn_id": "conversation-1:turn:1", "expected_plan_version": 0},
    )
    assert applied.status_code == 200
    assert applied.json()["applied"] is True
    assert runtime.applied == [("conversation-1", "conversation-1:turn:1", 0)]
