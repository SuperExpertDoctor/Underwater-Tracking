"""Typed contracts for the unified expert conversation client."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from underwater_tracking.domain.agent_models import ExpertDirective
from underwater_tracking.domain.memory_models import (
    MemoryContext,
    MemoryEvidenceTrace,
    MemoryStreamStatus,
    UserId,
)
from underwater_tracking.domain.execution_models import (
    ExecutionDecisionRecord,
)
from underwater_tracking.domain.models import StrictModel

ConversationKind = Literal["plan_revision", "evidence_query", "mixed", "clarification"]
ConversationRole = Literal["expert", "user", "assistant"]
AssistantMode = Literal["auto", "plan_revision", "evidence_query"]


class ConversationClassification(StrictModel):
    """Structured LLM classification used to route one expert message."""

    classification: ConversationKind
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    target_scope: tuple[str, ...] = ()
    region_scope: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    memory_ids: tuple[str, ...] = ()
    proposal: ExpertDirective | None = None
    expected_plan_version: int | None = Field(default=None, ge=0)
    clarification_question: str | None = None

    @property
    def target_ids(self) -> tuple[str, ...]:
        return self.target_scope

    @property
    def region_ids(self) -> tuple[str, ...]:
        return self.region_scope


class RegionEntryDiagnosis(StrictModel):
    """Public region-entry evidence used by deterministic assistant answers."""

    region_id: str = Field(min_length=1)
    probability: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    confirmations: int = Field(default=0, ge=0)
    required_confirmations: int = Field(default=2, ge=1)
    lifecycle: str = Field(min_length=1)
    coverage_ratio: float | None = Field(
        default=None, ge=0.0, le=1.0, allow_inf_nan=False
    )
    source_backed_ping_count: int = Field(default=0, ge=0)
    blocked_reason: str | None = None


class HandoffDiagnosis(StrictModel):
    """Current owner-to-successor handoff evidence from the runtime."""

    owner_group_id: str = Field(min_length=1)
    successor_group_id: str | None = None
    required_uuv_ids: tuple[str, ...] = ()
    observed_uuv_ids: tuple[str, ...] = ()
    blocked_reason: str | None = None


class OperationalDiagnosis(StrictModel):
    """A bounded, frame-bound diagnosis for the four operational questions."""

    scenario_id: str = Field(min_length=1)
    sim_time_s: int = Field(ge=0)
    frame_id: int = Field(ge=0)
    execution_revision: int | None = Field(default=None, ge=1)
    execution_health_status: str = Field(min_length=1)
    execution_health_reasons: tuple[str, ...] = ()
    planning_data_status: str = Field(min_length=1)
    planning_data_age_s: int | None = Field(default=None, ge=0)
    target_estimate_status: str = Field(min_length=1)
    target_estimate_age_s: int | None = Field(default=None, ge=0)
    active_group_count: int = Field(default=0, ge=0)
    passive_group_count: int = Field(default=0, ge=0)
    tracking_owner_group_id: str | None = None
    region_entry_progress: tuple[RegionEntryDiagnosis, ...] = ()
    handoff_progress: HandoffDiagnosis | None = None


class ConversationAnswer(StrictModel):
    """Read-only answer payload returned by the evidence branch."""

    answer: str
    evidence_ids: tuple[str, ...] = ()
    counterfactual_plan_id: str | None = None
    counterfactual_summary: str | None = None
    memory_ids: tuple[str, ...] = ()
    memory_status: MemoryStreamStatus | None = None
    evidence_trace: tuple[MemoryEvidenceTrace, ...] = ()
    execution_revision: int | None = Field(default=None, ge=1)
    frame_id: int | None = Field(default=None, ge=0)
    unresolved_evidence: tuple[str, ...] = ()
    decision_record: ExecutionDecisionRecord | None = None
    diagnosis: OperationalDiagnosis | None = None


class ConversationProposal(StrictModel):
    """A directive preview that requires explicit operator confirmation."""

    proposal_id: str
    directive: ExpertDirective
    expected_plan_version: int = Field(ge=0)
    summary: str
    diff: dict[str, object] | None = None
    status: Literal["preview", "applied", "rejected", "needs_clarification"] = "preview"


class ConversationMessage(StrictModel):
    """One rendered conversation turn with scope and provenance."""

    message_id: str
    conversation_id: str
    user_id: UserId = "operator"
    assistant_mode: AssistantMode = "auto"
    turn_id: str | None = None
    role: ConversationRole
    text: str
    classification: ConversationKind | None = None
    target_scope: tuple[str, ...] = ()
    region_scope: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    proposal: ConversationProposal | None = None
    expected_plan_version: int | None = Field(default=None, ge=0)
    execution_revision: int | None = Field(default=None, ge=1)
    frame_id: int | None = Field(default=None, ge=0)

    @field_validator("proposal", mode="before")
    @classmethod
    def wrap_legacy_proposal(cls, value: object) -> object:
        if isinstance(value, ExpertDirective):
            return ConversationProposal(
                proposal_id=value.directive_id,
                directive=value,
                expected_plan_version=0,
                summary=value.raw_text,
                status=value.status,
            )
        return value

    @field_validator("user_id")
    @classmethod
    def reject_blank_user_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("user_id must not be blank")
        return value

    @property
    def target_ids(self) -> tuple[str, ...]:
        return self.target_scope

    @property
    def region_ids(self) -> tuple[str, ...]:
        return self.region_scope


class ConversationTurnResult(StrictModel):
    """Complete result for one message and its explicit next action."""

    conversation_id: str
    turn_id: str
    user_id: UserId = "operator"
    assistant_mode: AssistantMode = "auto"
    classification: ConversationClassification
    messages: tuple[ConversationMessage, ...]
    target_scope: tuple[str, ...] = ()
    region_scope: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    proposal: ConversationProposal | None = None
    answer: ConversationAnswer | None = None
    clarification_question: str | None = None
    expected_plan_version: int = Field(ge=0)
    applied: bool = False
    memory_context: MemoryContext | None = None
    memory_stream_cursor: int | None = Field(default=None, ge=0)
    queued_memory_work_id: str | None = Field(default=None, min_length=1, max_length=240)
    execution_revision: int | None = Field(default=None, ge=1)
    frame_id: int | None = Field(default=None, ge=0)
    unresolved_evidence: tuple[str, ...] = ()
    decision_record: ExecutionDecisionRecord | None = None
    diagnosis: OperationalDiagnosis | None = None

    @field_validator("proposal", mode="before")
    @classmethod
    def wrap_legacy_proposal(cls, value: object) -> object:
        if isinstance(value, ExpertDirective):
            return ConversationProposal(
                proposal_id=value.directive_id,
                directive=value,
                expected_plan_version=0,
                summary=value.raw_text,
                status=value.status,
            )
        return value

    @field_validator("user_id")
    @classmethod
    def reject_blank_user_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("user_id must not be blank")
        return value

    @model_validator(mode="after")
    def validate_user_scope(self) -> ConversationTurnResult:
        if any(message.user_id != self.user_id for message in self.messages):
            raise ValueError("messages user_id values must match ConversationTurnResult.user_id")
        if self.memory_context is not None and self.memory_context.user_id != self.user_id:
            raise ValueError("memory_context.user_id must match ConversationTurnResult.user_id")
        if self.answer is not None:
            if (
                self.execution_revision is not None
                and self.answer.execution_revision is not None
                and self.execution_revision != self.answer.execution_revision
            ):
                raise ValueError("answer execution_revision must match conversation turn")
            if (
                self.frame_id is not None
                and self.answer.frame_id is not None
                and self.frame_id != self.answer.frame_id
            ):
                raise ValueError("answer frame_id must match conversation turn")
        return self

    @property
    def target_ids(self) -> tuple[str, ...]:
        return self.target_scope

    @property
    def region_ids(self) -> tuple[str, ...]:
        return self.region_scope
