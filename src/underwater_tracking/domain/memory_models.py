"""Strict contracts for short-term and long-term assistant memory."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from underwater_tracking.domain.models import StrictModel


UserId = Annotated[str, Field(min_length=1, max_length=120)]
_Identifier = Annotated[
    str,
    Field(
        min_length=1,
        max_length=240,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:/-]*$",
    ),
]
_MemorySummary = Annotated[str, Field(min_length=1, max_length=4000)]
_UnitInterval = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
MEMORY_WORK_PAYLOAD_MAX_JSON_BYTES = 8192


def _utc_now() -> datetime:
    return datetime.now(UTC)


class MemoryType(StrEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    DELETED = "deleted"


class MemoryWorkType(StrEnum):
    OBSERVATION = "observation"
    CONVERSATION_TURN = "conversation_turn"
    MAINTENANCE = "maintenance"


class MemoryWorkStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    DEGRADED = "degraded"
    FAILED = "failed"


class MemoryStreamStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    DEGRADED = "degraded"
    FAILED = "failed"


class MemoryStreamEventType(StrEnum):
    WORK_QUEUED = "work_queued"
    WORK_PROCESSING = "work_processing"
    WORK_COMPLETED = "work_completed"
    WORK_DEGRADED = "work_degraded"
    WORK_RETRY_SCHEDULED = "work_retry_scheduled"
    WORKER_RECOVERED = "worker_recovered"
    SOURCE_READ_DEGRADED = "source_read_degraded"
    CONTEXT_LOADED = "context_loaded"
    RETRIEVAL_STARTED = "retrieval_started"
    RETRIEVAL_COMPLETED = "retrieval_completed"
    MEMORY_FILTERED = "memory_filtered"
    MEMORY_EXTRACTED = "memory_extracted"
    MEMORY_VERSION_CREATED = "memory_version_created"
    MEMORY_VERSION_SUPERSEDED = "memory_version_superseded"
    SHORT_TERM_COMPRESSION_STARTED = "short_term_compression_started"
    SHORT_TERM_COMPRESSED = "short_term_compressed"
    COMPRESSION_DEGRADED = "compression_degraded"
    MEMORY_ACCESSED = "memory_accessed"
    MEMORY_ARCHIVED = "memory_archived"
    MEMORY_DELETED = "memory_deleted"
    EVIDENCE_TRACE_STARTED = "evidence_trace_started"
    EVIDENCE_TRACE_COMPLETED = "evidence_trace_completed"


class MemoryStreamReasonCode(StrEnum):
    FILTERED_LOW_IMPORTANCE = "filtered_low_importance"
    FILTERED_TRANSIENT = "filtered_transient"
    EXPLICIT_REMEMBER = "explicit_remember"
    SOURCE_UNAVAILABLE = "source_unavailable"
    EMBEDDING_UNAVAILABLE = "embedding_unavailable"
    LLM_UNAVAILABLE = "llm_unavailable"
    RETRY_SCHEDULED = "retry_scheduled"
    LEASE_EXPIRED = "lease_expired"


class _MemoryModel(StrictModel):
    @model_validator(mode="before")
    @classmethod
    def discard_unsupported_source_fields(cls, value: object) -> object:
        if isinstance(value, dict):
            return {
                key: item
                for key, item in value.items()
                if not (key.startswith("source_") and key not in cls.model_fields)
            }
        return value

    @field_validator("user_id", check_fields=False)
    @classmethod
    def reject_blank_user_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("user_id must not be blank")
        return value


class ShortTermMessage(StrictModel):
    """A bounded original message retained beside a rolling summary."""

    message_id: _Identifier
    scenario_id: _Identifier | None = None
    turn_id: _Identifier | None = None
    role: Literal["expert", "user", "assistant"]
    text: str = Field(min_length=1, max_length=4000)
    source_evidence_ids: tuple[_Identifier, ...] = Field(default=(), max_length=64)
    execution_revision: int | None = Field(default=None, ge=1)
    frame_id: int | None = Field(default=None, ge=0)
    created_at: datetime = Field(default_factory=_utc_now)


class ShortTermContext(_MemoryModel):
    """Per-user, per-conversation short-term context and compression state."""

    user_id: UserId = "operator"
    scenario_id: _Identifier | None = None
    conversation_id: _Identifier
    summary_text: str = Field(default="", max_length=12_000)
    summary_version: int = Field(default=0, ge=0)
    recent_messages: tuple[ShortTermMessage, ...] = Field(default=(), max_length=128)
    message_count: int = Field(default=0, ge=0)
    compressed_message_count: int = Field(default=0, ge=0)
    estimated_tokens: int = Field(default=0, ge=0)
    compression_count: int = Field(default=0, ge=0)
    last_compressed_at: datetime | None = None
    compression_status: MemoryStreamStatus = MemoryStreamStatus.PENDING
    execution_revision: int | None = Field(default=None, ge=1)
    frame_id: int | None = Field(default=None, ge=0)
    updated_at: datetime = Field(default_factory=_utc_now)


class MemoryVersion(_MemoryModel):
    """An immutable version of one logical long-term memory family."""

    memory_id: _Identifier
    memory_family_id: _Identifier
    version: int = Field(ge=1)
    user_id: UserId = "operator"
    scenario_id: _Identifier | None = None
    memory_type: MemoryType
    summary: _MemorySummary
    importance_score: _UnitInterval
    embedding: tuple[float, ...] = Field(default=(), max_length=16_384)
    embedding_version: str = Field(default="v1", min_length=1, max_length=120)
    status: MemoryStatus = MemoryStatus.ACTIVE
    supersedes_memory_id: _Identifier | None = None
    source_message_ids: tuple[_Identifier, ...] = Field(default=(), max_length=128)
    source_event_ids: tuple[_Identifier, ...] = Field(default=(), max_length=128)
    source_decision_ids: tuple[_Identifier, ...] = Field(default=(), max_length=128)
    source_plan_ids: tuple[_Identifier, ...] = Field(default=(), max_length=128)
    change_reason: str = Field(default="created", min_length=1, max_length=500)
    created_at: datetime = Field(default_factory=_utc_now)
    last_accessed_at: datetime | None = None
    access_count: int = Field(default=0, ge=0)
    sim_time_s: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    execution_revision: int | None = Field(default=None, ge=1)
    frame_id: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_version_lineage(self) -> "MemoryVersion":
        if self.version == 1 and self.supersedes_memory_id is not None:
            raise ValueError("version 1 must not define supersedes_memory_id")
        if self.version > 1 and self.supersedes_memory_id is None:
            raise ValueError("version greater than 1 requires supersedes_memory_id")
        return self


class MemoryRetrievalHit(StrictModel):
    """A scored memory version that may be supplied as bounded context."""

    memory: MemoryVersion
    similarity_score: _UnitInterval
    rerank_score: _UnitInterval
    retrieval_reason: str = Field(min_length=1, max_length=500)


class MemoryEvidenceTrace(_MemoryModel):
    """A constrained trail from a question through memory to source IDs."""

    trace_id: _Identifier
    user_id: UserId = "operator"
    status: MemoryStreamStatus
    memory_ids: tuple[_Identifier, ...] = Field(default=(), max_length=64)
    source_message_ids: tuple[_Identifier, ...] = Field(default=(), max_length=64)
    source_event_ids: tuple[_Identifier, ...] = Field(default=(), max_length=64)
    source_decision_ids: tuple[_Identifier, ...] = Field(default=(), max_length=64)
    source_plan_ids: tuple[_Identifier, ...] = Field(default=(), max_length=64)
    created_at: datetime = Field(default_factory=_utc_now)


class MemoryContext(_MemoryModel):
    """The separate short-term, retrieved long-term, and evidence inputs."""

    user_id: UserId = "operator"
    scenario_id: _Identifier | None = None
    short_term_context: ShortTermContext | None = None
    long_term_material: tuple[MemoryRetrievalHit, ...] = Field(default=(), max_length=64)
    retrieved_memory_ids: tuple[_Identifier, ...] = Field(default=(), max_length=64)
    memory_status: MemoryStreamStatus = MemoryStreamStatus.DEGRADED
    degraded_reason: str | None = Field(default=None, max_length=1000)
    evidence_trace: tuple[MemoryEvidenceTrace, ...] = Field(default=(), max_length=64)
    execution_revision: int | None = Field(default=None, ge=1)
    frame_id: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_user_scope(self) -> "MemoryContext":
        if (
            self.short_term_context is not None
            and self.short_term_context.user_id != self.user_id
        ):
            raise ValueError("short_term_context.user_id must match MemoryContext.user_id")
        if any(hit.memory.user_id != self.user_id for hit in self.long_term_material):
            raise ValueError("long_term_material memory users must match MemoryContext.user_id")
        if any(trace.user_id != self.user_id for trace in self.evidence_trace):
            raise ValueError("evidence_trace users must match MemoryContext.user_id")
        if self.short_term_context is not None:
            if (
                self.execution_revision is not None
                and self.short_term_context.execution_revision is not None
                and self.execution_revision != self.short_term_context.execution_revision
            ):
                raise ValueError("short_term_context execution_revision must match MemoryContext")
            if (
                self.frame_id is not None
                and self.short_term_context.frame_id is not None
                and self.frame_id != self.short_term_context.frame_id
            ):
                raise ValueError("short_term_context frame_id must match MemoryContext")
        return self


class MemoryWorkPayload(StrictModel):
    """References queued for processing, never an unrestricted raw request."""

    source_type: str | None = Field(default=None, min_length=1, max_length=120)
    source_text: str | None = Field(default=None, max_length=4000)
    source_payload: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict, max_length=32
    )
    source_message_ids: tuple[_Identifier, ...] = Field(default=(), max_length=64)
    source_event_ids: tuple[_Identifier, ...] = Field(default=(), max_length=64)
    source_decision_ids: tuple[_Identifier, ...] = Field(default=(), max_length=64)
    source_plan_ids: tuple[_Identifier, ...] = Field(default=(), max_length=64)
    source_cursor: int | None = Field(default=None, ge=0)
    execution_revision: int | None = Field(default=None, ge=1)
    frame_id: int | None = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def discard_unsupported_source_fields(cls, value: object) -> object:
        if isinstance(value, dict):
            return {
                key: item
                for key, item in value.items()
                if not (key.startswith("source_") and key not in cls.model_fields)
            }
        return value

    @model_validator(mode="after")
    def _total_json_bytes_are_bounded(self) -> "MemoryWorkPayload":
        encoded = json.dumps(
            self.model_dump(mode="json"), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if len(encoded) > MEMORY_WORK_PAYLOAD_MAX_JSON_BYTES:
            raise ValueError(
                "memory work payload total JSON exceeds "
                f"{MEMORY_WORK_PAYLOAD_MAX_JSON_BYTES} bytes"
            )
        return self


class MemoryWorkItem(_MemoryModel):
    """A durable work contract for later persistence and worker execution."""

    work_id: _Identifier
    user_id: UserId = "operator"
    conversation_id: _Identifier | None = None
    scenario_id: _Identifier | None = None
    work_type: MemoryWorkType
    payload: MemoryWorkPayload = Field(default_factory=MemoryWorkPayload)
    status: MemoryWorkStatus = MemoryWorkStatus.PENDING
    attempts: int = Field(default=0, ge=0)
    available_at: datetime = Field(default_factory=_utc_now)
    created_at: datetime = Field(default_factory=_utc_now)
    completed_at: datetime | None = None
    last_error: str | None = Field(default=None, min_length=1, max_length=1000)


class MemoryStreamPayload(StrictModel):
    """Safe event metadata; raw request text and LLM thoughts have no field."""

    reason_code: MemoryStreamReasonCode | None = None
    hit_count: int | None = Field(default=None, ge=0)
    memory_ids: tuple[_Identifier, ...] = Field(default=(), max_length=64)
    memory_family_id: _Identifier | None = None
    work_id: _Identifier | None = None
    memory_type: MemoryType | None = None
    version: int | None = Field(default=None, ge=1)
    summary_version: int | None = Field(default=None, ge=0)
    source_ids: tuple[_Identifier, ...] = Field(default=(), max_length=64)
    source_message_ids: tuple[_Identifier, ...] = Field(default=(), max_length=64)
    source_event_ids: tuple[_Identifier, ...] = Field(default=(), max_length=64)
    source_decision_ids: tuple[_Identifier, ...] = Field(default=(), max_length=64)
    source_plan_ids: tuple[_Identifier, ...] = Field(default=(), max_length=64)
    plan_version: int | None = Field(default=None, ge=0)
    execution_revision: int | None = Field(default=None, ge=1)
    frame_id: int | None = Field(default=None, ge=0)
    operation: Literal["create", "update", "ignore"] | None = None

    @model_validator(mode="before")
    @classmethod
    def discard_unsupported_source_fields(cls, value: object) -> object:
        if isinstance(value, dict):
            return {
                key: item
                for key, item in value.items()
                if not (key.startswith("source_") and key not in cls.model_fields)
            }
        return value


class MemoryStreamEvent(_MemoryModel):
    """A bounded stream event distinct from any LLM thinking stream."""

    cursor: int = Field(ge=0)
    event_id: _Identifier
    user_id: UserId = "operator"
    scenario_id: _Identifier | None = None
    status: MemoryStreamStatus
    type: MemoryStreamEventType
    payload: MemoryStreamPayload = Field(default_factory=MemoryStreamPayload)
    conversation_id: _Identifier | None = None
    memory_id: _Identifier | None = None
    memory_family_id: _Identifier | None = None
    version: int | None = Field(default=None, ge=1)
    created_at: datetime = Field(default_factory=_utc_now)
    sim_time_s: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    execution_revision: int | None = Field(default=None, ge=1)
    frame_id: int | None = Field(default=None, ge=0)


class MemoryFilterDecision(StrictModel):
    should_store: bool
    explicit_remember: bool = False
    memory_type: MemoryType | None = None
    operation: Literal["create", "update", "ignore"]
    family_key: _Identifier | None = None
    candidate_memory_id: _Identifier | None = None
    importance_score: _UnitInterval = 0.0
    tags: tuple[str, ...] = Field(default=(), max_length=32)
    reason: str = Field(min_length=1, max_length=500)


class MemoryExtractionResult(StrictModel):
    summary: _MemorySummary
    tags: tuple[str, ...] = Field(default=(), max_length=32)
    source_message_ids: tuple[_Identifier, ...] = Field(default=(), max_length=128)
    source_event_ids: tuple[_Identifier, ...] = Field(default=(), max_length=128)
    source_decision_ids: tuple[_Identifier, ...] = Field(default=(), max_length=128)
    source_plan_ids: tuple[_Identifier, ...] = Field(default=(), max_length=128)
    change_reason: str = Field(min_length=1, max_length=500)


class ShortTermCompressionResult(StrictModel):
    summary_text: str = Field(min_length=1, max_length=12_000)
    retained_messages: tuple[ShortTermMessage, ...] = Field(default=(), max_length=128)
    source_message_ids: tuple[_Identifier, ...] = Field(default=(), max_length=128)
