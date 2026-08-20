"""Ports used by the HTTP/WebSocket adapter.

The API layer depends on small runtime ports instead of importing the
LangGraph graph.  This keeps request handling non-blocking and lets tests
inject deterministic fakes while the real ``CarrierRuntime`` remains the
owner of graph state and SQLite persistence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

from underwater_tracking.agent.nodes.questions import QuestionAnswer
from underwater_tracking.domain.agent_models import ExpertDirective, TrackingPlan
from underwater_tracking.domain.conversation_models import (
    ConversationMessage,
    ConversationTurnResult,
)
from underwater_tracking.domain.ui_models import OperationalFrame


class RuntimePort(Protocol):
    """The read-only and human-input operations exposed by CarrierRuntime."""

    def active_plan(self) -> TrackingPlan | None:
        """Return the currently broadcast plan, if one exists."""

    def conversation_message(self, message: ConversationMessage) -> ConversationTurnResult:
        """Classify one conversation turn and return a preview/read-only result."""

    def apply_conversation(
        self,
        conversation_id: str,
        turn_id: str,
        expected_plan_version: int,
        *,
        user_id: str = "operator",
    ) -> ConversationTurnResult:
        """Apply a stored preview only when it belongs to the requesting user."""

    def ask(
        self,
        raw_text: str,
        counterfactual: Mapping[str, object] | None = None,
    ) -> QuestionAnswer:
        """Answer one evidence-backed question without changing the plan."""

    def preview_directive(self, raw_text: str) -> ExpertDirective:
        """Parse and validate one directive preview."""

    def apply_directive(self, directive_id: str) -> ExpertDirective:
        """Apply one already-reviewed directive preview."""

    def preview_assignment(
        self, *, uuv_ids: Sequence[str], target_id: str
    ) -> ExpertDirective:
        """Build a typed human-assignment preview without an LLM call."""

    def submit_sensor_mode(
        self,
        *,
        uuv_id: str,
        mode: Literal["passive", "active"],
        target_id: str | None,
        expected_plan_version: int,
    ) -> None:
        """Queue a direct sonar mode control for the next engine boundary."""


class ReplayPort(Protocol):
    """Read-only operational frame replay."""

    def range(
        self,
        start_s: float = 0.0,
        end_s: float | None = None,
        *,
        offset: int = 0,
        limit: int | None = 1000,
    ) -> list[OperationalFrame]:
        """Return validated frames in the requested inclusive time range."""

    def count(self, start_s: float = 0.0, end_s: float | None = None) -> int:
        """Return the number of frames in the requested inclusive time range."""


class DirectiveQueuePort(Protocol):
    """Asynchronous directive preview/apply queue."""

    def submit(
        self,
        *,
        text: str,
        author: str,
        expected_plan_version: int,
        target_ids: Sequence[str],
    ) -> str:
        """Queue a preview and return its request id immediately."""

    def status(self, request_id: str) -> dict[str, object]:
        """Return the current queue state for one request."""

    def apply(self, request_id: str) -> None:
        """Request explicit application of a reviewed preview."""

    def submit_assignment(
        self,
        *,
        uuv_ids: Sequence[str],
        target_id: str,
        expected_plan_version: int,
    ) -> str:
        """Queue a typed assignment preview without blocking the event loop."""


class QuestionPort(Protocol):
    """Question service port, kept separate for read-only test fakes."""

    def ask(
        self,
        raw_text: str,
        counterfactual: Mapping[str, object] | None = None,
    ) -> QuestionAnswer:
        """Return an evidence-backed answer."""
