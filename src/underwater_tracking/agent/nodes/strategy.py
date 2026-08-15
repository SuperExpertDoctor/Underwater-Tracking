# src/underwater_tracking/agent/nodes/strategy.py
"""Strategy generation node (spec 15.2, 8.1).

``StrategyGenerationNode`` requests exactly three candidate concepts —
``quality_first``, ``balanced``, ``resource_saving`` — for strategic
events (state route STRATEGIC, spec 8.2), and a single ``hold_current``
or modified proposal for periodic reviews. Each request carries a
curated payload: the requested concept, sorted target intent summaries
and evidence ids, and the trigger events, plus model, temperature, and
the immutable strategy system prompt. All behavior is deterministic in
the state input; the LLM responses are the only external input.

The node attaches per-concept model and prompt versions plus the
canonical request/response hashes to the state's ``llm_provenance``
(spec 16). It never validates or repairs proposals — schema and semantic
verification live in the Verify subgraph (Task 6).
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import ValidationError

from underwater_tracking.agent.llm import (
    LLMCallMetadata,
    LLMContentError,
    StructuredLLM,
)
from underwater_tracking.agent.prompts import (
    STRATEGY_PROMPT_VERSION,
    STRATEGY_SYSTEM_PROMPT,
    canonical_digest,
)
from underwater_tracking.agent.state import CarrierState
from underwater_tracking.domain.agent_models import (
    Concept,
    PredictedTrackRef,
    StrategyProposal,
    StrategySet,
)
from underwater_tracking.domain.models import EventLevel, RuntimeEvent

_STRATEGIC_CONCEPTS: tuple[Concept, ...] = (
    "quality_first",
    "balanced",
    "resource_saving",
)
_PERIODIC_CONCEPTS: tuple[Concept, ...] = ("hold_current",)


class StrategyGenerationNode:
    """Semantic strategy-generation node (LangGraph node: state in, state out).

    ``build_payload`` is the pure payload builder; ``__call__`` requests one
    proposal per concept and assembles the ordered ``StrategySet``.
    """

    def __init__(
        self,
        llm: StructuredLLM[StrategyProposal],
        *,
        model_id: str = "underwater-assistant-model",
        prompt_version: str = STRATEGY_PROMPT_VERSION,
        temperature: float = 0.2,
        allowed_soft_constraints: tuple[str, ...] = ("energy_reserve_0.1",),
    ) -> None:
        self._llm = llm
        self._model_id = model_id
        self._prompt_version = prompt_version
        self._temperature = temperature
        self._allowed_soft_constraints = allowed_soft_constraints

    def __call__(self, state: CarrierState) -> CarrierState:
        strategic = self._is_strategic(state)
        concepts = _STRATEGIC_CONCEPTS if strategic else _PERIODIC_CONCEPTS
        proposals: list[StrategyProposal] = []
        provenance: dict[str, LLMCallMetadata] = {}
        for concept in concepts:
            payload = self.build_payload(state, concept)
            proposal = self._invoke_strategy(concept, payload)
            proposals.append(proposal)
            provenance[f"strategy:{concept}"] = LLMCallMetadata(
                operation="strategy",
                model=self._model_id,
                prompt_version=self._prompt_version,
                request_hash=canonical_digest(payload),
                response_hash=canonical_digest(proposal.model_dump(mode="json")),
                sim_time_s=self._sim_time(state),
                scenario_id=state.get("scenario_id", ""),
            )
        return {
            "strategy_set": StrategySet(
                trigger_event_ids=self._trigger_event_ids(state),
                proposals=tuple(proposals),
            ),
            "llm_provenance": {**state.get("llm_provenance", {}), **provenance},
        }

    def build_payload(self, state: CarrierState, concept: Concept) -> dict[str, object]:
        """Curated strategy payload for one requested concept.

        IDs are sorted; only the fields the prompt may use are serialized —
        intent summaries and trigger events, never raw snapshots or hidden
        ground reality.
        """
        return {
            "model": self._model_id,
            "temperature": self._temperature,
            "system_prompt": STRATEGY_SYSTEM_PROMPT,
            "scenario_id": state.get("scenario_id", ""),
            "sim_time_s": self._sim_time(state),
            "mode": "strategic" if self._is_strategic(state) else "periodic",
            "requested_concept": concept,
            "targets": [
                {
                    "target_id": target_id,
                    "label": hypothesis.label,
                    "confidence": hypothesis.confidence,
                    "evidence_ids": sorted(hypothesis.evidence_ids),
                }
                for target_id, hypothesis in sorted(
                    state.get("intent_hypotheses", {}).items()
                )
            ],
            "trigger_events": [
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "level": event.level.value,
                    "sim_time_s": event.sim_time_s,
                }
                for event in sorted(self._events(state), key=lambda event: event.event_id)
            ],
            "predicted_tracks": _predicted_track_summary(
                state.get("predictions", {})
            ),
            "allowed_soft_constraints": sorted(self._allowed_soft_constraints),
            "evidence_ids": sorted(
                {
                    evidence_id
                    for hypothesis in state.get("intent_hypotheses", {}).values()
                    for evidence_id in hypothesis.evidence_ids
                }
            ),
        }

    def _invoke_strategy(
        self, concept: Concept, payload: dict[str, object]
    ) -> StrategyProposal:
        """One structured strategy call; on schema failure, exactly ONE re-ask.

        The LLM port raises ``LLMContentError`` for schema violations (spec
        8.3 content path). Mirroring the question node's bounded correction,
        the detailed validation errors are appended as
        ``correction_feedback`` and the model answers exactly once more; a
        second content failure is a hard error — never an unbounded loop.
        Transport and config errors are untouched (the port retries those
        internally against its own budget).
        """
        try:
            return self._llm.invoke_structured(
                "strategy",
                payload,
                StrategyProposal,
                prompt_version=self._prompt_version,
            )
        except LLMContentError as exc:
            return self._llm.invoke_structured(
                "strategy",
                {**payload, "correction_feedback": _content_error_feedback(exc)},
                StrategyProposal,
                prompt_version=self._prompt_version,
            )

    def _is_strategic(self, state: CarrierState) -> bool:
        return state.get("route") == EventLevel.STRATEGIC

    def _events(self, state: CarrierState) -> tuple[RuntimeEvent, ...]:
        events = state.get("coalesced_events")
        if not events:
            events = state.get("pending_events")
        return events or ()

    def _trigger_event_ids(self, state: CarrierState) -> tuple[str, ...]:
        return tuple(sorted(event.event_id for event in self._events(state)))

    def _sim_time(self, state: CarrierState) -> int:
        return max((event.sim_time_s for event in self._events(state)), default=0)


def _predicted_track_summary(
    predictions: Mapping[str, PredictedTrackRef],
) -> list[dict[str, object]]:
    """Downsampled predicted-track summary for the strategy payload (R3).

    At most 24 samples per target keep the payload bounded; the corridor
    array is downsampled with the same stride.
    """
    summaries: list[dict[str, object]] = []
    for target_id, prediction in sorted(predictions.items()):
        points = prediction.points_xy
        stride = max(1, (len(points) + 23) // 24)
        summaries.append(
            {
                "target_id": target_id,
                "sim_time_s": prediction.sim_time_s,
                "horizon_s": prediction.horizon_s,
                "sample_step_s": prediction.sample_step_s,
                "points_xy": [list(point) for point in points[::stride]],
                "corridor_radius_m": list(prediction.corridor_radius_m[::stride]),
                "fallback_used": prediction.fallback_used,
            }
        )
    return summaries


def _content_error_feedback(exc: LLMContentError) -> str:
    """The detailed validation errors behind a content failure, for the re-ask.

    The port's own message is generic; the chained pydantic
    ``ValidationError`` carries the per-field problems the model must fix.
    """
    cause = exc.__cause__
    if isinstance(cause, ValidationError):
        return str(cause)
    return str(exc)
