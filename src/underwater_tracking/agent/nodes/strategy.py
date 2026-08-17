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

import math
from collections.abc import Callable, Mapping

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
from underwater_tracking.agent.nodes.snapshot import PlanningSnapshot

_STRATEGIC_CONCEPTS: tuple[Concept, ...] = (
    "quality_first",
    "balanced",
    "resource_saving",
)
_PERIODIC_CONCEPTS: tuple[Concept, ...] = ("hold_current",)
_MAX_SCHEME_CONSTRAINTS = 16
_MAX_INTELLIGENCE_REPORTS = 16
_MAX_ASSESSMENT_ITEMS = 8
_MAX_ASSESSMENT_STRING_LENGTH = 160


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
        snapshot_provider: Callable[[str], PlanningSnapshot] | None = None,
    ) -> None:
        self._llm = llm
        self._model_id = model_id
        self._prompt_version = prompt_version
        self._temperature = temperature
        self._allowed_soft_constraints = allowed_soft_constraints
        self._snapshot_provider = snapshot_provider

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
            "decision_factors": self._decision_factors(state),
            "allowed_soft_constraints": sorted(self._allowed_soft_constraints),
            "evidence_ids": sorted(
                {
                    evidence_id
                    for hypothesis in state.get("intent_hypotheses", {}).values()
                    for evidence_id in hypothesis.evidence_ids
                }
            ),
        }

    def _decision_factors(self, state: CarrierState) -> dict[str, object]:
        """Expose bounded estimator/resource factors without raw snapshots.

        Strategy generation may consider resource pressure and observability,
        but it still cannot see hidden reality or emit numeric assignments.
        Direct node tests without a snapshot provider retain an empty context.
        """
        provider = self._snapshot_provider
        snapshot_ref = state.get("snapshot_ref")
        if provider is None or not snapshot_ref:
            return {}
        snapshot = provider(snapshot_ref)
        situation = snapshot.situation
        target_ids = {report.target_id for report in situation.group_reports}
        quality: dict[str, object] = {}
        for report in sorted(situation.group_reports, key=lambda item: item.target_id):
            quality[report.target_id] = {
                "instant": report.quality.instant,
                "window_mean": report.quality.window_mean,
                "ewma": report.quality.ewma,
                "fim_min_eigenvalue": report.belief.fim_min_eigenvalue,
                "fim_condition": (
                    report.belief.fim_condition
                    if math.isfinite(report.belief.fim_condition)
                    else 1.0e6
                ),
                "hard_guard_reasons": sorted(report.quality.hard_guard_reasons),
            }
        scheme = _valid_scheme_summary(snapshot)
        return {
            "target_quality": quality,
            "resource_summary": {
                "fleet_count": len(situation.uuvs),
                "available_count": sum(
                    uuv.status.value not in {"failed", "returning"}
                    for uuv in situation.uuvs
                ),
                "reserved_count": sum(uuv.reserved for uuv in situation.uuvs),
                "low_energy_count": sum(uuv.energy_fraction < 0.2 for uuv in situation.uuvs),
                "mean_energy_fraction": (
                    sum(uuv.energy_fraction for uuv in situation.uuvs) / len(situation.uuvs)
                    if situation.uuvs
                    else 0.0
                ),
            },
            "active_plan_version": (
                snapshot.active_plan.revision if snapshot.active_plan is not None else 0
            ),
            "applied_expert_constraints": [
                {
                    "directive_type": directive.directive_type,
                    "target_scope": sorted(directive.target_scope),
                    "disabled_uuv_count": len(directive.disabled_uuv_ids),
                    "minimum_quality": dict(sorted(directive.minimum_quality.items())),
                }
                for directive in snapshot.applied_directives
            ],
            "capability_summary": _capability_summary(snapshot),
            "operational_scheme": scheme,
            "intelligence_summaries": _intelligence_summaries(snapshot, target_ids),
            "required_quality_constraints": _required_quality_constraints(
                snapshot, target_ids, scheme
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


def _valid_scheme_summary(snapshot: PlanningSnapshot) -> dict[str, object] | None:
    """Return a bounded active scheme summary, never an expired one."""
    scheme = snapshot.situation.operational_scheme
    if scheme is None or not (
        scheme.valid_from_s <= snapshot.sim_time_s < scheme.valid_until_s
    ):
        return None
    target_ids = {report.target_id for report in snapshot.situation.group_reports}
    return {
        "scheme_id": scheme.scheme_id,
        "version": scheme.version,
        "valid_until_s": scheme.valid_until_s,
        "target_priorities": {
            target: scheme.target_priorities[target]
            for target in sorted(target_ids & set(scheme.target_priorities))
        },
        "minimum_quality": {
            target: scheme.minimum_quality[target]
            for target in sorted(target_ids & set(scheme.minimum_quality))
        },
        "constraints": list(sorted(scheme.constraints)[:_MAX_SCHEME_CONSTRAINTS]),
    }


def _intelligence_summaries(
    snapshot: PlanningSnapshot,
    target_ids: set[str],
) -> list[dict[str, object]]:
    """Expose only currently valid, target-scoped, bounded intelligence."""
    now = snapshot.sim_time_s
    reports = sorted(
        (
            report
            for report in snapshot.situation.intelligence_reports
            if report.target_id in target_ids
            and report.issued_at_s <= now < report.valid_until_s
        ),
        key=lambda report: report.report_id,
    )[:_MAX_INTELLIGENCE_REPORTS]
    summaries: list[dict[str, object]] = []
    for report in reports:
        summary: dict[str, object] = {
            "report_id": report.report_id,
            "source": report.source.value,
            "target_id": report.target_id,
            "confidence": report.confidence,
            "valid_until_s": report.valid_until_s,
            "assessment": _bounded_assessment(report.assessment),
        }
        summaries.append(summary)
    return summaries


def _bounded_assessment(value: object, depth: int = 0) -> object:
    """Retain compact operational content while placing a finite payload bound."""
    if depth >= 2:
        return "[truncated]"
    if isinstance(value, Mapping):
        return {
            str(key): _bounded_assessment(child, depth + 1)
            for key, child in sorted(value.items())[:_MAX_ASSESSMENT_ITEMS]
        }
    if isinstance(value, (list, tuple)):
        return [
            _bounded_assessment(child, depth + 1)
            for child in value[:_MAX_ASSESSMENT_ITEMS]
        ]
    if isinstance(value, str):
        return value[:_MAX_ASSESSMENT_STRING_LENGTH]
    return value


def _capability_summary(snapshot: PlanningSnapshot) -> dict[str, object]:
    """Aggregate sensing and maneuver resources without exposing assignments."""
    uuvs = snapshot.situation.uuvs
    return {
        "uuv_count": len(uuvs),
        "passive_range_m": _numeric_summary(
            [uuv.capability.passive_range_m for uuv in uuvs]
        ),
        "active_range_m": _numeric_summary(
            [uuv.capability.active_range_m for uuv in uuvs]
        ),
        "bearing_variance_rad2": _numeric_summary(
            [uuv.capability.bearing_variance_rad2 for uuv in uuvs]
        ),
        "max_speed_mps": _numeric_summary(
            [uuv.capability.max_speed_mps for uuv in uuvs]
        ),
        "max_turn_rate_rad_s": _numeric_summary(
            [uuv.capability.max_turn_rate_rad_s for uuv in uuvs]
        ),
        "passive_only_count": sum(
            not uuv.capability.active_sonar_available for uuv in uuvs
        ),
        "passive_sonar_available_count": sum(
            uuv.capability.passive_sonar_available for uuv in uuvs
        ),
        "active_sonar_available_count": sum(
            uuv.capability.active_sonar_available for uuv in uuvs
        ),
        "endurance_s": _numeric_summary([uuv.capability.endurance_s for uuv in uuvs]),
        "availability": _numeric_summary([uuv.capability.availability for uuv in uuvs]),
    }


def _numeric_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"minimum": 0.0, "maximum": 0.0, "mean": 0.0}
    return {
        "minimum": min(values),
        "maximum": max(values),
        "mean": sum(values) / len(values),
    }


def _required_quality_constraints(
    snapshot: PlanningSnapshot,
    target_ids: set[str],
    scheme: dict[str, object] | None,
) -> dict[str, float]:
    """Aggregate active scheme and applied-directive quality floors by target."""
    minimums = {target: 0.0 for target in target_ids}
    if scheme is not None:
        minimum_quality = scheme.get("minimum_quality")
        if isinstance(minimum_quality, dict):
            for target, quality in minimum_quality.items():
                if isinstance(target, str) and isinstance(quality, (int, float)):
                    minimums[target] = max(minimums.get(target, 0.0), float(quality))
    for directive in snapshot.applied_directives:
        for target, quality in directive.minimum_quality.items():
            if target in minimums:
                minimums[target] = max(minimums[target], quality)
    return {
        target: quality for target, quality in sorted(minimums.items()) if quality > 0.0
    }

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
