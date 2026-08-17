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
from typing import Any, cast

from pydantic import ValidationError

from underwater_tracking.agent.llm import (
    LLMCallMetadata,
    LLMContentError,
    StructuredLLM,
)
from underwater_tracking.agent.prompts import (
    STRATEGY_PROMPT_VERSION,
    STRATEGY_SYSTEM_PROMPT,
    SUGGESTIONS_PROMPT_VERSION,
    SUGGESTIONS_SYSTEM_PROMPT,
    canonical_digest,
)
from underwater_tracking.agent.state import CarrierState
from underwater_tracking.domain.agent_models import (
    Concept,
    PlanAdjustmentSuggestion,
    PlanAdjustmentSuggestionSet,
    PredictedTrackRef,
    StrategyProposal,
    StrategySet,
)
from underwater_tracking.domain.models import EventLevel, RuntimeEvent
from underwater_tracking.domain.platforms import (
    PlatformKind,
    PlatformSnapshot,
    USVPlatformState,
    UUVPlatformState,
)
from underwater_tracking.agent.nodes.snapshot import PlanningSnapshot
from underwater_tracking.knowledge.client import KnowledgeProvider, KnowledgeQueryResult

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
        llm: StructuredLLM[Any],
        *,
        model_id: str = "underwater-assistant-model",
        prompt_version: str = STRATEGY_PROMPT_VERSION,
        temperature: float = 0.2,
        allowed_soft_constraints: tuple[str, ...] = ("energy_reserve_0.1",),
        snapshot_provider: Callable[[str], PlanningSnapshot] | None = None,
        knowledge_provider: KnowledgeProvider | None = None,
    ) -> None:
        self._llm = llm
        self._model_id = model_id
        self._prompt_version = prompt_version
        self._temperature = temperature
        self._allowed_soft_constraints = allowed_soft_constraints
        self._snapshot_provider = snapshot_provider
        self._knowledge_provider = knowledge_provider

    def __call__(self, state: CarrierState) -> CarrierState:
        strategic = self._is_strategic(state)
        concepts = _STRATEGIC_CONCEPTS if strategic else _PERIODIC_CONCEPTS
        # Every strategy generation is a plan-adjustment opportunity.  The
        # ontology client is bounded and auditable; if it is configured, its
        # failure remains an LLMError so the carrier pauses instead of making
        # an ungrounded local substitute decision.
        external_knowledge = self._query_knowledge(state)
        proposals: list[StrategyProposal] = []
        provenance: dict[str, LLMCallMetadata] = {}
        for concept in concepts:
            payload = self.build_payload(
                state, concept, external_knowledge=external_knowledge
            )
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
        suggestions: tuple[PlanAdjustmentSuggestion, ...] = ()
        if self._knowledge_provider is not None:
            suggestion_concept: Concept = "balanced" if strategic else "hold_current"
            suggestion_payload = self.build_payload(
                state,
                suggestion_concept,
                external_knowledge=external_knowledge,
            )
            suggestion_payload.update(
                {
                    "candidate_strategies": [
                        proposal.model_dump(mode="json") for proposal in proposals
                    ],
                    "suggestion_categories": [
                        "tracking_quality",
                        "segmented_handoff",
                        "resource_rotation",
                        "commander_preference",
                    ],
                    "required_suggestion_count": 4,
                    "system_prompt": SUGGESTIONS_SYSTEM_PROMPT,
                }
            )
            suggestion_set = self._invoke_suggestions(suggestion_payload)
            suggestions = suggestion_set.suggestions
            suggestion_metadata = LLMCallMetadata(
                operation="plan_adjustment_suggestions",
                model=self._model_id,
                prompt_version=SUGGESTIONS_PROMPT_VERSION,
                request_hash=canonical_digest(suggestion_payload),
                response_hash=canonical_digest(suggestion_set.model_dump(mode="json")),
                sim_time_s=self._sim_time(state),
                scenario_id=state.get("scenario_id", ""),
            )
            provenance["plan_adjustment_suggestions"] = suggestion_metadata
        return {
            "strategy_set": StrategySet(
                trigger_event_ids=self._trigger_event_ids(state),
                proposals=tuple(proposals),
            ),
            "llm_provenance": {**state.get("llm_provenance", {}), **provenance},
            "knowledge_query_ids": tuple(item.query_id for item in external_knowledge),
            "plan_adjustment_suggestions": suggestions,
        }

    def build_payload(
        self,
        state: CarrierState,
        concept: Concept,
        *,
        external_knowledge: tuple[KnowledgeQueryResult, ...] = (),
    ) -> dict[str, object]:
        """Curated strategy payload for one requested concept.

        IDs are sorted; only the fields the prompt may use are serialized —
        intent summaries and trigger events, never raw snapshots or hidden
        ground reality.
        """
        evidence_ids = {
            evidence_id
            for hypothesis in state.get("intent_hypotheses", {}).values()
            for evidence_id in hypothesis.evidence_ids
        }
        evidence_ids.update(event.event_id for event in self._events(state))
        evidence_ids.update(item.query_id for item in external_knowledge)
        if not evidence_ids and state.get("snapshot_ref"):
            evidence_ids.add(str(state["snapshot_ref"]))
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
            "evidence_ids": sorted(evidence_ids),
            "external_knowledge": [item.to_prompt_dict() for item in external_knowledge],
        }

    def _query_knowledge(self, state: CarrierState) -> tuple[KnowledgeQueryResult, ...]:
        provider = self._knowledge_provider
        if provider is None:
            return ()
        scenario_id = state.get("scenario_id", "")
        sim_time_s = self._sim_time(state)
        events = self._events(state)
        event_summary = ", ".join(
            f"{event.event_type}:{event.event_id}"
            for event in sorted(events, key=lambda item: item.event_id)[:8]
        ) or "no explicit event ids"
        targets = ", ".join(sorted(state.get("intent_hypotheses", {}))) or "the current target estimate"
        query = (
            "For an underwater multi-UUV and USV relay tracking mission, provide "
            "general expert guidance for the next segmented tracking-plan adjustment. "
            f"Scenario {scenario_id}; simulation time {sim_time_s}s; targets {targets}; "
            f"trigger events {event_summary}. Focus on passive-continuous tracking, "
            "selective active sonar, relay connectivity, handoff timing, energy reserve, "
            "and uncertainty management. Return applicable principles and sources; do "
            "not issue numeric waypoints or replace current estimator evidence."
        )
        return (
            provider.query(
                query_text=query,
                sim_time_s=sim_time_s,
                scenario_id=scenario_id,
            ),
        )

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
                "remaining_range_m": {
                    uuv.uuv_id: round(uuv.remaining_range_m, 1)
                    for uuv in sorted(situation.uuvs, key=lambda item: item.uuv_id)
                },
            },
            "active_plan_version": (
                snapshot.active_plan.revision if snapshot.active_plan is not None else 0
            ),
            "applied_expert_constraints": [
                {
                    "directive_type": directive.directive_type,
                    "target_scope": sorted(directive.target_scope),
                    "disabled_uuv_count": len(directive.disabled_uuv_ids),
                    "return_uuv_ids": sorted(directive.return_uuv_ids),
                    "minimum_quality": dict(sorted(directive.minimum_quality.items())),
                }
                for directive in snapshot.applied_directives
            ],
            "capability_summary": _capability_summary(snapshot),
            "observability_feedback": [
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "level": event.level.value,
                    "sim_time_s": event.sim_time_s,
                    "payload": _bounded_assessment(event.payload),
                }
                for event in situation.pending_events
                if event.event_type.startswith("observability_")
            ][-_MAX_ASSESSMENT_ITEMS:],
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
            return cast(
                StrategyProposal,
                self._llm.invoke_structured(
                    "strategy",
                    payload,
                    StrategyProposal,
                    prompt_version=self._prompt_version,
                ),
            )
        except LLMContentError as exc:
            return cast(
                StrategyProposal,
                self._llm.invoke_structured(
                    "strategy",
                    {**payload, "correction_feedback": _content_error_feedback(exc)},
                    StrategyProposal,
                    prompt_version=self._prompt_version,
                ),
            )

    def _invoke_suggestions(
        self, payload: dict[str, object]
    ) -> PlanAdjustmentSuggestionSet:
        """Request exactly four operator suggestions with one bounded re-ask."""
        try:
            return cast(
                PlanAdjustmentSuggestionSet,
                self._llm.invoke_structured(
                    "plan_adjustment_suggestions",
                    payload,
                    PlanAdjustmentSuggestionSet,
                    prompt_version=SUGGESTIONS_PROMPT_VERSION,
                ),
            )
        except LLMContentError as exc:
            return cast(
                PlanAdjustmentSuggestionSet,
                self._llm.invoke_structured(
                    "plan_adjustment_suggestions",
                    {**payload, "correction_feedback": _content_error_feedback(exc)},
                    PlanAdjustmentSuggestionSet,
                    prompt_version=SUGGESTIONS_PROMPT_VERSION,
                ),
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
    platform_snapshot = snapshot.situation.platform_snapshot
    if platform_snapshot is not None:
        return _platform_core_capability_summary(platform_snapshot)
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


def _platform_core_capability_summary(platform_snapshot: PlatformSnapshot) -> dict[str, object]:
    """Expose platform-core capabilities and connectivity without target data."""
    links = [link.model_dump(mode="json") for link in platform_snapshot.communication_links]
    carrier_id = platform_snapshot.carrier.carrier_id
    carrier_links = {
        link.target_id: link.distance_m
        for link in platform_snapshot.communication_links
        if link.source_id == carrier_id and link.medium == "surface"
    }
    platforms: list[dict[str, object]] = []
    by_kind: dict[str, dict[str, object]] = {}
    for kind, states in (
        (PlatformKind.USV, platform_snapshot.roster.usvs),
        (PlatformKind.UUV, platform_snapshot.roster.uuvs),
    ):
        kind_platforms: list[dict[str, object]] = []
        for state in states:
            sonar = state.capability.sonar
            motion = state.capability.motion
            communications = state.capability.communications
            operational_available = state.deployment_state != "failed" and state.energy_fraction > 0.0
            state_summary: dict[str, object] = {
                "platform_id": state.platform_id,
                "platform_index": state.platform_index,
                "kind": kind.value,
                "passive_range_m": sonar.passive_range_m,
                "active_source_range_m": sonar.active_source_range_m,
                "active_receive_range_m": sonar.active_receive_range_m,
                "passive_available": operational_available,
                "active_available": operational_available and sonar.active_capable,
                "bearing_quality": {
                    "passive_variance_rad2": sonar.passive_bearing_variance_rad2,
                    "active_sigma_rad": sonar.active_bearing_sigma_rad,
                    "active_range_sigma_m": sonar.active_range_sigma_m,
                },
                "speed_mps": state.speed_mps,
                "max_speed_mps": motion.max_speed_mps,
                "max_turn_rate_rad_s": motion.max_turn_rate_rad_s,
                "energy_fraction": state.energy_fraction,
                "endurance_s": (
                    state.energy_fraction / sonar.ping_energy_cost_fraction * sonar.ping_cooldown_s
                    if sonar.ping_energy_cost_fraction > 0.0
                    else None
                ),
                "surface_communication_range_m": communications.surface_range_m,
                "acoustic_communication_range_m": communications.acoustic_range_m,
                "deployment_state": state.deployment_state,
                "sensor_mode": state.sensor_mode,
                "operational_available": operational_available,
            }
            if kind is PlatformKind.USV:
                usv_state = cast(USVPlatformState, state)
                state_summary["distance_to_carrier_m"] = usv_state.distance_to_carrier_m
                state_summary["carrier_connected"] = state.platform_id in carrier_links
            else:
                uuv_state = cast(UUVPlatformState, state)
                state_summary["distance_to_carrier_m"] = carrier_links.get(state.platform_id)
                state_summary["is_group_leader"] = uuv_state.is_group_leader
                state_summary["master_connected"] = uuv_state.master_connected
                state_summary["leader_connectivity"] = {
                    "connected": uuv_state.master_connected,
                    "is_group_leader": uuv_state.is_group_leader,
                }
            kind_platforms.append(state_summary)
            platforms.append(state_summary)
        by_kind[kind.value] = {
            "platforms": kind_platforms,
            "aggregate": _platform_aggregate(kind_platforms),
        }

    return {
        "carrier": {
            "carrier_id": carrier_id,
            "support_radius_m": platform_snapshot.carrier.support_radius_m,
            "surface_connected_platform_count": len(carrier_links),
        },
        "platforms": platforms,
        "by_kind": by_kind,
        "communication_links": links,
    }


def _platform_aggregate(platforms: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate fields that are meaningful across a homogeneous platform kind."""
    def values(field: str) -> list[float]:
        return [
            float(value)
            for platform in platforms
            if isinstance(value := platform.get(field), (int, float))
        ]

    def optional_values(field: str) -> list[float]:
        return [
            float(value)
            for platform in platforms
            if isinstance(value := platform.get(field), (int, float))
        ]

    def nested_values(field: str, nested_field: str) -> list[float]:
        result: list[float] = []
        for platform in platforms:
            nested = platform.get(field)
            if isinstance(nested, Mapping):
                value = nested.get(nested_field)
                if isinstance(value, (int, float)):
                    result.append(float(value))
        return result

    def counts(field: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for platform in platforms:
            value = str(platform[field])
            result[value] = result.get(value, 0) + 1
        return result

    return {
        "count": len(platforms),
        "passive_range_m": _numeric_summary(values("passive_range_m")),
        "active_source_range_m": _numeric_summary(values("active_source_range_m")),
        "active_receive_range_m": _numeric_summary(values("active_receive_range_m")),
        "bearing_quality": {
            "passive_variance_rad2": _numeric_summary(
                nested_values("bearing_quality", "passive_variance_rad2")
            ),
            "active_sigma_rad": _numeric_summary(
                nested_values("bearing_quality", "active_sigma_rad")
            ),
        },
        "speed_mps": _numeric_summary(values("speed_mps")),
        "max_speed_mps": _numeric_summary(values("max_speed_mps")),
        "max_turn_rate_rad_s": _numeric_summary(values("max_turn_rate_rad_s")),
        "energy_fraction": _numeric_summary(values("energy_fraction")),
        "endurance_s": _numeric_summary(optional_values("endurance_s")),
        "surface_communication_range_m": _numeric_summary(
            values("surface_communication_range_m")
        ),
        "acoustic_communication_range_m": _numeric_summary(
            values("acoustic_communication_range_m")
        ),
        "distance_to_carrier_m": _numeric_summary(optional_values("distance_to_carrier_m")),
        "passive_available_count": sum(bool(platform["passive_available"]) for platform in platforms),
        "active_available_count": sum(bool(platform["active_available"]) for platform in platforms),
        "operational_available_count": sum(
            bool(platform["operational_available"]) for platform in platforms
        ),
        "deployment_state_counts": counts("deployment_state"),
        "sensor_mode_counts": counts("sensor_mode"),
        "carrier_connected_count": sum(bool(platform.get("carrier_connected", False)) for platform in platforms),
        "master_connected_count": sum(bool(platform.get("master_connected", False)) for platform in platforms),
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
