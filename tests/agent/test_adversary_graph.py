"""Target-local, high-level adversary graph contract tests."""

from __future__ import annotations

import json
from typing import Any

import pytest

from underwater_tracking.agent.graphs.adversary import build_adversary_graph
from underwater_tracking.agent.nodes.adversary import (
    ADVERSARY_PROMPT_VERSION,
    AdversaryDecisionGate,
    build_adversary_payload,
    validate_adversary_decision,
)
from underwater_tracking.domain.adversary_models import (
    AdversaryBelief,
    AdversaryDecisionRecord,
    AdversaryEscapeInput,
    AdversaryIntentDecision,
    AdversaryKinematicLimits,
    AdversaryMissionState,
    AdversaryObservation,
    AdversaryOperatingBoundary,
    AdversaryTrigger,
    CommunicationsAcousticExposure,
    PlatformThreatSummary,
    TargetLocalContact,
)


class RecordingStructuredLLM:
    def __init__(
        self,
        decision: AdversaryIntentDecision | None = None,
        error: Exception | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._decision = decision
        self._error = error

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: type[Any],
        *,
        prompt_version: str = "",
    ) -> Any:
        self.calls.append(
            {
                "operation": operation,
                "payload": payload,
                "response_model": response_model,
                "prompt_version": prompt_version,
            }
        )
        if self._error is not None:
            raise self._error
        if self._decision is None:
            raise AssertionError("test recorder needs a typed decision")
        return self._decision


def make_context() -> AdversaryEscapeInput:
    return AdversaryEscapeInput(
        target_id="SUB-1",
        sim_time_s=600,
        mission_state=AdversaryMissionState(
            target_id="SUB-1",
            task_region_id="mission_east",
            task_region_polygon_xy=((0.0, 0.0), (5000.0, 0.0), (5000.0, 5000.0)),
            mission_route_xy=((1200.0, 800.0), (2500.0, 1500.0), (4500.0, 2000.0)),
            escape_regions={
                "escape_north": ((2000.0, 3500.0), (3000.0, 3500.0), (2500.0, 4500.0)),
                "escape_south": ((2000.0, 100.0), (3000.0, 100.0), (2500.0, 800.0)),
            },
            current_route_index=0,
            local_contact_ids=("UUV-1",),
        ),
        belief=AdversaryBelief(
            target_id="SUB-1",
            as_of_s=600,
            estimated_position_xy=(1200.0, 800.0),
            estimated_velocity_xy=(3.0, 0.5),
            position_uncertainty_m=180.0,
            velocity_uncertainty_mps=0.7,
            estimated_heading=0.1,
            estimated_speed_mps=3.0,
            intent_hypothesis="reposition",
            intent_confidence=0.62,
        ),
        local_contacts=(
            TargetLocalContact(
                platform_id="UUV-1",
                platform_kind="uuv",
                first_seen_s=580,
                last_seen_s=590,
                estimated_range_m=2300.0,
                relative_bearing_rad=1.2,
                threat_level="high",
                status="active",
            ),
        ),
        observations=(
            AdversaryObservation(
                observation_id="OBS-1",
                observed_at_s=590,
                kind="passive_sonar",
                bearing_rad=1.2,
                range_m=2300.0,
                confidence=0.68,
                assessment="platform",
            ),
        ),
        platform_threats=(
            PlatformThreatSummary(
                platform_id="UUV-1",
                platform_kind="uuv",
                observed_at_s=590,
                threat_level="high",
                estimated_range_m=2300.0,
                relative_bearing_rad=1.2,
                passive_detection_risk=0.72,
                active_ping_risk=0.25,
                relay_detection_risk=0.15,
                surface_relay_available=False,
            ),
        ),
        communications_acoustic_exposure=CommunicationsAcousticExposure(
            as_of_s=600,
            passive_signature_level=0.22,
            active_emitter_exposure=0.1,
            communication_intercept_risk=0.38,
            relay_detection_risk=0.64,
            acoustic_clutter_level=0.42,
            last_burst_age_s=85.0,
            own_emission_mode="passive",
        ),
        decision_history=(
            AdversaryDecisionRecord(
                decision_id="DEC-1",
                sim_time_s=540,
                maneuver="course_change",
                intent="reposition",
                segment="deterministic-guidance",
                speed=3.1,
                heading=0.1,
                decoy_action="none",
                decoy_count=0,
                outcome="inconclusive",
            ),
        ),
        trigger_events=(
            AdversaryTrigger(
                trigger_id="target_mission_initialized:SUB-1:0",
                event_type="target_mission_initialized",
                sim_time_s=0,
                severity="strategic",
                summary="target mission state initialized after bootstrap",
            ),
        ),
        kinematic_limits=AdversaryKinematicLimits(
            max_speed_mps=5.0,
            max_turn_rate_rad_s=0.05,
            decision_horizon_s=60.0,
            max_decoy_count=2,
            decoy_inventory=2,
        ),
        operating_boundary=AdversaryOperatingBoundary(
            min_x=0.0,
            max_x=5000.0,
            min_y=0.0,
            max_y=5000.0,
        ),
    )


def make_decision(intent: str = "avoid_contact") -> AdversaryIntentDecision:
    return AdversaryIntentDecision(
        decision_id="DEC-2",
        target_id="SUB-1",
        intent=intent,  # type: ignore[arg-type]
        confidence=0.74,
        rationale="The local contact episode and its threat level favor avoiding contact.",
    )


def test_payload_contains_mission_and_target_local_evidence_only() -> None:
    context = make_context()
    payload = build_adversary_payload(context)
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).casefold()

    assert payload["mission_state"] == context.mission_state.model_dump(mode="json")
    assert payload["own_position_xy"] == context.belief.estimated_position_xy
    assert payload["local_contacts"]
    assert "blue_plan" not in encoded
    assert "uuv_inventory" not in encoded
    assert "target_estimate" not in encoded
    assert "true_position" not in encoded
    assert "depth_change" not in encoded
    assert "usv" not in encoded


def test_graph_calls_high_level_typed_contract() -> None:
    recorder = RecordingStructuredLLM(make_decision())
    graph = build_adversary_graph(recorder)

    result = graph.invoke({"context": make_context()})

    assert result["decision"] == make_decision().model_copy(
        update={"trigger_event_ids": ("target_mission_initialized:SUB-1:0",)}
    )
    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["operation"] == "adversary_mission_decision"
    assert call["response_model"] is AdversaryIntentDecision
    assert call["prompt_version"] == ADVERSARY_PROMPT_VERSION
    assert {"build_payload", "decide", "validate"} <= set(graph.get_graph().nodes)


def test_llm_failure_propagates_without_fabricated_decision() -> None:
    recorder = RecordingStructuredLLM(error=RuntimeError("provider unavailable"))
    graph = build_adversary_graph(recorder)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        graph.invoke({"context": make_context()})
    assert len(recorder.calls) == 1


def test_intent_validation_rejects_unknown_escape_region() -> None:
    context = make_context()
    decision = AdversaryIntentDecision(
        decision_id="DEC-2",
        target_id="SUB-1",
        intent="escape_to_region",
        escape_region_id="unknown",
        confidence=0.74,
        rationale="The local contact episode requires an escape region.",
    )
    with pytest.raises(ValueError, match="configured escape region"):
        validate_adversary_decision(decision, context)


def test_non_escape_intent_rejects_escape_region() -> None:
    with pytest.raises(ValueError, match="escape_region_id"):
        AdversaryIntentDecision(
            decision_id="DEC-3",
            target_id="SUB-1",
            intent="continue_mission",
            escape_region_id="escape_north",
            confidence=0.8,
            rationale="Continue the configured mission route.",
        )


def test_adversary_gate_requires_trigger_or_material_local_revision() -> None:
    gate = AdversaryDecisionGate(cooldown_s=60)
    context = make_context()

    assert gate.should_request(context) is True
    gate.record_decision(context)
    assert gate.should_request(context.model_copy(update={"sim_time_s": 620})) is False

    threat_changed = context.model_copy(
        update={
            "sim_time_s": 630,
            "platform_threats": (
                context.platform_threats[0].model_copy(update={"threat_level": "critical"}),
            ),
            "local_contacts": (
                context.local_contacts[0].model_copy(update={"threat_level": "critical"}),
            ),
            "trigger_events": (
                AdversaryTrigger(
                    trigger_id="threat-change",
                    event_type="target_contact_threat_changed",
                    sim_time_s=630,
                    severity="tactical",
                    summary="local threat level changed",
                ),
            ),
        }
    )
    assert gate.should_request(threat_changed) is True


def test_adversary_gate_does_not_invoke_without_local_evidence_or_mission_trigger() -> None:
    base = make_context()
    empty = base.model_copy(
        update={
            "observations": (),
            "platform_threats": (),
            "local_contacts": (),
            "trigger_events": (),
            "communications_acoustic_exposure": base.communications_acoustic_exposure.model_copy(
                update={"active_emitter_exposure": 0.0}
            ),
        }
    )
    assert AdversaryDecisionGate().should_request(empty) is False
