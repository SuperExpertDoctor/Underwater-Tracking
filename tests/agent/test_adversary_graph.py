"""First adversary-graph contract tests.

The test double is a typed recorder only.  It is injected by these tests and
is never imported by production modules or used as an unavailable-LLM path.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from underwater_tracking.agent.graphs.adversary import build_adversary_graph
from underwater_tracking.agent.nodes.adversary import (
    ADVERSARY_PROMPT_VERSION,
    AdversaryDecisionGate,
    build_adversary_payload,
)
from underwater_tracking.domain.adversary_models import (
    AdversaryBelief,
    AdversaryDecisionRecord,
    AdversaryEscapeDecision,
    AdversaryEscapeInput,
    AdversaryKinematicLimits,
    AdversaryObservation,
    AdversaryOperatingBoundary,
    CommunicationsAcousticExposure,
    PlatformThreatSummary,
    AdversaryTrigger,
)


class RecordingStructuredLLM:
    """Typed test double that records the exact structured-call contract."""

    def __init__(
        self,
        decision: AdversaryEscapeDecision | None = None,
        error: Exception | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self._decision = decision
        self._error = error

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: type[AdversaryEscapeDecision],
        *,
        prompt_version: str = "",
    ) -> AdversaryEscapeDecision:
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
            AdversaryObservation(
                observation_id="OBS-2",
                observed_at_s=570,
                kind="communication_intercept",
                bearing_rad=-0.4,
                range_m=None,
                confidence=0.45,
                assessment="communication",
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
                PlatformThreatSummary(
                    platform_id="MOTHER-1",
                    platform_kind="mother_ship",
                observed_at_s=560,
                threat_level="medium",
                estimated_range_m=5400.0,
                relative_bearing_rad=-1.0,
                passive_detection_risk=0.35,
                active_ping_risk=0.58,
                relay_detection_risk=0.8,
                surface_relay_available=True,
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
                segment="S-0",
                speed=3.1,
                heading=0.1,
                decoy_action="none",
                decoy_count=0,
                outcome="inconclusive",
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


def make_decision() -> AdversaryEscapeDecision:
    return AdversaryEscapeDecision(
        target_id="SUB-1",
        maneuver="course_change",
        intent="break_contact",
        waypoint=(1350.0, 900.0),
        segment="S-1",
        speed=4.2,
        heading=0.8,
        decoy_action="deploy",
        decoy_count=1,
        confidence=0.74,
        rationale="The recent passive bearing and relay risk favor a bounded course change with one decoy.",
        communications_discipline="burst_only",
    )


def test_payload_contains_only_target_side_evidence() -> None:
    context = make_context()
    payload = build_adversary_payload(context)
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).casefold()

    assert set(payload) == {
        "prompt_version",
        "system_prompt",
        "target_id",
        "sim_time_s",
        "belief",
        "observations",
        "platform_threats",
        "trigger_events",
        "communications_acoustic_exposure",
        "decision_history",
        "kinematic_limits",
        "operating_boundary",
    }
    assert "truth" not in encoded
    assert "ground_truth" not in encoded
    assert "true_position" not in encoded
    assert "simulation_truth" not in encoded
    assert "usv" not in encoded
    assert payload["belief"] == context.belief.model_dump(mode="json")
    assert payload["platform_threats"]


def test_graph_calls_typed_structured_llm_and_wires_nodes() -> None:
    recorder = RecordingStructuredLLM(make_decision())
    graph = build_adversary_graph(recorder)

    result = graph.invoke({"context": make_context()})

    assert result["decision"] == make_decision()
    assert len(recorder.calls) == 1
    call = recorder.calls[0]
    assert call["operation"] == "adversary_escape"
    assert call["response_model"] is AdversaryEscapeDecision
    assert call["prompt_version"] == ADVERSARY_PROMPT_VERSION
    node_names = set(graph.get_graph().nodes)
    assert {"build_payload", "decide", "validate"} <= node_names


def test_llm_failure_propagates_without_fallback() -> None:
    failure = RuntimeError("structured provider unavailable")
    recorder = RecordingStructuredLLM(error=failure)
    graph = build_adversary_graph(recorder)

    with pytest.raises(RuntimeError, match="structured provider unavailable"):
        graph.invoke({"context": make_context()})
    assert len(recorder.calls) == 1


def test_semantic_guards_reject_impossible_speed_turn_waypoint_and_decoys() -> None:
    context = make_context()
    cases = (
        make_decision().model_copy(update={"speed": 5.1}),
        make_decision().model_copy(update={"heading": 3.2}),
        make_decision().model_copy(update={"waypoint": (5001.0, 900.0)}),
        make_decision().model_copy(update={"decoy_count": 3}),
        make_decision().model_copy(update={"decoy_action": "none", "decoy_count": 1}),
    )
    for decision in cases:
        recorder = RecordingStructuredLLM(decision)
        graph = build_adversary_graph(recorder)
        with pytest.raises(ValueError):
            graph.invoke({"context": context})


def test_contract_rejects_extra_private_state_and_non_finite_decision_values() -> None:
    with pytest.raises(ValueError):
        AdversaryEscapeDecision.model_validate(
            {**make_decision().model_dump(), "private_state": "unavailable"}
        )
    with pytest.raises(ValueError):
        AdversaryEscapeDecision.model_validate(
            {**make_decision().model_dump(), "speed": float("nan")}
        )
    with pytest.raises(ValueError):
        AdversaryEscapeInput.model_validate(
            {
                **make_context().model_dump(),
                "belief": {
                    **make_context().belief.model_dump(),
                    "estimated_position_xy": (float("nan"), 800.0),
                },
            }
        )


def test_adversary_gate_requires_cooldown_or_a_hysteretic_revision() -> None:
    gate = AdversaryDecisionGate(cooldown_s=60, heading_revision_rad=0.1, speed_revision_mps=0.5)
    context = make_context()

    assert gate.should_request(context) is True
    gate.record_decision(context)
    assert gate.should_request(context.model_copy(update={"sim_time_s": 620})) is False

    revised = context.model_copy(
        update={
            "sim_time_s": 630,
            "belief": context.belief.model_copy(
                update={"estimated_heading": 0.3, "estimated_speed_mps": 4.0}
            ),
        }
    )
    assert gate.should_request(revised) is False
    assert gate.should_request(revised.model_copy(update={"sim_time_s": 660})) is False
    assert gate.should_request(revised.model_copy(update={"sim_time_s": 690})) is True

    gate.record_decision(revised)
    triggered = revised.model_copy(
        update={
            "sim_time_s": 645,
            "trigger_events": context.trigger_events + (
                AdversaryTrigger(
                    trigger_id="PING-1",
                    event_type="active_ping",
                    sim_time_s=645,
                    severity="tactical",
                    summary="active sonar emission observed",
                ),
            ),
        }
    )
    assert gate.should_request(triggered) is False


def test_adversary_gate_does_not_bypass_cooldown_for_new_active_ping_ids() -> None:
    gate = AdversaryDecisionGate(cooldown_s=60)
    context = make_context()
    gate.record_decision(context)

    for sim_time_s in (610, 620, 630):
        active_ping = AdversaryTrigger(
            trigger_id=f"PING-{sim_time_s}",
            event_type="active_ping",
            sim_time_s=sim_time_s,
            severity="tactical",
            summary="active sonar emission observed",
        )
        ping_context = context.model_copy(
            update={"sim_time_s": sim_time_s, "trigger_events": (active_ping,)},
        )
        assert gate.should_request(ping_context) is False

    regional_feedback = AdversaryTrigger(
        trigger_id="REGIONAL-635",
        event_type="regional_feedback_received",
        sim_time_s=635,
        severity="informational",
        summary="blue regional feedback",
    )
    assert gate.should_request(
        context.model_copy(update={"sim_time_s": 635, "trigger_events": (regional_feedback,)})
    ) is False

    strategic_trigger = AdversaryTrigger(
        trigger_id="DETECTION-641",
        event_type="target_detection",
        sim_time_s=641,
        severity="strategic",
        summary="new target-side detection change",
    )
    assert gate.should_request(
        context.model_copy(update={"sim_time_s": 641, "trigger_events": (strategic_trigger,)})
    ) is True


def test_adversary_gate_does_not_invoke_without_local_evidence() -> None:
    gate = AdversaryDecisionGate()
    base = make_context()
    empty_exposure = base.communications_acoustic_exposure.model_copy(
        update={
            "active_emitter_exposure": 0.0,
            "own_emission_mode": "passive",
        }
    )
    context = base.model_copy(
        update={
            "observations": (),
            "platform_threats": (),
            "trigger_events": (),
            "communications_acoustic_exposure": empty_exposure,
        }
    )

    assert gate.should_request(context) is False
