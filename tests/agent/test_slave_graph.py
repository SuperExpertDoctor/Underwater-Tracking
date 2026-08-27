"""Executable slave graph tests with a typed recording LLM double."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeVar, cast

import pytest
from pydantic import ValidationError

from underwater_tracking.agent.graphs.slave import build_slave_graph
from underwater_tracking.agent.llm import LLMContentError, StructuredLLM
from underwater_tracking.domain.slave_models import (
    SlaveBeliefSummary,
    SlaveCommunicationLink,
    SlaveHandoffSegment,
    SlavePlatformCapability,
    SlaveSonarContext,
    SlaveSonarDecision,
)

T = TypeVar("T", bound=SlaveSonarDecision)


@dataclass
class RecordingStructuredLLM(StructuredLLM[SlaveSonarDecision]):
    """Typed test double used only to inspect graph wiring and payloads."""

    response: SlaveSonarDecision | Exception
    calls: list[tuple[str, dict[str, object], type[SlaveSonarDecision], str]] = field(
        default_factory=list
    )

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: type[T],
        *,
        prompt_version: str = "",
    ) -> T:
        assert response_model is SlaveSonarDecision
        self.calls.append((operation, payload, response_model, prompt_version))
        if isinstance(self.response, Exception):
            raise self.response
        return cast(T, self.response)


@dataclass
class SequencedStructuredLLM(StructuredLLM[SlaveSonarDecision]):
    """Return a planned sequence so bounded LLM repairs can be tested."""

    responses: list[SlaveSonarDecision]
    calls: list[tuple[str, dict[str, object], type[SlaveSonarDecision], str]] = field(
        default_factory=list
    )

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: type[T],
        *,
        prompt_version: str = "",
    ) -> T:
        assert response_model is SlaveSonarDecision
        self.calls.append((operation, payload, response_model, prompt_version))
        if not self.responses:
            raise AssertionError("test LLM response sequence was exhausted")
        return cast(T, self.responses.pop(0))


@dataclass
class ContentRepairStructuredLLM(StructuredLLM[SlaveSonarDecision]):
    """Fail with provider content errors before returning a typed response."""

    failures: int
    response: SlaveSonarDecision
    calls: list[tuple[str, dict[str, object], type[SlaveSonarDecision], str]] = field(
        default_factory=list
    )

    def invoke_structured(
        self,
        operation: str,
        payload: dict[str, object],
        response_model: type[T],
        *,
        prompt_version: str = "",
    ) -> T:
        assert response_model is SlaveSonarDecision
        self.calls.append((operation, payload, response_model, prompt_version))
        if self.failures:
            self.failures -= 1
            raise LLMContentError("incomplete sonar decision")
        return cast(T, self.response)


def _platform(
    platform_id: str,
    platform_kind: str,
    *,
    active: bool = True,
) -> SlavePlatformCapability:
    return SlavePlatformCapability(
        platform_id=platform_id,
        platform_kind=platform_kind,  # type: ignore[arg-type]
        passive_capable=True,
        active_capable=active,
        active_receive_capable=True,
        passive_range_m=4500.0,
        active_range_m=3200.0,
        max_speed_mps=4.0 if platform_kind == "uuv" else 7.0,
        max_turn_rate_rad_s=0.2,
        endurance_s=7200.0,
        passive_bearing_variance_rad2=0.015,
        active_bearing_sigma_rad=0.04,
        active_range_sigma_m=35.0,
        clutter_sensitivity=0.3,
        energy_fraction=0.8,
        ping_energy_cost_fraction=0.08,
        exposure_cost=0.25,
        ping_cooldown_s=20,
        cooldown_remaining_s=0,
        available=True,
        sensor_mode="passive",
        deployment_state="deployed",
        group_id="group-01" if platform_kind == "uuv" else None,
        is_group_leader=platform_id == "uuv-02",
        master_connected=False,
        carrier_connected=platform_kind == "usv",
        distance_to_carrier_m=700.0 if platform_kind == "usv" else None,
        carrier_support_radius_m=1500.0 if platform_kind == "usv" else None,
    )


def _context(*, active_emitter_capable: bool = True) -> SlaveSonarContext:
    return SlaveSonarContext(
        scenario_id="scenario-01",
        sim_time_s=120,
        group_id="group-01",
        target_id="target-01",
        master_id="carrier-master",
        master_connected=False,
        platforms=(
            _platform("uuv-01", "uuv", active=active_emitter_capable),
            _platform("uuv-02", "uuv"),
        ),
        communication_links=(
            SlaveCommunicationLink(
                source_id="carrier-master",
                target_id="uuv-01",
                medium="acoustic",
                distance_m=700.0,
                range_m=1000.0,
            ),
            SlaveCommunicationLink(
                source_id="uuv-01",
                target_id="uuv-02",
                medium="acoustic",
                distance_m=500.0,
                range_m=1000.0,
            ),
        ),
        belief=SlaveBeliefSummary(
            target_id="target-01",
            quality=0.52,
            covariance_trace_m2=480000.0,
            covariance_max_eigenvalue_m2=350000.0,
            last_observation_age_s=18.0,
            passive_snr_db=-3.0,
            background_noise_db=72.0,
            active_clutter_level=0.35,
            target_lost=False,
            candidate_count=2,
            candidate_ids=("candidate-a", "candidate-b"),
            association_confidence=0.48,
        ),
        handoff_segments=(
            SlaveHandoffSegment(
                segment_id="segment-01",
                start_s=0,
                end_s=180,
                predicted_quality=0.55,
                predicted_covariance_trace_m2=480000.0,
                owner_group_id="group-01",
            ),
            SlaveHandoffSegment(
                segment_id="segment-02",
                start_s=180,
                end_s=360,
                predicted_quality=0.72,
                predicted_covariance_trace_m2=310000.0,
                owner_group_id="group-02",
            ),
        ),
        current_segment_id="segment-01",
        predicted_intent="withdraw",
        intent_confidence=0.74,
    )


def _passive_decision() -> SlaveSonarDecision:
    return SlaveSonarDecision(
        mode="passive",
        emitter=None,
        receiver_ids=("uuv-01", "uuv-02"),
        target_id="target-01",
        group_id="group-01",
        handoff_segment="segment-01",
        rationale="Keep the passive mesh open while the group resolves candidate ambiguity.",
        confidence=0.76,
        expected_information_gain=0.42,
        energy_cost_fraction=0.0,
        exposure_cost=0.0,
        cooldown_s=0,
    )


def _active_decision() -> SlaveSonarDecision:
    return SlaveSonarDecision(
        mode="active",
        emitter="uuv-01",
        receiver_ids=("uuv-01", "uuv-02"),
        target_id="target-01",
        group_id="group-01",
        handoff_segment="segment-01",
        rationale="Passive evidence is ambiguous and stale, so one bounded active ping is justified before handoff.",
        confidence=0.70,
        expected_information_gain=0.68,
        energy_cost_fraction=0.08,
        exposure_cost=0.25,
        cooldown_s=20,
    )


def test_slave_graph_calls_injected_llm_and_preserves_truth_safe_payload() -> None:
    llm = RecordingStructuredLLM(_passive_decision())
    result = build_slave_graph(llm, model_id="slave-test-model").invoke(
        {"context": _context()}
    )

    assert result["decision"] == _passive_decision()
    assert len(llm.calls) == 1
    operation, payload, response_model, prompt_version = llm.calls[0]
    assert operation == "slave_sonar_decision"
    assert response_model is SlaveSonarDecision
    assert prompt_version == "slave-sonar-v1"
    assert payload["model"] == "slave-test-model"
    assert payload["output_token_budget"] == 1024
    assert payload["thinking_mode"] == "disabled"
    assert {
        "platform_capabilities",
        "connectivity",
        "belief_derived_quality",
        "passive_acoustic",
        "active_acoustic",
        "track_status",
        "rotation_and_future_segments",
        "master_connection",
    }.issubset(payload)
    assert not _all_mapping_keys(payload).intersection(
        {"true_position", "target_truth", "ground_truth", "position_xy", "mean"}
    )


def test_active_decision_is_feasible_and_uses_the_same_graph_boundary() -> None:
    llm = RecordingStructuredLLM(_active_decision())
    result = build_slave_graph(llm).invoke({"context": _context()})

    decision = result["decision"]
    assert isinstance(decision, SlaveSonarDecision)
    assert decision.mode == "active"
    assert decision.emitter == "uuv-01"


def test_active_decision_rejects_non_capable_emitter_without_replacement() -> None:
    llm = RecordingStructuredLLM(_active_decision())

    with pytest.raises(ValueError, match="active emitter"):
        build_slave_graph(llm).invoke(
            {"context": _context(active_emitter_capable=False)}
        )
    assert len(llm.calls) == 3


def test_boundary_invalid_slave_decision_gets_two_bounded_llm_repairs() -> None:
    invalid = _active_decision().model_copy(update={"emitter": None})
    llm = SequencedStructuredLLM([invalid, invalid, _active_decision()])

    result = build_slave_graph(llm).invoke({"context": _context()})

    assert result["decision"] == _active_decision()
    assert len(llm.calls) == 3
    assert "active mode requires an emitter" in str(llm.calls[1][1]["correction_feedback"])
    assert "active mode requires an emitter" in str(llm.calls[2][1]["correction_feedback"])


def test_doctrine_violation_gets_two_bounded_llm_repairs() -> None:
    stable_belief = _context().belief.model_copy(
        update={
            "quality": 0.92,
            "covariance_growth_factor": 1.02,
            "background_noise_db": 1.0,
            "target_lost": False,
            "candidate_count": 1,
            "candidate_ids": ("target-01",),
        }
    )
    stable_context = _context().model_copy(update={"belief": stable_belief})
    invalid = _active_decision()
    llm = SequencedStructuredLLM([invalid, invalid, _passive_decision()])

    result = build_slave_graph(llm).invoke({"context": stable_context})

    assert result["decision"] == _passive_decision()
    assert len(llm.calls) == 3
    assert all(
        "MUST return mode=passive" in str(call[1]["correction_feedback"])
        for call in llm.calls[1:]
    )


def test_slave_content_error_gets_two_bounded_llm_repairs() -> None:
    llm = ContentRepairStructuredLLM(failures=2, response=_passive_decision())

    result = build_slave_graph(llm).invoke({"context": _context()})

    assert result["decision"] == _passive_decision()
    assert len(llm.calls) == 3
    assert all(
        "correction_feedback" in call[1] for call in llm.calls[1:]
    )


def test_active_decision_rejects_distance_disconnected_receiver() -> None:
    context = _context()
    disconnected = context.model_copy(
        update={
            "communication_links": tuple(
                link.model_copy(update={"distance_m": link.range_m + 1.0})
                for link in context.communication_links
            )
        }
    )

    with pytest.raises(ValueError, match="disconnected"):
        build_slave_graph(RecordingStructuredLLM(_active_decision())).invoke(
            {"context": disconnected}
        )


def test_decision_rejects_unknown_receiver() -> None:
    invalid = _passive_decision().model_copy(
        update={"receiver_ids": ("uuv-01", "not-in-roster")}
    )
    llm = RecordingStructuredLLM(invalid)

    with pytest.raises(ValueError, match="unknown platforms"):
        build_slave_graph(llm).invoke({"context": _context()})


def test_active_mode_is_rejected_when_no_doctrine_exception_is_present() -> None:
    stable_belief = _context().belief.model_copy(
        update={
            "quality": 0.92,
            "covariance_growth_factor": 1.02,
            "background_noise_db": 1.0,
            "target_lost": False,
            "candidate_count": 1,
            "candidate_ids": ("target-01",),
        }
    )
    stable_context = _context().model_copy(update={"belief": stable_belief})

    with pytest.raises(ValueError, match="outside doctrine exception"):
        build_slave_graph(RecordingStructuredLLM(_active_decision())).invoke(
            {"context": stable_context}
        )


def test_passive_continuity_is_a_strict_output_requirement() -> None:
    raw = _passive_decision().model_dump(mode="python")
    raw["passive_continuous"] = False

    with pytest.raises(ValidationError):
        SlaveSonarDecision.model_validate(raw)


def test_context_rejects_truth_side_fields() -> None:
    raw = _context().model_dump(mode="python")
    raw["true_position"] = (100.0, 200.0)

    with pytest.raises(ValidationError):
        SlaveSonarContext.model_validate(raw)


def test_llm_failure_propagates_without_fallback() -> None:
    failure = RuntimeError("provider unavailable")
    llm = RecordingStructuredLLM(failure)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        build_slave_graph(llm).invoke({"context": _context()})
    assert len(llm.calls) == 1


def _all_mapping_keys(value: object) -> set[object]:
    if isinstance(value, dict):
        result: set[object] = set(value)
        for child in value.values():
            result.update(_all_mapping_keys(child))
        return result
    if isinstance(value, list):
        result = set()
        for child in value:
            result.update(_all_mapping_keys(child))
        return result
    return set()
