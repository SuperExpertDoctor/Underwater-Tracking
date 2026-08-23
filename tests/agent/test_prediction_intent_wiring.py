from collections.abc import Sequence

from underwater_tracking.agent.graphs.central import PredictionIntentWiringNode
from underwater_tracking.agent.nodes.event_monitor import EventMonitor
from underwater_tracking.agent.nodes.intent import IntentAnalysisNode
from underwater_tracking.domain.agent_models import (
    IntentHypothesis,
    TrajectoryDiffGateState,
    TrajectoryDiffResult,
)
from underwater_tracking.domain.models import (
    EventLevel,
    GroupQuality,
    GroupReport,
    RuntimeEvent,
    SituationSnapshot,
    TargetBelief,
)


class ScriptedIntentLLM:
    def __init__(self, responses: Sequence[IntentHypothesis]) -> None:
        self.responses = list(responses)
        self.operations: list[str] = []
        self.payloads: list[dict[str, object]] = []

    def invoke_structured(self, operation, payload, schema, *, prompt_version):
        del schema, prompt_version
        self.operations.append(operation)
        self.payloads.append(payload)
        return self.responses.pop(0)


def _snapshot() -> SituationSnapshot:
    reports = tuple(
        GroupReport(
            group_id=f"G-{target_id}",
            target_id=target_id,
            sim_time_s=60,
            member_ids=(),
            belief=TargetBelief(
                target_id=target_id,
                sim_time_s=60,
                mean=(60.0, 0.0, 1.0, 0.0),
                covariance=((100.0, 0.0), (0.0, 100.0)),
                model_probabilities={"cv": 0.8, "left_turn": 0.1, "right_turn": 0.1},
                source_observation_ids=(f"O:{target_id}:60",),
            ),
            quality=GroupQuality(
                instant=0.8,
                window_mean=0.8,
                ewma=0.8,
                components={},
            ),
            plan_revision=1,
        )
        for target_id in ("T1", "T2")
    )
    return SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=2,
        sim_time_s=60,
        uuvs=(),
        group_reports=reports,
        pending_events=(),
    )


def _history(_snapshot, target_id):
    offset = 0.0 if target_id == "T1" else 100.0
    return ((0, offset, 0.0), (30, offset + 30.0, 0.0), (60, offset + 60.0, 0.0))


def _diff() -> TrajectoryDiffResult:
    return TrajectoryDiffResult(
        diff_id="D1",
        target_id="T1",
        previous_prediction_id="P1",
        current_prediction_id="P2",
        previous_sim_time_s=30,
        current_sim_time_s=60,
        status="comparable",
        absolute_rms_m=300.0,
        normalized_rms=3.0,
        previous_evidence_ids=("O:T1:30",),
        current_evidence_ids=("O:T1:60",),
        normalized_threshold=2.45,
        absolute_floor_m=250.0,
        reset_normalized_threshold=1.75,
        reset_absolute_floor_m=150.0,
        threshold_schema_version="trajectory-diff-v1",
        confirmation_cycles=2,
        exceeded=True,
        consecutive_count=2,
        latched=True,
        gate_transition="suspected",
    )


def _suspicion() -> RuntimeEvent:
    return RuntimeEvent(
        event_id="S1:target_intent_change_suspected:T1:60",
        scenario_id="S1",
        sim_time_s=60,
        event_type="target_intent_change_suspected",
        entity_id="T1",
        level=EventLevel.TACTICAL,
        payload={
            "diff_id": "D1",
            "previous_prediction_id": "P1",
            "current_prediction_id": "P2",
            "observation_ids": ("O:T1:60",),
            "absolute_rms_m": 300.0,
            "normalized_rms": 3.0,
            "absolute_floor_m": 250.0,
            "normalized_threshold": 2.45,
            "consecutive_count": 2,
            "source": "trajectory_diff",
        },
    )


def _hypothesis(
    label: str,
    confidence: float,
    *,
    diff_id: str = "D1",
) -> IntentHypothesis:
    return IntentHypothesis(
        label=label,
        confidence=confidence,
        evidence_ids=(
            diff_id,
            "O:T1:60",
            "S1:target_intent_change_suspected:T1:60",
        ),
        alternatives={"transit" if label != "transit" else "evade": 0.1},
        model_id="real-intent-model",
        prompt_version="intent-v1",
    )


def _state() -> dict:
    return {
        "scenario_id": "S1",
        "snapshot_ref": "R2",
        "confirmed_intent_labels": {"T1": "transit"},
        "prediction_diffs": {"T1": _diff()},
        "prediction_diff_gates": {
            "T1": TrajectoryDiffGateState(
                target_id="T1",
                consecutive_count=2,
                latched=True,
                verification_pending=True,
                suspicion_event_id="S1:target_intent_change_suspected:T1:60",
                suspicion_diff_id="D1",
                latest_diff_id="D1",
            )
        },
        "prediction_intent_verification_target_ids": ("T1",),
        "coalesced_events": (_suspicion(),),
    }


def _wiring(responses: Sequence[IntentHypothesis]):
    snapshot = _snapshot()
    llm = ScriptedIntentLLM(responses)
    inner = IntentAnalysisNode(
        llm,
        model_id="real-intent-model",
        belief_history=_history,
        snapshot_provider=lambda _ref: snapshot,
    )
    return PredictionIntentWiringNode(
        inner,
        EventMonitor(scenario_id="S1"),
        lambda _ref: snapshot,
    ), llm


def test_intent_node_filter_and_payload_are_bounded_to_suspected_target() -> None:
    wiring, llm = _wiring((_hypothesis("evade", 0.8),))

    result = wiring(_state())

    assert llm.operations == ["intent"]
    assert llm.payloads[0]["target_id"] == "T1"
    assert llm.payloads[0]["trajectory_diff"]["diff_id"] == "D1"
    assert set(llm.payloads[0]["evidence_ids"]) >= {
        "D1",
        "O:T1:60",
        "S1:target_intent_change_suspected:T1:60",
    }
    assert result["prediction_intent_confirmed"] is False
    assert result["prediction_intent_verification_target_ids"] == ("T1",)


def test_two_real_port_analyses_confirm_changed_semantic_label() -> None:
    wiring, llm = _wiring(
        (
            _hypothesis("evade", 0.8),
            _hypothesis("evade", 0.85, diff_id="D2"),
        )
    )
    first = wiring(_state())
    second_diff = _diff().model_copy(
        update={"diff_id": "D2", "current_prediction_id": "P3"}
    )
    second = wiring(
        {
            **_state(),
            **first,
            "prediction_diffs": {"T1": second_diff},
        }
    )

    assert llm.operations == ["intent", "intent"]
    assert first["prediction_intent_confirmed"] is False
    assert second["prediction_intent_confirmed"] is True
    assert second["confirmed_intent_labels"]["T1"] == "evade"
    assert second["prediction_intent_verification_target_ids"] == ()
    assert second["prediction_diffs"]["T1"].gate_transition == "confirmed"
    event = second["coalesced_events"][-1]
    assert event.event_type == "target_intent_changed"
    assert event.payload["diff_id"] == "D1"
    assert event.payload["verification_diff_id"] == "D2"
    assert event.payload["suspicion_event_id"] == _suspicion().event_id
    assert event.payload["llm_request_hash"]
    assert event.payload["llm_response_hash"]
    assert len(event.payload["intent_llm_calls"]) == 2
    assert all(call["operation"] == "intent" for call in event.payload["intent_llm_calls"])
    assert event.payload["source"] == "real_intent_llm"


def test_unchanged_or_low_confidence_label_ends_verification() -> None:
    for hypothesis in (_hypothesis("transit", 0.8), _hypothesis("evade", 0.6)):
        wiring, _llm = _wiring((hypothesis,))
        result = wiring(_state())

        assert result["prediction_intent_confirmed"] is False
        assert result["prediction_intent_verification_target_ids"] == ()
        assert result["prediction_diff_gates"]["T1"].verification_pending is False
        assert [event.event_type for event in result["coalesced_events"]] == [
            "target_intent_change_suspected"
        ]
