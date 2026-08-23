from underwater_tracking.config.models import TrajectoryDiffConfig
from underwater_tracking.domain.agent_models import (
    TrajectoryDiffGateState,
    TrajectoryDiffResult,
)
from underwater_tracking.prediction.diff_gate import advance_diff_gate


CONFIG = TrajectoryDiffConfig()


def comparable_diff(
    diff_id: str,
    *,
    normalized: float,
    absolute: float,
    exceeded: bool,
) -> TrajectoryDiffResult:
    return TrajectoryDiffResult(
        diff_id=diff_id,
        target_id="T1",
        previous_prediction_id=f"{diff_id}-previous",
        current_prediction_id=f"{diff_id}-current",
        previous_sim_time_s=0,
        current_sim_time_s=30,
        status="comparable",
        normalized_rms=normalized,
        absolute_rms_m=absolute,
        normalized_threshold=CONFIG.normalized_threshold,
        absolute_floor_m=CONFIG.absolute_floor_m,
        reset_normalized_threshold=CONFIG.reset_normalized_threshold,
        reset_absolute_floor_m=CONFIG.reset_absolute_floor_m,
        threshold_schema_version=CONFIG.schema_version,
        confirmation_cycles=CONFIG.confirmation_cycles,
        exceeded=exceeded,
    )


def exceeded_diff(diff_id: str) -> TrajectoryDiffResult:
    return comparable_diff(diff_id, normalized=3.0, absolute=300.0, exceeded=True)


def test_gate_requires_two_exceedances_and_latches_once() -> None:
    first = advance_diff_gate(None, exceeded_diff("D1"), CONFIG)
    assert first.state.consecutive_count == 1
    assert first.state.latched is False
    assert first.emit_suspicion is False
    assert first.request_intent_verification is False

    second = advance_diff_gate(first.state, exceeded_diff("D2"), CONFIG)
    assert second.state.consecutive_count == 2
    assert second.state.latched is True
    assert second.emit_suspicion is True
    assert second.request_intent_verification is True
    assert second.state.suspicion_diff_id == "D2"

    third = advance_diff_gate(second.state, exceeded_diff("D3"), CONFIG)
    assert third.state.consecutive_count == 2
    assert third.emit_suspicion is False
    assert third.request_intent_verification is True
    assert third.state.latest_diff_id == "D3"
    assert third.state.suspicion_diff_id == "D2"


def test_gate_releases_when_either_lower_threshold_is_crossed() -> None:
    state = TrajectoryDiffGateState(
        target_id="T1",
        consecutive_count=2,
        latched=True,
        verification_pending=True,
    )
    normalized_reset = advance_diff_gate(
        state,
        comparable_diff("D1", normalized=1.7, absolute=500.0, exceeded=False),
        CONFIG,
    )
    assert normalized_reset.reset is True
    assert normalized_reset.state.latched is False
    assert normalized_reset.state.verification_pending is False

    absolute_reset = advance_diff_gate(
        state,
        comparable_diff("D2", normalized=4.0, absolute=149.0, exceeded=False),
        CONFIG,
    )
    assert absolute_reset.reset is True
    assert absolute_reset.state.latched is False


def test_non_exceeded_comparable_diff_clears_accumulation() -> None:
    first = advance_diff_gate(None, exceeded_diff("D1"), CONFIG)
    cleared = advance_diff_gate(
        first.state,
        comparable_diff("D2", normalized=2.0, absolute=200.0, exceeded=False),
        CONFIG,
    )

    assert cleared.reset is True
    assert cleared.state.consecutive_count == 0
    assert cleared.state.latest_diff_id == "D2"


def test_non_comparable_diff_resets_the_baseline() -> None:
    state = TrajectoryDiffGateState(target_id="T1", consecutive_count=1)
    diff = comparable_diff("D2", normalized=0.0, absolute=0.0, exceeded=False).model_copy(
        update={"status": "predictor_regime_reset"}
    )

    decision = advance_diff_gate(state, diff, CONFIG)

    assert decision.reset is True
    assert decision.state == TrajectoryDiffGateState(target_id="T1", latest_diff_id="D2")
    assert decision.request_intent_verification is False
