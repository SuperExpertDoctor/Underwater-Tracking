"""Persistent hysteresis gate for consecutive trajectory divergences."""

from __future__ import annotations

from dataclasses import dataclass

from underwater_tracking.config.models import TrajectoryDiffConfig
from underwater_tracking.domain.agent_models import (
    TrajectoryDiffGateState,
    TrajectoryDiffResult,
)


@dataclass(frozen=True, slots=True)
class TrajectoryDiffGateDecision:
    state: TrajectoryDiffGateState
    emit_suspicion: bool
    request_intent_verification: bool
    reset: bool


def advance_diff_gate(
    previous: TrajectoryDiffGateState | None,
    diff: TrajectoryDiffResult,
    config: TrajectoryDiffConfig,
) -> TrajectoryDiffGateDecision:
    """Advance one target gate by one newly evidenced forecast comparison."""
    state = previous or TrajectoryDiffGateState(target_id=diff.target_id)
    if state.target_id != diff.target_id:
        state = TrajectoryDiffGateState(target_id=diff.target_id)

    if diff.status != "comparable":
        cleared = TrajectoryDiffGateState(
            target_id=diff.target_id,
            latest_diff_id=diff.diff_id,
            intent_baseline_label=state.intent_baseline_label,
        )
        return TrajectoryDiffGateDecision(
            state=cleared,
            emit_suspicion=False,
            request_intent_verification=False,
            reset=state.latched or state.consecutive_count > 0,
        )

    normalized = diff.normalized_rms or 0.0
    absolute = diff.absolute_rms_m or 0.0
    if state.latched and (
        normalized < config.reset_normalized_threshold or absolute < config.reset_absolute_floor_m
    ):
        cleared = TrajectoryDiffGateState(
            target_id=diff.target_id,
            latest_diff_id=diff.diff_id,
            intent_baseline_label=state.intent_baseline_label,
        )
        return TrajectoryDiffGateDecision(
            state=cleared,
            emit_suspicion=False,
            request_intent_verification=False,
            reset=True,
        )

    if state.latched:
        kept = state.model_copy(update={"latest_diff_id": diff.diff_id})
        return TrajectoryDiffGateDecision(
            state=kept,
            emit_suspicion=False,
            request_intent_verification=kept.verification_pending,
            reset=False,
        )

    if not diff.exceeded:
        cleared = TrajectoryDiffGateState(
            target_id=diff.target_id,
            latest_diff_id=diff.diff_id,
            intent_baseline_label=state.intent_baseline_label,
        )
        return TrajectoryDiffGateDecision(
            state=cleared,
            emit_suspicion=False,
            request_intent_verification=False,
            reset=state.consecutive_count > 0,
        )

    consecutive_count = state.consecutive_count + 1
    emit_suspicion = consecutive_count >= config.confirmation_cycles
    updated = TrajectoryDiffGateState(
        target_id=diff.target_id,
        consecutive_count=consecutive_count,
        latched=emit_suspicion,
        verification_pending=emit_suspicion,
        suspicion_diff_id=diff.diff_id if emit_suspicion else None,
        latest_diff_id=diff.diff_id,
        intent_baseline_label=state.intent_baseline_label,
    )
    return TrajectoryDiffGateDecision(
        state=updated,
        emit_suspicion=emit_suspicion,
        request_intent_verification=emit_suspicion,
        reset=False,
    )
