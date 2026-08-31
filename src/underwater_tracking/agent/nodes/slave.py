"""LLM-backed decision node for the executable group-slave brain."""

from __future__ import annotations

from typing import TypedDict

from underwater_tracking.agent.llm import LLMContentError, StructuredLLM
from underwater_tracking.domain.slave_models import (
    SlaveSonarContext,
    SlaveSonarDecision,
    SlaveDecisionValidationError,
    validate_slave_decision,
)

SLAVE_PROMPT_VERSION = "slave-sonar-v1"
_MAX_CONTENT_REPAIRS = 2
_MAX_BOUNDARY_REPAIRS = 2

SLAVE_SYSTEM_PROMPT = """
    You are the group slave brain for a UUV underwater tracking group.
Return exactly one JSON object matching the supplied SlaveSonarDecision schema.
Use only the operational estimates in the user payload. The target's hidden
ground state, true position, true trajectory, simulator truth and evaluation-
only labels are not available and must never be inferred or cited.

Passive listening is continuous and is the default. For mode=passive, emitter
must be null, energy_cost_fraction and exposure_cost must be zero, cooldown_s
must be zero, and all receivers must be passive-capable. Use mode=active only
when passive SNR is insufficient, background noise is high, covariance or
track age is growing, the target is lost, or multiple candidates are
ambiguous. Active sonar is still an exception: account for active clutter and
false alarms, and keep active energy, exposure and cooldown within the
emitter's capability. Active mode requires an available active-capable
emitter, available active receivers and distance-derived connectivity between
the emitter and each other receiver.

Compare UUV passive/active capability, speed, turn rate, endurance, energy,
deployment state, sensor mode and carrier support radius. Use per-platform
master_connected/is_group_leader and the
group-level master_connected flag: when disconnected, choose only a locally
executable action and do not invent remote approval. Use belief-derived
quality/covariance, passive SNR/background noise, active clutter, candidate
count/loss state, intent confidence and current/future rotation segments to
preserve continuity and prepare the handoff before the target can outrun the
current group. Never invent IDs or segments.
""".strip()


class SlaveState(TypedDict, total=False):
    """LangGraph state for one local slave decision cycle."""

    context: SlaveSonarContext
    situation: SlaveSonarContext
    decision: SlaveSonarDecision
    prompt_payload: dict[str, object]


class SlaveSonarDecisionNode:
    """Call the injected real structured LLM and reject unsafe output."""

    def __init__(
        self,
        llm: StructuredLLM[SlaveSonarDecision],
        *,
        model_id: str = "underwater-slave-model",
        prompt_version: str = SLAVE_PROMPT_VERSION,
        temperature: float = 0.2,
    ) -> None:
        if llm is None:
            raise TypeError("slave brain requires an injected StructuredLLM")
        self._llm = llm
        self._model_id = model_id
        self._prompt_version = prompt_version
        self._temperature = temperature

    def build_payload(self, context: SlaveSonarContext) -> dict[str, object]:
        """Serialize only bounded operational factors for the LLM."""

        return {
            "model": self._model_id,
            "temperature": self._temperature,
            "output_token_budget": 1024,
            "thinking_mode": "disabled",
            "prompt_version": self._prompt_version,
            "system_prompt": SLAVE_SYSTEM_PROMPT,
            "scenario_id": context.scenario_id,
            "sim_time_s": context.sim_time_s,
            "group_id": context.group_id,
            "target_id": context.target_id,
            "master_connection": {
                "master_id": context.master_id,
                "connected": context.master_connected,
                "local_execution_required": not context.master_connected,
            },
            "platform_capabilities": [
                {
                    "platform_id": platform.platform_id,
                    "platform_kind": platform.platform_kind,
                    "passive_capable": platform.passive_capable,
                    "active_capable": platform.active_capable,
                    "active_receive_capable": platform.active_receive_capable,
                    "passive_range_m": platform.passive_range_m,
                    "active_range_m": platform.active_range_m,
                    "max_speed_mps": platform.max_speed_mps,
                    "max_turn_rate_rad_s": platform.max_turn_rate_rad_s,
                    "endurance_s": platform.endurance_s,
                    "passive_bearing_variance_rad2": platform.passive_bearing_variance_rad2,
                    "active_bearing_sigma_rad": platform.active_bearing_sigma_rad,
                    "active_range_sigma_m": platform.active_range_sigma_m,
                    "clutter_sensitivity": platform.clutter_sensitivity,
                    "deployment_state": platform.deployment_state,
                    "energy_fraction": platform.energy_fraction,
                    "ping_energy_cost_fraction": platform.ping_energy_cost_fraction,
                    "exposure_cost": platform.exposure_cost,
                    "ping_cooldown_s": platform.ping_cooldown_s,
                    "cooldown_remaining_s": platform.cooldown_remaining_s,
                    "available": platform.available,
                    "sensor_mode": platform.sensor_mode,
                    "group_id": platform.group_id,
                    "is_group_leader": platform.is_group_leader,
                    "master_connected": platform.master_connected,
                    "carrier_connected": platform.carrier_connected,
                    "distance_to_carrier_m": platform.distance_to_carrier_m,
                    "carrier_support_radius_m": platform.carrier_support_radius_m,
                }
                for platform in sorted(context.platforms, key=lambda item: item.platform_id)
            ],
            "connectivity": [
                {
                    "source_id": link.source_id,
                    "target_id": link.target_id,
                    "medium": link.medium,
                    "distance_m": link.distance_m,
                    "range_m": link.range_m,
                    "connected": link.connected,
                }
                for link in sorted(
                    context.communication_links,
                    key=lambda item: (item.source_id, item.target_id, item.medium),
                )
            ],
            "belief_derived_quality": {
                "target_id": context.belief.target_id,
                "quality": context.belief.quality,
                "covariance_trace_m2": context.belief.covariance_trace_m2,
                "covariance_max_eigenvalue_m2": context.belief.covariance_max_eigenvalue_m2,
                "last_observation_age_s": context.belief.last_observation_age_s,
                "association_confidence": context.belief.association_confidence,
            },
            "passive_acoustic": {
                "snr_db": context.belief.passive_snr_db,
                "background_noise_db": context.belief.background_noise_db,
                "covariance_growth_factor": context.belief.covariance_growth_factor,
            },
            "active_acoustic": {
                "clutter_level": context.belief.active_clutter_level,
                "candidate_count": context.belief.candidate_count,
                "candidate_ids": list(context.belief.candidate_ids),
            },
            "track_status": {
                "lost": context.belief.target_lost,
                "candidate_count": context.belief.candidate_count,
            },
            "rotation_and_future_segments": {
                "current_segment_id": context.current_segment_id,
                "predicted_intent": context.predicted_intent,
                "intent_confidence": context.intent_confidence,
                "segments": [
                    {
                        "segment_id": segment.segment_id,
                        "start_s": segment.start_s,
                        "end_s": segment.end_s,
                        "predicted_quality": segment.predicted_quality,
                        "predicted_covariance_trace_m2": segment.predicted_covariance_trace_m2,
                        "owner_group_id": segment.owner_group_id,
                        "intercept_xy": segment.intercept_xy,
                        "is_current": segment.segment_id == context.current_segment_id,
                    }
                    for segment in sorted(
                        context.handoff_segments, key=lambda item: item.start_s
                    )
                ],
            },
            "decision_factors": {
                "passive_is_default": context.passive_continuous,
                "continuous_passive_required": context.passive_continuous,
                "active_is_exceptional": context.active_only_on_exception,
                "distance_derived_connectivity": True,
                "future_segment_handoff_required": True,
                "carrier_support_radius_is_hard_limit": True,
                "active_quality_floor": context.active_quality_floor,
                "active_covariance_growth_factor": context.active_covariance_growth_factor,
                "active_background_noise_db": context.active_background_noise_db,
                "max_active_exposure_cost": context.max_active_exposure_cost,
                "require_connected_emitter_receiver": context.require_connected_emitter_receiver,
                "local_autonomy_when_disconnected": context.local_autonomy_when_disconnected,
            },
        }

    def __call__(self, state: SlaveState) -> dict[str, object]:
        context = state.get("context") or state.get("situation")
        if context is None:
            raise ValueError("slave graph state requires context")
        payload = self.build_payload(context)
        # Let every real LLM error escape. The runtime owns retry/pause policy;
        # this node never substitutes a rule-based decision.
        decision = self._invoke(payload)
        for repair_attempt in range(_MAX_BOUNDARY_REPAIRS + 1):
            try:
                validate_slave_decision(decision, context)
                return {"decision": decision, "prompt_payload": payload}
            except SlaveDecisionValidationError as exc:
                if repair_attempt >= _MAX_BOUNDARY_REPAIRS:
                    raise
                decision = self._invoke(
                    {
                        **payload,
                        "correction_feedback": self._boundary_correction_feedback(
                            context,
                            exc,
                            repair_attempt=repair_attempt + 1,
                        ),
                    }
                )

        raise AssertionError("unreachable slave boundary validation state")

    @staticmethod
    def _boundary_correction_feedback(
        context: SlaveSonarContext,
        error: SlaveDecisionValidationError,
        *,
        repair_attempt: int,
    ) -> str:
        active_emitters = [
            platform.platform_id
            for platform in context.platforms
            if platform.available
            and platform.deployment_state != "failed"
            and platform.active_capable
        ]
        active_receivers = [
            platform.platform_id
            for platform in context.platforms
            if platform.available
            and platform.deployment_state != "failed"
            and platform.active_receive_capable
        ]
        return (
            f"Boundary repair {repair_attempt} failed validation: {error}. "
            "Return a new complete JSON decision; do not repeat the rejected "
            "field values. The decision must use only the supplied roster, "
            "connectivity, doctrine exceptions, and handoff segments. "
            f"Available active emitters are exactly {active_emitters!r}; "
            f"available active receivers are exactly {active_receivers!r}. "
            "For mode=active, emitter is required and must be one of the "
            "available active emitters. For mode=passive, emitter must be null "
            "and active-only fields must remain zero. "
            + (
                "This context does not satisfy any active-sonar exception trigger; "
                "you MUST return mode=passive, emitter=null, zero energy_cost_fraction, "
                "zero exposure_cost, and cooldown_s=0."
                if "outside doctrine exception triggers" in str(error)
                else ""
            )
        )

    def _invoke(self, payload: dict[str, object]) -> SlaveSonarDecision:
        """Allow bounded LLM-only content repairs; never synthesize a decision."""
        for repair_attempt in range(_MAX_CONTENT_REPAIRS + 1):
            try:
                raw = self._llm.invoke_structured(
                    "slave_sonar_decision",
                    payload,
                    SlaveSonarDecision,
                    prompt_version=self._prompt_version,
                )
                return SlaveSonarDecision.model_validate(raw)
            except LLMContentError as exc:
                if repair_attempt >= _MAX_CONTENT_REPAIRS:
                    raise
                payload = {
                    **payload,
                    "correction_feedback": (
                        f"Bounded content repair {repair_attempt + 1} failed: {exc}. "
                        "Return only one complete JSON object matching the supplied "
                        "SlaveSonarDecision schema."
                    ),
                }
        raise AssertionError("unreachable slave content validation state")
