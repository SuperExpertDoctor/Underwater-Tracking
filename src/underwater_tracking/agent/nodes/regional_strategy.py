from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from underwater_tracking.agent.llm import LLMCallMetadata, LLMContentError, StructuredLLM
from underwater_tracking.agent.nodes.snapshot import PlanningSnapshot
from underwater_tracking.agent.prompts import (
    REGIONAL_STRATEGY_PROMPT_VERSION,
    REGIONAL_STRATEGY_SYSTEM_PROMPT,
    canonical_digest,
)
from underwater_tracking.agent.state import CarrierState
from underwater_tracking.domain.agent_models import IntentHypothesis
from underwater_tracking.domain.regional_models import (
    RegionalStrategySet,
    TargetRegionPlan,
)


def validate_regional_strategy(
    target_region_plan: TargetRegionPlan,
    strategy: RegionalStrategySet,
) -> RegionalStrategySet:
    """Validate exact region coverage and evidence references from the LLM."""
    expected = set(target_region_plan.region_ids)
    actual = [policy.region_id for policy in strategy.policies]
    if len(actual) != len(set(actual)):
        raise ValueError("duplicate regional policy")
    unknown = set(actual) - expected
    if unknown:
        raise ValueError(f"unknown region policy: {sorted(unknown)}")
    missing = expected - set(actual)
    if missing:
        raise ValueError(f"missing regional policy: {sorted(missing)}")
    allowed_evidence = set(target_region_plan.evidence_ids)
    for policy in strategy.policies:
        if not set(policy.evidence_ids) <= allowed_evidence:
            raise ValueError(f"regional policy {policy.region_id} cites unknown evidence")
    return strategy


class RegionalStrategyGenerationNode:
    """Request bounded regional policies from the real structured LLM port."""

    def __init__(
        self,
        llm: StructuredLLM[RegionalStrategySet],
        *,
        model_id: str = "underwater-assistant-model",
        prompt_version: str = REGIONAL_STRATEGY_PROMPT_VERSION,
        temperature: float = 0.2,
        snapshot_provider: Callable[[str], PlanningSnapshot] | None = None,
    ) -> None:
        self._llm = llm
        self._model_id = model_id
        self._prompt_version = prompt_version
        self._temperature = temperature
        self._snapshot_provider = snapshot_provider

    def build_payload(
        self,
        snapshot: PlanningSnapshot,
        target_region_plan: TargetRegionPlan,
        intents: Mapping[str, IntentHypothesis],
    ) -> dict[str, object]:
        intent = intents.get(target_region_plan.target_id)
        evidence_ids = set(target_region_plan.evidence_ids)
        if intent is not None:
            evidence_ids.update(intent.evidence_ids)
        return {
            "model": self._model_id,
            "temperature": self._temperature,
            "system_prompt": REGIONAL_STRATEGY_SYSTEM_PROMPT,
            "scenario_id": snapshot.scenario_id,
            "sim_time_s": snapshot.sim_time_s,
            "target_id": target_region_plan.target_id,
            "prediction_id": target_region_plan.prediction_id,
            "intent": (
                {
                    "label": intent.label,
                    "confidence": intent.confidence,
                    "evidence_ids": sorted(intent.evidence_ids),
                }
                if intent is not None
                else {
                    "label": target_region_plan.intent_label,
                    "confidence": target_region_plan.intent_confidence,
                    "evidence_ids": sorted(target_region_plan.evidence_ids),
                }
            ),
            "regions": [self._region_payload(cell) for cell in target_region_plan.cells],
            "operational_constraints": {
                "require_uuv_per_region": target_region_plan.grid_spec.require_uuv_per_region,
                "require_usv_per_region": target_region_plan.grid_spec.require_usv_per_region,
                "relay_overlap_policy": target_region_plan.grid_spec.relay_overlap_policy,
                "passive_sonar_required": True,
            },
            "evidence_ids": sorted(evidence_ids),
        }

    def invoke_for_plan(
        self,
        snapshot: PlanningSnapshot,
        target_region_plan: TargetRegionPlan,
        intents: Mapping[str, IntentHypothesis],
    ) -> RegionalStrategySet:
        payload = self.build_payload(snapshot, target_region_plan, intents)
        strategy = self._invoke(payload)
        return validate_regional_strategy(target_region_plan, strategy)

    def __call__(self, state: CarrierState) -> CarrierState:
        if self._snapshot_provider is None:
            raise ValueError("regional strategy requires a snapshot provider")
        snapshot_ref = state.get("snapshot_ref")
        if not snapshot_ref:
            raise ValueError("regional strategy requires snapshot_ref")
        snapshot = self._snapshot_provider(snapshot_ref)
        plans = state.get("regional_plans", {})
        policies: dict[str, RegionalStrategySet] = {}
        provenance: dict[str, LLMCallMetadata] = {}
        for target_id, target_plan in sorted(plans.items()):
            payload = self.build_payload(snapshot, target_plan, state.get("intent_hypotheses", {}))
            strategy = validate_regional_strategy(target_plan, self._invoke(payload))
            policies[target_id] = strategy
            provenance[f"regional_strategy:{target_id}"] = LLMCallMetadata(
                operation="regional_strategy",
                model=self._model_id,
                prompt_version=self._prompt_version,
                request_hash=canonical_digest(payload),
                response_hash=canonical_digest(strategy.model_dump(mode="json")),
                sim_time_s=snapshot.sim_time_s,
                scenario_id=snapshot.scenario_id,
            )
        return {
            "regional_policies": policies,
            "llm_provenance": {**state.get("llm_provenance", {}), **provenance},
        }

    def _invoke(self, payload: dict[str, object]) -> RegionalStrategySet:
        try:
            return self._llm.invoke_structured(
                "regional_strategy",
                payload,
                RegionalStrategySet,
                prompt_version=self._prompt_version,
            )
        except LLMContentError as exc:
            correction_payload = {**payload, "correction_feedback": str(exc)}
            return self._llm.invoke_structured(
                "regional_strategy",
                correction_payload,
                RegionalStrategySet,
                prompt_version=self._prompt_version,
            )

    @staticmethod
    def _region_payload(cell: Any) -> dict[str, object]:
        return {
            "region_id": cell.region_id,
            "geometry": {
                "grid_x": cell.grid_x,
                "grid_y": cell.grid_y,
                "bounds": {
                    "min_x": cell.min_x,
                    "max_x": cell.max_x,
                    "min_y": cell.min_y,
                    "max_y": cell.max_y,
                },
                "center_xy": list(cell.center_xy),
                "cell_size_m": cell.cell_size_m,
            },
            "time": {
                "first_entry_s": cell.first_entry_s,
                "last_exit_s": cell.last_exit_s,
                "visit_windows": [window.model_dump(mode="json") for window in cell.visit_windows],
            },
            "occupancy_likelihood": cell.occupancy_likelihood,
            "intent_labels": list(cell.intent_labels),
            "evidence_ids": sorted(cell.evidence_ids),
            "predecessor_region_ids": sorted(cell.predecessor_region_ids),
            "successor_region_ids": sorted(cell.successor_region_ids),
        }
