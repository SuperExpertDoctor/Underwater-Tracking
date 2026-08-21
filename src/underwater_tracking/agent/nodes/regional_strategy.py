from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from underwater_tracking.agent.llm import LLMCallMetadata, LLMContentError, StructuredLLM
from underwater_tracking.agent.nodes.snapshot import PlanningSnapshot
from underwater_tracking.agent.prompts import (
    REGIONAL_STRATEGY_PROMPT_VERSION,
    REGIONAL_STRATEGY_SYSTEM_PROMPT,
    UUV_REGIONAL_STRATEGY_PROMPT_VERSION,
    UUV_REGIONAL_STRATEGY_SYSTEM_PROMPT,
    canonical_digest,
)
from underwater_tracking.agent.state import CarrierState
from underwater_tracking.domain.agent_models import IntentHypothesis
from underwater_tracking.domain.regional_models import (
    RegionalMissionCandidate,
    RegionalPolicy,
    RegionalStrategySet,
    TargetRegionPlan,
    TimeWindow,
    UUVRegionalStrategySet,
    UUVRegionalPolicy,
)
from underwater_tracking.planning.candidate_regions import (
    CandidateRegion,
    candidate_region_to_mission_candidate,
)
from underwater_tracking.planning.regional_plan_validator import (
    AvailableUUVs,
    ValidatedRegionalStrategy,
    validate_uuv_strategy,
)

# A regional policy carries several nested objects and LongCat may include
# reasoning tokens in the same completion budget. Keep each response small
# enough to finish as valid JSON under the configured 4096-token limit.
_REGIONS_PER_LLM_REQUEST = 4


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


def validate_regional_strategy_batch(
    target_region_plan: TargetRegionPlan,
    strategy: RegionalStrategySet,
    expected_region_ids: Sequence[str],
) -> RegionalStrategySet:
    """Validate one bounded response before merging it into the full plan."""
    expected = set(expected_region_ids)
    actual = [policy.region_id for policy in strategy.policies]
    if len(actual) != len(set(actual)):
        raise ValueError("duplicate regional policy in batch")
    unknown = set(actual) - expected
    if unknown:
        raise ValueError(f"unknown regional policy in batch: {sorted(unknown)}")
    missing = expected - set(actual)
    if missing:
        raise ValueError(f"missing regional policy in batch: {sorted(missing)}")
    allowed_evidence = set(target_region_plan.evidence_ids)
    for policy in strategy.policies:
        if not set(policy.evidence_ids) <= allowed_evidence:
            raise ValueError(
                f"regional policy {policy.region_id} cites unknown evidence"
            )
    return strategy


class RegionalStrategyGenerationNode:
    """Request bounded regional policies from the real structured LLM port."""

    def __init__(
        self,
        llm: StructuredLLM[Any],
        *,
        model_id: str = "underwater-assistant-model",
        prompt_version: str = REGIONAL_STRATEGY_PROMPT_VERSION,
        temperature: float = 0.2,
        snapshot_provider: Callable[[str], PlanningSnapshot] | None = None,
        uuv_only: bool = False,
    ) -> None:
        self._llm = llm
        self._model_id = model_id
        self._prompt_version = prompt_version
        self._temperature = temperature
        self._snapshot_provider = snapshot_provider
        self._uuv_only = uuv_only

    def build_payload(
        self,
        snapshot: PlanningSnapshot,
        target_region_plan: TargetRegionPlan,
        intents: Mapping[str, IntentHypothesis],
        *,
        cells: Sequence[Any] | None = None,
        batch_index: int | None = None,
        batch_count: int | None = None,
        uuv_only: bool | None = None,
    ) -> dict[str, object]:
        if self._uuv_only if uuv_only is None else uuv_only:
            selected_cells = tuple(
                target_region_plan.cells if cells is None else cells
            )
            return self.build_uuv_payload(
                snapshot,
                tuple(_cell_to_mission_candidate(cell) for cell in selected_cells),
                intents,
                target_id=target_region_plan.target_id,
                batch_index=batch_index,
                batch_count=batch_count,
            )
        intent = intents.get(target_region_plan.target_id)
        evidence_ids = set(target_region_plan.evidence_ids)
        if intent is not None:
            evidence_ids.update(intent.evidence_ids)
        platform_candidates = _platform_candidates(snapshot)
        selected_cells = tuple(target_region_plan.cells if cells is None else cells)
        active_tasks = (
            snapshot.active_plan.region_tasks.values()
            if snapshot.active_plan is not None
            else ()
        )
        regional_effects = [
            {
                "region_id": task.region_id,
                "assignment_status": task.assignment_status,
                "degraded_reasons": sorted(task.degraded_reasons),
                "plan_revision": task.plan_revision,
            }
            for task in sorted(active_tasks, key=lambda item: item.region_id)
            if task.target_id == target_region_plan.target_id
        ]
        expert_feedback = [
            {
                "directive_id": directive.directive_id,
                "region_ids": sorted(directive.feedback_region_ids),
                "feedback": directive.feedback_text or directive.raw_text,
            }
            for directive in snapshot.applied_directives
            if (
                directive.directive_type == "feedback"
                and target_region_plan.target_id in directive.target_scope
            )
        ]
        payload: dict[str, object] = {
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
            "regions": [self._region_payload(cell) for cell in selected_cells],
            "operational_constraints": {
                "allowed_tracking_modes": [
                    "heuristic_uuv",
                ],
                "require_uuv_per_region": target_region_plan.grid_spec.require_uuv_per_region,
                "passive_sonar_required": True,
            },
            "platform_candidates": platform_candidates,
            "regional_context": {
                "snapshot_revision": snapshot.snapshot_revision,
                "target_plan_revision": target_region_plan.plan_revision,
                "prediction_id": target_region_plan.prediction_id,
                "previous_region_effects": regional_effects,
                "expert_feedback": expert_feedback,
            },
            "evidence_ids": sorted(evidence_ids),
        }
        if batch_index is not None and batch_count is not None:
            payload["region_batch"] = {
                "index": batch_index,
                "count": batch_count,
                "region_ids": [cell.region_id for cell in selected_cells],
            }
        return payload

    def build_uuv_payload(
        self,
        snapshot: PlanningSnapshot,
        candidate_regions: Sequence[RegionalMissionCandidate | CandidateRegion],
        intents: Mapping[str, IntentHypothesis],
        *,
        target_id: str | None = None,
        available_uuv_ids: AvailableUUVs | None = None,
        batch_index: int | None = None,
        batch_count: int | None = None,
    ) -> dict[str, object]:
        """Build a candidate-only payload for a UUV regional strategy call."""
        candidates = tuple(
            candidate
            if isinstance(candidate, RegionalMissionCandidate)
            else candidate_region_to_mission_candidate(candidate)
            for candidate in candidate_regions
        )
        if not candidates:
            raise ValueError("UUV regional strategy requires candidate regions")
        resolved_target_id = target_id or _target_id_from_candidate(candidates[0])
        intent = intents.get(resolved_target_id)
        platform_candidates = _platform_candidates(snapshot)
        if available_uuv_ids is not None:
            available_ids = (
                set(available_uuv_ids)
                if isinstance(available_uuv_ids, Mapping)
                else set(available_uuv_ids)
            )
            known_ids = {str(item["platform_id"]) for item in platform_candidates}
            for platform_id in sorted(available_ids - known_ids):
                platform_candidates.append(
                    {
                        "platform_id": platform_id,
                        "kind": "uuv",
                    }
                )
            platform_candidates.sort(key=lambda item: str(item["platform_id"]))
        evidence_ids: set[str] = set()
        if intent is not None:
            evidence_ids.update(intent.evidence_ids)
        candidate_payloads = [self._candidate_payload(candidate) for candidate in candidates]
        payload: dict[str, object] = {
            "model": self._model_id,
            "temperature": self._temperature,
            "system_prompt": UUV_REGIONAL_STRATEGY_SYSTEM_PROMPT,
            "scenario_id": snapshot.scenario_id,
            "sim_time_s": snapshot.sim_time_s,
            "target_id": resolved_target_id,
            "intent": (
                {
                    "label": intent.label,
                    "confidence": intent.confidence,
                    "evidence_ids": sorted(intent.evidence_ids),
                }
                if intent is not None
                else None
            ),
            "candidate_regions": candidate_payloads,
            "operational_constraints": {
                "allowed_tracking_modes": [
                    "active_scan",
                    "passive_track",
                    "handoff_reserve",
                ],
                "passive_sonar_required": True,
                "candidate_geometry_locked": True,
            },
            "platform_candidates": platform_candidates,
            "regional_context": {
                "snapshot_revision": snapshot.snapshot_revision,
                "candidate_ids": [candidate.candidate_id for candidate in candidates],
            },
            "evidence_ids": sorted(evidence_ids),
        }
        if batch_index is not None and batch_count is not None:
            payload["candidate_batch"] = {
                "index": batch_index,
                "count": batch_count,
                "candidate_ids": [candidate.candidate_id for candidate in candidates],
            }
        return payload

    def invoke_for_candidates(
        self,
        snapshot: PlanningSnapshot,
        candidate_regions: Sequence[RegionalMissionCandidate | CandidateRegion],
        intents: Mapping[str, IntentHypothesis],
        *,
        available_uuv_ids: AvailableUUVs,
    ) -> ValidatedRegionalStrategy:
        """Invoke the UUV-only LLM port and validate its candidate output."""
        payload = self.build_uuv_payload(
            snapshot,
            candidate_regions,
            intents,
            available_uuv_ids=available_uuv_ids,
        )
        strategy = self._invoke_uuv(payload)
        return validate_uuv_strategy(candidate_regions, strategy, available_uuv_ids)

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
        if self._uuv_only:
            return self._call_uuv_only(state, snapshot)
        plans = state.get("regional_plans", {})
        policies: dict[str, RegionalStrategySet | UUVRegionalStrategySet] = {}
        provenance: dict[str, LLMCallMetadata] = {}
        for target_id, target_plan in sorted(plans.items()):
            cells = tuple(target_plan.cells)
            batches = tuple(
                cells[index : index + _REGIONS_PER_LLM_REQUEST]
                for index in range(0, len(cells), _REGIONS_PER_LLM_REQUEST)
            ) or ((),)
            batch_payloads: list[dict[str, object]] = []
            merged_policies: list[RegionalPolicy] = []
            for batch_index, batch in enumerate(batches):
                payload = self.build_payload(
                    snapshot,
                    target_plan,
                    state.get("intent_hypotheses", {}),
                    cells=batch,
                    batch_index=batch_index if len(batches) > 1 else None,
                    batch_count=len(batches) if len(batches) > 1 else None,
                )
                strategy = validate_regional_strategy_batch(
                    target_plan,
                    self._invoke(payload),
                    tuple(cell.region_id for cell in batch),
                )
                batch_payloads.append(payload)
                merged_policies.extend(strategy.policies)
            strategy = validate_regional_strategy(
                target_plan,
                RegionalStrategySet(policies=tuple(merged_policies)),
            )
            policies[target_id] = strategy
            provenance[f"regional_strategy:{target_id}"] = LLMCallMetadata(
                operation="regional_strategy",
                model=self._model_id,
                prompt_version=self._prompt_version,
                request_hash=canonical_digest(batch_payloads),
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
            return cast(
                RegionalStrategySet,
                self._llm.invoke_structured(
                    "regional_strategy",
                    payload,
                    RegionalStrategySet,
                    prompt_version=self._prompt_version,
                ),
            )
        except LLMContentError as exc:
            correction_payload = {**payload, "correction_feedback": str(exc)}
            return cast(
                RegionalStrategySet,
                self._llm.invoke_structured(
                    "regional_strategy",
                    correction_payload,
                    RegionalStrategySet,
                    prompt_version=self._prompt_version,
                ),
            )

    def _invoke_uuv(self, payload: dict[str, object]) -> UUVRegionalStrategySet:
        try:
            return cast(
                UUVRegionalStrategySet,
                self._llm.invoke_structured(
                    "regional_strategy",
                    payload,
                    UUVRegionalStrategySet,
                    prompt_version=UUV_REGIONAL_STRATEGY_PROMPT_VERSION,
                ),
            )
        except LLMContentError as exc:
            correction_payload = {**payload, "correction_feedback": str(exc)}
            return cast(
                UUVRegionalStrategySet,
                self._llm.invoke_structured(
                    "regional_strategy",
                    correction_payload,
                    UUVRegionalStrategySet,
                    prompt_version=UUV_REGIONAL_STRATEGY_PROMPT_VERSION,
                ),
            )

    def _call_uuv_only(
        self,
        state: CarrierState,
        snapshot: PlanningSnapshot,
    ) -> CarrierState:
        candidate_map = state.get("regional_candidates") or {}
        if not candidate_map:
            candidate_map = {
                target_id: tuple(_cell_to_mission_candidate(cell) for cell in plan.cells)
                for target_id, plan in (state.get("regional_plans") or {}).items()
            }
        resources = _uuv_platform_resources(snapshot)
        policies: dict[str, RegionalStrategySet | UUVRegionalStrategySet] = {}
        provenance: dict[str, LLMCallMetadata] = {}
        for target_id, candidates in sorted(candidate_map.items()):
            normalized_candidates = tuple(candidates)
            batches = tuple(
                normalized_candidates[index : index + _REGIONS_PER_LLM_REQUEST]
                for index in range(0, len(normalized_candidates), _REGIONS_PER_LLM_REQUEST)
            ) or ((),)
            batch_payloads: list[dict[str, object]] = []
            merged_policies: list[UUVRegionalPolicy] = []
            for batch_index, batch in enumerate(batches):
                payload = self.build_uuv_payload(
                    snapshot,
                    batch,
                    state.get("intent_hypotheses", {}),
                    target_id=target_id,
                    available_uuv_ids=resources,
                    batch_index=batch_index if len(batches) > 1 else None,
                    batch_count=len(batches) if len(batches) > 1 else None,
                )
                strategy = validate_uuv_strategy(
                    batch, self._invoke_uuv(payload), resources
                )
                batch_payloads.append(payload)
                merged_policies.extend(strategy.policies)
            # Re-run the validator over the merged response so a UUV selected
            # in two separate LLM batches is still rejected deterministically.
            strategy = validate_uuv_strategy(
                normalized_candidates,
                UUVRegionalStrategySet(policies=tuple(merged_policies)),
                resources,
            )
            policies[target_id] = strategy
            provenance[f"regional_strategy:{target_id}"] = LLMCallMetadata(
                operation="regional_strategy",
                model=self._model_id,
                prompt_version=UUV_REGIONAL_STRATEGY_PROMPT_VERSION,
                request_hash=canonical_digest(batch_payloads),
                response_hash=canonical_digest(strategy.model_dump(mode="json")),
                sim_time_s=snapshot.sim_time_s,
                scenario_id=snapshot.scenario_id,
            )
        return {
            "regional_policies": policies,
            "llm_provenance": {**state.get("llm_provenance", {}), **provenance},
        }

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

    @staticmethod
    def _candidate_payload(candidate: RegionalMissionCandidate) -> dict[str, object]:
        return {
            "candidate_id": candidate.candidate_id,
            "cell_ids": list(candidate.cell_ids),
            "time_window": candidate.time_window.model_dump(mode="json"),
            "perimeter_points": [list(point) for point in candidate.perimeter_points],
            "predecessor_candidate_ids": list(candidate.predecessor_candidate_ids),
            "successor_candidate_ids": list(candidate.successor_candidate_ids),
        }


def _platform_candidates(
    snapshot: PlanningSnapshot,
) -> list[dict[str, object]]:
    """Expose live platform capabilities needed for LLM regional grouping."""
    situation = getattr(snapshot, "situation", None)
    platform_snapshot = getattr(situation, "platform_snapshot", None)
    if platform_snapshot is None:
        return []
    candidates: list[dict[str, object]] = []
    platforms = platform_snapshot.roster.uuvs
    for platform in platforms:
        candidates.append(
            {
                "platform_id": platform.platform_id,
                "kind": (
                    platform.capability.kind.value
                    if hasattr(platform.capability.kind, "value")
                    else str(platform.capability.kind)
                ),
                "deployment_state": platform.deployment_state,
                "energy_fraction": platform.energy_fraction,
                "speed_mps": platform.speed_mps,
                "max_speed_mps": platform.capability.motion.max_speed_mps,
                "passive_range_m": platform.capability.sonar.passive_range_m,
                "active_capable": platform.capability.sonar.active_capable,
                "acoustic_range_m": platform.capability.communications.acoustic_range_m,
                "surface_range_m": platform.capability.communications.surface_range_m,
            }
        )
    return sorted(candidates, key=lambda item: str(item["platform_id"]))


def _uuv_platform_resources(snapshot: PlanningSnapshot) -> dict[str, object]:
    situation = getattr(snapshot, "situation", None)
    platform_snapshot = getattr(situation, "platform_snapshot", None)
    if platform_snapshot is None:
        return {}
    return {
        platform.platform_id: platform
        for platform in platform_snapshot.roster.uuvs
    }


def _cell_to_mission_candidate(cell: Any) -> RegionalMissionCandidate:
    return RegionalMissionCandidate(
        candidate_id=cell.region_id,
        cell_ids=(cell.region_id,),
        time_window=TimeWindow(
            start_s=cell.first_entry_s,
            end_s=max(cell.first_entry_s + 1, cell.last_exit_s),
        ),
        perimeter_points=tuple(
            sorted(
                (
                    (cell.min_x, cell.min_y),
                    (cell.min_x, cell.max_y),
                    (cell.max_x, cell.min_y),
                    (cell.max_x, cell.max_y),
                )
            )
        ),
        predecessor_candidate_ids=tuple(cell.predecessor_region_ids),
        successor_candidate_ids=tuple(cell.successor_region_ids),
    )


def _target_id_from_candidate(candidate: RegionalMissionCandidate) -> str:
    marker = ":r"
    if marker in candidate.candidate_id:
        return candidate.candidate_id.split(marker, 1)[0]
    return candidate.candidate_id.split(":", 1)[0]
