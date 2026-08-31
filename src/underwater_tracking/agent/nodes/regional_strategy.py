from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, cast

from underwater_tracking.agent.llm import LLMCallMetadata, LLMContentError, StructuredLLM
from underwater_tracking.agent.nodes.snapshot import PlanningSnapshot
from underwater_tracking.agent.prompts import (
    EXECUTION_STRATEGY_PROMPT_VERSION,
    REGIONAL_STRATEGY_PROMPT_VERSION,
    REGIONAL_STRATEGY_SYSTEM_PROMPT,
    UUV_REGIONAL_STRATEGY_PROMPT_VERSION,
    UUV_REGIONAL_STRATEGY_SYSTEM_PROMPT,
    canonical_digest,
)
from underwater_tracking.agent.state import CarrierState
from underwater_tracking.domain.agent_models import IntentHypothesis
from underwater_tracking.domain.models import ContactClassification
from underwater_tracking.domain.regional_models import (
    RegionalMissionCandidate,
    RegionalPolicy,
    RegionalStrategySet,
    TargetRegionPlan,
    TimeWindow,
    StrategyValidationReport,
    UUVRegionalPolicyDecision,
    UUVRegionalStrategyDecisionSet,
    UUVRegionalStrategySet,
)
from underwater_tracking.planning.candidate_regions import (
    CandidateRegion,
    candidate_region_to_mission_candidate,
)
from underwater_tracking.planning.region_cap import MAX_EXECUTABLE_REGIONS_PER_TARGET
from underwater_tracking.planning.plan_stability import rectangle_iou
from underwater_tracking.planning.regional_plan_validator import (
    AvailableUUVs,
    RegionalPlanError,
    RegionalSemanticRejection,
    ValidatedRegionalStrategy,
    resolve_uuv_strategy,
    validate_uuv_decision_batch,
)
from underwater_tracking.planning.execution_strategy import ExecutionStrategyRevisionNode

# A regional policy carries several nested objects and LongCat may include
# reasoning tokens in the same completion budget. Keep each response small
# enough to finish as valid JSON under the configured 4096-token limit.
_REGIONS_PER_LLM_REQUEST = 4
_UUV_REGIONS_PER_LLM_REQUEST = 2
_UUV_PROVIDER_CANDIDATE_CAP = MAX_EXECUTABLE_REGIONS_PER_TARGET


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
        batch_size: int = _REGIONS_PER_LLM_REQUEST,
        max_concurrency: int = 3,
        semantic_correction_attempts: int = 1,
    ) -> None:
        if not 1 <= batch_size <= _REGIONS_PER_LLM_REQUEST:
            raise ValueError(
                f"regional batch_size must be between 1 and {_REGIONS_PER_LLM_REQUEST}"
            )
        if not 1 <= max_concurrency <= 3:
            raise ValueError("regional max_concurrency must be between 1 and 3")
        if not 0 <= semantic_correction_attempts <= 1:
            raise ValueError("semantic_correction_attempts must be 0 or 1")
        self._llm = llm
        self._model_id = model_id
        self._prompt_version = prompt_version
        self._temperature = temperature
        self._snapshot_provider = snapshot_provider
        self._uuv_only = uuv_only
        self._batch_size = batch_size
        # The transport concurrency cap is independent from the provider input
        # cap. The latter bounds bootstrap latency to the executable mission
        # surface while the complete deterministic region graph stays in state.
        self._max_concurrency = max_concurrency
        self._semantic_correction_attempts = semantic_correction_attempts
        self._execution_strategy = ExecutionStrategyRevisionNode(
            llm,
            model_id=model_id,
            prompt_version=EXECUTION_STRATEGY_PROMPT_VERSION,
        )

    @property
    def execution_strategy(self) -> ExecutionStrategyRevisionNode:
        """Expose the constrained semantic revision port for new execution runs."""

        return self._execution_strategy

    def build_execution_strategy_payload(self, **kwargs: object) -> dict[str, object]:
        """Build a payload that contains no geometry or resource assignments."""

        return self._execution_strategy.build_payload(**kwargs)  # type: ignore[arg-type]

    def invoke_execution_strategy(self, **kwargs: object) -> StrategyValidationReport:
        """Validate one semantic revision without touching legacy regional plans."""

        return self._execution_strategy.revise(**kwargs)  # type: ignore[arg-type]

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
            "output_token_budget": 2048,
            "thinking_mode": "disabled",
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
                    "alternatives": dict(sorted(intent.alternatives.items())),
                    "ranked_motives": [
                        motive.model_dump(mode="json")
                        for motive in intent.ranked_motives
                    ],
                    "planning_effects": list(intent.planning_effects),
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
            known_platforms = {
                str(item["platform_id"]): item for item in platform_candidates
            }
            platform_candidates = [
                known_platforms.get(
                    platform_id,
                    {"platform_id": platform_id, "kind": "uuv"},
                )
                for platform_id in sorted(available_ids)
            ]
        evidence_ids: set[str] = set()
        if intent is not None:
            evidence_ids.update(intent.evidence_ids)
        candidate_payloads = [self._candidate_payload(candidate) for candidate in candidates]
        payload: dict[str, object] = {
            "model": self._model_id,
            "temperature": self._temperature,
            "output_token_budget": 1024,
            "thinking_mode": "disabled",
            "system_prompt": UUV_REGIONAL_STRATEGY_SYSTEM_PROMPT,
            "scenario_id": snapshot.scenario_id,
            "sim_time_s": snapshot.sim_time_s,
            "target_id": resolved_target_id,
            "intent": (
                {
                    "label": intent.label,
                    "confidence": intent.confidence,
                    "evidence_ids": sorted(intent.evidence_ids),
                    "alternatives": dict(sorted(intent.alternatives.items())),
                    "ranked_motives": [
                        motive.model_dump(mode="json")
                        for motive in intent.ranked_motives
                    ],
                    "planning_effects": list(intent.planning_effects),
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
                "provider_candidate_cap": _UUV_PROVIDER_CANDIDATE_CAP,
                "provider_batch_cap": _UUV_REGIONS_PER_LLM_REQUEST,
                "executable_region_cap": MAX_EXECUTABLE_REGIONS_PER_TARGET,
                "resource_allocation": "deterministic_mission_optimizer",
            },
            "platform_candidates": platform_candidates,
            "regional_context": {
                "snapshot_revision": snapshot.snapshot_revision,
                "candidate_ids": [candidate.candidate_id for candidate in candidates],
                "rolling_change_control": _rolling_change_control(
                    snapshot, resolved_target_id, candidates
                ),
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
        decisions = self._validate_uuv_batch(
            candidate_regions,
            payload,
            self._invoke_uuv(payload),
            available_uuv_ids,
            require_active_scan=True,
        )
        return resolve_uuv_strategy(candidate_regions, decisions, available_uuv_ids)

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
                cells[index : index + self._batch_size]
                for index in range(0, len(cells), self._batch_size)
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

    def _invoke(
        self,
        payload: dict[str, object],
        *,
        correction_attempts: int | None = None,
    ) -> RegionalStrategySet:
        remaining_attempts = (
            self._semantic_correction_attempts
            if correction_attempts is None
            else correction_attempts
        )
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
            if remaining_attempts <= 0:
                raise
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

    def _invoke_uuv(
        self,
        payload: dict[str, object],
        *,
        correction_attempts: int | None = None,
    ) -> UUVRegionalStrategyDecisionSet:
        remaining_attempts = (
            self._semantic_correction_attempts
            if correction_attempts is None
            else correction_attempts
        )
        try:
            response = self._llm.invoke_structured(
                "regional_strategy",
                payload,
                UUVRegionalStrategyDecisionSet,
                prompt_version=UUV_REGIONAL_STRATEGY_PROMPT_VERSION,
            )
            decisions = self._coerce_uuv_decision_set(response)
            if _requires_rolling_reflection(payload):
                reflection_payload = {
                    **payload,
                    "rolling_reflection": {
                        "draft_policies": decisions.model_dump(mode="json"),
                        "instruction": (
                            "Critique the draft against rolling_change_control, "
                            "candidate time windows, tracking completion, and UUV "
                            "availability. Return the minimally changed final policy "
                            "set using the same strict schema."
                        ),
                    },
                }
                response = self._llm.invoke_structured(
                    "regional_strategy",
                    reflection_payload,
                    UUVRegionalStrategyDecisionSet,
                    prompt_version=UUV_REGIONAL_STRATEGY_PROMPT_VERSION,
                )
                return self._coerce_uuv_decision_set(response)
            return decisions
        except LLMContentError as exc:
            if remaining_attempts <= 0:
                raise
            correction_payload = {**payload, "correction_feedback": str(exc)}
            response = self._llm.invoke_structured(
                "regional_strategy",
                correction_payload,
                UUVRegionalStrategyDecisionSet,
                prompt_version=UUV_REGIONAL_STRATEGY_PROMPT_VERSION,
            )
            return self._coerce_uuv_decision_set(response)

    @staticmethod
    def _coerce_uuv_decision_set(response: Any) -> UUVRegionalStrategyDecisionSet:
        """Accept old persisted response objects outside the live schema."""
        if isinstance(response, UUVRegionalStrategyDecisionSet):
            return response
        if isinstance(response, UUVRegionalStrategySet):
            return UUVRegionalStrategyDecisionSet(
                policies=tuple(
                    UUVRegionalPolicyDecision.model_validate(
                        policy.model_dump(
                            mode="json",
                            exclude={"predecessor_candidate_id", "successor_candidate_id"},
                        )
                    )
                    for policy in response.policies
                )
            )
        return UUVRegionalStrategyDecisionSet.model_validate(response)

    def _validate_uuv_batch(
        self,
        batch: Sequence[RegionalMissionCandidate | CandidateRegion],
        payload: dict[str, object],
        decisions: UUVRegionalStrategyDecisionSet,
        resources: AvailableUUVs,
        *,
        require_active_scan: bool = False,
    ) -> UUVRegionalStrategyDecisionSet:
        try:
            return validate_uuv_decision_batch(
                batch,
                decisions,
                resources,
                require_active_scan=require_active_scan,
            )
        except RegionalPlanError as error:
            if self._semantic_correction_attempts <= 0:
                raise RegionalSemanticRejection(
                    f"regional semantic correction disabled: {error}"
                ) from error
            correction_payload = {
                **payload,
                "correction_feedback": {
                    "category": "semantic",
                    "message": str(error),
                    "allowed_candidate_ids": [candidate.candidate_id for candidate in batch],
                },
            }
            corrected = self._invoke_uuv(correction_payload, correction_attempts=0)
            try:
                return validate_uuv_decision_batch(
                    batch,
                    corrected,
                    resources,
                    require_active_scan=require_active_scan,
                )
            except RegionalPlanError as second_error:
                raise RegionalSemanticRejection(
                    f"bounded regional semantic correction rejected: {second_error}"
                ) from second_error

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
            normalized_candidates = _select_uuv_provider_candidates(
                tuple(candidates), snapshot=snapshot, target_id=target_id
            )
            uuv_batch_size = min(self._batch_size, _UUV_REGIONS_PER_LLM_REQUEST)
            batches = tuple(
                normalized_candidates[index : index + uuv_batch_size]
                for index in range(0, len(normalized_candidates), uuv_batch_size)
            ) or ((),)
            batch_payloads: list[dict[str, object]] = []
            merged_decisions: list[UUVRegionalPolicyDecision] = []
            batch_results = self._run_uuv_batches(
                snapshot,
                target_id,
                batches,
                state.get("intent_hypotheses", {}),
                resources,
            )
            for payload, decisions in batch_results:
                batch_payloads.append(payload)
                merged_decisions.extend(decisions.policies)
            # Resolve only after the complete candidate graph is available, so
            # cross-batch predecessor/successor links remain valid.
            strategy = resolve_uuv_strategy(
                normalized_candidates,
                UUVRegionalStrategyDecisionSet(policies=tuple(merged_decisions)),
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

    def _run_uuv_batches(
        self,
        snapshot: PlanningSnapshot,
        target_id: str,
        batches: Sequence[Sequence[RegionalMissionCandidate]],
        intents: Mapping[str, IntentHypothesis],
        resources: AvailableUUVs,
    ) -> tuple[
        tuple[dict[str, object], UUVRegionalStrategyDecisionSet],
        ...,
    ]:
        """Run bounded regional calls and return results in batch order.

        LongCat latency is high enough that serializing seven candidate
        batches can consume the bootstrap deadline before a plan can commit.
        Futures are collected by completion, but the returned tuple is sorted
        by the immutable batch index so request/response hashes and resolved
        policy order remain replayable.
        """
        batch_count = len(batches)
        resource_batches = _partition_uuv_resources(resources, batch_count)

        def run_batch(
            batch_index: int,
            batch: Sequence[RegionalMissionCandidate],
        ) -> tuple[dict[str, object], UUVRegionalStrategyDecisionSet]:
            batch_resources = resource_batches[batch_index]
            payload = self.build_uuv_payload(
                snapshot,
                batch,
                intents,
                target_id=target_id,
                available_uuv_ids=batch_resources,
                batch_index=batch_index if batch_count > 1 else None,
                batch_count=batch_count if batch_count > 1 else None,
            )
            decisions = self._validate_uuv_batch(
                batch,
                payload,
                self._invoke_uuv(payload),
                batch_resources,
                require_active_scan=batch_index == 0,
            )
            return payload, decisions

        if batch_count == 1:
            return (run_batch(0, batches[0]),)

        executor = ThreadPoolExecutor(
            max_workers=min(self._max_concurrency, batch_count),
            thread_name_prefix="regional-strategy",
        )
        futures: dict[Future[tuple[dict[str, object], UUVRegionalStrategyDecisionSet]], int] = {}
        try:
            for batch_index, batch in enumerate(batches):
                future = executor.submit(run_batch, batch_index, batch)
                futures[future] = batch_index
            completed: dict[
                int, tuple[dict[str, object], UUVRegionalStrategyDecisionSet]
            ] = {}
            for future in as_completed(futures):
                completed[futures[future]] = future.result()
            return tuple(completed[index] for index in range(batch_count))
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

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
            "top_left_xy": list(candidate.top_left_xy),
            "bottom_right_xy": list(candidate.bottom_right_xy),
            "predecessor_candidate_ids": list(candidate.predecessor_candidate_ids),
            "successor_candidate_ids": list(candidate.successor_candidate_ids),
        }


def _rolling_change_control(
    snapshot: PlanningSnapshot,
    target_id: str,
    candidates: Sequence[RegionalMissionCandidate],
) -> dict[str, object]:
    """Expose IoU and UUV reassignment costs for rolling LLM decisions."""
    active_plan = snapshot.active_plan
    previous_plan = (
        None
        if active_plan is None
        else getattr(active_plan, "regional_plans", {}).get(target_id)
    )
    previous_regions = () if previous_plan is None else previous_plan.task_regions
    previous_bounds = {
        region.region_id: (
            region.lower_left_xy[0],
            region.upper_right_xy[0],
            region.lower_left_xy[1],
            region.upper_right_xy[1],
        )
        for region in previous_regions
    }
    assignments = {} if active_plan is None else getattr(active_plan, "region_tasks", {})
    comparisons: list[dict[str, object]] = []
    for candidate in candidates:
        xs, ys = zip(*candidate.perimeter_points, strict=True)
        bounds = (min(xs), max(xs), min(ys), max(ys))
        best_region_id, best_iou = max(
            (
                (region_id, rectangle_iou(bounds, previous))
                for region_id, previous in previous_bounds.items()
            ),
            key=lambda item: (item[1], item[0]),
            default=(None, 0.0),
        )
        prior_region = next(
            (region for region in previous_regions if region.region_id == best_region_id),
            None,
        )
        prior_uuv_ids = (
            []
            if prior_region is None
            else sorted(
                {
                    uuv_id
                    for cell_id in prior_region.cell_ids
                    for uuv_id in getattr(assignments.get(cell_id), "assigned_uuv_ids", ())
                }
            )
        )
        comparisons.append(
            {
                "candidate_id": candidate.candidate_id,
                "best_previous_region_id": best_region_id,
                "iou_with_previous": round(best_iou, 6),
                "previous_assigned_uuv_ids": prior_uuv_ids,
            }
        )
    return {
        "objective": "minimize_region_and_uuv_reassignment_subject_to_tracking_completion",
        "iou_retention_threshold": 0.6,
        "candidate_comparisons": comparisons,
    }


def _requires_rolling_reflection(payload: Mapping[str, object]) -> bool:
    """Run the bounded revision pass only when an old region can be retained."""
    context = payload.get("regional_context")
    if not isinstance(context, Mapping):
        return False
    change_control = context.get("rolling_change_control")
    if not isinstance(change_control, Mapping):
        return False
    comparisons = change_control.get("candidate_comparisons")
    return isinstance(comparisons, list) and any(
        isinstance(item, Mapping) and item.get("best_previous_region_id") is not None
        for item in comparisons
    )


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


def _partition_uuv_resources(
    resources: AvailableUUVs,
    partition_count: int,
) -> tuple[AvailableUUVs, ...]:
    """Give concurrent LLM batches deterministic, non-overlapping UUV pools."""
    if partition_count <= 0:
        raise ValueError("UUV resource partition count must be positive")
    if isinstance(resources, Mapping):
        ordered_ids = sorted(
            resources,
            key=lambda platform_id: (
                not _resource_active_capable(resources[platform_id]),
                str(platform_id),
            ),
        )
        buckets: list[dict[str, object]] = [dict() for _ in range(partition_count)]
        for index, platform_id in enumerate(ordered_ids):
            buckets[index % partition_count][platform_id] = resources[platform_id]
        return tuple(buckets)
    ordered_ids = sorted(set(resources))
    id_buckets: list[list[str]] = [[] for _ in range(partition_count)]
    for index, platform_id in enumerate(ordered_ids):
        id_buckets[index % partition_count].append(platform_id)
    return tuple(tuple(bucket) for bucket in id_buckets)


def _resource_active_capable(resource: Any) -> bool:
    if isinstance(resource, Mapping):
        if "active_capable" in resource:
            return bool(resource["active_capable"])
        capability = resource.get("capability")
        if isinstance(capability, Mapping):
            sonar = capability.get("sonar")
            if isinstance(sonar, Mapping) and "active_capable" in sonar:
                return bool(sonar["active_capable"])
        return True
    capability = getattr(resource, "capability", None)
    sonar = getattr(capability, "sonar", None)
    active_capable = getattr(sonar, "active_capable", None)
    return True if active_capable is None else bool(active_capable)


def _select_uuv_provider_candidates(
    candidates: Sequence[RegionalMissionCandidate],
    *,
    snapshot: Any,
    target_id: str,
) -> tuple[RegionalMissionCandidate, ...]:
    """Bound provider input while retaining the complete planner candidate set.

    The deterministic optimizer already limits one target to four executable
    regions. Sending the full prediction corridor to a real provider can turn
    that same limit into dozens of sequential semantic requests. Protect the
    currently active regions, then choose the nearest time-window neighbors;
    the full candidate graph remains available to the optimizer and audit.
    """
    ordered = tuple(
        sorted(candidates, key=lambda item: (item.time_window.start_s, item.candidate_id))
    )
    if len(ordered) <= _UUV_PROVIDER_CANDIDATE_CAP:
        return ordered

    active_plan = getattr(snapshot, "active_plan", None)
    active_tasks = getattr(active_plan, "region_tasks", {}) if active_plan else {}
    candidate_ids = {candidate.candidate_id for candidate in ordered}
    protected_ids = frozenset(
        task.region_id
        for task in active_tasks.values()
        if getattr(task, "target_id", target_id) == target_id
        and getattr(task, "assignment_status", "") in {"active", "degraded"}
        and task.region_id in candidate_ids
    )
    if len(protected_ids) > _UUV_PROVIDER_CANDIDATE_CAP:
        raise RegionalPlanError(
            f"active UUV regions for target {target_id!r} exceed the provider cap"
        )

    selected: list[RegionalMissionCandidate] = []
    selected_ids: set[str] = set()

    if protected_ids:
        # Keep every active/degraded physical task. Future candidates are
        # added after the latest protected task; if the horizon ends there,
        # fill the remaining slots immediately before the earliest one.
        selected.extend(
            candidate
            for candidate in ordered
            if candidate.candidate_id in protected_ids
        )
        selected_ids.update(protected_ids)
        anchor = selected[-1]
    else:
        anchor = _provider_anchor_candidate(ordered, snapshot, target_id)
        selected.append(anchor)
        selected_ids.add(anchor.candidate_id)

    while len(selected) < _UUV_PROVIDER_CANDIDATE_CAP:
        following = tuple(
            candidate
            for candidate in ordered
            if candidate.candidate_id not in selected_ids
            and candidate.time_window.start_s > anchor.time_window.start_s
        )
        if not following:
            break
        next_candidate = min(
            following,
            key=lambda candidate: (
                _candidate_center_distance_squared(candidate, anchor),
                candidate.time_window.start_s,
                candidate.candidate_id,
            ),
        )
        selected.append(next_candidate)
        selected_ids.add(next_candidate.candidate_id)
        anchor = next_candidate

    if len(selected) < _UUV_PROVIDER_CANDIDATE_CAP:
        earliest = min(selected, key=_candidate_temporal_key)
        preceding = tuple(
            candidate
            for candidate in ordered
            if candidate.candidate_id not in selected_ids
            and candidate.time_window.start_s < earliest.time_window.start_s
        )
        while len(selected) < _UUV_PROVIDER_CANDIDATE_CAP and preceding:
            previous = min(
                preceding,
                key=lambda candidate: (
                    _candidate_center_distance_squared(candidate, earliest),
                    -candidate.time_window.start_s,
                    candidate.candidate_id,
                ),
            )
            selected.append(previous)
            selected_ids.add(previous.candidate_id)
            preceding = tuple(
                candidate
                for candidate in preceding
                if candidate.candidate_id != previous.candidate_id
            )
            earliest = previous

    if len(selected) < _UUV_PROVIDER_CANDIDATE_CAP:
        # A single time window may legitimately contain parallel coverage
        # alternatives. Keep the provider view bounded without inventing
        # same-time handoffs between those alternatives.
        selected_anchor = selected[-1]
        remaining = [
            candidate
            for candidate in ordered
            if candidate.candidate_id not in selected_ids
        ]
        for candidate in sorted(
            remaining,
            key=lambda item: (
                _candidate_center_distance_squared(item, selected_anchor),
                _candidate_temporal_key(item),
            ),
        ):
            if len(selected) >= _UUV_PROVIDER_CANDIDATE_CAP:
                break
            selected.append(candidate)
            selected_ids.add(candidate.candidate_id)

    # A clipped provider view must not contain arbitrary graph jumps. Rebuild
    # the public handoff chain from the selected time-ordered candidates so the
    # deterministic optimizer receives only adjacent, auditable transitions.
    selected = sorted(selected, key=_candidate_temporal_key)
    return tuple(
        candidate.model_copy(
            update={
                "predecessor_candidate_ids": (
                    (selected[index - 1].candidate_id,)
                    if index
                    and selected[index - 1].time_window.start_s
                    < candidate.time_window.start_s
                    else ()
                ),
                "successor_candidate_ids": (
                    (selected[index + 1].candidate_id,)
                    if index + 1 < len(selected)
                    and candidate.time_window.start_s
                    < selected[index + 1].time_window.start_s
                    else ()
                ),
            }
        )
        for index, candidate in enumerate(selected)
    )


def _provider_anchor_candidate(
    candidates: Sequence[RegionalMissionCandidate],
    snapshot: Any,
    target_id: str,
) -> RegionalMissionCandidate:
    """Select the current public window that starts the provider path."""
    situation = getattr(snapshot, "situation", snapshot)
    sim_time_s = int(
        getattr(situation, "sim_time_s", getattr(snapshot, "sim_time_s", 0))
    )
    current = tuple(
        candidate
        for candidate in candidates
        if candidate.time_window.start_s <= sim_time_s < candidate.time_window.end_s
    )
    pool = current or tuple(
        candidate
        for candidate in candidates
        if candidate.time_window.start_s
        == min(item.time_window.start_s for item in candidates)
    )
    public_point = _known_submarine_contact_point(snapshot, target_id)
    if public_point is None:
        public_point = _active_public_prior_point(snapshot, target_id)
    if public_point is None:
        return min(pool, key=_candidate_temporal_key)
    return min(
        pool,
        key=lambda candidate: (
            _candidate_public_distance(candidate, public_point),
            _candidate_center_distance_to_point_squared(candidate, public_point),
            candidate.candidate_id,
        ),
    )


def _candidate_temporal_key(
    candidate: RegionalMissionCandidate,
) -> tuple[int, str]:
    return candidate.time_window.start_s, candidate.candidate_id


def _candidate_center(candidate: RegionalMissionCandidate) -> tuple[float, float]:
    point_count = len(candidate.perimeter_points)
    return (
        sum(point[0] for point in candidate.perimeter_points) / point_count,
        sum(point[1] for point in candidate.perimeter_points) / point_count,
    )


def _candidate_center_distance_squared(
    left: RegionalMissionCandidate,
    right: RegionalMissionCandidate,
) -> float:
    left_x, left_y = _candidate_center(left)
    right_x, right_y = _candidate_center(right)
    return (left_x - right_x) ** 2 + (left_y - right_y) ** 2


def _candidate_center_distance_to_point_squared(
    candidate: RegionalMissionCandidate,
    point: tuple[float, float],
) -> float:
    center_x, center_y = _candidate_center(candidate)
    return (center_x - point[0]) ** 2 + (center_y - point[1]) ** 2


def _active_public_prior_point(
    snapshot: Any,
    target_id: str,
) -> tuple[float, float] | None:
    """Return the latest active public prior center for one target."""
    situation = getattr(snapshot, "situation", snapshot)
    sim_time_s = int(getattr(situation, "sim_time_s", getattr(snapshot, "sim_time_s", 0)))
    priors = tuple(getattr(situation, "target_search_priors", ()) or ())
    active_priors = tuple(
        prior
        for prior in priors
        if getattr(prior, "target_id", None) == target_id
        and int(getattr(prior, "issued_at_s", 0)) <= sim_time_s
        and sim_time_s < int(getattr(prior, "valid_until_s", 0))
    )
    if not active_priors:
        return None
    prior = max(
        active_priors,
        key=lambda item: (
            float(getattr(item, "confidence", 0.0)),
            int(getattr(item, "issued_at_s", 0)),
            str(getattr(item, "prior_id", "")),
        ),
    )
    center = getattr(prior, "center_xy", None)
    if center is None or len(center) < 2:
        return None
    return float(center[0]), float(center[1])


def _known_submarine_contact_point(
    snapshot: Any,
    target_id: str,
) -> tuple[float, float] | None:
    """Return the current position of an identified public submarine contact."""
    situation = getattr(snapshot, "situation", snapshot)
    contacts = tuple(getattr(situation, "contacts", ()) or ())
    matching = tuple(
        contact
        for contact in contacts
        if getattr(contact, "contact_id", None) == target_id
        and getattr(contact, "classification", None)
        is ContactClassification.SUBMARINE
        and getattr(contact, "estimated_position_xy", None) is not None
    )
    if not matching:
        return None
    contact = max(matching, key=lambda item: int(getattr(item, "sim_time_s", 0)))
    point = contact.estimated_position_xy
    return float(point[0]), float(point[1])


def _candidate_public_distance(
    candidate: RegionalMissionCandidate,
    point: tuple[float, float],
) -> float:
    """Compute distance from a public point to a candidate bounding box."""
    min_x = min(item[0] for item in candidate.perimeter_points)
    max_x = max(item[0] for item in candidate.perimeter_points)
    min_y = min(item[1] for item in candidate.perimeter_points)
    max_y = max(item[1] for item in candidate.perimeter_points)
    dx = max(min_x - point[0], 0.0, point[0] - max_x)
    dy = max(min_y - point[1], 0.0, point[1] - max_y)
    return dx * dx + dy * dy


def _cell_to_mission_candidate(cell: Any) -> RegionalMissionCandidate:
    return RegionalMissionCandidate(
        candidate_id=cell.region_id,
        cell_ids=(cell.region_id,),
        time_window=TimeWindow(
            start_s=cell.first_entry_s,
            end_s=max(cell.first_entry_s + 1, cell.last_exit_s),
        ),
        perimeter_points=tuple(
            (
                (cell.min_x, cell.min_y),
                (cell.max_x, cell.min_y),
                (cell.max_x, cell.max_y),
                (cell.min_x, cell.max_y),
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
