from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from underwater_tracking.domain.regional_models import (
    RegionalMissionCandidate,
    UUVRegionalPolicy,
    UUVRegionalPolicyDecision,
    UUVRegionalStrategyDecisionSet,
    UUVRegionalStrategySet,
)
from underwater_tracking.planning.candidate_regions import (
    CandidateRegion,
    candidate_region_to_mission_candidate,
)


class RegionalPlanError(ValueError):
    """Raised when an LLM regional candidate cannot be verified."""


class RegionalSemanticRejection(RegionalPlanError):
    """Raised after one bounded correction still fails semantic validation."""


class ValidatedRegionalStrategy(UUVRegionalStrategySet):
    """Marker type proving the strict candidate checks have completed."""


CandidateInput = RegionalMissionCandidate | CandidateRegion
AvailableUUVs = Iterable[str] | Mapping[str, Any]
DecisionInput = UUVRegionalStrategyDecisionSet | Mapping[str, object]


def validate_uuv_decision_batch(
    candidate_set: Sequence[CandidateInput],
    decisions: DecisionInput,
    available_uuv_ids: AvailableUUVs,
    *,
    require_active_scan: bool = False,
) -> UUVRegionalStrategyDecisionSet:
    """Validate one topology-free LLM response against its local batch.

    Candidate relationships are intentionally not checked here.  A response
    for a four-candidate batch cannot know whether a predecessor or successor
    lives in another batch; the complete candidate graph is checked by
    :func:`resolve_uuv_strategy` after all batches have been merged.
    """
    candidates = _normalize_candidates(candidate_set)
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    parsed = _parse_decision_set(decisions)
    expected_ids = set(by_id)
    actual_ids = [policy.candidate_id for policy in parsed.policies]
    if len(actual_ids) != len(set(actual_ids)):
        raise RegionalPlanError("duplicate regional policy in batch")
    unknown_ids = set(actual_ids) - expected_ids
    if unknown_ids:
        raise RegionalPlanError(f"unknown regional policy in batch: {sorted(unknown_ids)}")
    missing_ids = expected_ids - set(actual_ids)
    if missing_ids:
        raise RegionalPlanError(f"missing regional policy in batch: {sorted(missing_ids)}")

    available_ids, resources = _available_uuvs(available_uuv_ids)
    assigned_ids: set[str] = set()
    for policy in parsed.policies:
        _validate_decision_policy(policy, available_ids, resources)
        overlap = assigned_ids.intersection(policy.assigned_uuv_ids)
        if overlap:
            raise RegionalPlanError(f"overlapping UUV assignments: {sorted(overlap)}")
        assigned_ids.update(policy.assigned_uuv_ids)
    if require_active_scan and any(_active_capable(resource) for resource in resources.values()):
        if not any(policy.active_scan_uuv_count > 0 for policy in parsed.policies):
            raise RegionalPlanError(
                "current UUV window requires at least one active-scan allocation"
            )
    return parsed


def resolve_uuv_strategy(
    candidates: Sequence[CandidateInput],
    decisions: UUVRegionalStrategyDecisionSet | Mapping[str, object],
    available_uuvs: AvailableUUVs,
) -> ValidatedRegionalStrategy:
    """Resolve topology and policy order from the complete candidate graph."""
    normalized_candidates = _normalize_candidates(candidates)
    candidate_ids = {candidate.candidate_id for candidate in normalized_candidates}
    for candidate in normalized_candidates:
        for relation in (
            *candidate.predecessor_candidate_ids,
            *candidate.successor_candidate_ids,
        ):
            if relation not in candidate_ids:
                raise RegionalPlanError(
                    f"candidate {candidate.candidate_id} references unknown topology node {relation}"
                )

    parsed = validate_uuv_decision_batch(
        normalized_candidates,
        decisions,
        available_uuvs,
    )
    policies_by_id = {policy.candidate_id: policy for policy in parsed.policies}
    resolved: list[UUVRegionalPolicy] = []
    for candidate in normalized_candidates:
        decision = policies_by_id[candidate.candidate_id]
        resolved.append(
            UUVRegionalPolicy(
                candidate_id=decision.candidate_id,
                coverage_mode=decision.coverage_mode,
                tracking_mode=decision.tracking_mode,
                priority=decision.priority,
                required_quality=decision.required_quality,
                active_scan_uuv_count=decision.active_scan_uuv_count,
                passive_track_uuv_count=decision.passive_track_uuv_count,
                reserve_uuv_count=decision.reserve_uuv_count,
                optional_uuv_count=decision.optional_uuv_count,
                assigned_uuv_ids=decision.assigned_uuv_ids,
                predecessor_candidate_id=_first_relation(
                    candidate.predecessor_candidate_ids
                ),
                successor_candidate_id=_first_relation(candidate.successor_candidate_ids),
                rationale=decision.rationale,
                evidence_ids=decision.evidence_ids,
            )
        )
    return ValidatedRegionalStrategy(policies=tuple(resolved))


def validate_uuv_strategy(
    candidate_set: Sequence[CandidateInput],
    strategy: UUVRegionalStrategySet | Mapping[str, object],
    available_uuv_ids: AvailableUUVs,
) -> ValidatedRegionalStrategy:
    """Validate an LLM strategy against generated regions and live UUVs.

    The function is deliberately the only boundary that turns semantic LLM
    output into an accepted regional strategy.  Its schema forbids legacy
    surface-platform fields before assignment and handoff checks run.
    """
    candidates = _normalize_candidates(candidate_set)
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    parsed = _parse_strategy(strategy)
    expected_ids = set(by_id)
    actual_ids = [policy.candidate_id for policy in parsed.policies]
    if len(actual_ids) != len(set(actual_ids)):
        raise RegionalPlanError("duplicate regional policy")
    unknown_regions = set(actual_ids) - expected_ids
    if unknown_regions:
        raise RegionalPlanError(f"unknown region: {sorted(unknown_regions)}")
    missing_regions = expected_ids - set(actual_ids)
    if missing_regions:
        raise RegionalPlanError(f"missing region policy: {sorted(missing_regions)}")

    available_ids, resources = _available_uuvs(available_uuv_ids)
    assigned_ids: set[str] = set()
    for policy in parsed.policies:
        _validate_policy(
            policy,
            by_id[policy.candidate_id],
            expected_ids,
            available_ids,
            resources,
        )
        overlap = assigned_ids.intersection(policy.assigned_uuv_ids)
        if overlap:
            raise RegionalPlanError(f"overlapping UUV assignments: {sorted(overlap)}")
        assigned_ids.update(policy.assigned_uuv_ids)

    return ValidatedRegionalStrategy(
        policies=parsed.policies,
        request_hash=parsed.request_hash,
        response_hash=parsed.response_hash,
    )


def _normalize_candidates(
    candidate_set: Sequence[CandidateInput],
) -> tuple[RegionalMissionCandidate, ...]:
    normalized: list[RegionalMissionCandidate] = []
    for candidate in candidate_set:
        try:
            normalized.append(
                candidate
                if isinstance(candidate, RegionalMissionCandidate)
                else candidate_region_to_mission_candidate(candidate)
            )
        except (TypeError, ValueError) as exc:
            raise RegionalPlanError(f"invalid regional candidate: {exc}") from exc
    ids = [candidate.candidate_id for candidate in normalized]
    if len(ids) != len(set(ids)):
        raise RegionalPlanError("duplicate regional candidate")
    if not normalized:
        raise RegionalPlanError("regional candidate set is empty")
    return tuple(normalized)


def _parse_strategy(
    strategy: UUVRegionalStrategySet | Mapping[str, object],
) -> UUVRegionalStrategySet:
    try:
        return (
            strategy
            if isinstance(strategy, UUVRegionalStrategySet)
            else UUVRegionalStrategySet.model_validate(strategy)
        )
    except (TypeError, ValidationError) as exc:
        raise RegionalPlanError("strict UUV strategy schema rejected the response") from exc


def _parse_decision_set(decisions: DecisionInput) -> UUVRegionalStrategyDecisionSet:
    try:
        return (
            decisions
            if isinstance(decisions, UUVRegionalStrategyDecisionSet)
            else UUVRegionalStrategyDecisionSet.model_validate(decisions)
        )
    except (TypeError, ValidationError) as exc:
        raise RegionalPlanError("strict UUV regional decision schema rejected the response") from exc


def _available_uuvs(
    available_uuv_ids: AvailableUUVs,
) -> tuple[set[str], Mapping[str, Any]]:
    if isinstance(available_uuv_ids, Mapping):
        return set(available_uuv_ids), available_uuv_ids
    ids = set(available_uuv_ids)
    return ids, {platform_id: None for platform_id in ids}


def _validate_policy(
    policy: UUVRegionalPolicy,
    candidate: RegionalMissionCandidate,
    candidate_ids: set[str],
    available_ids: set[str],
    resources: Mapping[str, Any],
) -> None:
    # The LLM selects the candidate-region policy, not the final fleet
    # allocation.  An empty assignment is therefore valid: it means the
    # deterministic mission optimizer must decide whether this candidate can
    # be covered in the current rolling window.  Non-empty assignments remain
    # strict locks and are checked below.
    unknown_uuv = set(policy.assigned_uuv_ids) - available_ids
    if unknown_uuv:
        raise RegionalPlanError(
            f"unknown UUV in {policy.candidate_id}: {sorted(unknown_uuv)}"
        )
    if policy.tracking_mode == "active_scan":
        unavailable = [
            platform_id
            for platform_id in policy.assigned_uuv_ids
            if not _active_capable(resources.get(platform_id))
        ]
        if unavailable:
            raise RegionalPlanError(
                f"active scan requires active-capable UUVs: {sorted(unavailable)}"
            )
    for relation_name, relation_id in (
        ("predecessor", policy.predecessor_candidate_id),
        ("successor", policy.successor_candidate_id),
    ):
        if relation_id is None:
            continue
        if relation_id not in candidate_ids:
            raise RegionalPlanError(
                f"handoff {relation_name} references unknown candidate {relation_id}"
            )
    if (
        policy.predecessor_candidate_id is not None
        and candidate.predecessor_candidate_ids
        and policy.predecessor_candidate_id not in candidate.predecessor_candidate_ids
    ):
        raise RegionalPlanError(
            f"handoff predecessor is not declared by candidate {candidate.candidate_id}"
        )
    if (
        policy.successor_candidate_id is not None
        and candidate.successor_candidate_ids
        and policy.successor_candidate_id not in candidate.successor_candidate_ids
    ):
        raise RegionalPlanError(
            f"handoff successor is not declared by candidate {candidate.candidate_id}"
        )


def _validate_decision_policy(
    policy: UUVRegionalPolicyDecision,
    available_ids: set[str],
    resources: Mapping[str, Any],
) -> None:
    unknown_uuv = set(policy.assigned_uuv_ids) - available_ids
    if unknown_uuv:
        raise RegionalPlanError(
            f"unknown UUV in {policy.candidate_id}: {sorted(unknown_uuv)}"
        )
    declared_capacity = (
        policy.active_scan_uuv_count
        + policy.passive_track_uuv_count
        + policy.reserve_uuv_count
        + policy.optional_uuv_count
    )
    if declared_capacity == 0 and policy.assigned_uuv_ids:
        raise RegionalPlanError(
            f"regional policy {policy.candidate_id} assigns UUVs with zero capacity"
        )
    if len(policy.assigned_uuv_ids) > declared_capacity:
        raise RegionalPlanError(
            f"regional policy {policy.candidate_id} assigns more UUVs than declared"
        )
    if policy.tracking_mode == "active_scan":
        if policy.active_scan_uuv_count < 1:
            raise RegionalPlanError(
                f"active scan requires an active UUV count for {policy.candidate_id}"
            )
        unavailable = [
            platform_id
            for platform_id in policy.assigned_uuv_ids
            if not _active_capable(resources.get(platform_id))
        ]
        if unavailable:
            raise RegionalPlanError(
                f"active scan requires active-capable UUVs: {sorted(unavailable)}"
            )


def _first_relation(relation_ids: Sequence[str]) -> str | None:
    return min(relation_ids) if relation_ids else None


def _active_capable(resource: Any) -> bool:
    if resource is None:
        return True
    if isinstance(resource, Mapping):
        if "active_capable" in resource:
            return bool(resource["active_capable"])
        capability = resource.get("capability")
        if isinstance(capability, Mapping):
            sonar = capability.get("sonar")
            if isinstance(sonar, Mapping) and "active_capable" in sonar:
                return bool(sonar["active_capable"])
        return True
    direct = getattr(resource, "active_capable", None)
    if direct is not None:
        return bool(direct)
    capability = getattr(resource, "capability", None)
    sonar = getattr(capability, "sonar", None)
    active_capable = getattr(sonar, "active_capable", None)
    return True if active_capable is None else bool(active_capable)
