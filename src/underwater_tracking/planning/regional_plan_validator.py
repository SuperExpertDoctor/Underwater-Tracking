from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import ValidationError

from underwater_tracking.domain.regional_models import (
    RegionalMissionCandidate,
    UUVRegionalPolicy,
    UUVRegionalStrategySet,
)
from underwater_tracking.planning.candidate_regions import (
    CandidateRegion,
    candidate_region_to_mission_candidate,
)


class RegionalPlanError(ValueError):
    """Raised when an LLM regional candidate cannot be verified."""


class ValidatedRegionalStrategy(UUVRegionalStrategySet):
    """Marker type proving the strict candidate checks have completed."""


CandidateInput = RegionalMissionCandidate | CandidateRegion
AvailableUUVs = Iterable[str] | Mapping[str, Any]


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
    if not policy.assigned_uuv_ids:
        raise RegionalPlanError(
            f"regional policy {policy.candidate_id} must assign at least one UUV"
        )
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
