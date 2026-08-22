"""Semantic revalidation of slow UUV mission plans."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Literal

from pydantic import ConfigDict, Field

from underwater_tracking.domain.mission_models import ExecutableMissionPlan
from underwater_tracking.domain.models import SituationSnapshot, StrictModel
from underwater_tracking.domain.planning_epoch_models import PlanningEpoch
from underwater_tracking.persistence.sqlite import json_dumps
from underwater_tracking.runtime.mission_controller import MissionSnapshot


class RevalidationIssue(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: Literal[
        "target_missing",
        "region_missing",
        "carrier_missing",
        "uuv_missing",
        "owner_changed",
        "deployment_changed",
        "resource_unavailable",
        "estimate_outside_envelope",
        "prior_expired",
        "prior_changed",
        "active_plan_advanced",
        "expert_version_advanced",
        "trigger_recovered",
    ]
    entity_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class MissionRevalidationReport(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    report_id: str = Field(min_length=1)
    epoch_id: str = Field(min_length=1)
    current_physics_revision: int = Field(ge=0)
    current_plan_version: int = Field(ge=0)
    valid: bool
    issues: tuple[RevalidationIssue, ...] = ()
    rebased_plan: ExecutableMissionPlan | None = None


def revalidate_executable_mission_plan(
    *,
    epoch: PlanningEpoch,
    candidate: ExecutableMissionPlan,
    current_situation: SituationSnapshot,
    current_mission: MissionSnapshot,
    current_expert_request_version: int | None,
    recovered_event_ids: frozenset[str],
) -> MissionRevalidationReport:
    """Check semantic assumptions while preserving the candidate strategy.

    The function intentionally does not compare ``epoch.base_physics_revision``
    with the current revision.  Physics drift is expected while a provider is
    running; only changes to entities, resources, evidence, or authority
    invalidate the candidate.
    """

    issues: list[RevalidationIssue] = []

    def issue(code: str, entity_id: str, message: str) -> None:
        issues.append(RevalidationIssue(code=code, entity_id=entity_id, message=message))  # type: ignore[arg-type]

    if current_mission.plan_revision > epoch.active_plan_version:
        issue(
            "active_plan_advanced",
            current_mission.scenario_id,
            "a newer executable plan is already active",
        )
    if (
        current_expert_request_version is not None
        and epoch.expert_request_version is not None
        and current_expert_request_version > epoch.expert_request_version
    ):
        issue(
            "expert_version_advanced",
            current_mission.scenario_id,
            "a newer expert request supersedes this epoch",
        )

    for event_id in sorted(set(epoch.critical_event_ids) & set(recovered_event_ids)):
        issue("trigger_recovered", event_id, "the epoch trigger was recovered before commit")

    candidate_targets = {
        assignment.target_id for assignment in candidate.region_assignments
    }
    known_targets = {
        report.target_id for report in current_situation.group_reports
    }
    known_targets.update(
        str(getattr(prior, "target_id"))
        for prior in getattr(current_situation, "target_search_priors", ())
    )
    known_targets.update(
        str(getattr(estimate, "target_id"))
        for estimate in getattr(current_situation, "target_estimates", ())
    )
    for target_id in sorted(candidate_targets - known_targets if known_targets else set()):
        issue("target_missing", target_id, "candidate target is absent from the current public situation")

    candidate_regions = {assignment.region_id for assignment in candidate.region_assignments}
    known_regions = set(getattr(current_situation, "known_region_ids", ()))
    for region_id in sorted(candidate_regions - known_regions if known_regions else set()):
        issue("region_missing", region_id, "candidate region is absent from the current region catalog")

    current_carriers = set(current_mission.carrier_missions)
    current_carriers.update(
        str(carrier.carrier_id)
        for carrier in (current_situation.carriers or ((current_situation.carrier,) if current_situation.carrier else ()))
    )
    candidate_carriers = set(candidate.uuv_batches_by_carrier) | set(candidate.carrier_missions)
    for carrier_id in sorted(candidate_carriers - current_carriers if current_carriers else set()):
        issue("carrier_missing", carrier_id, "candidate carrier is absent from the current mission")

    current_resources = dict(current_mission.uuv_resources)
    candidate_uuvs = {
        uuv_id
        for batch in candidate.batches
        for uuv_id in batch.uuv_ids
    }
    candidate_uuvs.update(candidate.reserved_uuv_ids)
    for assignment in candidate.region_assignments:
        candidate_uuvs.update(
            (
                *assignment.active_scan_uuv_ids,
                *assignment.passive_track_uuv_ids,
                *assignment.reserve_uuv_ids,
            )
        )
    for carrier in candidate.carrier_missions.values():
        candidate_uuvs.update(
            (
                *carrier.onboard_uuv_ids,
                *carrier.ready_uuv_ids,
                *carrier.reserved_uuv_ids,
                *carrier.recoverable_uuv_ids,
            )
        )
    candidate_owner_by_uuv = {
        uuv_id: carrier_id
        for carrier_id, carrier in candidate.carrier_missions.items()
        for uuv_id in (
            *carrier.onboard_uuv_ids,
            *carrier.ready_uuv_ids,
            *carrier.reserved_uuv_ids,
            *carrier.recoverable_uuv_ids,
        )
    }
    candidate_owner_by_uuv.update(
        {uuv_id: batch.carrier_id for batch in candidate.batches for uuv_id in batch.uuv_ids}
    )
    for uuv_id in sorted(candidate_uuvs):
        resource = current_resources.get(uuv_id)
        if resource is None:
            issue("uuv_missing", uuv_id, "candidate UUV is absent from the current inventory")
            continue
        expected_owner = candidate_owner_by_uuv.get(uuv_id)
        if expected_owner is not None and resource.carrier_id not in {None, expected_owner}:
            issue("owner_changed", uuv_id, "permanent UUV carrier ownership changed")
        deployment = str(resource.deployment_state).lower()
        if deployment in {"failed", "returning", "recovering"} or not resource.healthy:
            issue("resource_unavailable", uuv_id, "UUV is unhealthy or unavailable")
        if resource.energy_fraction <= 0.10:
            issue("resource_unavailable", uuv_id, "UUV energy is below the reserve threshold")
        expected_episode = candidate.resource_episode_by_uuv.get(uuv_id)
        if expected_episode is not None and expected_episode != resource.resource_episode:
            issue("deployment_changed", uuv_id, "UUV resource episode changed")

    active_prior_ids = {
        str(getattr(prior, "prior_id"))
        for prior in getattr(current_situation, "target_search_priors", ())
    }
    if epoch.public_target_prior_ids and not set(epoch.public_target_prior_ids).issubset(active_prior_ids):
        issue("prior_changed", "target-prior", "captured public target prior is no longer active")
    for prior in getattr(current_situation, "target_search_priors", ()):
        valid_until = getattr(prior, "valid_until_s", None)
        if valid_until is not None and current_situation.sim_time_s >= valid_until:
            if getattr(prior, "prior_id", None) in epoch.public_target_prior_ids:
                issue("prior_expired", str(prior.prior_id), "captured target prior has expired")

    _check_estimate_envelopes(current_situation, candidate, issue)
    unique_issues = tuple(
        sorted(issues, key=lambda item: (item.code, item.entity_id, item.message))
    )
    valid = not unique_issues
    rebased_plan = None
    if valid:
        rebased_payload = candidate.model_dump(mode="json")
        rebased_payload["revision"] = current_mission.plan_revision + 1
        rebased_plan = ExecutableMissionPlan.model_validate(rebased_payload)
    report_id = "validation:" + hashlib.sha256(
        json_dumps(
            {
                "epoch_id": epoch.epoch_id,
                "revision": current_situation.snapshot_revision,
                "issues": [item.model_dump(mode="json") for item in unique_issues],
            }
        ).encode("utf-8")
    ).hexdigest()[:24]
    return MissionRevalidationReport(
        report_id=report_id,
        epoch_id=epoch.epoch_id,
        current_physics_revision=current_situation.snapshot_revision,
        current_plan_version=current_mission.plan_revision,
        valid=valid,
        issues=unique_issues,
        rebased_plan=rebased_plan,
    )


def _check_estimate_envelopes(
    situation: SituationSnapshot,
    candidate: ExecutableMissionPlan,
    issue: Callable[[str, str, str], None],
) -> None:
    bounds_by_region = getattr(situation, "region_bounds_by_id", {})
    if not bounds_by_region:
        return
    assignments = {assignment.region_id for assignment in candidate.region_assignments}
    for estimate in getattr(situation, "target_estimates", ()):
        mean = getattr(estimate, "mean", None)
        if mean is None:
            mean = getattr(estimate, "position_xy", None)
        if mean is None:
            continue
        if not any(
            region_id in assignments and _inside_buffer(mean, bounds_by_region[region_id], 500.0)
            for region_id in bounds_by_region
        ):
            issue(
                "estimate_outside_envelope",
                str(getattr(estimate, "target_id", "target")),
                "target estimate is outside the selected region envelope",
            )


def _inside_buffer(point: object, bounds: object, margin: float) -> bool:
    if not isinstance(point, (tuple, list)) or len(point) != 2:
        return True
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 4:
        return True
    try:
        x, y = float(point[0]), float(point[1])
        min_x, max_x, min_y, max_y = (float(value) for value in bounds)
    except (TypeError, ValueError):
        return True
    return min_x - margin <= x <= max_x + margin and min_y - margin <= y <= max_y + margin
