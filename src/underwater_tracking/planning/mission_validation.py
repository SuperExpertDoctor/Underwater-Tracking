"""Independent validation for executable UUV-only mission plans.

The UUV-only execution contract is intentionally separate from the legacy
group/USV ``TrackingPlan`` contract.  This module validates the typed mission
plan against the immutable live planning snapshot before the plan is stored
or handed to the physical carrier fleet.  The legacy plan projection is
therefore an audit view only; it is never used as the source of UUV members,
carrier routes, or execution commands.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from underwater_tracking.agent.nodes.snapshot import PlanningSnapshot
from underwater_tracking.domain.mission_models import ExecutableMissionPlan


def validate_executable_mission_plan(
    snapshot: PlanningSnapshot,
    plan: ExecutableMissionPlan,
    *,
    candidate_ids: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return deterministic validation errors for one executable UUV plan.

    The optimizer may leave regions uncovered when resources are insufficient,
    so an empty batch is valid.  Every resource that *is* assigned must,
    however, be a known healthy UUV with an active capability and a live
    carrier ownership record.  Carrier routes are deliberately not required
    here: the physical engine materializes safe routes from the typed stop
    points at the execution boundary.
    """

    issues: list[str] = []
    situation = snapshot.situation
    platform_snapshot = situation.platform_snapshot
    if platform_snapshot is None:
        return ("platform_snapshot_missing",)
    if plan.revision != snapshot.snapshot_revision:
        issues.append("mission_revision_mismatch")

    live_uuvs = {uuv.uuv_id: uuv for uuv in situation.uuvs}
    platform_uuvs = {
        platform.platform_id: platform for platform in platform_snapshot.roster.uuvs
    }
    known_uuv_ids = set(live_uuvs) & set(platform_uuvs)
    carriers = tuple(platform_snapshot.carriers or (platform_snapshot.carrier,))
    carrier_ids = {carrier.carrier_id for carrier in carriers}
    ownership = _carrier_ownership(carriers, known_uuv_ids)
    primary_carrier_id = platform_snapshot.carrier.carrier_id
    for uuv_id in sorted(known_uuv_ids - set(ownership)):
        # Older snapshots do not list per-carrier inventory.  Match the
        # optimizer's deterministic compatibility rule and assign those
        # resources to the primary carrier.
        ownership[uuv_id] = primary_carrier_id

    expected_candidates = set(candidate_ids)
    assigned_uuv_ids: set[str] = set()
    assigned_candidate_ids: set[str] = set()
    for carrier_id, batches in sorted(plan.uuv_batches_by_carrier.items()):
        if carrier_id not in carrier_ids:
            issues.append(f"unknown_carrier:{carrier_id}")
        for batch in batches:
            assigned_candidate_ids.add(batch.candidate_id)
            for uuv_id in batch.uuv_ids:
                assigned_uuv_ids.add(uuv_id)
                _check_uuv(
                    uuv_id,
                    carrier_id,
                    live_uuvs,
                    platform_uuvs,
                    ownership,
                    issues,
                    requires_active=uuv_id in batch.active_scan_uuv_ids,
                )

    for uuv_id in plan.reserved_uuv_ids:
        assigned_uuv_ids.add(uuv_id)
        _check_uuv(
            uuv_id,
            None,
            live_uuvs,
            platform_uuvs,
            ownership,
            issues,
            requires_active=False,
        )

    for assignment in plan.region_assignments:
        assigned_candidate_ids.add(assignment.region_id)
        for uuv_id in (
            *assignment.active_scan_uuv_ids,
            *assignment.passive_track_uuv_ids,
            *assignment.reserve_uuv_ids,
        ):
            if uuv_id not in assigned_uuv_ids:
                issues.append(
                    f"region_resource_not_in_batch:{assignment.region_id}:{uuv_id}"
                )
            _check_uuv(
                uuv_id,
                None,
                live_uuvs,
                platform_uuvs,
                ownership,
                issues,
                requires_active=uuv_id in assignment.active_scan_uuv_ids,
            )

    for carrier_id, mission in sorted(plan.carrier_missions.items()):
        if carrier_id not in carrier_ids:
            issues.append(f"unknown_carrier_mission:{carrier_id}")
        if mission.carrier_id != carrier_id:
            issues.append(f"carrier_mission_key_mismatch:{carrier_id}")
        for uuv_id in (
            *mission.onboard_uuv_ids,
            *mission.ready_uuv_ids,
            *mission.reserved_uuv_ids,
            *mission.recoverable_uuv_ids,
        ):
            if uuv_id not in known_uuv_ids:
                issues.append(f"unknown_carrier_inventory_uuv:{uuv_id}")
            elif ownership.get(uuv_id) not in {None, carrier_id}:
                issues.append(
                    f"carrier_inventory_ownership_mismatch:{uuv_id}:{carrier_id}"
                )

    if expected_candidates:
        for candidate_id in sorted(assigned_candidate_ids - expected_candidates):
            issues.append(f"unknown_mission_candidate:{candidate_id}")

    return tuple(sorted(set(issues)))


def _carrier_ownership(
    carriers: Sequence[object], known_uuv_ids: set[str]
) -> dict[str, str]:
    ownership: dict[str, str] = {}
    for carrier in sorted(carriers, key=lambda item: str(item.carrier_id)):
        for uuv_id in (
            *carrier.onboard_platform_ids,
            *carrier.deployed_platform_ids,
        ):
            if uuv_id in known_uuv_ids and uuv_id not in ownership:
                ownership[uuv_id] = carrier.carrier_id
    return ownership


def _check_uuv(
    uuv_id: str,
    carrier_id: str | None,
    live_uuvs: Mapping[str, object],
    platform_uuvs: Mapping[str, object],
    ownership: Mapping[str, str],
    issues: list[str],
    *,
    requires_active: bool,
) -> None:
    live = live_uuvs.get(uuv_id)
    platform = platform_uuvs.get(uuv_id)
    if live is None or platform is None:
        issues.append(f"unknown_uuv:{uuv_id}")
        return
    deployment_state = getattr(live.deployment_state, "value", live.deployment_state)
    if deployment_state not in {"onboard", "deployed"}:
        issues.append(f"uuv_not_deployable:{uuv_id}")
    if live.energy_fraction <= 0.10 or platform.energy_fraction <= 0.10:
        issues.append(f"uuv_energy_below_reserve:{uuv_id}")
    if requires_active and not platform.capability.sonar.active_capable:
        issues.append(f"uuv_active_capability_missing:{uuv_id}")
    owner = ownership.get(uuv_id)
    if owner is None:
        issues.append(f"uuv_carrier_ownership_missing:{uuv_id}")
    elif carrier_id is not None and owner != carrier_id:
        issues.append(f"uuv_carrier_mismatch:{uuv_id}:{carrier_id}")
