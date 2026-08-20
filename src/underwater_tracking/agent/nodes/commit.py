# src/underwater_tracking/agent/nodes/commit.py
"""Independent plan validation and atomic versioned commit (spec 15.3).

``CommitNode`` re-checks the candidate plan against the immutable
planning snapshot alone — target coverage, 2-4 members per group, member
uniqueness and resource health, the 10% energy return reserve, waypoint
bounds, separation and kinematics, evidence references, and the base
snapshot revision — then commits it through ``PlanRepository.commit``,
whose single immediate transaction also supersedes the previous
broadcast plan. Only after the transaction succeeded is one
``PlanCommand`` per group created and persisted, and commands are
published only after they were persisted: a stale plan (the stored
scenario revision moved after the snapshot) surfaces as ``stale`` with
nothing written, and any validation issue surfaces as ``rejected``. A
periodic review that selected the broadcast plan itself returns
``hold_current`` without a write and without a new revision.

The checks are independent: they never trust the candidate's own metric
fields (``predicted_*``) and only use the raw snapshot inputs.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Literal, TypedDict

from underwater_tracking.agent.nodes.optimize import (
    PlanningConfig,
    _capability_feasible,
    _effective_speed_mps,
)
from underwater_tracking.agent.nodes.snapshot import PlanningSnapshot
from underwater_tracking.agent.nodes.verify import validate_regional_tasks
from underwater_tracking.agent.state import CarrierState
from underwater_tracking.domain.availability import deployability_conflict, is_deployable
from underwater_tracking.domain.agent_models import (
    PlanCommand,
    TrackingPlan,
    ValidationIssue,
)
from underwater_tracking.domain.models import DeploymentState, GroupReport, TargetBelief
from underwater_tracking.planning.allocation import AllocationInput, projected_tracking_quality
from underwater_tracking.persistence.plans import PlanRepository, StaleSnapshotError

# Shared immutable default for node constructors (B008: no call in defaults).
_DEFAULT_CONFIG = PlanningConfig()

# Float boundary tolerance on the planner's own feasibility bounds (the
# waypoint planner admits ``max_step`` and ``min_separation`` boundary
# cases up to ~1e-3 m).
_BOUND_TOLERANCE_M = 1e-3


class CommitResult(TypedDict):
    """Outcome of one plan commit attempt (spec 15.3)."""

    commit_status: Literal["committed", "hold_current", "stale", "rejected"]
    plan_id: str | None
    issues: tuple[ValidationIssue, ...]


def validate_plan(
    snapshot: PlanningSnapshot,
    plan: TrackingPlan,
    config: PlanningConfig = _DEFAULT_CONFIG,
) -> tuple[ValidationIssue, ...]:
    """Independently validate one candidate plan against the snapshot.

    Every check is recomputed from the raw snapshot inputs: target
    coverage (a degraded emergency plan may retain a subset of the
    tracked targets), legacy group size, member uniqueness and resource health,
    the energy return reserve and range bound per member, waypoint
    bounds/separation/kinematics, evidence references, and the base
    snapshot revision. Issues are returned sorted by (code, field,
    message) for deterministic output.
    """
    issues: list[ValidationIssue] = []
    issues.extend(validate_regional_tasks(snapshot, plan))
    _check_base_revision(snapshot, plan, issues)
    _check_coverage(snapshot, plan, issues)
    _check_groups_and_members(snapshot, plan, config, issues)
    _check_rotation(snapshot, plan, config, issues)
    _check_required_quality(snapshot, plan, issues)
    _check_waypoints(snapshot, plan, config, issues)
    _check_segments(snapshot, plan, config, issues)
    _check_evidence(snapshot, plan, issues)
    return tuple(sorted(issues, key=lambda issue: (issue.code, issue.field, issue.message)))


def build_commands(
    snapshot: PlanningSnapshot,
    plan: TrackingPlan,
) -> tuple[PlanCommand, ...]:
    """Create one versioned execution command per group (spec 5.2).

    Commands are ordered by target id; each carries the group's final
    member ids, the per-member waypoint sequences, and one deterministic
    action per member (a return action when assigned, ``rotate`` when the
    target has a rotation condition, otherwise ``track``). Released
    members never appear in a committed plan (the allocator drops them to
    the reserve pool), so no command carries a ``release`` action.
    """
    commands: list[PlanCommand] = []
    targets = sorted(set(plan.member_ids_by_target) | set(plan.usv_ids_by_target))
    for target in targets:
        members = plan.member_ids_by_target.get(target, ())
        usv_ids = plan.usv_ids_by_target.get(target, ())
        if not members and not usv_ids:
            continue
        group_id = _report(snapshot, target).group_id
        commands.append(
            PlanCommand(
                command_id=f"{plan.plan_id}:group:{group_id}",
                plan_id=plan.plan_id,
                plan_revision=plan.revision,
                scenario_id=plan.scenario_id,
                group_id=group_id,
                region_id=_region_id_for_command(plan, target, members, usv_ids),
                target_id=target,
                sim_time_s=plan.valid_from_s,
                member_ids=members,
                usv_ids=usv_ids,
                usv_roles_by_member={
                    usv_id: plan.roles_by_member[usv_id]
                    for usv_id in usv_ids
                    if usv_id in plan.roles_by_member
                },
                usv_actions={
                    usv_id: _usv_action(plan, usv_id)
                    for usv_id in usv_ids
                },
                waypoints_by_member={
                    member: plan.waypoints_by_member[member]
                    for member in (*members, *usv_ids)
                    if member in plan.waypoints_by_member
                },
                actions={member: _member_action(plan, target, member) for member in members},
                sensor_mode="passive",
            )
        )
    return tuple(commands)


def _region_id_for_command(
    plan: TrackingPlan,
    target_id: str,
    members: tuple[str, ...],
    usv_ids: tuple[str, ...],
) -> str | None:
    """Retain a precise regional address when a legacy group maps to one task."""
    matching = tuple(
        task.region_id
        for task in sorted(plan.region_tasks.values(), key=lambda item: item.region_id)
        if task.target_id == target_id
        and tuple(sorted(task.assigned_uuv_ids)) == tuple(sorted(members))
        and tuple(sorted(task.assigned_usv_ids)) == tuple(sorted(usv_ids))
    )
    return matching[0] if len(matching) == 1 else None


def _usv_action(plan: TrackingPlan, usv_id: str) -> str:
    return "relay" if plan.roles_by_member.get(usv_id) == "surface_relay" else "track"


class CommitNode:
    """LangGraph node: validate, then atomically commit the selected plan.

    ``repository.commit`` is the single transaction (plan insert plus
    supersede of the previous broadcast plan); commands are persisted and
    published only after it succeeded. ``publish`` receives every
    committed command exactly once, in target order.
    """

    def __init__(
        self,
        *,
        repository: PlanRepository,
        snapshot_provider: Callable[[str], PlanningSnapshot],
        config: PlanningConfig = _DEFAULT_CONFIG,
        publish: Callable[[PlanCommand], None] | None = None,
        current_snapshot_revision: Callable[[], int] | None = None,
    ) -> None:
        self._repository = repository
        self._snapshot_provider = snapshot_provider
        self._config = config
        self._publish = publish
        self._current_snapshot_revision = current_snapshot_revision

    def snapshot_is_current(self, snapshot_revision: int) -> bool:
        """Return whether the live situation has not advanced past a snapshot."""
        provider = self._current_snapshot_revision
        return provider is None or provider() <= snapshot_revision

    def __call__(self, state: CarrierState, candidate: TrackingPlan) -> CommitResult:
        ref = state.get("snapshot_ref")
        if ref is None:
            raise ValueError("CommitNode requires snapshot_ref in state")
        snapshot = self._snapshot_provider(ref)
        active = snapshot.active_plan
        if active is not None and candidate.plan_id == active.plan_id:
            return {
                "commit_status": "hold_current",
                "plan_id": candidate.plan_id,
                "issues": (),
            }
        if not self.snapshot_is_current(snapshot.snapshot_revision):
            return {
                "commit_status": "stale",
                "plan_id": candidate.plan_id,
                "issues": (),
            }
        issues = validate_plan(snapshot, candidate, self._config)
        if issues:
            return {
                "commit_status": "rejected",
                "plan_id": candidate.plan_id,
                "issues": issues,
            }
        try:
            self._repository.commit(candidate)
        except StaleSnapshotError:
            return {
                "commit_status": "stale",
                "plan_id": candidate.plan_id,
                "issues": (),
            }
        commands = build_commands(snapshot, candidate)
        for command in commands:
            self._repository.save_command(command)
        if self._publish is not None:
            for command in commands:
                self._publish(command)
        return {
            "commit_status": "committed",
            "plan_id": candidate.plan_id,
            "issues": (),
        }


def _check_base_revision(
    snapshot: PlanningSnapshot,
    plan: TrackingPlan,
    issues: list[ValidationIssue],
) -> None:
    """The plan must build on exactly this snapshot's revision."""
    if plan.base_snapshot_revision != snapshot.snapshot_revision:
        issues.append(
            ValidationIssue(
                code="base_revision_mismatch",
                field="base_snapshot_revision",
                message=f"plan {plan.plan_id} builds on a different snapshot revision",
                observed=str(plan.base_snapshot_revision),
                expected=str(snapshot.snapshot_revision),
            )
        )


def _check_coverage(
    snapshot: PlanningSnapshot,
    plan: TrackingPlan,
    issues: list[ValidationIssue],
) -> None:
    """Every tracked target needs a plan group — except in a degraded
    emergency plan, which legitimately retains a feasible subset."""
    tracked = {report.target_id for report in snapshot.situation.group_reports}
    for target in sorted(set(plan.member_ids_by_target) - tracked):
        issues.append(
            ValidationIssue(
                code="unknown_target",
                field=f"member_ids_by_target[{target}]",
                message=f"no group report for target {target}",
            )
        )
    if plan.status != "degraded":
        for target in sorted(tracked - set(plan.member_ids_by_target)):
            issues.append(
                ValidationIssue(
                    code="coverage_missing",
                    field=f"member_ids_by_target[{target}]",
                    message=f"tracked target {target} has no plan group",
                )
            )


def _check_groups_and_members(
    snapshot: PlanningSnapshot,
    plan: TrackingPlan,
    config: PlanningConfig,
    issues: list[ValidationIssue],
) -> None:
    """Legacy group sizes, member safety, and return reserve.

    Regional task membership is validated by ``validate_regional_tasks``;
    its derived target-level view must not reapply the legacy 2-4 UUV limit.
    """
    uuvs_by_id = {uuv.uuv_id: uuv for uuv in snapshot.situation.uuvs}
    disabled = {
        uuv_id
        for directive in snapshot.applied_directives
        for uuv_id in (*directive.disabled_uuv_ids, *directive.return_uuv_ids)
    }
    assigned_to: dict[str, str] = {}
    has_authoritative_regional_tasks = bool(plan.regional_plans or plan.region_tasks)
    explicit_dispatch = snapshot.situation.platform_snapshot is not None
    active_members = (
        {
            member
            for members in (snapshot.active_plan.member_ids_by_target.values() if snapshot.active_plan else ())
            for member in members
        }
        if explicit_dispatch
        else set()
    )
    for target in sorted(plan.member_ids_by_target):
        members = plan.member_ids_by_target[target]
        if not has_authoritative_regional_tasks and not 2 <= len(members) <= 4:
            issues.append(
                ValidationIssue(
                    code="group_size",
                    field=f"member_ids_by_target[{target}]",
                    message=f"group of target {target} must have 2-4 members,"
                    f" has {len(members)}",
                    observed=str(len(members)),
                    expected="2..4",
                )
            )
        for member in members:
            if member in assigned_to:
                issues.append(
                    ValidationIssue(
                        code="duplicate_member",
                        field=f"member_ids_by_target[{target}]",
                        message=f"uuv {member} is assigned to multiple targets",
                    )
                )
            assigned_to[member] = target
    for member in sorted(assigned_to):
        uuv = uuvs_by_id.get(member)
        if uuv is None:
            issues.append(
                ValidationIssue(
                    code="unknown_member",
                    field=f"member_ids_by_target[{assigned_to[member]}]",
                    message=f"no resource state for uuv {member}",
                )
            )
            continue
        onboard_dispatch = (
            explicit_dispatch and uuv.deployment_state is DeploymentState.ONBOARD
        )
        planned_transit = onboard_dispatch or (
            explicit_dispatch and member in active_members
        )
        if (not is_deployable(uuv) and not onboard_dispatch) or member in disabled:
            issues.append(
                ValidationIssue(
                    code="unavailable_member",
                    field=f"member_ids_by_target[{assigned_to[member]}]",
                    message=(
                        f"uuv {member} is {deployability_conflict(uuv)}"
                        if not is_deployable(uuv)
                        else f"uuv {member} is disabled"
                    ),
                )
            )
        target = assigned_to[member]
        report = _report_or_none(snapshot, target)
        if report is None:
            continue  # already flagged as unknown_target
        mean = report.belief.mean
        if uuv.energy_fraction < config.return_reserve:
            issues.append(
                ValidationIssue(
                    code="return_reserve",
                    field=f"member_ids_by_target[{target}]",
                    message=f"uuv {member} energy below the return reserve",
                    observed=f"{uuv.energy_fraction:.3f}",
                    expected=f">= {config.return_reserve}",
                )
            )
        if (
            not planned_transit
            and _distance(uuv.position_xy, (mean[0], mean[1])) > config.max_range_m
        ):
            issues.append(
                ValidationIssue(
                    code="range_exceeded",
                    field=f"member_ids_by_target[{target}]",
                    message=f"uuv {member} beyond max tracking range",
                    observed=f"{_distance(uuv.position_xy, (mean[0], mean[1])):.1f} m",
                    expected=f"<= {config.max_range_m} m",
                )
            )
        if not uuv.capability.passive_sonar_available:
            issues.append(
                ValidationIssue(
                    code="passive_sonar",
                    field=f"member_ids_by_target[{target}]",
                    message=f"uuv {member} has no passive sonar",
                )
            )
        if uuv.capability.availability <= 0.0:
            issues.append(
                ValidationIssue(
                    code="capability_unavailable",
                    field=f"member_ids_by_target[{target}]",
                    message=f"uuv {member} has no operational capability availability",
                )
            )
        if uuv.capability.endurance_s < config.plan_horizon_s:
            issues.append(
                ValidationIssue(
                    code="endurance",
                    field=f"member_ids_by_target[{target}]",
                    message=f"uuv {member} cannot cover the plan horizon",
                    observed=f"{uuv.capability.endurance_s:.1f} s",
                    expected=f">= {config.plan_horizon_s} s",
                )
            )
        if (
            not planned_transit
            and _distance(uuv.position_xy, (mean[0], mean[1]))
            > uuv.capability.passive_range_m
        ):
            issues.append(
                ValidationIssue(
                    code="capability_range",
                    field=f"member_ids_by_target[{target}]",
                    message=f"uuv {member} exceeds its passive sensing range",
                    observed=f"{_distance(uuv.position_xy, (mean[0], mean[1])):.1f} m",
                    expected=f"<= {uuv.capability.passive_range_m:.1f} m",
                )
            )
        elif (
            not planned_transit
            and uuv.capability.passive_sonar_available
            and uuv.capability.availability > 0.0
            and uuv.capability.endurance_s >= config.plan_horizon_s
            and not _capability_feasible(uuv, (mean[0], mean[1]), config)
        ):
            issues.append(
                ValidationIssue(
                    code="capability_kinematics",
                    field=f"member_ids_by_target[{target}]",
                    message=f"uuv {member} cannot reach its target within capability limits",
                    observed=f"{_effective_speed_mps(uuv):.3f} m/s",
                    expected="within speed and turn-rate limits",
                )
            )


def _check_rotation(
    snapshot: PlanningSnapshot,
    plan: TrackingPlan,
    config: PlanningConfig,
    issues: list[ValidationIssue],
) -> None:
    """Rotation ids must be assigned members with a real rotation trigger."""
    assigned_to = {
        member: target
        for target, members in plan.member_ids_by_target.items()
        for member in members
    }
    uuvs_by_id = {uuv.uuv_id: uuv for uuv in snapshot.situation.uuvs}
    for member in plan.rotation_uuv_ids:
        target = assigned_to.get(member)
        if target is None:
            issues.append(
                ValidationIssue(
                    code="rotation_member",
                    field="rotation_uuv_ids",
                    message=f"rotation uuv {member} is not a member of any target group",
                )
            )
            continue
        if plan.rotation_conditions and target not in plan.rotation_conditions:
            issues.append(
                ValidationIssue(
                    code="rotation_condition",
                    field="rotation_uuv_ids",
                    message=f"rotation uuv {member} has no rotation condition for target {target}",
                )
            )
        uuv = uuvs_by_id.get(member)
        if uuv is not None and uuv.energy_fraction >= config.rotation_threshold:
            issues.append(
                ValidationIssue(
                    code="rotation_energy",
                    field="rotation_uuv_ids",
                    message=f"uuv {member} is not below the rotation energy threshold",
                    observed=f"{uuv.energy_fraction:.3f}",
                    expected=f"< {config.rotation_threshold:.3f}",
                )
            )


def _projected_quality(
    snapshot: PlanningSnapshot,
    target: str,
    members: Sequence[str],
) -> float:
    """Use the allocator's quality formula with actual UUV speeds."""
    uuvs_by_id = {uuv.uuv_id: uuv for uuv in snapshot.situation.uuvs}
    known_members = tuple(member for member in members if member in uuvs_by_id)
    problem = AllocationInput(
        uuv_ids=known_members,
        target_ids=(target,),
        quality_by_target={target: _report(snapshot, target).quality.ewma},
        uuv_bearing_variance_rad2={
            member: uuvs_by_id[member].capability.bearing_variance_rad2
            for member in known_members
        },
        uuv_speed_mps={member: _effective_speed_mps(uuvs_by_id[member]) for member in known_members},
        uuv_max_turn_rate_rad_s={
            member: uuvs_by_id[member].capability.max_turn_rate_rad_s
            for member in known_members
        },
    )
    return projected_tracking_quality(problem, target, known_members)


def _check_required_quality(
    snapshot: PlanningSnapshot,
    plan: TrackingPlan,
    issues: list[ValidationIssue],
) -> None:
    """Recompute target quality floors from the snapshot, never plan metrics."""
    for target, members in sorted(plan.member_ids_by_target.items()):
        report = _report_or_none(snapshot, target)
        if report is None or not members:
            continue
        # A DEGRADED explicit-platform plan can be a staged dispatch: its
        # members are still onboard or are continuing along the previous
        # segment toward the estimated sector.  Requiring present passive
        # quality at that point would reject the very plan that restores it.
        if plan.status == "degraded" and _is_staged_dispatch(snapshot, members):
            continue
        required = plan.required_quality.get(target, 0.0)
        scheme = snapshot.situation.operational_scheme
        if scheme is not None and scheme.valid_from_s <= snapshot.sim_time_s < scheme.valid_until_s:
            required = max(required, scheme.minimum_quality.get(target, 0.0))
        for directive in snapshot.applied_directives:
            required = max(required, directive.minimum_quality.get(target, 0.0))
        if required <= 0.0:
            continue
        projected = _projected_quality(snapshot, target, members)
        if projected < required:
            issues.append(
                ValidationIssue(
                    code="required_quality",
                    field=f"member_ids_by_target[{target}]",
                    message=f"target {target} projected quality is below its required floor",
                    observed=f"{projected:.3f}",
                    expected=f">= {required:.3f}",
                )
            )


def _is_staged_dispatch(
    snapshot: PlanningSnapshot, members: Sequence[str]
) -> bool:
    """Whether every member is still in a permitted explicit-world transit."""
    if snapshot.situation.platform_snapshot is None:
        return False
    active_members = {
        member
        for active in ((snapshot.active_plan,) if snapshot.active_plan else ())
        for group in active.member_ids_by_target.values()
        for member in group
    }
    onboard = {
        uuv.uuv_id
        for uuv in snapshot.situation.uuvs
        if uuv.deployment_state is DeploymentState.ONBOARD
    }
    return bool(members) and set(members) <= onboard | active_members


def _check_waypoints(
    snapshot: PlanningSnapshot,
    plan: TrackingPlan,
    config: PlanningConfig,
    issues: list[ValidationIssue],
) -> None:
    """Waypoint bounds, kinematics, and per-step group separation."""
    xmin, xmax, ymin, ymax = config.bounds
    uuvs_by_id = {uuv.uuv_id: uuv for uuv in snapshot.situation.uuvs}
    for target, members in plan.member_ids_by_target.items():
        for member in members:
            if member not in plan.waypoints_by_member:
                issues.append(
                    ValidationIssue(
                        code="missing_waypoints",
                        field=f"waypoints_by_member[{member}]",
                        message=f"no waypoint sequence for member uuv {member}",
                    )
                )
    for member in sorted(plan.waypoints_by_member):
        if member not in uuvs_by_id:
            continue
        uuv = uuvs_by_id[member]
        max_step = _effective_speed_mps(uuv) * config.replan_period_s
        for step, waypoint in enumerate(plan.waypoints_by_member[member]):
            if not (
                xmin <= waypoint.x <= xmax
                and ymin <= waypoint.y <= ymax
            ):
                issues.append(
                    ValidationIssue(
                        code="waypoint_out_of_bounds",
                        field=f"waypoints_by_member[{member}]",
                        message=f"waypoint {step} of uuv {member} outside the scenario box",
                    )
                )
            if step == 0 and _distance(
                uuv.position_xy, (waypoint.x, waypoint.y)
            ) > max_step + _BOUND_TOLERANCE_M:
                issues.append(
                    ValidationIssue(
                        code="waypoint_kinematics",
                        field=f"waypoints_by_member[{member}]",
                        message=f"first waypoint of uuv {member} beyond one replan step",
                        observed=f"{_distance(uuv.position_xy, (waypoint.x, waypoint.y)):.1f} m",
                        expected=f"<= {max_step:.1f} m",
                    )
                )
    for target in sorted(plan.member_ids_by_target):
        members = plan.member_ids_by_target[target]
        staged_dispatch = _is_staged_dispatch(snapshot, members)
        sequences = [
            plan.waypoints_by_member[member]
            for member in members
            if member in plan.waypoints_by_member
        ]
        if staged_dispatch:
            # The carrier launch transient cannot reach 300 m separation in
            # one 30 s window; the dispatch path fans the group out before
            # the next strict on-station validation cycle.
            continue
        for step in range(_longest(sequences)):
            points: list[tuple[float, float]] = []
            for sequence in sequences:
                if step < len(sequence):
                    points.append((sequence[step].x, sequence[step].y))
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    if _distance(points[i], points[j]) < config.min_separation_m - _BOUND_TOLERANCE_M:
                        issues.append(
                            ValidationIssue(
                                code="waypoint_separation",
                                field=f"waypoints_by_member[{target}]",
                                message=f"group of target {target} violates minimum"
                                f" separation at step {step}",
                            )
                        )


def _check_segments(
    snapshot: PlanningSnapshot,
    plan: TrackingPlan,
    config: PlanningConfig,
    issues: list[ValidationIssue],
) -> None:
    """Segment-plan sanity: contiguous indices, ordered times, bounded intercepts.

    No horizon cap is enforced: a segment may end past ``valid_until_s``,
    so relay plans remain committable while the prediction horizon exceeds
    the plan window.
    """
    segment_plan = plan.segment_plan
    if segment_plan is None:
        return
    xmin, xmax, ymin, ymax = config.bounds
    for index, segment in enumerate(segment_plan.segments):
        if segment.index != index:
            issues.append(
                ValidationIssue(
                    code="segment_index_gap",
                    field=f"segment_plan.segments[{index}]",
                    message="segment indices must be contiguous from 0",
                )
            )
        if segment.end_s <= segment.start_s:
            issues.append(
                ValidationIssue(
                    code="segment_time_invalid",
                    field=f"segment_plan.segments[{index}]",
                    message="segment end must follow its start",
                )
            )
        x, y = segment.intercept_xy
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            issues.append(
                ValidationIssue(
                    code="segment_out_of_bounds",
                    field=f"segment_plan.segments[{index}]",
                    message="segment intercept outside the scenario box",
                )
            )
        if segment.start_s < plan.valid_from_s:
            issues.append(
                ValidationIssue(
                    code="segment_past",
                    field=f"segment_plan.segments[{index}]",
                    message="segment starts before the plan window",
                )
            )


def _check_evidence(
    snapshot: PlanningSnapshot,
    plan: TrackingPlan,
    issues: list[ValidationIssue],
) -> None:
    """Every referenced evidence observation must exist in the situation."""
    known_ids = {
        observation_id
        for report in snapshot.situation.group_reports
        for observation_id in report.belief.source_observation_ids
    }
    for evidence_id in plan.evidence_ids:
        if evidence_id not in known_ids and not evidence_id.startswith(
            f"{snapshot.scenario_id}:knowledge:"
        ):
            issues.append(
                ValidationIssue(
                    code="evidence_unresolved",
                    field="evidence_ids",
                    message=f"observation {evidence_id} is unknown",
                )
            )


def _member_action(plan: TrackingPlan, target: str, member: str) -> str:
    """Deterministic per-member execution action from the plan's action maps."""
    if member in plan.return_actions:
        return plan.return_actions[member]
    if member in plan.rotation_uuv_ids:
        return "rotate"
    return "track"


def _longest(sequences: Sequence[Sequence[object]]) -> int:
    return max((len(sequence) for sequence in sequences), default=0)


def _report(snapshot: PlanningSnapshot, target: str) -> GroupReport:
    report = _report_or_none(snapshot, target)
    if report is None:
        raise ValueError(f"no group report for target {target!r}")
    return report


def _report_or_none(snapshot: PlanningSnapshot, target: str) -> GroupReport | None:
    for report in snapshot.situation.group_reports:
        if report.target_id == target:
            return report
    return None


def _belief(snapshot: PlanningSnapshot, target: str) -> TargetBelief:
    return _report(snapshot, target).belief


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
