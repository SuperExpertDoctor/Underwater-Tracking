from dataclasses import replace

import pytest

from underwater_tracking.planning.allocation import (
    AllocationInput,
    AllocationObjective,
    AllocationSolution,
    allocate_groups,
)
from underwater_tracking.agent.nodes.optimize import CandidateEvaluation, CandidateMetrics, _sort_key
from underwater_tracking.domain.agent_models import TrackingPlan
from underwater_tracking.planning.validator import validate_allocation


def test_allocator_never_assigns_a_uuv_without_passive_sonar() -> None:
    problem = AllocationInput(
        uuv_ids=("uuv_0", "uuv_1", "uuv_2"),
        target_ids=("target_0",),
        quality_by_target={"target_0": 0.8},
        uuv_passive_sonar_available={"uuv_0": False},
    )

    solution = allocate_groups(problem)

    assert solution.hard_violations == ()
    assert "uuv_0" not in solution.members_by_target["target_0"]


@pytest.mark.parametrize("priority", [float("nan"), float("inf")])
def test_allocator_rejects_non_finite_target_priority(priority: float) -> None:
    with pytest.raises(ValueError, match="priority"):
        AllocationInput(
            uuv_ids=("uuv_0", "uuv_1"),
            target_ids=("target_0",),
            quality_by_target={"target_0": 0.8},
            target_priority_by_target={"target_0": priority},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("quality_by_target", {"target_0": float("nan")}),
        ("assignment_age_s", {"target_0": float("inf")}),
        ("energy_cost", {("uuv_0", "target_0"): float("nan")}),
        ("travel_cost", {("uuv_0", "target_0"): float("inf")}),
        ("rotation_cost", {("uuv_0", "target_0"): float("nan")}),
        ("uuv_energy_fraction", {"uuv_0": float("inf")}),
        ("quality_warning", float("nan")),
        ("quality_release", float("inf")),
        ("release_hold_s", float("nan")),
        ("reassignment_penalty", float("inf")),
        ("required_quality_by_target", {"target_0": float("nan")}),
        ("target_priority_by_target", {"target_0": float("inf")}),
        ("uuv_passive_range_m", {"uuv_0": float("nan")}),
        ("uuv_bearing_variance_rad2", {"uuv_0": float("inf")}),
        ("uuv_speed_mps", {"uuv_0": float("nan")}),
        ("uuv_max_turn_rate_rad_s", {"uuv_0": float("inf")}),
        ("uuv_endurance_s", {"uuv_0": float("nan")}),
        ("uuv_availability", {"uuv_0": float("inf")}),
        ("plan_horizon_s", float("nan")),
        ("rotation_threshold", float("inf")),
    ],
)
def test_allocator_rejects_non_finite_float_inputs(field, value) -> None:
    problem = AllocationInput.synthetic(uuv_count=2, target_count=1)
    with pytest.raises(ValueError):
        replace(problem, **{field: value})


def test_allocator_rejects_availability_above_one() -> None:
    problem = AllocationInput.synthetic(uuv_count=2, target_count=1)
    with pytest.raises(ValueError, match="availability"):
        replace(problem, uuv_availability={"uuv_0": 1.1})


def test_target_priority_changes_normal_milp_assignment() -> None:
    problem = AllocationInput(
        uuv_ids=("uuv_0", "uuv_1", "uuv_2", "uuv_3"),
        target_ids=("high", "low"),
        quality_by_target={"high": 0.8, "low": 0.8},
        target_priority_by_target={"high": 10.0, "low": 0.0},
        uuv_bearing_variance_rad2={
            "uuv_0": 0.1,
            "uuv_1": 0.1,
            "uuv_2": 0.005,
            "uuv_3": 0.005,
        },
        reassignment_penalty=1.0,
    )

    solution = allocate_groups(problem)

    assert solution.solver_status == "milp"
    assert set(solution.members_by_target["high"]) == {"uuv_2", "uuv_3"}


def test_allocator_uses_two_members_when_quality_is_feasible():
    problem = AllocationInput.synthetic(uuv_count=6, target_count=2, feasible_pair_quality=0.8)
    solution = allocate_groups(problem)
    assert all(len(members) == 2 for members in solution.members_by_target.values())
    assert len(solution.reserve_ids) == 2
    assert solution.hard_violations == ()


def test_target_growth_from_two_to_four_keeps_two_members_per_target():
    for target_count in (2, 3, 4):
        problem = AllocationInput.synthetic(
            uuv_count=12, target_count=target_count, feasible_pair_quality=0.8
        )
        solution = allocate_groups(problem)
        assert solution.hard_violations == ()
        assert all(len(members) == 2 for members in solution.members_by_target.values())
        assert len(solution.reserve_ids) == 12 - 2 * target_count


def test_failed_member_is_excluded_from_groups_and_reserves():
    uuv_ids = tuple(f"uuv_{i}" for i in range(12))
    problem = AllocationInput(
        uuv_ids=uuv_ids,
        target_ids=("target_0", "target_1"),
        quality_by_target={"target_0": 0.8, "target_1": 0.8},
        uuv_available={uuv: uuv != "uuv_1" for uuv in uuv_ids},
    )
    solution = allocate_groups(problem)
    assert solution.hard_violations == ()
    members = [uuv for group in solution.members_by_target.values() for uuv in group]
    assert "uuv_1" not in members
    assert "uuv_1" not in solution.reserve_ids
    assert all(len(group) >= 2 for group in solution.members_by_target.values())
    assert len(solution.reserve_ids) == 7


def test_prior_three_member_group_stays_unchanged_between_warning_and_release():
    uuv_ids = tuple(f"uuv_{i}" for i in range(12))
    problem = AllocationInput(
        uuv_ids=uuv_ids,
        target_ids=("target_0",),
        quality_by_target={"target_0": 0.7},
        prior_members={"target_0": ("uuv_0", "uuv_1", "uuv_2")},
    )
    solution = allocate_groups(problem)
    assert solution.hard_violations == ()
    assert tuple(solution.members_by_target["target_0"]) == ("uuv_0", "uuv_1", "uuv_2")
    assert len(solution.reserve_ids) == 9


def test_release_only_after_the_configured_hold_time():
    uuv_ids = tuple(f"uuv_{i}" for i in range(12))
    prior = {"target_0": ("uuv_0", "uuv_1", "uuv_2")}

    young = AllocationInput(
        uuv_ids=uuv_ids,
        target_ids=("target_0",),
        quality_by_target={"target_0": 0.8},
        prior_members=prior,
        assignment_age_s={"target_0": 300.0},
    )
    solution = allocate_groups(young)
    assert solution.hard_violations == ()
    assert tuple(solution.members_by_target["target_0"]) == ("uuv_0", "uuv_1", "uuv_2")

    mature = AllocationInput(
        uuv_ids=uuv_ids,
        target_ids=("target_0",),
        quality_by_target={"target_0": 0.8},
        prior_members=prior,
        assignment_age_s={"target_0": 601.0},
    )
    solution = allocate_groups(mature)
    assert solution.hard_violations == ()
    members = tuple(solution.members_by_target["target_0"])
    assert len(members) == 2
    released = [uuv for uuv in prior["target_0"] if uuv not in members]
    assert len(released) == 1
    assert released[0] in solution.reserve_ids


def test_low_quality_grows_group_to_three_members():
    problem = AllocationInput.synthetic(
        uuv_count=12, target_count=1, feasible_pair_quality=0.5
    )
    solution = allocate_groups(problem)
    assert solution.hard_violations == ()
    assert len(solution.members_by_target["target_0"]) == 3
    assert len(solution.reserve_ids) == 9


def test_prior_four_member_group_stays_until_release_conditions_are_met():
    uuv_ids = tuple(f"uuv_{i}" for i in range(12))
    prior = {"target_0": ("uuv_0", "uuv_1", "uuv_2", "uuv_3")}

    mid = AllocationInput(
        uuv_ids=uuv_ids,
        target_ids=("target_0",),
        quality_by_target={"target_0": 0.7},
        prior_members=prior,
    )
    solution = allocate_groups(mid)
    assert tuple(solution.members_by_target["target_0"]) == ("uuv_0", "uuv_1", "uuv_2", "uuv_3")

    mature = AllocationInput(
        uuv_ids=uuv_ids,
        target_ids=("target_0",),
        quality_by_target={"target_0": 0.8},
        prior_members=prior,
        assignment_age_s={"target_0": 601.0},
    )
    solution = allocate_groups(mature)
    assert solution.hard_violations == ()
    members = tuple(solution.members_by_target["target_0"])
    assert len(members) == 3
    released = [uuv for uuv in prior["target_0"] if uuv not in members]
    assert len(released) == 1
    assert released[0] in solution.reserve_ids


def test_validator_flags_hard_constraint_violations():
    problem = AllocationInput.synthetic(uuv_count=6, target_count=2, feasible_pair_quality=0.8)
    broken = AllocationSolution(
        members_by_target={"target_0": ("uuv_0",), "target_1": ("uuv_1", "uuv_2", "uuv_0")},
        reserve_ids=("uuv_3", "uuv_4", "uuv_5"),
        objective=AllocationObjective(
            active_count=4,
            energy_cost=0.0,
            travel_cost=0.0,
            reassignment_cost=0.0,
            rotation_cost=0.0,
            reserve_health=3.0,
        ),
        solver_status="milp",
        hard_violations=(),
    )
    violations = validate_allocation(problem, broken)
    assert "target target_0: requires 2..3 members, got 1" in violations
    assert any("assigned to more than one target" in message for message in violations)

    constrained = AllocationInput(
        uuv_ids=("uuv_0", "uuv_1", "uuv_2", "uuv_3"),
        target_ids=("target_0",),
        quality_by_target={"target_0": 0.8},
        uuv_available={"uuv_2": False},
        feasible_pairs={("uuv_0", "target_0")},
    )
    broken = AllocationSolution(
        members_by_target={"target_0": ("uuv_1", "uuv_2")},
        reserve_ids=("uuv_0", "uuv_3"),
        objective=AllocationObjective(
            active_count=2,
            energy_cost=0.0,
            travel_cost=0.0,
            reassignment_cost=0.0,
            rotation_cost=0.0,
            reserve_health=2.0,
        ),
        solver_status="milp",
        hard_violations=(),
    )
    violations = validate_allocation(constrained, broken)
    assert "uuv uuv_2: unavailable but assigned to target target_0" in violations
    assert "uuv uuv_1: assigned to infeasible target pair (uuv_1, target_0)" in violations


def test_infeasible_problem_reports_violations():
    problem = AllocationInput.synthetic(uuv_count=1, target_count=2, feasible_pair_quality=0.8)
    solution = allocate_groups(problem)
    assert solution.solver_status == "infeasible"
    assert solution.hard_violations != ()
    assert all(not members for members in solution.members_by_target.values())


def test_fallback_produces_a_valid_solution_when_milp_is_unavailable(monkeypatch):
    import underwater_tracking.planning.allocation as allocation_module

    def raise_unavailable(*args, **kwargs):
        raise RuntimeError("milp unavailable")

    monkeypatch.setattr(allocation_module, "milp", raise_unavailable)
    problem = AllocationInput.synthetic(uuv_count=6, target_count=2, feasible_pair_quality=0.8)
    solution = allocate_groups(problem)
    assert solution.solver_status == "fallback"
    assert all(len(members) == 2 for members in solution.members_by_target.values())
    assert len(solution.reserve_ids) == 2
    assert solution.hard_violations == ()


def test_fallback_prefers_healthy_reserves_and_matches_milp(monkeypatch):
    import underwater_tracking.planning.allocation as allocation_module

    # Uniform costs (the synthetic default): the tier-4 tie-break is the
    # only thing separating solutions. The two least healthy UUVs must be
    # assigned so the healthiest UUVs stay in reserve.
    problem = AllocationInput(
        uuv_ids=tuple(f"uuv_{i}" for i in range(6)),
        target_ids=("target_0",),
        quality_by_target={"target_0": 0.8},
        uuv_energy_fraction={
            "uuv_0": 0.95,
            "uuv_1": 0.10,
            "uuv_2": 0.11,
            "uuv_3": 0.12,
            "uuv_4": 0.13,
            "uuv_5": 0.14,
        },
    )
    milp_solution = allocate_groups(problem)
    assert milp_solution.solver_status == "milp"

    def raise_unavailable(*args, **kwargs):
        raise RuntimeError("milp unavailable")

    monkeypatch.setattr(allocation_module, "milp", raise_unavailable)
    fallback_solution = allocate_groups(problem)
    assert fallback_solution.solver_status == "fallback"
    assert fallback_solution.hard_violations == ()
    assert set(fallback_solution.members_by_target["target_0"]) == {"uuv_1", "uuv_2"}
    assert "uuv_0" in fallback_solution.reserve_ids
    assert fallback_solution.members_by_target == milp_solution.members_by_target


def test_allocation_is_deterministic():
    problem = AllocationInput.synthetic(uuv_count=12, target_count=4, feasible_pair_quality=0.8)
    first = allocate_groups(problem)
    second = allocate_groups(problem)
    assert first.members_by_target == second.members_by_target
    assert first.reserve_ids == second.reserve_ids
    assert first.objective == second.objective
    assert first.solver_status == second.solver_status


def test_reserved_uuvs_are_never_assigned():
    """Human-assigned UUVs (spec 17.2) are excluded from every solution."""
    problem = AllocationInput.synthetic(
        uuv_count=6,
        target_count=2,
        reserved_uuv_ids=frozenset({"uuv_1", "uuv_4"}),
    )
    solution = allocate_groups(problem)
    assert solution.solver_status == "milp"
    for members in solution.members_by_target.values():
        assert not (set(members) & {"uuv_1", "uuv_4"})


def test_fully_reserved_problem_cannot_form_a_group():
    problem = AllocationInput.synthetic(
        uuv_count=2,
        target_count=1,
        reserved_uuv_ids=frozenset({"uuv_0", "uuv_1"}),
    )
    solution = allocate_groups(problem)
    assert solution.members_by_target.get("target_0", ()) == ()


def test_candidate_order_accounts_for_quality_deficit_and_priority_loss():
    plan = TrackingPlan(
        plan_id="P1",
        scenario_id="S1",
        revision=1,
        base_snapshot_revision=0,
    )
    quality_met = CandidateEvaluation(
        plan=plan,
        metrics=CandidateMetrics(
            hard_violations=(),
            active_count=3,
            economic_cost=10.0,
            quality_deficit=0.0,
            priority_loss=0.0,
        ),
        index=1,
    )
    quality_shortfall = CandidateEvaluation(
        plan=plan,
        metrics=CandidateMetrics(
            hard_violations=(),
            active_count=2,
            economic_cost=1.0,
            quality_deficit=0.1,
            priority_loss=0.2,
        ),
        index=0,
    )

    assert _sort_key(quality_met) < _sort_key(quality_shortfall)


def test_required_quality_grows_group_and_reports_projected_deficit():
    problem = AllocationInput(
        uuv_ids=tuple(f"uuv_{i}" for i in range(6)),
        target_ids=("target_0",),
        quality_by_target={"target_0": 0.5},
        required_quality_by_target={"target_0": 0.6},
    )

    solution = allocate_groups(problem)

    assert len(solution.members_by_target["target_0"]) == 3
    assert solution.hard_violations == ()
