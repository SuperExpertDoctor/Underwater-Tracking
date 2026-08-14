# tests/property/test_foundation_invariants.py
"""Property tests for the deterministic foundation's approved invariants.

Each approved invariant is checked against randomized inputs via
Hypothesis: angle wrapping, Fisher information positive
semidefiniteness, quality score bounds, allocation stability under
UUV-input permutation, waypoint motion/boundary constraints, and the
absence of ``TargetTruth``/``domain.truth`` imports in operational
packages. Every property must hold against the existing implementation;
a failing example here is a genuine invariant violation, not a flake.
"""

import ast
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from hypothesis import assume, given, settings, strategies as st

from underwater_tracking.planning.allocation import AllocationInput, allocate_groups
from underwater_tracking.planning.fim import bearing_fim, fim_metrics
from underwater_tracking.planning.waypoints import plan_group_waypoints
from underwater_tracking.tracking.angles import wrap_angle
from underwater_tracking.tracking.quality import QualityCalculator, QualityInputs

# Repository root resolved from this file, so the scan works from any CWD.
_REPO_ROOT = Path(__file__).parents[2]
_SRC_ROOT = _REPO_ROOT / "src"

# Scenario box used by the waypoint properties.
_BOUNDS = (-5000.0, 5000.0, -5000.0, 5000.0)


# --- Angle wrapping ---------------------------------------------------------


@given(st.floats(-1e6, 1e6))
def test_wrap_angle_scalar_lies_in_half_open_unit_circle(value: float) -> None:
    wrapped = float(wrap_angle(value))
    assert -math.pi <= wrapped < math.pi


@given(st.lists(st.floats(-1e6, 1e6), min_size=1, max_size=64))
def test_wrap_angle_array_matches_scalar_elementwise(values: list[float]) -> None:
    wrapped = wrap_angle(np.asarray(values))
    assert wrapped.shape == (len(values),)
    for value, result in zip(values, wrapped, strict=True):
        assert result == pytest.approx(float(wrap_angle(value)))


@given(st.floats(-1e6, 1e6))
def test_wrap_angle_is_idempotent(value: float) -> None:
    once = float(wrap_angle(value))
    twice = float(wrap_angle(once))
    assert twice == pytest.approx(once, abs=1e-12)


@given(st.floats(-1e6, 1e6))
def test_wrap_angle_is_periodic_in_2pi(value: float) -> None:
    # ``value + 2*pi`` is itself rounded, so the wrapped images agree
    # only up to float noise, never exactly.
    assert float(wrap_angle(value + 2.0 * math.pi)) == pytest.approx(
        float(wrap_angle(value)), abs=1e-6
    )


# --- Fisher information positive semidefiniteness ---------------------------


@given(
    st.tuples(st.floats(-2000.0, 2000.0), st.floats(-2000.0, 2000.0)),
    st.lists(
        st.tuples(st.floats(-2000.0, 2000.0), st.floats(-2000.0, 2000.0)),
        min_size=1,
        max_size=8,
    ),
    st.lists(st.floats(1e-6, 1.0), min_size=1, max_size=8),
)
@settings(max_examples=100)
def test_bearing_fim_is_positive_semidefinite(
    target: tuple[float, float],
    observers: list[tuple[float, float]],
    variances: list[float],
) -> None:
    assume(len(observers) == len(variances))
    target_array = np.asarray(target, dtype=float)
    observer_array = np.asarray(observers, dtype=float)
    for observer in observer_array:
        assume(float(np.sum((observer - target_array) ** 2)) >= 1e-6)
    fim = bearing_fim(target_array, observer_array, np.asarray(variances, dtype=float))
    # The information matrix is a sum of rank-one PSD outer products, so
    # every eigenvalue must be non-negative up to float rounding.
    eigenvalues = np.linalg.eigvalsh(fim)
    assert eigenvalues[0] >= -1e-9
    # The metrics reduction never reports a negative minimum eigenvalue.
    metrics = fim_metrics(fim)
    assert metrics.min_eigenvalue >= 0.0
    # The matrix is symmetric by construction.
    np.testing.assert_array_equal(fim, fim.T)


# --- Quality score bounds ----------------------------------------------------


@st.composite
def quality_updates(draw) -> tuple[list[tuple[float, QualityInputs]], int, float]:
    """A random calculator configuration and an increasing update timeline."""
    window_s = draw(st.integers(1, 1000))
    ewma_alpha = draw(st.floats(0.05, 1.0))
    count = draw(st.integers(1, 30))
    gap = draw(st.floats(1.0, 600.0))
    updates: list[tuple[float, QualityInputs]] = []
    time_s = 0.0
    for _ in range(count):
        inputs = QualityInputs(
            covariance_trace=draw(st.floats(0.0, 1e8)),
            fim_min_eigenvalue=draw(st.floats(0.0, 1e3)),
            fim_condition=draw(st.floats(1.0, 1e12)),
            detection_rate=draw(st.floats(0.0, 1.0)),
            normalized_nis=draw(st.floats(0.0, 1.0)),
            age_s=draw(st.floats(0.0, 1e6)),
        )
        updates.append((time_s, inputs))
        time_s += gap
    return updates, window_s, ewma_alpha


@given(quality_updates())
@settings(max_examples=100, deadline=None)
def test_quality_instant_window_and_ewma_stay_in_unit_interval(group) -> None:
    updates, window_s, ewma_alpha = group
    calculator = QualityCalculator(window_s=window_s, ewma_alpha=ewma_alpha)
    for time_s, inputs in updates:
        result = calculator.update(time_s, inputs)
        assert 0.0 <= result.instant <= 1.0
        assert 0.0 <= result.window_mean <= 1.0
        assert 0.0 <= result.ewma <= 1.0
        assert all(0.0 <= value <= 1.0 for value in result.components.values())


# --- Allocation stability under UUV input permutation -----------------------


@st.composite
def allocation_problems(draw) -> AllocationInput:
    """A random allocation problem keyed entirely by uuv and target ids."""
    uuv_count = draw(st.integers(6, 10))
    target_count = draw(st.integers(1, 3))
    uuv_ids = tuple(f"uuv_{i}" for i in range(uuv_count))
    target_ids = tuple(f"target_{i}" for i in range(target_count))
    quality_by_target = {
        target: draw(st.floats(0.0, 1.0)) for target in target_ids
    }
    uuv_available = {uuv: draw(st.booleans()) for uuv in uuv_ids}
    draw_feasible = draw(st.booleans())
    feasible_pairs = None
    if draw_feasible:
        feasible_pairs = {
            (uuv, target)
            for uuv in uuv_ids
            for target in target_ids
            if draw(st.booleans())
        }
    prior_members = {
        target: tuple(
            uuv
            for uuv in uuv_ids
            if draw(st.booleans()) and uuv_available.get(uuv, True)
        )
        for target in target_ids
    }
    assignment_age_s = {
        target: draw(st.floats(0.0, 1200.0)) for target in target_ids
    }
    target_degraded = frozenset(
        target for target in target_ids if draw(st.booleans())
    )
    energy_cost = {
        (uuv, target): draw(st.floats(0.0, 100.0))
        for uuv in uuv_ids
        for target in target_ids
        if draw(st.booleans())
    }
    uuv_energy_fraction = {
        uuv: draw(st.floats(0.0, 1.0)) for uuv in uuv_ids
    }
    return AllocationInput(
        uuv_ids=uuv_ids,
        target_ids=target_ids,
        quality_by_target=quality_by_target,
        uuv_available=uuv_available,
        prior_members=prior_members,
        assignment_age_s=assignment_age_s,
        feasible_pairs=feasible_pairs,
        target_degraded=target_degraded,
        energy_cost=energy_cost,
        uuv_energy_fraction=uuv_energy_fraction,
    )


@given(allocation_problems(), st.data())
@settings(max_examples=50, deadline=None)
def test_allocation_is_invariant_under_uuv_id_permutation(problem, data) -> None:
    permutation = data.draw(st.permutations(range(len(problem.uuv_ids))))
    permuted_ids = tuple(problem.uuv_ids[i] for i in permutation)
    permuted = replace(problem, uuv_ids=permuted_ids)
    base = allocate_groups(problem)
    reordered = allocate_groups(permuted)
    assert reordered.members_by_target == base.members_by_target
    assert reordered.reserve_ids == base.reserve_ids
    assert reordered.objective == base.objective
    assert reordered.solver_status == base.solver_status


# --- Waypoint bounds ----------------------------------------------------------


@st.composite
def waypoint_groups(draw) -> tuple[np.ndarray, np.ndarray, float, float]:
    """A feasible two/three-UUV tracking group near an asymmetric sigma set.

    Every UUV sits 500-2000 m from the sigma mean (so the planner's
    lattice always has reachable candidates), pairwise at least 400 m
    apart, and the drawn step/separation bounds match the existing
    planner tests so a feasible joint assignment always exists.
    """
    offsets = np.array([[50.0, 0.0], [0.0, 130.0], [-70.0, -40.0]], dtype=float)
    sigma_points = offsets - offsets.mean(axis=0)
    sigma_mean = sigma_points.mean(axis=0)
    n_uuvs = draw(st.integers(2, 3))
    angles = draw(
        st.lists(st.floats(0.0, 2.0 * math.pi), min_size=n_uuvs, max_size=n_uuvs)
    )
    distances = draw(
        st.lists(st.floats(500.0, 2000.0), min_size=n_uuvs, max_size=n_uuvs)
    )
    positions = np.stack(
        [
            sigma_mean + distance * np.array([math.cos(angle), math.sin(angle)])
            for angle, distance in zip(angles, distances)
        ]
    )
    for i in range(n_uuvs):
        for j in range(i + 1, n_uuvs):
            assume(np.linalg.norm(positions[i] - positions[j]) >= 400.0)
    max_step_m = draw(st.floats(600.0, 1200.0))
    min_separation_m = draw(st.floats(200.0, 300.0))
    return positions, sigma_points, max_step_m, min_separation_m


@given(waypoint_groups())
@settings(max_examples=50, deadline=None)
def test_every_waypoint_respects_scenario_bounds_and_motion(group) -> None:
    positions, sigma_points, max_step_m, min_separation_m = group
    result = plan_group_waypoints(
        positions,
        sigma_points,
        None,
        max_step_m,
        min_separation_m,
        1e-3,
        32,
        min_range_m=800.0,
        max_range_m=2000.0,
        range_bins=5,
        horizon_steps=3,
        bounds=_BOUNDS,
    )
    sequence = result.sequence_xy
    assert sequence.shape == (positions.shape[0], 3, 2)
    np.testing.assert_array_equal(result.waypoints_xy, sequence[:, 0, :])
    assert result.separation_violated is False
    for step in range(sequence.shape[1]):
        origin = positions if step == 0 else sequence[:, step - 1, :]
        step_lengths = np.linalg.norm(sequence[:, step, :] - origin, axis=1)
        assert np.all(step_lengths <= max_step_m * (1.0 + 1e-9) + 1e-6)
        assert np.all(sequence[:, step, 0] >= _BOUNDS[0] - 1e-6)
        assert np.all(sequence[:, step, 0] <= _BOUNDS[1] + 1e-6)
        assert np.all(sequence[:, step, 1] >= _BOUNDS[2] - 1e-6)
        assert np.all(sequence[:, step, 1] <= _BOUNDS[3] + 1e-6)
        for i in range(positions.shape[0]):
            for j in range(i + 1, positions.shape[0]):
                separation = float(
                    np.linalg.norm(sequence[i, step, :] - sequence[j, step, :])
                )
                assert separation >= min_separation_m * (1.0 - 1e-9) - 1e-6


# --- Absence of TargetTruth in operational packages ---------------------------


def _operational_modules() -> list[Path]:
    """Every source module outside the ``domain`` package, in stable order."""
    package = _SRC_ROOT / "underwater_tracking"
    return sorted(
        path
        for path in package.rglob("*.py")
        if path.relative_to(package).parts[0] != "domain"
    )


def _imported_names(path: Path) -> set[str]:
    """Fully qualified names imported by ``path``, discovered via AST.

    ``from underwater_tracking.domain import truth`` and
    ``from underwater_tracking.domain.truth import TargetTruth`` both
    collapse to names starting with ``underwater_tracking.domain.truth``,
    so a single prefix check covers every import form.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                qualified = f"{module}.{alias.name}" if module else alias.name
                names.add(qualified)
    return names


def test_no_operational_package_imports_target_truth() -> None:
    target = "underwater_tracking.domain.truth"
    offenders = [
        f"{path}: imports {name!r}"
        for path in _operational_modules()
        for name in _imported_names(path)
        if name == target or name.startswith(target + ".")
    ]
    assert offenders == [], (
        "operational packages must not import ground-truth objects:\n"
        + "\n".join(offenders)
    )
