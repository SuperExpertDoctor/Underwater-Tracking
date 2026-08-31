import math

import numpy as np
import pytest
from hypothesis import assume, given, settings, strategies as st
from underwater_tracking.planning.fim import bearing_fim, fim_metrics
from underwater_tracking.planning.waypoints import _score_joint_batch, plan_group_waypoints


def test_two_uuv_waypoints_avoid_collinear_geometry():
    result = plan_group_waypoints(
        uuv_positions=np.array([[-1000.0, 0.0], [-2000.0, 0.0]]),
        target_sigma_points=np.array([[0.0, 0.0], [100.0, 50.0], [-100.0, -50.0]]),
        previous_waypoints=None, max_step_m=900.0, min_separation_m=300.0,
        bearing_variance=1e-3, beam_width=16,
    )
    vectors = result.waypoints_xy - np.array([0.0, 0.0])
    cosine = abs(float(vectors[0] @ vectors[1])) / (np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1]))
    assert cosine < 0.8


def test_waypoints_xy_are_the_committed_first_points():
    result = plan_group_waypoints(
        uuv_positions=np.array([[-1000.0, 0.0], [-2000.0, 0.0]]),
        target_sigma_points=np.array([[0.0, 0.0], [100.0, 50.0], [-100.0, -50.0]]),
        previous_waypoints=None,
        max_step_m=900.0,
        min_separation_m=300.0,
        bearing_variance=1e-3,
        beam_width=16,
    )
    np.testing.assert_array_equal(result.waypoints_xy, result.sequence_xy[:, 0, :])


def test_planner_is_deterministic():
    uuv_positions = np.array([[-1000.0, 0.0], [-2000.0, 0.0]])
    target_sigma_points = np.array([[0.0, 0.0], [100.0, 50.0], [-100.0, -50.0]])
    first = plan_group_waypoints(
        uuv_positions, target_sigma_points, None, 900.0, 300.0, 1e-3, 16
    )
    second = plan_group_waypoints(
        uuv_positions, target_sigma_points, None, 900.0, 300.0, 1e-3, 16
    )
    np.testing.assert_array_equal(first.waypoints_xy, second.waypoints_xy)
    np.testing.assert_array_equal(first.sequence_xy, second.sequence_xy)


def test_score_batch_matches_task7_fim_functions():
    # The planner scores candidate joints with a batched evaluation of
    # the same Fisher information operator as fim.bearing_fim and the
    # same reductions as fim.fim_metrics; pin them against each other.
    sigma_points = np.array([[0.0, 0.0], [100.0, 50.0], [-100.0, -50.0]])
    variance = 1e-3
    joints = np.array(
        [
            [[-250.0, 433.0], [-1328.0, 356.0]],
            [[-500.0, 0.0], [-1375.0, 0.0]],
        ]
    )
    positions = np.array([[-1000.0, 0.0], [-2000.0, 0.0]])
    scores = _score_joint_batch(joints, sigma_points, positions, None, variance)
    for row, waypoints in enumerate(joints):
        min_eigenvalue = float("inf")
        logdet = float("inf")
        for sigma in sigma_points:
            metrics = fim_metrics(
                bearing_fim(sigma, waypoints, np.full(2, variance))
            )
            min_eigenvalue = min(min_eigenvalue, metrics.min_eigenvalue)
            logdet = min(logdet, metrics.logdet)
        assert scores[row, 0] == pytest.approx(min_eigenvalue, rel=1e-9)
        assert scores[row, 1] == pytest.approx(logdet)
        energy = float(np.sum((waypoints - positions) ** 2))
        assert scores[row, 2] == pytest.approx(-energy)
        assert scores[row, 3] == 0.0


def test_score_batch_matches_fim_functions_singular():
    # One-observer and collinear-observer joints are singular for at
    # least one sigma point: logdet is -inf, exactly as fim_metrics
    # reports them. The worst-case min_eigenvalue is exactly zero when
    # the smallest eigenvalue clamps, or a sub-1e-15 floating-point
    # residual when rounding leaves it infinitesimally positive (the
    # determinant of a rank-one matrix is pure rounding noise, and the
    # batch path reproduces fim_metrics' arithmetic bit-for-bit).
    sigma_points = np.array([[0.0, 0.0], [100.0, 50.0], [-100.0, -50.0]])
    variance = 1e-3
    positions = np.array([[-1000.0, 0.0], [-2000.0, 0.0]])
    one_observer = np.array(
        [
            [[-250.0, 433.0]],
            [[-1328.0, 356.0]],
        ]
    )
    scores = _score_joint_batch(
        one_observer, sigma_points, positions[:1], None, variance
    )
    assert np.all(scores[:, 0] == 0.0)
    assert np.all(scores[:, 1] == float("-inf"))
    # Two observers collinear with the origin sigma point: rank-one FIM.
    collinear = np.array([[[-250.0, 433.0], [-500.0, 866.0]]])
    scores = _score_joint_batch(collinear, sigma_points, positions, None, variance)
    assert scores[0, 0] == pytest.approx(0.0, abs=1e-15)
    assert scores[0, 1] == float("-inf")


@st.composite
def random_score_batches(draw):
    """Random joint batch, sigma set, positions, and variance.

    The batch keeps at least one observer per joint (including
    one-observer, necessarily singular joints) and never places a
    waypoint on a sigma point, which the FIM evaluator rejects.
    """
    n_sigma = draw(st.integers(1, 4))
    k = draw(st.integers(1, 3))
    n_joints = draw(st.integers(1, 4))
    sigma_points = np.asarray(
        draw(
            st.lists(
                st.tuples(st.floats(-2000.0, 2000.0), st.floats(-2000.0, 2000.0)),
                min_size=n_sigma,
                max_size=n_sigma,
            )
        )
    )
    waypoints = np.asarray(
        draw(
            st.lists(
                st.lists(
                    st.tuples(st.floats(-2000.0, 2000.0), st.floats(-2000.0, 2000.0)),
                    min_size=k,
                    max_size=k,
                ),
                min_size=n_joints,
                max_size=n_joints,
            )
        )
    )
    positions = np.asarray(
        draw(
            st.lists(
                st.tuples(st.floats(-2000.0, 2000.0), st.floats(-2000.0, 2000.0)),
                min_size=k,
                max_size=k,
            )
        )
    )
    for waypoint in waypoints.reshape(-1, 2):
        for sigma in sigma_points:
            assume(float(np.sum((waypoint - sigma) ** 2)) >= 1e-6)
    variance = draw(st.floats(1e-4, 1e-2))
    return waypoints, sigma_points, positions, variance


@given(random_score_batches())
@settings(max_examples=50, deadline=None)
def test_score_batch_matches_fim_functions_randomized(group):
    # Pins the batched planner scores (via fim.bearing_fim_batch)
    # against the per-call fim.bearing_fim/fim.fim_metrics loop over
    # randomized joint batches, including one-observer and near-singular
    # geometries.
    waypoints, sigma_points, positions, variance = group
    scores = _score_joint_batch(waypoints, sigma_points, positions, None, variance)
    for row, joint in enumerate(waypoints):
        min_eigenvalue = float("inf")
        logdet = float("inf")
        for sigma in sigma_points:
            metrics = fim_metrics(
                bearing_fim(sigma, joint, np.full(joint.shape[0], variance))
            )
            min_eigenvalue = min(min_eigenvalue, metrics.min_eigenvalue)
            logdet = min(logdet, metrics.logdet)
        assert scores[row, 0] == pytest.approx(min_eigenvalue, rel=1e-9, abs=1e-15)
        if math.isinf(logdet):
            assert scores[row, 1] == logdet
        else:
            assert scores[row, 1] == pytest.approx(logdet, rel=1e-9)
        energy = float(np.sum((joint - positions) ** 2))
        assert scores[row, 2] == pytest.approx(-energy, rel=1e-9)
        assert scores[row, 3] == 0.0


def test_plan_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        plan_group_waypoints(
            np.zeros((2, 3)), np.zeros((3, 2)), None, 900.0, 300.0, 1e-3, 16
        )
    with pytest.raises(ValueError):
        plan_group_waypoints(
            np.zeros((2, 2)), np.zeros((3, 2)), None, 0.0, 300.0, 1e-3, 16
        )
    with pytest.raises(ValueError):
        plan_group_waypoints(
            np.zeros((2, 2)), np.zeros((3, 2)), np.zeros((1, 2)), 900.0, 300.0, 1e-3, 16
        )
    with pytest.raises(ValueError):
        plan_group_waypoints(
            np.zeros((2, 2)),
            np.zeros((3, 2)),
            None,
            900.0,
            300.0,
            1e-3,
            16,
            min_range_m=5000.0,
            max_range_m=100.0,
        )
    with pytest.raises(ValueError):
        plan_group_waypoints(
            np.zeros((2, 2)),
            np.zeros((3, 2)),
            None,
            900.0,
            300.0,
            1e-3,
            16,
            uuv_ids=["uuv_0"],
        )


def test_separation_violated_flag_when_no_feasible_joint_exists():
    # min_separation_m (3000 m) exceeds the maximum separation any two
    # reachable waypoints can achieve (start points 800 m apart plus two
    # 900 m steps), so no joint assignment satisfies it. The planner
    # returns the best fallback and flags the violation; the exhaustive
    # certification (2 UUVs, a few dozen candidates: far below the
    # 200_000-leaf cap) runs and finds no feasible joint.
    sigma_points = np.array([[0.0, 1500.0], [100.0, 1550.0], [-100.0, 1450.0]])
    positions = np.array([[0.0, 0.0], [800.0, 0.0]])
    result = plan_group_waypoints(
        positions,
        sigma_points,
        None,
        max_step_m=900.0,
        min_separation_m=3000.0,
        bearing_variance=1e-3,
        beam_width=32,
        min_range_m=800.0,
        max_range_m=2000.0,
        range_bins=5,
        horizon_steps=1,
        bounds=BOUNDS,
    )
    assert result.separation_violated is True
    # The fallback is still a valid plan: inside the scenario box and
    # within the per-step motion bound of the UUV positions.
    step = result.sequence_xy[:, 0, :]
    assert np.all(step[:, 0] >= BOUNDS[0] - 1e-6)
    assert np.all(step[:, 0] <= BOUNDS[1] + 1e-6)
    assert np.all(step[:, 1] >= BOUNDS[2] - 1e-6)
    assert np.all(step[:, 1] <= BOUNDS[3] + 1e-6)
    assert np.all(np.linalg.norm(step - positions, axis=1) <= 900.0 * (1.0 + 1e-9) + 1e-6)


def test_separation_violated_flag_when_certification_cap_exceeded():
    # Three UUVs with a dense lattice (range_bins=20) give hundreds of
    # candidates per UUV, so the joint product exceeds the exhaustive
    # certification cap (200_000 leaves) and the violating beam result
    # is returned unrepaired, flagged for the caller. min_separation_m
    # again exceeds any achievable separation (start triangle side 800 m
    # plus two 900 m steps).
    sigma_points = np.array([[0.0, 1500.0], [100.0, 1550.0], [-100.0, 1450.0]])
    positions = np.array([[0.0, 0.0], [800.0, 0.0], [400.0, 692.8203230275509]])
    result = plan_group_waypoints(
        positions,
        sigma_points,
        None,
        max_step_m=900.0,
        min_separation_m=3000.0,
        bearing_variance=1e-3,
        beam_width=32,
        min_range_m=800.0,
        max_range_m=2000.0,
        range_bins=20,
        horizon_steps=1,
        bounds=BOUNDS,
    )
    assert result.separation_violated is True
    step = result.sequence_xy[:, 0, :]
    assert np.all(step[:, 0] >= BOUNDS[0] - 1e-6)
    assert np.all(step[:, 0] <= BOUNDS[1] + 1e-6)
    assert np.all(step[:, 1] >= BOUNDS[2] - 1e-6)
    assert np.all(step[:, 1] <= BOUNDS[3] + 1e-6)
    assert np.all(np.linalg.norm(step - positions, axis=1) <= 900.0 * (1.0 + 1e-9) + 1e-6)


# --- Hypothesis property tests -------------------------------------------------

# Scenario box used by the property tests; the drawn groups stay well inside.
BOUNDS = (-5000.0, 5000.0, -5000.0, 5000.0)
WIDE_BOUNDS = (-10000.0, 10000.0, -10000.0, 10000.0)


@st.composite
def tracking_groups(draw, max_step_bounds=(600.0, 1200.0)):
    """A feasible two/three-UUV tracking group.

    Sigma points are the three vertices of a fixed asymmetric triangle
    (pairwise distinct distances from the centroid, so no reflection or
    rotation maps the set onto itself), randomly rotated. Every UUV sits
    500-2000 m from the sigma mean, so each UUV always has several
    lattice candidates within the step bound (the planner's lattice with
    ``min_range_m=800`` is used by all property tests). UUV start
    positions are pairwise at least 400 m apart, enforced by assumption
    so Hypothesis shrinking can never drive the draw into an unsatisfied
    rejection loop.
    """
    # A symmetric sigma set would tie the best joint choices at float-
    # noise level and let rounding pick a frame-dependent winner, so the
    # sigma points are deliberately asymmetric and the rigid-transform
    # tests stay exact.
    rotation = draw(st.floats(0.0, 2.0 * math.pi))
    cosine, sine = math.cos(rotation), math.sin(rotation)
    offsets = np.array([[50.0, 0.0], [0.0, 130.0], [-70.0, -40.0]])
    offsets = offsets - offsets.mean(axis=0)
    sigma_points = np.column_stack(
        [
            cosine * offsets[:, 0] - sine * offsets[:, 1],
            sine * offsets[:, 0] + cosine * offsets[:, 1],
        ]
    )
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
    max_step_m = draw(st.floats(*max_step_bounds))
    min_separation_m = draw(st.floats(200.0, 300.0))
    return positions, sigma_points, max_step_m, min_separation_m


@given(tracking_groups())
@settings(max_examples=100)
def test_every_waypoint_respects_motion_bounds_and_separation(group):
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
        bounds=BOUNDS,
    )
    sequence = result.sequence_xy
    assert sequence.shape == (positions.shape[0], 3, 2)
    # Every step satisfies the separation constraint, so the planner
    # must not have flagged a violation.
    assert result.separation_violated is False
    for step in range(sequence.shape[1]):
        origin = positions if step == 0 else sequence[:, step - 1, :]
        step_lengths = np.linalg.norm(sequence[:, step, :] - origin, axis=1)
        assert np.all(step_lengths <= max_step_m * (1.0 + 1e-9) + 1e-6)
        assert np.all(sequence[:, step, 0] >= BOUNDS[0] - 1e-6)
        assert np.all(sequence[:, step, 0] <= BOUNDS[1] + 1e-6)
        assert np.all(sequence[:, step, 1] >= BOUNDS[2] - 1e-6)
        assert np.all(sequence[:, step, 1] <= BOUNDS[3] + 1e-6)
        for i in range(positions.shape[0]):
            for j in range(i + 1, positions.shape[0]):
                separation = np.linalg.norm(sequence[i, step, :] - sequence[j, step, :])
                assert separation >= min_separation_m * (1.0 - 1e-9) - 1e-6


def _worst_case_info(waypoints, sigma_points, bearing_variance=1e-3):
    """Worst-case FIM metrics of a waypoint set over all sigma points."""
    min_eigenvalue = float("inf")
    logdet = float("inf")
    for sigma in sigma_points:
        metrics = fim_metrics(
            bearing_fim(sigma, waypoints, np.full(waypoints.shape[0], bearing_variance))
        )
        min_eigenvalue = min(min_eigenvalue, metrics.min_eigenvalue)
        logdet = min(logdet, metrics.logdet)
    return min_eigenvalue, logdet


@given(
    tracking_groups(max_step_bounds=(600.0, 900.0)),
    st.floats(-1000.0, 1000.0),
    st.floats(-1000.0, 1000.0),
    st.tuples(st.floats(-300.0, 300.0), st.floats(-300.0, 300.0)),
)
@settings(max_examples=50, deadline=None)
def test_translation_moves_waypoints_exactly(group, tx, ty, previous_offset):
    # The beam search is an approximation: pruning keeps only the top
    # `beam_width` prefixes, and near the pruning boundary a prefix that
    # just fits in one frame can fall out of the other, so the pruned
    # result is not exactly translation-equivariant. The *unpruned*
    # argmax, however, is a pure function of geometry: translating every
    # input translates the optimum exactly. This test therefore runs the
    # planner with a beam wide enough to enumerate every joint in this
    # draw space (max_step <= 900 m gives at most ~27 candidates per UUV,
    # so at most 27**3 joints), pinning the exact equivariance property.
    positions, sigma_points, max_step_m, min_separation_m = group
    previous = positions + np.asarray(previous_offset)
    translation = np.array([tx, ty])

    base = plan_group_waypoints(
        positions,
        sigma_points,
        previous,
        max_step_m,
        min_separation_m,
        1e-3,
        100_000,
        min_range_m=800.0,
        max_range_m=2000.0,
        range_bins=5,
        horizon_steps=3,
        bounds=WIDE_BOUNDS,
    )
    moved = plan_group_waypoints(
        positions + translation,
        sigma_points + translation,
        previous + translation,
        max_step_m,
        min_separation_m,
        1e-3,
        100_000,
        min_range_m=800.0,
        max_range_m=2000.0,
        range_bins=5,
        horizon_steps=3,
        bounds=WIDE_BOUNDS,
    )
    np.testing.assert_allclose(
        moved.waypoints_xy, base.waypoints_xy + translation, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        moved.sequence_xy, base.sequence_xy + translation, rtol=1e-6, atol=1e-6
    )


@given(
    tracking_groups(max_step_bounds=(600.0, 900.0)),
    st.sampled_from(tuple(range(0, 360, 15))),
    st.floats(-1000.0, 1000.0),
    st.floats(-1000.0, 1000.0),
    st.tuples(st.floats(-300.0, 300.0), st.floats(-300.0, 300.0)),
)
@settings(max_examples=50, deadline=None)
def test_rotation_preserves_observability(group, rotation_deg, tx, ty, previous_offset):
    # The bearing lattice is rotationally closed for every 15-degree
    # step, so the rotated scenario has the same candidate set. The
    # score's energy term legitimately picks the lowest-travel member of
    # an information-equivalent family of joints, so pointwise
    # equivariance does not hold; the rotation-invariant quantity is the
    # observability of the chosen geometry, which must be preserved. The
    # beam width is the same full-enumeration value as the translation
    # test (at most ~27 candidates per UUV for max_step <= 900 m), so the
    # chosen joint is the unpruned argmax and its info equals the
    # rotated scenario's unpruned optimum exactly.
    positions, sigma_points, max_step_m, min_separation_m = group
    previous = positions + np.asarray(previous_offset)
    theta = math.radians(rotation_deg)
    rotation = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    translation = np.array([tx, ty])
    kwargs = {
        "max_step_m": max_step_m,
        "min_separation_m": min_separation_m,
        "bearing_variance": 1e-3,
        "beam_width": 100_000,
        "min_range_m": 800.0,
        "max_range_m": 2000.0,
        "range_bins": 5,
        "horizon_steps": 3,
        "bounds": WIDE_BOUNDS,
    }
    base = plan_group_waypoints(positions, sigma_points, previous, **kwargs)
    moved_sigma_points = sigma_points @ rotation.T + translation
    moved = plan_group_waypoints(
        positions @ rotation.T + translation,
        moved_sigma_points,
        previous @ rotation.T + translation,
        **kwargs,
    )
    base_min_eigenvalue, base_logdet = _worst_case_info(
        base.waypoints_xy, sigma_points
    )
    moved_min_eigenvalue, moved_logdet = _worst_case_info(
        moved.waypoints_xy, moved_sigma_points
    )
    assert moved_min_eigenvalue == pytest.approx(
        base_min_eigenvalue, rel=1e-6, abs=1e-12
    )
    if base_logdet == float("-inf"):
        assert moved_logdet == float("-inf")
    else:
        assert moved_logdet == pytest.approx(base_logdet, rel=1e-6, abs=1e-12)


@given(tracking_groups(), st.data())
@settings(max_examples=50, deadline=None)
def test_permuting_uuv_input_order_maps_results_by_id(group, data):
    positions, sigma_points, max_step_m, min_separation_m = group
    n_uuvs = positions.shape[0]
    uuv_ids = [f"uuv_{i}" for i in range(n_uuvs)]
    permutation = data.draw(st.permutations(range(n_uuvs)))
    previous_offsets = np.asarray(
        data.draw(
            st.lists(
                st.tuples(st.floats(-300.0, 300.0), st.floats(-300.0, 300.0)),
                min_size=n_uuvs,
                max_size=n_uuvs,
            )
        )
    )
    previous = positions + previous_offsets

    base = plan_group_waypoints(
        positions,
        sigma_points,
        previous,
        max_step_m,
        min_separation_m,
        1e-3,
        32,
        uuv_ids=uuv_ids,
        min_range_m=800.0,
        max_range_m=2000.0,
        range_bins=5,
        horizon_steps=3,
        bounds=BOUNDS,
    )
    permuted = plan_group_waypoints(
        positions[permutation],
        sigma_points,
        previous[permutation],
        max_step_m,
        min_separation_m,
        1e-3,
        32,
        uuv_ids=[uuv_ids[i] for i in permutation],
        min_range_m=800.0,
        max_range_m=2000.0,
        range_bins=5,
        horizon_steps=3,
        bounds=BOUNDS,
    )
    for i in range(n_uuvs):
        np.testing.assert_array_equal(
            permuted.waypoints_xy[i], base.waypoints_xy[permutation[i]]
        )
        np.testing.assert_array_equal(
            permuted.sequence_xy[i], base.sequence_xy[permutation[i]]
        )
