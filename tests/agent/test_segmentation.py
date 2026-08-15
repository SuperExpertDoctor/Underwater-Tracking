# tests/agent/test_segmentation.py
"""Trajectory segmentation for relay tracking (spec 6.7/14 amendment, R3).

``default_segment_plan`` splits one predicted track into equal time slices
across the available groups; the optimizer seeds each group's waypoint
lattice from its segment intercept; the verify scan rejects malformed
segments and exempts the ``group_id`` key (never any other segment field).
All tests are offline and deterministic.
"""

import math

from underwater_tracking.agent.nodes.commit import validate_plan
from underwater_tracking.agent.nodes.optimize import (
    PlanningConfig,
    optimize_candidates,
)
from underwater_tracking.agent.nodes.snapshot import build_planning_snapshot
from underwater_tracking.agent.nodes.verify import validate_strategy
from underwater_tracking.domain.agent_models import (
    PredictedTrackRef,
    Segment,
    SegmentPlan,
    StrategyProposal,
    StrategySet,
)
from underwater_tracking.domain.models import (
    GroupQuality,
    GroupReport,
    SituationSnapshot,
    TargetBelief,
    UUVState,
    UUVStatus,
)
from underwater_tracking.planning.segmentation import (
    default_segment_plan,
    initial_intercept,
)

TARGETS = ("T1",)
EVIDENCE = ("B:T1:900",)

PREDICTION = PredictedTrackRef(
    prediction_id="S1:track:T1:3",
    target_id="T1",
    sim_time_s=900,
    horizon_s=600.0,
    sample_step_s=30.0,
    times_s=(930.0, 960.0, 990.0, 1020.0, 1050.0),
    points_xy=(
        (140.0, 230.0),
        (150.0, 240.0),
        (160.0, 250.0),
        (170.0, 260.0),
        (180.0, 270.0),
    ),
    corridor_radius_m=(40.0, 42.0, 44.0, 46.0, 48.0),
)


def _situation() -> SituationSnapshot:
    uuvs = tuple(
        UUVState(
            uuv_id=f"uuv_{index:02d}",
            position_xy=(2000.0 * math.cos(2.0 * math.pi * index / 6),
                         2000.0 * math.sin(2.0 * math.pi * index / 6)),
            heading_rad=0.0,
            speed_mps=4.0,
            energy_fraction=1.0,
            status=UUVStatus.AVAILABLE,
        )
        for index in range(6)
    )
    return SituationSnapshot(
        scenario_id="S1",
        snapshot_revision=3,
        sim_time_s=900,
        uuvs=uuvs,
        group_reports=(
            GroupReport(
                group_id="G-T1",
                target_id="T1",
                sim_time_s=900,
                member_ids=("uuv_00", "uuv_01", "uuv_02"),
                belief=TargetBelief(
                    target_id="T1",
                    sim_time_s=900,
                    mean=(130.0, 220.0, 1.0, 0.5),
                    covariance=(
                        (400.0, 0.0, 0.0, 0.0),
                        (0.0, 400.0, 0.0, 0.0),
                        (0.0, 0.0, 1.0, 0.0),
                        (0.0, 0.0, 0.0, 1.0),
                    ),
                    model_probabilities={"cv": 0.7, "ct": 0.3},
                    source_observation_ids=("B:T1:900", "B:T1:870"),
                    fim_min_eigenvalue=0.005,
                    fim_condition=12.0,
                ),
                quality=GroupQuality(
                    instant=0.8,
                    window_mean=0.75,
                    ewma=0.76,
                    components={"cov": 0.7},
                    hard_guard_reasons=(),
                ),
                plan_revision=1,
            ),
        ),
        pending_events=(),
    )


def _proposal(*, segment_plan: SegmentPlan | None = None) -> StrategyProposal:
    return StrategyProposal(
        concept="balanced",
        target_priorities={"T1": 1.0},
        required_quality={"T1": 0.7},
        reinforcement_policy={"T1": "release_when_stable"},
        releasable_soft_constraints=("energy_reserve_0.1",),
        evidence_ids=EVIDENCE,
        rationale="relay tracking along the predicted track",
        segment_plan=segment_plan,
    )


def test_default_segment_plan_splits_track_uniformly():
    plan = default_segment_plan(PREDICTION, ("G-T1", "G-OTHER"))
    assert [segment.group_id for segment in plan.segments] == ["G-OTHER", "G-T1"]
    assert [segment.index for segment in plan.segments] == [0, 1]
    assert plan.segments[0].end_s == plan.segments[1].start_s
    assert plan.segments[0].start_s == 900
    assert plan.segments[1].end_s == 1500


def test_default_segment_plan_handles_empty_inputs():
    assert default_segment_plan(PREDICTION, ()).segments == ()


def test_initial_intercept_picks_earliest_segment_for_group():
    plan = SegmentPlan(
        segments=(
            Segment(index=0, start_s=900, end_s=1200, group_id="G-OTHER",
                    intercept_xy=(150.0, 240.0)),
            Segment(index=1, start_s=1200, end_s=1500, group_id="G-T1",
                    intercept_xy=(180.0, 270.0)),
        )
    )
    assert initial_intercept(plan, "T1") == (180.0, 270.0)
    assert initial_intercept(None, "T1") is None


def test_validate_strategy_accepts_valid_segments():
    plan = SegmentPlan(
        segments=(
            Segment(index=0, start_s=900, end_s=1200, group_id="G-T1",
                    intercept_xy=(150.0, 240.0)),
        )
    )
    report = validate_strategy(
        _proposal(segment_plan=plan).model_dump(),
        target_ids=TARGETS,
        evidence_ids=EVIDENCE,
        allowed_soft_constraints=("energy_reserve_0.1",),
    )
    assert report.valid is True


def test_validate_strategy_flags_bad_segments():
    bad = SegmentPlan(
        segments=(
            Segment(index=1, start_s=900, end_s=900, group_id="G-T1",
                    intercept_xy=(float("nan"), 240.0)),
        )
    )
    report = validate_strategy(
        _proposal(segment_plan=bad).model_dump(),
        target_ids=TARGETS,
        evidence_ids=EVIDENCE,
        allowed_soft_constraints=("energy_reserve_0.1",),
    )
    assert {issue.code for issue in report.issues} == {
        "segment_index_gap",
        "segment_time_invalid",
        "non_finite",
    }


def test_marker_scan_exempts_group_id_inside_segments():
    plan = SegmentPlan(
        segments=(
            Segment(index=0, start_s=900, end_s=1200, group_id="G-T1",
                    intercept_xy=(150.0, 240.0)),
        )
    )
    report = validate_strategy(
        _proposal(segment_plan=plan).model_dump(),
        target_ids=TARGETS,
        evidence_ids=EVIDENCE,
        allowed_soft_constraints=("energy_reserve_0.1",),
    )
    assert "member_or_waypoint" not in {issue.code for issue in report.issues}


def test_optimize_applies_proposal_segment_plan():
    proposal = _proposal(
        segment_plan=SegmentPlan(
            segments=(
                Segment(index=0, start_s=900, end_s=1500, group_id="G-T1",
                        intercept_xy=(150.0, 240.0)),
            )
        )
    )
    snapshot = build_planning_snapshot(_situation(), active_plan=None, applied_directives=())
    evaluations = optimize_candidates(
        snapshot,
        StrategySet(proposals=(proposal,)),
        config=PlanningConfig(),
    )
    plan = evaluations[0].plan
    assert plan.segment_plan is not None
    assert plan.segment_plan.segments[0].group_id == "G-T1"
    assert plan.member_ids_by_target["T1"]
    assert validate_plan(snapshot, plan, PlanningConfig()) == ()


def test_optimize_falls_back_to_default_segment_plan():
    proposal = _proposal(segment_plan=None)
    snapshot = build_planning_snapshot(_situation(), active_plan=None, applied_directives=())
    evaluations = optimize_candidates(
        snapshot,
        StrategySet(proposals=(proposal,)),
        config=PlanningConfig(),
        predictions={"T1": PREDICTION},
    )
    plan = evaluations[0].plan
    assert plan.segment_plan is not None
    assert plan.segment_plan.segments[0].group_id == "G-T1"
