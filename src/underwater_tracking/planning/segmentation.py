# src/underwater_tracking/planning/segmentation.py
"""Deterministic trajectory segmentation (spec 6.7 amendment, R3).

``SegmentPlan`` partitions a target's predicted track into time slices,
each assigned to one group with an intercept point where that group
initializes its waypoint standoff. LLM proposals may carry a
``segment_plan``; when they do not, ``default_segment_plan`` splits the
predicted track into equal time slices across the available groups — a
small pure function, no LLM.
"""

from __future__ import annotations

from collections.abc import Sequence

from underwater_tracking.domain.agent_models import (
    PredictedTrackRef,
    Segment,
    SegmentPlan,
)


def default_segment_plan(
    prediction: PredictedTrackRef,
    group_ids: Sequence[str],
) -> SegmentPlan:
    """Uniform time split of one predicted track across ``group_ids``.

    Times are absolute simulation times within the prediction horizon;
    each segment's intercept is the track point nearest its midpoint time
    (linear interpolation between bracketing samples). Empty inputs yield
    an empty ``SegmentPlan``. Deterministic: sorted, no randomness.
    """
    ordered = tuple(sorted(set(group_ids)))
    points = prediction.points_xy
    if not ordered or not points:
        return SegmentPlan(segments=())
    times = prediction.times_s
    t0 = prediction.sim_time_s
    horizon_s = float(prediction.horizon_s)
    step = horizon_s / len(ordered)
    segments: list[Segment] = []
    for index, group_id in enumerate(ordered):
        start_s = t0 + index * step
        end_s = t0 + (index + 1) * step
        intercept = _point_at(times, points, 0.5 * (start_s + end_s))
        segments.append(
            Segment(
                index=index,
                start_s=round(start_s),
                end_s=round(end_s),
                group_id=group_id,
                intercept_xy=intercept,
            )
        )
    return SegmentPlan(segments=tuple(segments))


def initial_intercept(
    segment_plan: SegmentPlan | None, target: str
) -> tuple[float, float] | None:
    """The intercept of the earliest segment assigned to this group (R3).

    The segment's intercept point becomes the group's initial waypoint
    target: the waypoint lattice is recentered there (see
    ``_plan_waypoints``), so the group's committed standoff converges on
    the predicted intercept instead of the current belief mean.
    """
    if segment_plan is None:
        return None
    for segment in sorted(segment_plan.segments, key=lambda s: s.index):
        if segment.group_id == f"G-{target}":
            return segment.intercept_xy
    return None


def _point_at(
    times: tuple[float, ...],
    points: tuple[tuple[float, float], ...],
    target_s: float,
) -> tuple[float, float]:
    """The track point nearest ``target_s``, interpolated between samples."""
    if not times:
        return points[len(points) // 2]
    if target_s <= times[0]:
        return points[0]
    if target_s >= times[-1]:
        return points[-1]
    for index in range(len(times) - 1):
        if times[index] <= target_s <= times[index + 1]:
            span = max(times[index + 1] - times[index], 1e-9)
            weight = (target_s - times[index]) / span
            return (
                points[index][0] + weight * (points[index + 1][0] - points[index][0]),
                points[index][1] + weight * (points[index + 1][1] - points[index][1]),
            )
    return points[-1]
