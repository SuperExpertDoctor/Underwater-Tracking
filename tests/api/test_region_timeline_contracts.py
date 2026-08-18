from __future__ import annotations

import pytest
from pydantic import ValidationError

from underwater_tracking.domain.ui_models import (
    MapBounds,
    OperationalFrame,
    Point2D,
    RegionAssignmentView,
    RegionTimelineView,
)


def _timeline(**overrides: object) -> RegionTimelineView:
    payload: dict[str, object] = {
        "region_id": "T1:cell:0:0",
        "target_id": "T1",
        "center": Point2D(x=50.0, y=50.0),
        "bounds": MapBounds(min_x=0.0, min_y=0.0, max_x=100.0, max_y=100.0),
        "start_offset_s": 0.0,
        "end_offset_s": 30.0,
        "status": "active",
        "coverage_mode": "required",
        "priority": 0.8,
        "occupancy_likelihood": 0.7,
        "uuv_assignments": (
            RegionAssignmentView(
                platform_id="uuv-1",
                platform_kind="uuv",
                role="passive_tracker",
                start_offset_s=0.0,
                end_offset_s=30.0,
            ),
        ),
    }
    payload.update(overrides)
    return RegionTimelineView(**payload)


def _frame(*, region_timeline: tuple[RegionTimelineView, ...] = ()) -> OperationalFrame:
    return OperationalFrame(
        frame_id=1,
        sim_time_s=100,
        plan_version=0,
        map_bounds=MapBounds(min_x=-1000.0, min_y=-1000.0, max_x=1000.0, max_y=1000.0),
        region_timeline=region_timeline,
    )


def test_region_timeline_round_trip_keeps_assignments_and_offsets() -> None:
    item = _timeline()
    frame = _frame(region_timeline=(item,))
    restored = OperationalFrame.model_validate_json(frame.model_dump_json())
    assert restored.region_timeline == (item,)


def test_old_operational_frame_without_region_timeline_is_compatible() -> None:
    payload = _frame().model_dump(mode="json")
    payload.pop("region_timeline", None)
    assert OperationalFrame.model_validate(payload).region_timeline == ()


def test_timeline_rejects_negative_duration() -> None:
    with pytest.raises(ValidationError, match="end_offset_s"):
        _timeline(start_offset_s=20.0, end_offset_s=10.0)


def test_timeline_rejects_non_finite_priority() -> None:
    with pytest.raises(ValidationError):
        _timeline(priority=float("nan"))
