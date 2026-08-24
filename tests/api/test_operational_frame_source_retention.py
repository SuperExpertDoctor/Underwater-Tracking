from underwater_tracking.api.live import compact_operational_frame
from underwater_tracking.domain.models import EventLevel
from underwater_tracking.domain.ui_models import (
    EventView,
    LedgerView,
    MapBounds,
    OperationalFrame,
    PlanTimelineView,
    TimelineFactorView,
)


def test_compact_frame_retains_event_sources_referenced_by_visible_views() -> None:
    events = tuple(
        EventView(
            event_id=f"event-{index}",
            sim_time_s=index,
            event_type="test_event",
            level=EventLevel.INFORMATIONAL,
        )
        for index in range(80)
    )
    frame = OperationalFrame(
        frame_id=80,
        sim_time_s=80,
        plan_version=2,
        map_bounds=MapBounds(min_x=-1.0, min_y=-1.0, max_x=1.0, max_y=1.0),
        events=events,
        ledger=(
            LedgerView(
                decision_id="decision-1",
                sim_time_s=1,
                trigger_event_ids=("event-0",),
            ),
        ),
        plan_timeline=(
            PlanTimelineView(
                adjustment_id="adjustment-1",
                sim_time_s=1,
                factors=(
                    TimelineFactorView(
                        kind="event",
                        ref_id="event-0",
                        label="trigger",
                    ),
                ),
            ),
        ),
        llm_thinking="current rationale",
        llm_thinking_source_event_ids=("event-1",),
    )

    compact = compact_operational_frame(frame)
    compact_ids = {event.event_id for event in compact.events}

    assert {"event-0", "event-1"} <= compact_ids


def test_compact_frame_retains_long_memory_audit_source_history() -> None:
    frame = OperationalFrame(
        frame_id=300,
        sim_time_s=300,
        plan_version=2,
        map_bounds=MapBounds(min_x=-1.0, min_y=-1.0, max_x=1.0, max_y=1.0),
        operator_audit_event_ids=tuple(f"audit-{index}" for index in range(300)),
    )

    compact = compact_operational_frame(frame)

    assert compact.operator_audit_event_ids[0] == "audit-0"
    assert compact.operator_audit_event_ids[-1] == "audit-299"
