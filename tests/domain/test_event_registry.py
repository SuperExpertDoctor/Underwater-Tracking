"""Authoritative runtime event registry coverage."""

from __future__ import annotations

import pytest

from underwater_tracking.domain.event_registry import (
    EVENT_REGISTRY,
    PRIVATE_AUDIENCES,
    event_definition,
)


RECOVERY_EVENT_TYPES = (
    "target_boundary_recovery_started",
    "target_boundary_turn_started",
    "target_boundary_recovery_completed",
    "target_navigation_recovery_failed",
)


def test_boundary_recovery_events_are_registered_as_private_nonplanning_events() -> None:
    for event_type in RECOVERY_EVENT_TYPES:
        assert event_type in EVENT_REGISTRY
        definition = event_definition(event_type)
        assert definition.audiences == PRIVATE_AUDIENCES
        assert definition.plan_impact_policy == "never"

    assert "target_navigation_guard_failed" not in EVENT_REGISTRY
    with pytest.raises(ValueError, match="unknown event type"):
        event_definition("target_navigation_guard_failed")
