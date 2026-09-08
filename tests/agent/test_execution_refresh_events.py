import pytest

from underwater_tracking.domain.event_registry import (
    EXECUTION_REFRESH_REASON_CODES,
    EXECUTION_REFRESH_STATUSES,
    event_definition,
    validate_event_payload,
)


REFRESH_EVENT_TYPES = (
    "execution_refresh_due",
    "execution_refresh_attempted",
    "execution_refresh_committed",
    "execution_refresh_rejected",
    "execution_refresh_waiting_for_source",
    "execution_snapshot_recovered",
)


def _payload(*, status: str = "waiting_for_source", reason: str = "public_source_expired") -> dict[str, object]:
    return {
        "attempt_id": "S1:execution-refresh:1",
        "refresh_status": status,
        "reason": reason,
        "reason_code": reason,
        "execution_revision": 1,
        "candidate_execution_revision": 2,
        "source_snapshot_revision": 4,
        "prediction_revision": 8,
    }


def test_execution_refresh_events_are_registered_as_public_memory_sources() -> None:
    for event_type in REFRESH_EVENT_TYPES:
        definition = event_definition(event_type)
        assert "operator_audit" in {audience.value for audience in definition.audiences}
        assert "memory_source" in {audience.value for audience in definition.audiences}


def test_execution_refresh_payload_accepts_only_bounded_status_and_reason_codes() -> None:
    validate_event_payload(
        "execution_refresh_waiting_for_source",
        _payload(),
    )
    assert "public_source_expired" in EXECUTION_REFRESH_REASON_CODES
    assert "waiting_for_source" in EXECUTION_REFRESH_STATUSES

    with pytest.raises(ValueError, match="reason"):
        validate_event_payload(
            "execution_refresh_waiting_for_source",
            _payload(reason="Traceback: provider prompt contents"),
        )


def test_execution_refresh_payload_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="refresh_status"):
        validate_event_payload(
            "execution_refresh_attempted",
            _payload(status="provider_stack_trace"),
        )
