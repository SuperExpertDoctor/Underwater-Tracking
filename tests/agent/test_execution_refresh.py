from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import ValidationError

from underwater_tracking.config.models import ExecutionRefreshConfig
from underwater_tracking.domain.execution_models import OperationalExecutionSnapshot
from underwater_tracking.runtime.execution_refresh import decide_execution_refresh


def _snapshot(
    *,
    valid_from_s: float = 100.0,
    valid_until_s: float = 550.0,
    source_snapshot_revision: int = 4,
    prediction_revision: int = 8,
) -> OperationalExecutionSnapshot:
    return cast(
        OperationalExecutionSnapshot,
        SimpleNamespace(
            valid_from_s=valid_from_s,
            valid_until_s=valid_until_s,
            source_snapshot_revision=source_snapshot_revision,
            prediction_revision=prediction_revision,
        ),
    )


def test_missing_snapshot_is_a_recovery_refresh() -> None:
    decision = decide_execution_refresh(
        None,
        sim_time_s=10.0,
        refresh_margin_s=120,
        source_track_revision=1,
        prediction_revision=1,
    )

    assert decision.due is True
    assert decision.recovery is True
    assert decision.reason == "missing_snapshot"
    assert decision.remaining_s is None


@pytest.mark.parametrize(
    ("sim_time_s", "reason", "recovery", "remaining_s"),
    (
        (550.0, "expired", True, 0.0),
        (450.0, "deadline_margin", False, 100.0),
        (200.0, "not_due", False, 350.0),
    ),
)
def test_refresh_decision_respects_deadline(
    sim_time_s: float,
    reason: str,
    recovery: bool,
    remaining_s: float,
) -> None:
    decision = decide_execution_refresh(
        _snapshot(),
        sim_time_s=sim_time_s,
        refresh_margin_s=120,
        source_track_revision=4,
        prediction_revision=8,
    )

    assert decision.due is (reason != "not_due")
    assert decision.recovery is recovery
    assert decision.reason == reason
    assert decision.remaining_s == remaining_s


@pytest.mark.parametrize(
    ("source_track_revision", "prediction_revision"),
    ((5, 8), (4, 9)),
)
def test_newer_public_revision_forces_refresh(
    source_track_revision: int,
    prediction_revision: int,
) -> None:
    decision = decide_execution_refresh(
        _snapshot(),
        sim_time_s=200.0,
        refresh_margin_s=120,
        source_track_revision=source_track_revision,
        prediction_revision=prediction_revision,
    )

    assert decision.due is True
    assert decision.recovery is False
    assert decision.reason == "source_revision_advanced"
    assert decision.remaining_s == 350.0


def test_simulation_time_rollback_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot move backwards"):
        decide_execution_refresh(
            _snapshot(),
            sim_time_s=99.0,
            refresh_margin_s=120,
            source_track_revision=4,
            prediction_revision=8,
        )


def test_execution_refresh_config_has_deadline_and_retry_windows() -> None:
    config = ExecutionRefreshConfig()

    assert config.validity_s == 450.0
    assert config.margin_s == 120.0
    assert config.retry_interval_s == 30.0


@pytest.mark.parametrize(
    "payload",
    (
        {"validity_s": 120, "margin_s": 120},
        {"validity_s": 450, "margin_s": 29},
        {"validity_s": 450, "margin_s": 120, "retry_interval_s": 29},
    ),
)
def test_execution_refresh_config_rejects_unsafe_windows(payload: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        ExecutionRefreshConfig(**payload)
