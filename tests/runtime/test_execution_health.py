from __future__ import annotations

from collections.abc import Mapping

import pytest

from tests.domain.test_execution_models import _snapshot as _domain_snapshot
from underwater_tracking.runtime.execution_health import classify_execution_health


def _snapshot(*, valid_from_s: float, valid_until_s: float):
    return _domain_snapshot().model_copy(
        update={
            "source_sim_time_s": valid_from_s,
            "generated_at_s": valid_from_s,
            "valid_from_s": valid_from_s,
            "valid_until_s": valid_until_s,
        }
    )


@pytest.mark.parametrize(
    ("age_s", "expected"),
    [
        (0, "current"),
        (450, "current"),
        (451, "degraded"),
        (900, "degraded"),
        (901, "expired"),
    ],
)
def test_execution_health_age_boundaries(age_s: float, expected: str) -> None:
    health = classify_execution_health(
        _snapshot(valid_from_s=1_000, valid_until_s=1_450),
        sim_time_s=1_000 + age_s,
        hard_stale_s=900,
    )

    assert health.status == expected
    assert health.age_s == age_s


def test_invalid_snapshot_is_failed_instead_of_expired() -> None:
    invalid: Mapping[str, object] = {
        **_snapshot(valid_from_s=1_000, valid_until_s=1_450).model_dump(mode="python"),
        "valid_until_s": 999.0,
    }

    health = classify_execution_health(
        invalid,
        sim_time_s=2_000,
        hard_stale_s=900,
    )

    assert health.status == "failed"
    assert health.reason_codes == ("execution_snapshot_validation_failed",)


def test_snapshot_instance_is_revalidated_before_classifying_health() -> None:
    invalid = _snapshot(valid_from_s=1_000, valid_until_s=1_450)
    invalid.__dict__["valid_until_s"] = 999.0

    health = classify_execution_health(
        invalid,
        sim_time_s=1_100,
        hard_stale_s=900,
    )

    assert health.status == "failed"
    assert health.reason_codes == ("execution_snapshot_validation_failed",)


def test_sim_time_before_valid_from_is_failed() -> None:
    health = classify_execution_health(
        _snapshot(valid_from_s=1_000, valid_until_s=1_450),
        sim_time_s=999,
        hard_stale_s=900,
    )

    assert health.status == "failed"
    assert health.reason_codes == ("execution_snapshot_not_yet_valid",)
    assert health.executable is False


@pytest.mark.parametrize(
    ("sim_time_s", "executable"),
    [(1_000, True), (1_451, True), (1_901, False)],
)
def test_only_current_and_degraded_snapshots_are_executable(
    sim_time_s: float,
    executable: bool,
) -> None:
    health = classify_execution_health(
        _snapshot(valid_from_s=1_000, valid_until_s=1_450),
        sim_time_s=sim_time_s,
        hard_stale_s=900,
    )

    assert health.executable is executable


def test_expired_and_failed_snapshots_are_not_executable() -> None:
    expired = classify_execution_health(
        _snapshot(valid_from_s=1_000, valid_until_s=1_450),
        sim_time_s=1_901,
        hard_stale_s=900,
    )
    failed = classify_execution_health(
        _snapshot(valid_from_s=1_000, valid_until_s=1_450),
        sim_time_s=999,
        hard_stale_s=900,
    )

    assert expired.status == "expired"
    assert failed.status == "failed"
    assert expired.executable is False
    assert failed.executable is False
