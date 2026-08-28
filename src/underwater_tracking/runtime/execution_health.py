"""Pure freshness classification for authoritative execution snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

from pydantic import ValidationError

from underwater_tracking.domain.execution_models import OperationalExecutionSnapshot


ExecutionHealthStatus = Literal["current", "degraded", "expired", "failed"]


@dataclass(frozen=True, slots=True)
class ExecutionHealth:
    status: ExecutionHealthStatus
    age_s: float
    reason_codes: tuple[str, ...]

    @property
    def executable(self) -> bool:
        return self.status in {"current", "degraded"}


def classify_execution_health(
    snapshot: OperationalExecutionSnapshot | Mapping[str, Any],
    *,
    sim_time_s: float,
    hard_stale_s: float,
) -> ExecutionHealth:
    """Classify model validity before applying age-based freshness rules."""

    if not isfinite(sim_time_s) or sim_time_s < 0.0:
        raise ValueError("sim_time_s must be finite and non-negative")
    if not isfinite(hard_stale_s) or hard_stale_s <= 0.0:
        raise ValueError("hard_stale_s must be finite and positive")

    try:
        validated = (
            snapshot
            if isinstance(snapshot, OperationalExecutionSnapshot)
            else OperationalExecutionSnapshot.model_validate(snapshot)
        )
    except (TypeError, ValueError, ValidationError):
        return ExecutionHealth(
            status="failed",
            age_s=0.0,
            reason_codes=("execution_snapshot_validation_failed",),
        )

    age_s = sim_time_s - float(validated.valid_from_s)
    if sim_time_s <= float(validated.valid_until_s):
        return ExecutionHealth(status="current", age_s=age_s, reason_codes=())
    if age_s <= hard_stale_s:
        return ExecutionHealth(
            status="degraded",
            age_s=age_s,
            reason_codes=("execution_snapshot_stale",),
        )
    return ExecutionHealth(
        status="expired",
        age_s=age_s,
        reason_codes=("execution_snapshot_expired",),
    )


__all__ = [
    "ExecutionHealth",
    "ExecutionHealthStatus",
    "classify_execution_health",
]
