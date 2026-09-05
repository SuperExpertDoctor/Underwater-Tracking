"""Pure deadline policy for authoritative execution snapshot refreshes."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from underwater_tracking.domain.execution_models import OperationalExecutionSnapshot


ExecutionRefreshReason = Literal[
    "not_due",
    "missing_snapshot",
    "deadline_margin",
    "expired",
    "source_revision_advanced",
]


@dataclass(frozen=True, slots=True)
class ExecutionRefreshDecision:
    """The deterministic decision made at one observation boundary."""

    due: bool
    recovery: bool
    reason: ExecutionRefreshReason
    remaining_s: float | None


def decide_execution_refresh(
    snapshot: OperationalExecutionSnapshot | None,
    *,
    sim_time_s: float,
    refresh_margin_s: int | float,
    source_track_revision: int,
    prediction_revision: int,
) -> ExecutionRefreshDecision:
    """Decide whether an execution snapshot must be refreshed.

    The public source and prediction revisions are compared with the revisions
    captured by the snapshot. A missing or expired snapshot is a recovery
    refresh, while a source gap never changes the snapshot's deadline.
    """

    if not isfinite(float(sim_time_s)):
        raise ValueError("sim_time_s must be finite")
    if not isfinite(float(refresh_margin_s)) or float(refresh_margin_s) < 0:
        raise ValueError("refresh_margin_s must be finite and non-negative")
    if snapshot is None:
        return ExecutionRefreshDecision(
            due=True,
            recovery=True,
            reason="missing_snapshot",
            remaining_s=None,
        )

    if sim_time_s < snapshot.valid_from_s:
        raise ValueError("simulation time cannot move backwards")
    remaining_s = float(snapshot.valid_until_s) - float(sim_time_s)
    if remaining_s <= 0:
        return ExecutionRefreshDecision(
            due=True,
            recovery=True,
            reason="expired",
            remaining_s=remaining_s,
        )
    captured_source_revision = getattr(
        getattr(snapshot, "target_track", None),
        "track_revision",
        snapshot.source_snapshot_revision,
    )
    if (
        source_track_revision > captured_source_revision
        or prediction_revision > snapshot.prediction_revision
    ):
        return ExecutionRefreshDecision(
            due=True,
            recovery=False,
            reason="source_revision_advanced",
            remaining_s=remaining_s,
        )
    if remaining_s <= float(refresh_margin_s):
        return ExecutionRefreshDecision(
            due=True,
            recovery=False,
            reason="deadline_margin",
            remaining_s=remaining_s,
        )
    return ExecutionRefreshDecision(
        due=False,
        recovery=False,
        reason="not_due",
        remaining_s=remaining_s,
    )
