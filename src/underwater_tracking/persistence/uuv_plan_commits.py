"""Atomic UUV executable-plan commit prepared on the plan repository connection."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from underwater_tracking.domain.agent_models import TrackingPlan
from underwater_tracking.domain.mission_models import ExecutableMissionPlan
from underwater_tracking.domain.planning_epoch_models import (
    EpochCommitResult,
    PlanningEpoch,
)
from underwater_tracking.persistence.plans import PlanRepository
from underwater_tracking.persistence.sqlite import json_dumps, now_ms
from underwater_tracking.planning.mission_revalidation import MissionRevalidationReport


class UUVPlanCommitRepository:
    """Stage report/audit rows and finish them with one SQLite transaction."""

    def __init__(self, plans: PlanRepository) -> None:
        self._plans = plans

    def prepare(
        self,
        *,
        epoch: PlanningEpoch,
        report: MissionRevalidationReport,
        audit_projection: TrackingPlan,
        executable_plan: ExecutableMissionPlan,
        expected_active_plan_revision: int,
    ) -> PreparedUUVCommit:
        if report.epoch_id != epoch.epoch_id:
            raise ValueError("revalidation report does not belong to epoch")
        if report.rebased_plan is not None and report.rebased_plan != executable_plan:
            raise ValueError("executable plan does not match revalidation report")
        active = self._plans.get_active(epoch.scenario_id)
        current_revision = active.revision if active is not None else 0
        if current_revision != expected_active_plan_revision:
            raise ValueError(
                f"active plan changed from {expected_active_plan_revision} to {current_revision}"
            )
        conn = self._plans.connection
        conn.execute("BEGIN IMMEDIATE")
        try:
            report_payload = json_dumps(report.model_dump(mode="json"))
            conn.execute(
                "INSERT INTO planning_revalidation_reports "
                "(report_id, epoch_id, valid, current_physics_revision, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    report.report_id,
                    epoch.epoch_id,
                    int(report.valid),
                    report.current_physics_revision,
                    report_payload,
                    now_ms(),
                ),
            )
            self._plans._insert_plan(audit_projection, "draft")  # noqa: SLF001
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        return PreparedUUVCommit(
            plans=self._plans,
            connection=conn,
            epoch=epoch,
            report=report,
            executable_plan=executable_plan,
            audit_projection=audit_projection,
        )


@dataclass
class PreparedUUVCommit:
    plans: PlanRepository
    connection: sqlite3.Connection
    epoch: PlanningEpoch
    report: MissionRevalidationReport
    executable_plan: ExecutableMissionPlan
    audit_projection: TrackingPlan
    _closed: bool = False

    def finish(self, result: EpochCommitResult) -> None:
        if self._closed:
            raise RuntimeError("prepared UUV commit is already closed")
        if result.status != "committed":
            raise ValueError("prepared executable plan can only finish as committed")
        if result.validation_report_id != self.report.report_id:
            raise ValueError("commit result does not reference prepared validation report")
        payload = self.audit_projection.model_dump(mode="json")
        payload["status"] = "active"
        self.connection.execute(
            "UPDATE plans SET status = 'active', payload = ? WHERE plan_id = ?",
            (json_dumps(payload), self.audit_projection.plan_id),
        )
        self.plans._supersede_previous(self.epoch.scenario_id, self.audit_projection.plan_id)  # noqa: SLF001
        result_payload = json_dumps(result.model_dump(mode="json"))
        self.connection.execute(
            "INSERT INTO planning_epoch_results "
            "(epoch_id, status, plan_id, plan_version, validation_report_id, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                result.epoch_id,
                result.status,
                result.plan_id,
                result.plan_version,
                result.validation_report_id,
                result_payload,
                now_ms(),
            ),
        )
        self.connection.execute(
            "UPDATE planning_epochs SET status = 'committed', payload = json_set(payload, '$.status', 'committed'), updated_at = ? "
            "WHERE epoch_id = ?",
            (now_ms(), result.epoch_id),
        )
        self.connection.execute("COMMIT")
        self._closed = True

    def rollback(self) -> None:
        if self._closed:
            return
        self.connection.execute("ROLLBACK")
        self._closed = True
