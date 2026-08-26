# src/underwater_tracking/persistence/plans.py
"""Versioned plan repository: snapshots, plans, and plan commands.

Implements the plan lifecycle (spec 15.3): candidate plans are validated in
graph memory and only the atomic ``commit`` writes a row to ``plans`` with
broadcast status ``active`` (or ``degraded`` for emergency plans); the
previous active/degraded plan is superseded in the same transaction. A plan
whose ``base_snapshot_revision`` does not match the stored snapshot revision
of its scenario is stale and rejected with :class:`StaleSnapshotError` —
commit-time enforcement of the strict-but-mutable ``TrackingPlan`` contract.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from underwater_tracking.domain.agent_models import PlanCommand, TrackingPlan
from underwater_tracking.domain.execution_models import OperationalExecutionSnapshot
from underwater_tracking.domain.regional_models import TargetRegionPlan
from underwater_tracking.persistence.sqlite import (
    json_dumps,
    now_ms,
    open_database,
    synchronized_database_method,
    transaction,
)

_BROADCAST_STATUSES = ("active", "degraded")
_BROADCAST_PLACEHOLDERS = ", ".join("?" for _ in _BROADCAST_STATUSES)


def _model_payload(value: Any) -> dict[str, Any]:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        payload = dump(mode="json")
    elif isinstance(value, dict):
        payload = value
    else:
        raise TypeError("execution result must provide model_dump or be a mapping")
    if not isinstance(payload, dict):
        raise TypeError("execution result payload must be a mapping")
    return payload


class StaleSnapshotError(RuntimeError):
    """Raised when a plan's base snapshot revision is older than the stored one."""


@dataclass(frozen=True)
class RegionalPlanRevision:
    """One target's regional plan as persisted in a versioned plan payload."""

    scenario_id: str
    plan_id: str
    plan_revision: int
    target_id: str
    regional_plan: TargetRegionPlan
    trigger_event_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    llm_hashes: tuple[str, str] | None


@dataclass(frozen=True)
class ExecutionRevisionRecord:
    """One persisted execution attempt, including rejected candidates."""

    commit_id: str
    scenario_id: str
    execution_revision: int
    candidate_execution_revision: int | None
    base_execution_revision: int | None
    status: str
    source_snapshot_revision: int | None
    active_plan_preserved: bool
    reason: str
    snapshot: OperationalExecutionSnapshot | None
    result_payload: dict[str, Any]


class PlanRepository:
    """Snapshot revisions, atomic plan commits, and per-group plan commands."""

    def __init__(self, database_path: str | Path) -> None:
        self._conn = open_database(database_path)

    def close(self) -> None:
        self._conn.close()

    @property
    def connection(self) -> sqlite3.Connection:
        """Return the shared WAL connection for coordinated UUV commits."""
        return self._conn

    @synchronized_database_method
    def set_snapshot_revision(
        self, scenario_id: str, revision: int, snapshot_hash: str = ""
    ) -> None:
        """Upsert the current situation snapshot revision for a scenario."""
        self._conn.execute(
            "INSERT INTO snapshots (scenario_id, revision, snapshot_hash, updated_at)"
            " VALUES (?, ?, ?, ?)"
            " ON CONFLICT (scenario_id) DO UPDATE SET"
            " revision = excluded.revision,"
            " snapshot_hash = excluded.snapshot_hash,"
            " updated_at = excluded.updated_at",
            (scenario_id, revision, snapshot_hash, now_ms()),
        )

    @synchronized_database_method
    def get_snapshot_revision(self, scenario_id: str) -> int:
        """Return the stored snapshot revision (0 when never recorded)."""
        row = self._conn.execute(
            "SELECT revision FROM snapshots WHERE scenario_id = ?", (scenario_id,)
        ).fetchone()
        return int(row["revision"]) if row is not None else 0

    @synchronized_database_method
    def commit(self, plan: TrackingPlan) -> None:
        """Atomically commit a validated plan or raise ``StaleSnapshotError``.

        The whole commit runs in one immediate transaction: the stored
        scenario revision is compared against ``plan.base_snapshot_revision``
        (any mismatch is stale), the plan is inserted with broadcast status
        (``degraded`` preserved for emergency plans, everything else
        ``active``), the previous broadcast plan is superseded, and only then
        is the transaction committed. Any failure rolls back every write.
        """
        stored_status = plan.status if plan.status == "degraded" else "active"
        with transaction(self._conn):
            stored_revision = self.get_snapshot_revision(plan.scenario_id)
            if stored_revision != plan.base_snapshot_revision:
                raise StaleSnapshotError(
                    f"plan {plan.plan_id} bases on snapshot revision"
                    f" {plan.base_snapshot_revision} but scenario {plan.scenario_id}"
                    f" is at revision {stored_revision}"
                )
            self._insert_plan(plan, stored_status)
            self._after_plan_insert()
            self._supersede_previous(plan.scenario_id, plan.plan_id)

    def _insert_plan(self, plan: TrackingPlan, stored_status: str) -> None:
        payload = plan.model_dump(mode="json")
        payload["status"] = stored_status
        self._conn.execute(
            "INSERT INTO plans (plan_id, scenario_id, revision, base_snapshot_revision,"
            " status, valid_from_s, valid_until_s, payload, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                plan.plan_id,
                plan.scenario_id,
                plan.revision,
                plan.base_snapshot_revision,
                stored_status,
                plan.valid_from_s,
                plan.valid_until_s,
                json_dumps(payload),
                now_ms(),
            ),
        )

    def _after_plan_insert(self) -> None:
        """Test seam between plan insert and supersede.

        Kept as a separate call so tests can inject a failure here and assert
        that the enclosing transaction rolls back every write.
        """

    def _supersede_previous(self, scenario_id: str, plan_id: str) -> None:
        self._conn.execute(
            f"UPDATE plans SET status = 'superseded'"
            f" WHERE scenario_id = ? AND status IN ({_BROADCAST_PLACEHOLDERS})"
            f" AND plan_id != ?",
            (scenario_id, *_BROADCAST_STATUSES, plan_id),
        )

    @synchronized_database_method
    def get_active(self, scenario_id: str) -> TrackingPlan | None:
        """Return the latest broadcast (active/degraded) plan for a scenario."""
        row = self._conn.execute(
            f"SELECT payload, status FROM plans"
            f" WHERE scenario_id = ? AND status IN ({_BROADCAST_PLACEHOLDERS})"
            f" ORDER BY revision DESC LIMIT 1",
            (scenario_id, *_BROADCAST_STATUSES),
        ).fetchone()
        return self._decode(row) if row is not None else None

    @synchronized_database_method
    def get_plan(self, plan_id: str) -> TrackingPlan | None:
        """Return a plan by id at its current lifecycle status."""
        row = self._conn.execute(
            "SELECT payload, status FROM plans WHERE plan_id = ?", (plan_id,)
        ).fetchone()
        return self._decode(row) if row is not None else None

    @synchronized_database_method
    def list_scenario_ids(self, limit: int = 100, *, offset: int = 0) -> tuple[str, ...]:
        """Return a bounded set of scenarios that have persisted plans."""
        bounded_limit = max(0, min(limit, 100))
        bounded_offset = max(0, offset)
        if bounded_limit == 0:
            return ()
        rows = self._conn.execute(
            "SELECT DISTINCT scenario_id FROM plans ORDER BY scenario_id LIMIT ? OFFSET ?",
            (bounded_limit, bounded_offset),
        ).fetchall()
        return tuple(row["scenario_id"] for row in rows)

    @synchronized_database_method
    def list_regional_revisions(
        self, scenario_id: str, *, target_id: str | None = None, limit: int = 100
    ) -> list[RegionalPlanRevision]:
        """Return persisted regional revisions, newest plan revision first.

        Regional data stays in the canonical ``plans.payload`` JSON written at
        commit time. This query projects that payload for replay without
        introducing a second serialization format or losing superseded plans.
        """
        rows = self._conn.execute(
            "SELECT payload, status FROM plans WHERE scenario_id = ?"
            " ORDER BY revision DESC, plan_id DESC",
            (scenario_id,),
        ).fetchall()
        revisions: list[RegionalPlanRevision] = []
        for row in rows:
            plan = self._decode(row)
            for regional_target_id, regional_plan in sorted(plan.regional_plans.items()):
                if target_id is not None and regional_target_id != target_id:
                    continue
                revisions.append(
                    RegionalPlanRevision(
                        scenario_id=plan.scenario_id,
                        plan_id=plan.plan_id,
                        plan_revision=plan.revision,
                        target_id=regional_target_id,
                        regional_plan=regional_plan,
                        trigger_event_ids=plan.trigger_event_ids,
                        evidence_ids=plan.evidence_ids,
                        llm_hashes=plan.regional_llm_hashes.get(regional_target_id),
                    )
                )
                if len(revisions) >= limit:
                    return revisions
        return revisions

    @synchronized_database_method
    def save_execution_commit(
        self,
        *,
        result: Any,
        audit_projection: TrackingPlan | None = None,
    ) -> None:
        """Persist one execution result and optional legacy audit projection atomically."""
        result_payload = _model_payload(result)
        snapshot = getattr(result, "snapshot", None)
        snapshot_payload = (
            snapshot.model_dump(mode="json") if snapshot is not None else None
        )
        scenario_id = str(result_payload.get("scenario_id", ""))
        commit_id = str(result_payload.get("commit_id", ""))
        if not scenario_id or not commit_id:
            raise ValueError("execution result requires scenario_id and commit_id")
        execution_revision = int(
            result_payload.get("execution_revision")
            or result_payload.get("candidate_execution_revision")
            or 0
        )
        status = str(result_payload.get("status", "failed"))
        source_snapshot_revision = (
            int(snapshot.source_snapshot_revision) if snapshot is not None else None
        )
        with transaction(self._conn):
            self._conn.execute(
                "INSERT INTO execution_revisions "
                "(commit_id, scenario_id, execution_revision, "
                " candidate_execution_revision, base_execution_revision, status, "
                " source_snapshot_revision, active_plan_preserved, reason, "
                " snapshot_payload, result_payload, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    commit_id,
                    scenario_id,
                    execution_revision,
                    result_payload.get("candidate_execution_revision"),
                    result_payload.get("base_execution_revision"),
                    status,
                    source_snapshot_revision,
                    int(bool(result_payload.get("active_plan_preserved"))),
                    str(result_payload.get("reason", ""))[:2000],
                    json_dumps(snapshot_payload) if snapshot_payload is not None else None,
                    json_dumps(result_payload),
                    now_ms(),
                ),
            )
            if audit_projection is not None:
                stored_revision = self.get_snapshot_revision(
                    audit_projection.scenario_id
                )
                if stored_revision > audit_projection.base_snapshot_revision:
                    raise StaleSnapshotError(
                        "audit projection bases on an older situation snapshot"
                    )
                if stored_revision != audit_projection.base_snapshot_revision:
                    self._conn.execute(
                        "INSERT INTO snapshots "
                        "(scenario_id, revision, snapshot_hash, updated_at) "
                        "VALUES (?, ?, '', ?) "
                        "ON CONFLICT (scenario_id) DO UPDATE SET "
                        "revision = excluded.revision, updated_at = excluded.updated_at",
                        (
                            audit_projection.scenario_id,
                            audit_projection.base_snapshot_revision,
                            now_ms(),
                        ),
                    )
                self._insert_plan(audit_projection, "active")
                self._supersede_previous(
                    audit_projection.scenario_id,
                    audit_projection.plan_id,
                )

    @synchronized_database_method
    def get_latest_execution_snapshot(
        self, scenario_id: str
    ) -> OperationalExecutionSnapshot | None:
        """Return the newest committed execution snapshot for a scenario."""
        row = self._conn.execute(
            "SELECT snapshot_payload FROM execution_revisions "
            "WHERE scenario_id = ? AND status = 'committed' "
            "AND snapshot_payload IS NOT NULL "
            "ORDER BY execution_revision DESC, created_at DESC LIMIT 1",
            (scenario_id,),
        ).fetchone()
        if row is None:
            return None
        return OperationalExecutionSnapshot.model_validate(
            json.loads(row["snapshot_payload"])
        )

    @synchronized_database_method
    def list_execution_revisions(
        self, scenario_id: str, *, limit: int = 100
    ) -> list[ExecutionRevisionRecord]:
        """Return execution attempts newest first, including preserved failures."""
        rows = self._conn.execute(
            "SELECT commit_id, scenario_id, execution_revision, "
            "candidate_execution_revision, base_execution_revision, status, "
            "source_snapshot_revision, active_plan_preserved, reason, "
            "snapshot_payload, result_payload FROM execution_revisions "
            "WHERE scenario_id = ? ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (scenario_id, max(0, limit)),
        ).fetchall()
        return [
            ExecutionRevisionRecord(
                commit_id=row["commit_id"],
                scenario_id=row["scenario_id"],
                execution_revision=int(row["execution_revision"]),
                candidate_execution_revision=(
                    int(row["candidate_execution_revision"])
                    if row["candidate_execution_revision"] is not None
                    else None
                ),
                base_execution_revision=(
                    int(row["base_execution_revision"])
                    if row["base_execution_revision"] is not None
                    else None
                ),
                status=row["status"],
                source_snapshot_revision=(
                    int(row["source_snapshot_revision"])
                    if row["source_snapshot_revision"] is not None
                    else None
                ),
                active_plan_preserved=bool(row["active_plan_preserved"]),
                reason=row["reason"],
                snapshot=(
                    OperationalExecutionSnapshot.model_validate(
                        json.loads(row["snapshot_payload"])
                    )
                    if row["snapshot_payload"] is not None
                    else None
                ),
                result_payload=json.loads(row["result_payload"]),
            )
            for row in rows
        ]

    list_execution_commits = list_execution_revisions

    @staticmethod
    def _decode(row: sqlite3.Row) -> TrackingPlan:
        payload = json.loads(row["payload"])
        payload["status"] = row["status"]
        return TrackingPlan.model_validate(payload)

    @synchronized_database_method
    def save_command(self, command: PlanCommand) -> None:
        """Persist a versioned per-group execution command (spec 5.2)."""
        self._conn.execute(
            "INSERT INTO plan_commands (command_id, plan_id, plan_revision,"
            " scenario_id, group_id, target_id, sim_time_s, payload, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                command.command_id,
                command.plan_id,
                command.plan_revision,
                command.scenario_id,
                command.group_id,
                command.target_id,
                command.sim_time_s,
                json_dumps(command.model_dump(mode="json")),
                now_ms(),
            ),
        )

    @synchronized_database_method
    def list_commands(self, plan_id: str) -> list[PlanCommand]:
        """Return the execution commands of one committed plan."""
        rows = self._conn.execute(
            "SELECT payload FROM plan_commands WHERE plan_id = ? ORDER BY rowid",
            (plan_id,),
        ).fetchall()
        return [PlanCommand.model_validate(json.loads(row["payload"])) for row in rows]
