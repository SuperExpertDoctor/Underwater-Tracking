"""SQLite persistence for immutable planning epoch captures and outcomes."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from underwater_tracking.domain.planning_epoch_models import (
    EpochCommitResult,
    PlanningEpoch,
    PlanningEpochCapture,
    PlanningEpochStatus,
)
from underwater_tracking.persistence.sqlite import json_dumps, now_ms, open_database, transaction


class PlanningEpochRepository:
    """Persist one immutable capture and one terminal result per epoch."""

    def __init__(self, database_path: str | Path | sqlite3.Connection) -> None:
        if isinstance(database_path, sqlite3.Connection):
            self._owns_connection = False
            self._conn: sqlite3.Connection = database_path
        else:
            self._owns_connection = True
            self._conn = open_database(database_path)

    def close(self) -> None:
        if self._owns_connection:
            self._conn.close()

    def create(self, capture: PlanningEpochCapture) -> None:
        epoch = capture.epoch
        epoch_payload = json_dumps(epoch.model_dump(mode="json"))
        situation_payload = json_dumps(capture.situation.model_dump(mode="json"))
        mission_payload = json_dumps(capture.mission.model_dump(mode="json"))
        created_at = now_ms()
        with transaction(self._conn):
            existing = self._conn.execute(
                "SELECT payload FROM planning_epochs WHERE epoch_id = ?",
                (epoch.epoch_id,),
            ).fetchone()
            if existing is not None:
                if existing["payload"] != epoch_payload:
                    raise ValueError(f"epoch {epoch.epoch_id!r} already exists with different payload")
                input_row = self._conn.execute(
                    "SELECT observation_batch_id, situation_payload, mission_payload "
                    "FROM planning_epoch_inputs WHERE epoch_id = ?",
                    (epoch.epoch_id,),
                ).fetchone()
                if input_row is None or (
                    input_row["observation_batch_id"] != epoch.observation_batch_id
                    or input_row["situation_payload"] != situation_payload
                    or input_row["mission_payload"] != mission_payload
                ):
                    raise ValueError(f"epoch {epoch.epoch_id!r} already exists with different capture")
                return
            self._conn.execute(
                "INSERT INTO planning_epochs "
                "(epoch_id, scenario_id, base_physics_revision, base_sim_time_s, status, payload, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    epoch.epoch_id,
                    epoch.scenario_id,
                    epoch.base_physics_revision,
                    epoch.base_sim_time_s,
                    epoch.status.value,
                    epoch_payload,
                    created_at,
                    created_at,
                ),
            )
            self._conn.execute(
                "INSERT INTO planning_epoch_inputs "
                "(epoch_id, observation_batch_id, situation_payload, mission_payload, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (epoch.epoch_id, epoch.observation_batch_id, situation_payload, mission_payload, created_at),
            )

    def get(self, epoch_id: str) -> PlanningEpoch:
        row = self._conn.execute(
            "SELECT payload FROM planning_epochs WHERE epoch_id = ?", (epoch_id,)
        ).fetchone()
        if row is None:
            raise KeyError(epoch_id)
        return PlanningEpoch.model_validate(json.loads(row["payload"]))

    def get_capture(self, epoch_id: str) -> PlanningEpochCapture:
        epoch = self.get(epoch_id)
        row = self._conn.execute(
            "SELECT observation_batch_id, situation_payload, mission_payload "
            "FROM planning_epoch_inputs WHERE epoch_id = ?",
            (epoch_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"missing capture for {epoch_id}")
        if row["observation_batch_id"] != epoch.observation_batch_id:
            raise ValueError(f"capture batch does not match epoch {epoch_id}")
        return PlanningEpochCapture(
            epoch=epoch,
            situation=json.loads(row["situation_payload"]),
            mission=json.loads(row["mission_payload"]),
        )

    def mark_running(self, epoch_id: str) -> None:
        with transaction(self._conn):
            epoch = self.get(epoch_id)
            if epoch.status in {
                PlanningEpochStatus.COMMITTED,
                PlanningEpochStatus.INVALIDATED,
                PlanningEpochStatus.REJECTED,
                PlanningEpochStatus.FAILED,
            }:
                raise ValueError(f"epoch {epoch_id!r} is already finished")
            self._set_status(epoch_id, PlanningEpochStatus.RUNNING)

    def finish(self, result: EpochCommitResult) -> None:
        with transaction(self._conn):
            epoch = self.get(result.epoch_id)
            row = self._conn.execute(
                "SELECT payload FROM planning_epoch_results WHERE epoch_id = ?",
                (result.epoch_id,),
            ).fetchone()
            result_payload = json_dumps(result.model_dump(mode="json"))
            if row is not None:
                if row["payload"] == result_payload:
                    return
                raise ValueError(f"epoch {result.epoch_id!r} is already finished")
            if epoch.status in {
                PlanningEpochStatus.COMMITTED,
                PlanningEpochStatus.INVALIDATED,
                PlanningEpochStatus.REJECTED,
                PlanningEpochStatus.FAILED,
            }:
                raise ValueError(f"epoch {result.epoch_id!r} is already finished")
            self._conn.execute(
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
            self._set_status(result.epoch_id, PlanningEpochStatus(result.status))

    def latest(self, scenario_id: str) -> tuple[PlanningEpoch, EpochCommitResult | None] | None:
        row = self._conn.execute(
            "SELECT payload, epoch_id FROM planning_epochs WHERE scenario_id = ? "
            "ORDER BY created_at DESC, epoch_id DESC LIMIT 1",
            (scenario_id,),
        ).fetchone()
        if row is None:
            return None
        epoch = PlanningEpoch.model_validate(json.loads(row["payload"]))
        result_row = self._conn.execute(
            "SELECT payload FROM planning_epoch_results WHERE epoch_id = ?",
            (row["epoch_id"],),
        ).fetchone()
        result = (
            EpochCommitResult.model_validate(json.loads(result_row["payload"]))
            if result_row is not None
            else None
        )
        return epoch, result

    def _set_status(self, epoch_id: str, status: PlanningEpochStatus) -> None:
        row = self._conn.execute(
            "SELECT payload FROM planning_epochs WHERE epoch_id = ?", (epoch_id,)
        ).fetchone()
        if row is None:
            raise KeyError(epoch_id)
        payload = json.loads(row["payload"])
        payload["status"] = status.value
        self._conn.execute(
            "UPDATE planning_epochs SET status = ?, payload = ?, updated_at = ? WHERE epoch_id = ?",
            (status.value, json_dumps(payload), now_ms(), epoch_id),
        )
