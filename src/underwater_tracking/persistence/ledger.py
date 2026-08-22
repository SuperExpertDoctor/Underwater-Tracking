# src/underwater_tracking/persistence/ledger.py
"""DecisionLedger: decisions, LLM metadata, expert directives, question runs.

The ledger (spec 16) makes every planning decision fully traceable: trigger
IDs, snapshot revision/hash, model/prompt/schema/config/code versions,
candidate concepts and plan IDs, rejected candidates with reasons, solver
metrics, verification/repair records, the final plan diff, and the expert
directives that shaped the decision. ``record`` writes the full
``DecisionRecord`` as one canonical JSON payload. The ``llm_calls``,
``expert_directives``, and ``question_runs`` tables hold the supporting
audit rows; LLM call rows store only metadata hashes, never request bodies,
headers, or secrets.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Literal

from underwater_tracking.domain.agent_models import DecisionRecord, ExpertDirective
from underwater_tracking.domain.ui_models import BrainActivityRecord
from underwater_tracking.persistence.sqlite import json_dumps, now_ms, open_database, transaction

_DEFAULT_LIMIT = 100


@dataclass(frozen=True)
class LlmCallRecord:
    """One persisted LLM call metadata row (no payloads or secrets)."""

    id: int
    operation: str
    model: str
    prompt_version: str
    request_hash: str
    response_hash: str
    latency_ms: int
    token_count: int
    error_category: str
    sim_time_s: int
    scenario_id: str
    created_at: int


@dataclass(frozen=True)
class QuestionRun:
    """One persisted expert question run with its evidence-backed answer."""

    run_id: str
    scenario_id: str
    question_text: str
    status: str
    payload: dict[str, Any]
    created_at: int


@dataclass(frozen=True)
class KnowledgeQueryRun:
    """One ontology query and its bounded answer or failure."""

    query_id: str
    scenario_id: str
    sim_time_s: int
    query_text: str
    mode: str
    status: str
    response: dict[str, Any]
    response_hash: str
    created_at: int


class DecisionLedger:
    """Durable audit ledger for decisions, LLM calls, directives, and questions."""

    def __init__(self, database_path: str | Path) -> None:
        self._conn = open_database(database_path)

    def close(self) -> None:
        self._conn.close()

    def record(self, decision: DecisionRecord) -> None:
        """Persist one planning decision (full traceability, spec 16)."""
        with transaction(self._conn):
            self._conn.execute(
                "INSERT INTO decision_records"
                " (decision_id, scenario_id, sim_time_s, snapshot_revision,"
                "  payload, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    decision.decision_id,
                    decision.scenario_id,
                    decision.sim_time_s,
                    decision.snapshot_revision,
                    json_dumps(decision.model_dump(mode="json")),
                    now_ms(),
                ),
            )

    def get(self, decision_id: str) -> DecisionRecord | None:
        row = self._conn.execute(
            "SELECT payload FROM decision_records WHERE decision_id = ?",
            (decision_id,),
        ).fetchone()
        return DecisionRecord.model_validate(json.loads(row["payload"])) if row else None

    def list_decisions(
        self, scenario_id: str | None = None, limit: int = _DEFAULT_LIMIT
    ) -> list[DecisionRecord]:
        """Return decisions newest first, optionally filtered by scenario."""
        if scenario_id is None:
            rows = self._conn.execute(
                "SELECT payload FROM decision_records ORDER BY sim_time_s DESC,"
                " decision_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT payload FROM decision_records WHERE scenario_id = ?"
                " ORDER BY sim_time_s DESC, decision_id DESC LIMIT ?",
                (scenario_id, limit),
            ).fetchall()
        return [DecisionRecord.model_validate(json.loads(row["payload"])) for row in rows]

    def list_scenario_ids(self, limit: int = 100, *, offset: int = 0) -> tuple[str, ...]:
        """Return a bounded set of scenarios that have persisted decisions."""
        bounded_limit = max(0, min(limit, 100))
        bounded_offset = max(0, offset)
        if bounded_limit == 0:
            return ()
        rows = self._conn.execute(
            "SELECT DISTINCT scenario_id FROM decision_records ORDER BY scenario_id LIMIT ? OFFSET ?",
            (bounded_limit, bounded_offset),
        ).fetchall()
        return tuple(row["scenario_id"] for row in rows)

    def record_llm_call(
        self,
        *,
        operation: str,
        model: str,
        prompt_version: str,
        request_hash: str = "",
        response_hash: str = "",
        latency_ms: int = 0,
        token_count: int = 0,
        error_category: str = "",
        sim_time_s: int = 0,
        scenario_id: str = "",
    ) -> int:
        """Record LLM call metadata (hashes only, never payloads or secrets).

        Returns the new row id. ``error_category`` distinguishes transient and
        content errors for the retry bookkeeping (spec 8.3).
        """
        with transaction(self._conn):
            cursor = self._conn.execute(
                "INSERT INTO llm_calls"
                " (operation, model, prompt_version, request_hash, response_hash,"
                "  latency_ms, token_count, error_category, sim_time_s, scenario_id,"
                "  created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    operation,
                    model,
                    prompt_version,
                    request_hash,
                    response_hash,
                    latency_ms,
                    token_count,
                    error_category,
                    sim_time_s,
                    scenario_id,
                    now_ms(),
                ),
            )
        return int(cursor.lastrowid or 0)

    def list_llm_calls(
        self,
        limit: int = _DEFAULT_LIMIT,
        *,
        scenario_id: str | None = None,
        operation: str | None = None,
    ) -> list[LlmCallRecord]:
        """List LLM metadata hashes, optionally for one scenario and operation."""
        clauses: list[str] = []
        params: list[object] = []
        if scenario_id is not None:
            clauses.append("scenario_id = ?")
            params.append(scenario_id)
        if operation is not None:
            clauses.append("operation = ?")
            params.append(operation)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            "SELECT id, operation, model, prompt_version, request_hash, response_hash,"
            " latency_ms, token_count, error_category, sim_time_s, scenario_id, created_at"
            f" FROM llm_calls{where} ORDER BY id DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
        return [
            LlmCallRecord(
                id=int(row["id"]),
                operation=row["operation"],
                model=row["model"],
                prompt_version=row["prompt_version"],
                request_hash=row["request_hash"],
                response_hash=row["response_hash"],
                latency_ms=int(row["latency_ms"]),
                token_count=int(row["token_count"]),
                error_category=row["error_category"],
                sim_time_s=int(row["sim_time_s"]),
                scenario_id=row["scenario_id"],
                created_at=int(row["created_at"]),
            )
            for row in rows
        ]

    def latest_role_activity(
        self, scenario_id: str
    ) -> Mapping[Literal["master", "slave", "adversary"], BrainActivityRecord]:
        """Return the newest durable activity record for each decision role.

        LLM rows contain metadata only, so evidence platform IDs are taken from
        persisted decision records when available. The publisher supplies the
        configured-role set separately and turns absent rows into ``ready``.
        """
        role_operations: dict[str, Literal["master", "slave", "adversary"]] = {
            "strategy": "master",
            "intent": "master",
            "regional_strategy": "master",
            "plan_adjustment_suggestions": "master",
            "commit": "master",
            "slave_sonar_decision": "slave",
            "adversary_escape": "adversary",
            "adversary_mission_decision": "adversary",
            "adversary_intent": "adversary",
        }
        activity: dict[
            Literal["master", "slave", "adversary"], BrainActivityRecord
        ] = {}
        for call in reversed(self.list_llm_calls(scenario_id=scenario_id, limit=1000)):
            call_role = role_operations.get(call.operation)
            if call_role is None:
                continue
            if call.error_category:
                status: Literal[
                    "unconfigured", "ready", "running", "succeeded", "degraded", "failed"
                ] = (
                    "failed"
                    if call.error_category in {"config", "content", "semantic"}
                    else "degraded"
                )
                message = f"{call.operation} failed: {call.error_category}"
            else:
                status = "succeeded"
                message = f"{call.operation} completed"
            activity[call_role] = BrainActivityRecord(
                brain_id=_brain_id(call_role),
                role=call_role,
                status=status,
                operation=call.operation,
                sim_time_s=call.sim_time_s,
                message=message,
            )

        for decision in self.list_decisions(scenario_id, limit=1000):
            decision_role: Literal["master", "slave", "adversary"] = "master"
            if decision.model_version.startswith("adversary"):
                decision_role = "adversary"
            elif decision.model_version.startswith("slave"):
                decision_role = "slave"
            if decision_role in activity and (
                activity[decision_role].sim_time_s or 0,
                activity[decision_role].operation or "",
            ) > (decision.sim_time_s, "commit"):
                continue
            evidence_platform_ids = tuple(
                sorted(
                    evidence_id
                    for evidence_id in decision.input_evidence_ids
                    if _looks_like_platform_id(evidence_id)
                )
            )
            activity[decision_role] = BrainActivityRecord(
                brain_id=_brain_id(decision_role),
                role=decision_role,
                status="succeeded" if decision.final_plan_id is not None else "degraded",
                operation="commit" if decision.final_plan_id is not None else "planning",
                sim_time_s=decision.sim_time_s,
                evidence_platform_ids=evidence_platform_ids,
                message=(
                    "committed plan"
                    if decision.final_plan_id is not None
                    else "planning decision recorded without a committed plan"
                ),
            )
        return activity

    def save_directive(self, directive: ExpertDirective, scenario_id: str) -> None:
        """Persist an expert directive; re-saving an id updates its state."""
        payload = directive.model_dump(mode="json")
        self._conn.execute(
            "INSERT INTO expert_directives"
            " (directive_id, scenario_id, status, confidence, payload, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (directive_id) DO UPDATE SET"
            " scenario_id = excluded.scenario_id,"
            " status = excluded.status,"
            " confidence = excluded.confidence,"
            " payload = excluded.payload,"
            " created_at = excluded.created_at",
            (
                directive.directive_id,
                scenario_id,
                directive.status,
                directive.confidence,
                json_dumps(payload),
                now_ms(),
            ),
        )

    def list_directives(
        self, scenario_id: str | None = None, status: str | None = None
    ) -> list[ExpertDirective]:
        clauses: list[str] = []
        params: list[object] = []
        if scenario_id is not None:
            clauses.append("scenario_id = ?")
            params.append(scenario_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT payload FROM expert_directives{where} ORDER BY created_at, directive_id",
            params,
        ).fetchall()
        return [ExpertDirective.model_validate(json.loads(row["payload"])) for row in rows]

    def save_question(
        self,
        *,
        run_id: str,
        scenario_id: str,
        question_text: str,
        payload: dict[str, Any],
        status: str = "completed",
    ) -> None:
        """Persist one expert question run with its evidence-backed answer."""
        self._conn.execute(
            "INSERT INTO question_runs"
            " (run_id, scenario_id, question_text, status, payload, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                run_id,
                scenario_id,
                question_text,
                status,
                json_dumps(payload),
                now_ms(),
            ),
        )

    def list_questions(
        self, scenario_id: str | None = None, limit: int = _DEFAULT_LIMIT
    ) -> list[QuestionRun]:
        if scenario_id is None:
            rows = self._conn.execute(
                "SELECT run_id, scenario_id, question_text, status, payload, created_at"
                " FROM question_runs ORDER BY created_at DESC, run_id LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT run_id, scenario_id, question_text, status, payload, created_at"
                " FROM question_runs WHERE scenario_id = ?"
                " ORDER BY created_at DESC, run_id LIMIT ?",
                (scenario_id, limit),
            ).fetchall()
        return [
            QuestionRun(
                run_id=row["run_id"],
                scenario_id=row["scenario_id"],
                question_text=row["question_text"],
                status=row["status"],
                payload=json.loads(row["payload"]),
                created_at=int(row["created_at"]),
            )
            for row in rows
        ]

    def save_knowledge_query(
        self,
        *,
        query_id: str,
        scenario_id: str,
        sim_time_s: int,
        query_text: str,
        mode: str,
        status: str,
        response: dict[str, Any],
    ) -> None:
        """Persist the ontology query and bounded response for battle replay."""
        payload = json_dumps(response)
        response_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self._conn.execute(
            "INSERT INTO knowledge_queries"
            " (query_id, scenario_id, sim_time_s, query_text, mode, status,"
            "  response_hash, payload, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (query_id) DO UPDATE SET"
            " scenario_id = excluded.scenario_id,"
            " sim_time_s = excluded.sim_time_s,"
            " query_text = excluded.query_text,"
            " mode = excluded.mode,"
            " status = excluded.status,"
            " response_hash = excluded.response_hash,"
            " payload = excluded.payload,"
            " created_at = excluded.created_at",
            (
                query_id,
                scenario_id,
                sim_time_s,
                query_text,
                mode,
                status,
                response_hash,
                payload,
                now_ms(),
            ),
        )

    def list_knowledge_queries(
        self, scenario_id: str | None = None, limit: int = _DEFAULT_LIMIT
    ) -> list[KnowledgeQueryRun]:
        if scenario_id is None:
            rows = self._conn.execute(
                "SELECT query_id, scenario_id, sim_time_s, query_text, mode, status,"
                " response_hash, payload, created_at FROM knowledge_queries"
                " ORDER BY sim_time_s DESC, query_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT query_id, scenario_id, sim_time_s, query_text, mode, status,"
                " response_hash, payload, created_at FROM knowledge_queries"
                " WHERE scenario_id = ? ORDER BY sim_time_s DESC, query_id DESC LIMIT ?",
                (scenario_id, limit),
            ).fetchall()
        return [
            KnowledgeQueryRun(
                query_id=row["query_id"],
                scenario_id=row["scenario_id"],
                sim_time_s=int(row["sim_time_s"]),
                query_text=row["query_text"],
                mode=row["mode"],
                status=row["status"],
                response=json.loads(row["payload"]),
                response_hash=row["response_hash"],
                created_at=int(row["created_at"]),
            )
            for row in rows
        ]


def _brain_id(role: Literal["master", "slave", "adversary"]) -> str:
    return {
        "master": "carrier-master",
        "slave": "group-slave",
        "adversary": "target-adversary",
    }[role]


def _looks_like_platform_id(value: str) -> bool:
    lowered = value.lower()
    return lowered.startswith(("uuv_", "uuv-", "usv_", "usv-", "carrier_", "carrier-"))
