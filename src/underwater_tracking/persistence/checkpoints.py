# src/underwater_tracking/persistence/checkpoints.py
"""LangGraph checkpoint and long-term-memory factories (spec 8.4).

The carrier graph persists checkpoints to SQLite so runs survive restart and
support recovery, state inspection, and time-travel debugging; the long-term
memory store backs scenario-level summary namespaces and expert requests
sharing one store per scenario. Both factories open their own connection to
the given database file with the same WAL + busy-timeout settings as the
agent repositories, because the checkpointer is the highest-frequency writer
in the runtime and shares the single WAL write lock with EventStore appends
(SqliteSaver has no retry, so a contention failure would crash a graph
step). Graph tests use the in-memory savers; the runtime uses these
SQLite-backed factories.

Import note: ``SqliteSaver`` lives in the ``langgraph-checkpoint-sqlite``
companion package (installed with langgraph 1.2.x); the import path is
``langgraph.checkpoint.sqlite``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

_BUSY_TIMEOUT_MS = 60_000

# LangGraph serializes typed checkpoint channels with msgpack extension
# records. Keep the allowlist explicit so a real run is warning-free and a
# future strict serializer cannot silently deserialize an unexpected class.
_ALLOWED_MSGPACK_MODULES = (
    ("underwater_tracking.domain.models", "EventLevel"),
    ("underwater_tracking.domain.models", "RuntimeEvent"),
    ("underwater_tracking.domain.models", "BearingObservation"),
    ("underwater_tracking.domain.models", "TargetBelief"),
    ("underwater_tracking.domain.models", "GroupQuality"),
    ("underwater_tracking.domain.models", "GroupReport"),
    ("underwater_tracking.groups.state", "FilterSnapshot"),
    ("underwater_tracking.domain.agent_models", "IntentHypothesis"),
    ("underwater_tracking.domain.agent_models", "PlanAdjustmentSuggestion"),
    ("underwater_tracking.domain.agent_models", "PredictedTrackRef"),
    ("underwater_tracking.domain.agent_models", "RegionalPlanMetrics"),
    ("underwater_tracking.domain.agent_models", "StrategySet"),
    ("underwater_tracking.domain.agent_models", "TrackingPlan"),
    ("underwater_tracking.domain.mission_models", "ExecutableMissionPlan"),
    ("underwater_tracking.domain.mission_models", "MissionCandidate"),
    ("underwater_tracking.domain.mission_models", "RegionMissionState"),
    ("underwater_tracking.domain.mission_models", "CarrierMissionModel"),
    ("underwater_tracking.domain.mission_models", "UUVMissionBatch"),
    ("underwater_tracking.domain.mission_models", "RegionLifecycle"),
    ("underwater_tracking.domain.mission_models", "CarrierRouteStatus"),
    ("underwater_tracking.domain.regional_models", "GridSpec"),
    ("underwater_tracking.domain.regional_models", "TimeWindow"),
    ("underwater_tracking.domain.regional_models", "RegionCell"),
    ("underwater_tracking.domain.regional_models", "RegionTask"),
    ("underwater_tracking.domain.regional_models", "CommunicationRequirement"),
    ("underwater_tracking.domain.regional_models", "TargetRegionPlan"),
    ("underwater_tracking.domain.regional_models", "RegionalMissionCandidate"),
    ("underwater_tracking.domain.regional_models", "UUVRegionalPolicy"),
    ("underwater_tracking.domain.regional_models", "UUVRegionalStrategySet"),
    ("underwater_tracking.planning.regional_plan_validator", "ValidatedRegionalStrategy"),
    ("underwater_tracking.agent.llm", "LLMCallMetadata"),
)


def _open_factory_conn(database_path: str | Path) -> sqlite3.Connection:
    """Open a factory connection in autocommit with WAL and a busy timeout.

    ``busy_timeout`` must be set before ``journal_mode=WAL``: when two
    connections first open the same fresh file, the loser of the WAL-mode
    conversion must wait on the write lock instead of failing instantly at
    the default timeout of zero.
    """
    conn = sqlite3.connect(
        str(database_path),
        check_same_thread=False,
        isolation_level=None,
        timeout=_BUSY_TIMEOUT_MS / 1000,
    )
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def create_checkpointer(database_path: str | Path) -> SqliteSaver:
    """Open a LangGraph SQLite checkpointer on the given database file."""
    conn = _open_factory_conn(database_path)
    saver = SqliteSaver(
        conn,
        serde=JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES),
    )
    saver.setup()
    return saver


def create_store(database_path: str | Path) -> SqliteStore:
    """Open a long-term memory store on the given database file."""
    return SqliteStore(_open_factory_conn(database_path))
