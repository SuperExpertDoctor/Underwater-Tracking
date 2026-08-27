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

from collections.abc import Iterator
from contextlib import contextmanager
import sqlite3
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore

from underwater_tracking.persistence.sqlite import connect_database, database_write_lock

# LangGraph serializes typed checkpoint channels with msgpack extension
# records. Keep the allowlist explicit so a real run is warning-free and a
# future strict serializer cannot silently deserialize an unexpected class.
_ALLOWED_MSGPACK_MODULES = (
    ("underwater_tracking.domain.models", "EventLevel"),
    ("underwater_tracking.domain.models", "EventAudience"),
    ("underwater_tracking.domain.models", "UUVStatus"),
    ("underwater_tracking.domain.models", "CarrierStatus"),
    ("underwater_tracking.domain.models", "DeploymentState"),
    ("underwater_tracking.domain.models", "ContactClassification"),
    ("underwater_tracking.domain.models", "IntelligenceSource"),
    ("underwater_tracking.domain.models", "SurveillanceCapability"),
    ("underwater_tracking.domain.models", "OperationalScheme"),
    ("underwater_tracking.domain.models", "IntelligenceReport"),
    ("underwater_tracking.domain.models", "Contact"),
    ("underwater_tracking.domain.models", "RuntimeEvent"),
    ("underwater_tracking.domain.models", "BearingObservation"),
    ("underwater_tracking.domain.models", "UUVState"),
    ("underwater_tracking.domain.models", "CarrierState"),
    ("underwater_tracking.domain.models", "SituationSnapshot"),
    ("underwater_tracking.domain.models", "TargetBelief"),
    ("underwater_tracking.domain.models", "GroupQuality"),
    ("underwater_tracking.domain.models", "GroupReport"),
    ("underwater_tracking.domain.planning_epoch_models", "PlanningEpoch"),
    ("underwater_tracking.domain.planning_epoch_models", "EpochCommitResult"),
    ("underwater_tracking.domain.planning_epoch_models", "PlanningEpochStatus"),
    ("underwater_tracking.runtime.mission_controller", "MissionSnapshot"),
    ("underwater_tracking.groups.state", "FilterSnapshot"),
    ("underwater_tracking.domain.agent_models", "IntentHypothesis"),
    ("underwater_tracking.domain.agent_models", "PlanAdjustmentSuggestion"),
    ("underwater_tracking.domain.agent_models", "PredictedTrackRef"),
    ("underwater_tracking.domain.agent_models", "TrajectoryDiffResult"),
    ("underwater_tracking.domain.agent_models", "TrajectoryDiffGateState"),
    ("underwater_tracking.domain.agent_models", "IntentVerificationCallRef"),
    ("underwater_tracking.intent.deterministic", "ConfirmedIntentRevision"),
    ("underwater_tracking.intent.deterministic", "IntentLatchState"),
    ("underwater_tracking.domain.agent_models", "Segment"),
    ("underwater_tracking.domain.agent_models", "SegmentPlan"),
    ("underwater_tracking.domain.agent_models", "StrategyProposal"),
    ("underwater_tracking.domain.agent_models", "RegionalPlanMetrics"),
    ("underwater_tracking.domain.agent_models", "StrategySet"),
    ("underwater_tracking.domain.agent_models", "Waypoint"),
    ("underwater_tracking.domain.agent_models", "PlanDiff"),
    ("underwater_tracking.domain.agent_models", "TrackingPlan"),
    ("underwater_tracking.domain.agent_models", "ExpertDirective"),
    ("underwater_tracking.domain.mission_models", "ExecutableMissionPlan"),
    ("underwater_tracking.domain.mission_models", "MissionCandidate"),
    ("underwater_tracking.domain.mission_models", "UUVMissionMode"),
    ("underwater_tracking.domain.mission_models", "UUVResourceState"),
    ("underwater_tracking.domain.mission_models", "PredictionGridCell"),
    ("underwater_tracking.domain.mission_models", "PredictionGrid"),
    ("underwater_tracking.domain.mission_models", "RegionMissionState"),
    ("underwater_tracking.domain.mission_models", "CarrierMissionModel"),
    ("underwater_tracking.domain.mission_models", "UUVMissionBatch"),
    ("underwater_tracking.domain.mission_models", "RegionLifecycle"),
    ("underwater_tracking.domain.mission_models", "CarrierRouteStatus"),
    ("underwater_tracking.domain.mission_models", "CarrierExecutionMode"),
    ("underwater_tracking.domain.regional_models", "GridSpec"),
    ("underwater_tracking.domain.regional_models", "TimeWindow"),
    ("underwater_tracking.domain.regional_models", "RegionCell"),
    ("underwater_tracking.domain.regional_models", "SonarPolicy"),
    ("underwater_tracking.domain.regional_models", "RegionTask"),
    ("underwater_tracking.domain.regional_models", "CommunicationRequirement"),
    ("underwater_tracking.domain.regional_models", "RegionalPolicy"),
    ("underwater_tracking.domain.regional_models", "RegionalStrategySet"),
    ("underwater_tracking.domain.regional_models", "TargetRegionPlan"),
    ("underwater_tracking.domain.regional_models", "RegionalMissionCandidate"),
    ("underwater_tracking.domain.regional_models", "UUVRegionalPolicy"),
    ("underwater_tracking.domain.regional_models", "UUVRegionalStrategySet"),
    ("underwater_tracking.domain.execution_models", "ExecutionRegion"),
    ("underwater_tracking.planning.dynamic_regions", "DynamicRegionChain"),
    ("underwater_tracking.planning.regional_plan_validator", "ValidatedRegionalStrategy"),
    ("underwater_tracking.domain.platforms", "PlatformKind"),
    ("underwater_tracking.domain.platforms", "MotionLimits"),
    ("underwater_tracking.domain.platforms", "SonarCapability"),
    ("underwater_tracking.domain.platforms", "CommunicationCapability"),
    ("underwater_tracking.domain.platforms", "PlatformCapability"),
    ("underwater_tracking.domain.platforms", "UUVPlatformState"),
    ("underwater_tracking.domain.platforms", "USVPlatformState"),
    ("underwater_tracking.domain.platforms", "CarrierPlatformState"),
    ("underwater_tracking.domain.platforms", "PlatformRoster"),
    ("underwater_tracking.domain.platforms", "CommunicationLink"),
    ("underwater_tracking.domain.platforms", "PlatformSnapshot"),
    ("underwater_tracking.domain.observations", "PassiveSonarObservation"),
    ("underwater_tracking.domain.adversary_models", "LocalPlatformDetection"),
    ("underwater_tracking.agent.llm", "LLMCallMetadata"),
    ("underwater_tracking.agent.nodes.snapshot", "PlanningSnapshot"),
    ("underwater_tracking.world_model.models", "DataStatus"),
    ("underwater_tracking.world_model.models", "HorizonName"),
    ("underwater_tracking.world_model.models", "EventType"),
    ("underwater_tracking.world_model.models", "RuleEvidence"),
    ("underwater_tracking.world_model.models", "PredictedEvent"),
    ("underwater_tracking.world_model.models", "HorizonCoverage"),
    ("underwater_tracking.world_model.models", "WorldModelForecast"),
)


def _open_factory_conn(database_path: str | Path) -> sqlite3.Connection:
    """Open a factory connection in autocommit with WAL and a busy timeout.

    ``busy_timeout`` must be set before ``journal_mode=WAL``: when two
    connections first open the same fresh file, the loser of the WAL-mode
    conversion must wait on the write lock instead of failing instantly at
    the default timeout of zero.
    """
    return connect_database(database_path)


class LockedSqliteSaver(SqliteSaver):
    """Serialize LangGraph checkpoint writes with repository transactions."""

    def __init__(self, *args: object, max_checkpoints: int = 2, **kwargs: object) -> None:
        if max_checkpoints < 1:
            raise ValueError("max_checkpoints must be positive")
        super().__init__(*args, **kwargs)
        self.max_checkpoints = max_checkpoints

    @contextmanager
    def cursor(self, transaction: bool = True) -> Iterator[sqlite3.Cursor]:
        with database_write_lock(self.conn):
            with super().cursor(transaction=transaction) as cursor:
                yield cursor

    def put(
        self,
        config: object,
        checkpoint: object,
        metadata: object,
        new_versions: object,
    ) -> object:
        result = super().put(config, checkpoint, metadata, new_versions)
        configurable = config["configurable"]  # type: ignore[index]
        self._prune_thread(
            str(configurable["thread_id"]),  # type: ignore[index]
            str(configurable.get("checkpoint_ns", "")),  # type: ignore[union-attr]
        )
        return result

    def put_writes(
        self,
        config: object,
        writes: object,
        task_id: str,
        task_path: str = "",
    ) -> None:
        super().put_writes(config, writes, task_id, task_path)  # type: ignore[arg-type]
        configurable = config["configurable"]  # type: ignore[index]
        self._prune_thread(
            str(configurable["thread_id"]),  # type: ignore[index]
            str(configurable.get("checkpoint_ns", "")),  # type: ignore[union-attr]
        )

    def _prune_thread(self, thread_id: str, checkpoint_ns: str = "") -> None:
        """Keep recent checkpoints and delete their dependent pending writes."""
        with database_write_lock(self.conn):
            rows = self.conn.execute(
                "SELECT checkpoint_id FROM checkpoints "
                "WHERE thread_id = ? AND checkpoint_ns = ? "
                "ORDER BY checkpoint_id DESC",
                (thread_id, checkpoint_ns),
            ).fetchall()
            stale_ids = [
                str(row[0]) for row in rows[self.max_checkpoints :]
            ]
            if not stale_ids:
                return
            self.conn.executemany(
                "DELETE FROM writes WHERE thread_id = ? AND checkpoint_ns = ? "
                "AND checkpoint_id = ?",
                ((thread_id, checkpoint_ns, checkpoint_id) for checkpoint_id in stale_ids),
            )
            self.conn.executemany(
                "DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? "
                "AND checkpoint_id = ?",
                ((thread_id, checkpoint_ns, checkpoint_id) for checkpoint_id in stale_ids),
            )


class LockedSqliteStore(SqliteStore):
    """Serialize LangGraph store batches with repository transactions."""

    @contextmanager
    def _cursor(self, *, transaction: bool = True) -> Iterator[sqlite3.Cursor]:
        with database_write_lock(self.conn):
            with super()._cursor(transaction=transaction) as cursor:
                yield cursor


def create_checkpointer(
    database_path: str | Path, *, max_checkpoints: int = 2
) -> SqliteSaver:
    """Open a LangGraph SQLite checkpointer on the given database file."""
    conn = _open_factory_conn(database_path)
    saver = LockedSqliteSaver(
        conn,
        max_checkpoints=max_checkpoints,
        serde=JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES),
    )
    with database_write_lock(conn):
        saver.setup()
    return saver


def create_store(database_path: str | Path) -> SqliteStore:
    """Open a long-term memory store on the given database file."""
    return LockedSqliteStore(_open_factory_conn(database_path))
