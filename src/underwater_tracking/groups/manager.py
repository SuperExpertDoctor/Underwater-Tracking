# src/underwater_tracking/groups/manager.py
"""Runtime registry for per-target group graphs.

``GroupManager`` owns one compiled group graph plus a ``thread_id`` per
target, so every group keeps an independent, checkpointed state. The engine
must invoke each group at most once per superstep: calling ``invoke`` twice
on the same target inside one parent step would run the stateful subgraph
twice on the same state, which is not supported.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from underwater_tracking.domain.models import BearingObservation, GroupReport
from underwater_tracking.config.models import RuntimeRetentionConfig
from underwater_tracking.groups.checkpoint import BoundedInMemorySaver
from underwater_tracking.groups.graph import build_group_graph
from underwater_tracking.groups.state import GroupState, PlanCommand


_GROUP_MSGPACK_MODULES = (
    ("underwater_tracking.domain.models", "EventLevel"),
    ("underwater_tracking.domain.models", "RuntimeEvent"),
    ("underwater_tracking.domain.models", "BearingObservation"),
    ("underwater_tracking.domain.models", "TargetBelief"),
    ("underwater_tracking.domain.models", "GroupQuality"),
    ("underwater_tracking.domain.models", "GroupReport"),
    ("underwater_tracking.groups.state", "FilterSnapshot"),
)


class GroupManager:
    """Create, invoke, complete, and list group runtimes by target id."""

    def __init__(
        self,
        checkpointer: BaseCheckpointSaver[Any] | None = None,
        *,
        retention: RuntimeRetentionConfig | None = None,
    ) -> None:
        effective_retention = retention or RuntimeRetentionConfig()
        self._checkpointer = (
            checkpointer
            if checkpointer is not None
            else BoundedInMemorySaver(
                max_checkpoints=effective_retention.group_checkpoint_limit,
                serde=JsonPlusSerializer(
                    allowed_msgpack_modules=_GROUP_MSGPACK_MODULES
                )
            )
        )
        self._graph: Any = build_group_graph(self._checkpointer)
        self._threads: dict[str, str] = {}
        self._event_history_limit = effective_retention.event_history_limit

    def create(
        self,
        target_id: str,
        *,
        scenario_id: str,
        group_id: str,
        member_ids: tuple[str, ...],
        coarse_prior: tuple[float, float],
        member_positions: dict[str, tuple[float, float]] | None = None,
    ) -> GroupReport:
        """Register a group runtime for ``target_id`` and run its init cycle.

        Raises ``ValueError`` when a runtime for the target already exists.
        """
        if target_id in self._threads:
            raise ValueError(f"group runtime for target {target_id!r} already exists")
        thread_id = f"{scenario_id}:{target_id}"
        self._threads[target_id] = thread_id
        state = GroupState.initial(
            scenario_id,
            group_id,
            target_id,
            member_ids,
            coarse_prior,
            member_positions=member_positions,
            event_history_limit=self._event_history_limit,
        )
        output = self._graph.invoke(state, config={"configurable": {"thread_id": thread_id}})
        report = output["report"]
        assert isinstance(report, GroupReport)
        return report

    def invoke(
        self,
        target_id: str,
        *,
        observations: tuple[BearingObservation, ...] = (),
        member_positions: dict[str, tuple[float, float]] | None = None,
        command: PlanCommand | None = None,
        sim_time_s: int | None = None,
    ) -> GroupReport:
        """Run one cycle of the target's group graph and return its report.

        Raises ``KeyError`` when no runtime exists for the target.
        """
        thread_id = self._threads.get(target_id)
        if thread_id is None:
            raise KeyError(f"no group runtime for target {target_id!r}")
        inputs: dict[str, object] = {}
        if observations:
            inputs["new_observations"] = tuple(observations)
        if member_positions is not None:
            inputs["member_positions"] = dict(member_positions)
        if command is not None:
            inputs["pending_command"] = command
        if sim_time_s is not None:
            inputs["cycle_sim_time_s"] = sim_time_s
        inputs["event_history_limit"] = self._event_history_limit
        output = self._graph.invoke(inputs, config={"configurable": {"thread_id": thread_id}})
        report = output["report"]
        assert isinstance(report, GroupReport)
        return report

    def complete(self, target_id: str) -> None:
        """Drop the target's group runtime and its checkpointed thread."""
        thread_id = self._threads.pop(target_id, None)
        if thread_id is None:
            raise KeyError(f"no group runtime for target {target_id!r}")
        try:
            self._checkpointer.delete_thread(thread_id)
        except NotImplementedError:
            # The checkpointer may not support thread deletion; the runtime
            # registry is still released.
            pass

    def list_groups(self) -> tuple[str, ...]:
        """Target ids of all active group runtimes, in stable sorted order."""
        return tuple(sorted(self._threads))
