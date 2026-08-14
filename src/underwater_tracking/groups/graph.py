# src/underwater_tracking/groups/graph.py
"""Compiled stateful graph for one per-target group runtime."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from underwater_tracking.groups.nodes import (
    apply_plan_command,
    build_report,
    calculate_quality,
    emit_events,
    ensure_initialized,
    ingest_observations,
    predict_and_update,
)
from underwater_tracking.groups.state import GroupState


def build_group_graph(checkpointer: BaseCheckpointSaver[Any] | None = None) -> Any:
    """Compile the per-target group graph.

    The graph is strictly linear: ``ingest_observations`` ->
    ``ensure_initialized`` -> ``predict_and_update`` -> ``calculate_quality``
    -> ``apply_plan_command`` -> ``build_report`` -> ``emit_events``.

    State persists across invokes through the checkpointer, so each target
    must run on its own ``thread_id``. When no checkpointer is supplied an
    ``InMemorySaver`` is used, which keeps every compiled graph stateful by
    default.
    """
    builder = StateGraph(GroupState)
    builder.add_node("ingest_observations", ingest_observations)
    builder.add_node("ensure_initialized", ensure_initialized)
    builder.add_node("predict_and_update", predict_and_update)
    builder.add_node("calculate_quality", calculate_quality)
    builder.add_node("apply_plan_command", apply_plan_command)
    builder.add_node("build_report", build_report)
    builder.add_node("emit_events", emit_events)
    builder.add_edge(START, "ingest_observations")
    builder.add_edge("ingest_observations", "ensure_initialized")
    builder.add_edge("ensure_initialized", "predict_and_update")
    builder.add_edge("predict_and_update", "calculate_quality")
    builder.add_edge("calculate_quality", "apply_plan_command")
    builder.add_edge("apply_plan_command", "build_report")
    builder.add_edge("build_report", "emit_events")
    builder.add_edge("emit_events", END)
    effective_checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
    return builder.compile(checkpointer=effective_checkpointer)
