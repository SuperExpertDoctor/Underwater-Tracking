"""LangGraph wiring for the first executable group-slave brain."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from underwater_tracking.agent.llm import StructuredLLM
from underwater_tracking.agent.nodes.slave import (
    SLAVE_PROMPT_VERSION,
    SlaveSonarDecisionNode,
    SlaveState,
)
from underwater_tracking.domain.slave_models import SlaveSonarDecision


def build_slave_graph(
    llm: StructuredLLM[SlaveSonarDecision],
    *,
    model_id: str = "underwater-slave-model",
    prompt_version: str = SLAVE_PROMPT_VERSION,
    temperature: float = 0.2,
) -> Any:
    """Compile the local graph with no fallback or rule-based branch."""

    builder = StateGraph(SlaveState)
    builder.add_node(
        "decide_sonar",
        SlaveSonarDecisionNode(
            llm,
            model_id=model_id,
            prompt_version=prompt_version,
            temperature=temperature,
        ),
    )
    builder.add_edge(START, "decide_sonar")
    builder.add_edge("decide_sonar", END)
    return builder.compile()


def build_group_slave_graph(
    llm: StructuredLLM[SlaveSonarDecision],
    *,
    model_id: str = "underwater-slave-model",
    prompt_version: str = SLAVE_PROMPT_VERSION,
    temperature: float = 0.2,
) -> Any:
    """Role-named alias for callers that name the graph by its group role."""

    return build_slave_graph(
        llm,
        model_id=model_id,
        prompt_version=prompt_version,
        temperature=temperature,
    )

