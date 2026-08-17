"""LangGraph wiring for one target's real-LLM adversary decision."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from underwater_tracking.agent.llm import StructuredLLM
from underwater_tracking.agent.nodes.adversary import (
    ADVERSARY_PROMPT_VERSION,
    AdversaryDecisionNode,
    AdversaryState,
    BuildAdversaryPayloadNode,
    ValidateAdversaryDecisionNode,
)
from underwater_tracking.domain.adversary_models import AdversaryEscapeDecision


def build_adversary_graph(
    llm: StructuredLLM[AdversaryEscapeDecision],
    *,
    operation: str = "adversary_escape",
    prompt_version: str = ADVERSARY_PROMPT_VERSION,
) -> Any:
    """Compile the target-side graph.

    There is intentionally no repair, fallback, or rule-based replacement:
    the injected structured LLM is the decision authority and every failure
    propagates to the caller.
    """
    builder = StateGraph(AdversaryState)
    builder.add_node("build_payload", BuildAdversaryPayloadNode())
    builder.add_node(
        "decide",
        AdversaryDecisionNode(
            llm,
            operation=operation,
            prompt_version=prompt_version,
        ),
    )
    builder.add_node(
        "validate",
        ValidateAdversaryDecisionNode(
            llm,
            operation=operation,
            prompt_version=prompt_version,
        ),
    )
    builder.add_edge(START, "build_payload")
    builder.add_edge("build_payload", "decide")
    builder.add_edge("decide", "validate")
    builder.add_edge("validate", END)
    return builder.compile()


__all__ = ["build_adversary_graph"]
