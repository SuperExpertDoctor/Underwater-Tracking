# src/underwater_tracking/agent/graphs/verify.py
"""Bounded semantic Verify subgraph (spec 8.3, plan Task 6).

Wiring: ``validate`` -> ``route_validity``; a valid candidate ends, an
invalid one goes to ``repair`` — and back to ``validate`` — while semantic
attempts remain, and an exhausted attempt budget goes to ``fallback``.
Transport retries inside the LLM port are independent: they run against the
port's own counter and never consume the semantic attempt budget. The graph
is stateless (no checkpointer): every invoke is a fresh validation cycle.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from underwater_tracking.agent.llm import StructuredLLM
from underwater_tracking.agent.nodes.verify import (
    FallbackNode,
    RepairNode,
    ValidateNode,
    VerifyContext,
    VerifyState,
    route_validity,
)
from underwater_tracking.agent.prompts import STRATEGY_PROMPT_VERSION
from underwater_tracking.domain.agent_models import ExpertDirective, StrategyProposal


def build_verify_graph(
    llm: StructuredLLM[StrategyProposal],
    *,
    model_id: str = "underwater-assistant-model",
    prompt_version: str = STRATEGY_PROMPT_VERSION,
    temperature: float = 0.2,
    target_ids: tuple[str, ...] = (),
    evidence_ids: tuple[str, ...] = (),
    allowed_soft_constraints: tuple[str, ...] = (),
    expert_directive: ExpertDirective | None = None,
) -> Any:
    """Compile the stateless bounded semantic Verify subgraph.

    ``target_ids``/``evidence_ids``/``allowed_soft_constraints``/
    ``expert_directive`` are the semantic validation context; the same
    fields on the invoke state override them per call.
    """
    context = VerifyContext(
        target_ids=target_ids,
        evidence_ids=evidence_ids,
        allowed_soft_constraints=allowed_soft_constraints,
        expert_directive=expert_directive,
    )
    builder = StateGraph(VerifyState)
    builder.add_node("validate", ValidateNode(context))
    builder.add_node(
        "repair",
        RepairNode(
            llm,
            model_id=model_id,
            prompt_version=prompt_version,
            temperature=temperature,
            context=context,
        ),
    )
    builder.add_node("fallback", FallbackNode(context))
    builder.add_edge(START, "validate")
    builder.add_conditional_edges(
        "validate",
        route_validity,
        {"end": END, "repair": "repair", "fallback": "fallback"},
    )
    builder.add_edge("repair", "validate")
    return builder.compile()
