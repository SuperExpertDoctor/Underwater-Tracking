# src/underwater_tracking/groups/state.py
"""Group state schema for one per-target tracking graph.

``GroupState`` is the only state that flows through a target's group graph.
It holds the scenario/group/target identity, the member roster (and their
current positions, fed by the engine), the serialized IMM-UIF filter, the
last observation batch, the quality history, the current plan revision, the
pending plan command, the last report, and the emitted events. It never
holds global resources, LLM messages, or truth.

The whole schema must survive LangGraph checkpointing, which serializes
state as JSON: numpy arrays therefore never appear in state. Means and
covariances are plain tuples, and every node converts to and from numpy at
its own boundary (see ``nodes.py``).
"""

from __future__ import annotations

from pydantic import Field

from underwater_tracking.domain.models import (
    BearingObservation,
    GroupQuality,
    GroupReport,
    RuntimeEvent,
    StrictModel,
    TargetBelief,
)


class PlanCommand(StrictModel):
    """Versioned planning-layer command for one target's group.

    ``member_replacements`` maps a failed member UUV id to its replacement;
    ``plan_revision`` is the new plan revision the command produces. The
    graph applies the command deterministically: replaced members leave the
    roster (and their positions), the revision is adopted, and one
    ``member_failed`` event is emitted per applied replacement.
    """

    command_id: str
    scenario_id: str
    target_id: str
    sim_time_s: int = Field(ge=0)
    plan_revision: int = Field(ge=0)
    member_replacements: dict[str, str] = Field(default_factory=dict)


class ModelFilterState(StrictModel):
    """Serializable mean/covariance of one IMM model filter."""

    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]


class FilterSnapshot(StrictModel):
    """Serializable snapshot of the whole IMM estimator (three models).

    The estimator is rebuilt from this snapshot at the start of every
    ``predict_and_update`` node so that no mutable estimator object ever
    lives inside checkpointed state.
    """

    filters: dict[str, ModelFilterState]
    model_probabilities: dict[str, float]


class GroupState(StrictModel):
    """Checkpoint-safe state of one target's group runtime.

    Every field has a default so that partial invoke inputs (for example
    ``{"new_observations": ...}`` against a checkpointed thread) validate.
    """

    scenario_id: str = ""
    group_id: str = ""
    target_id: str = ""
    member_ids: tuple[str, ...] = ()
    #: Observer positions per member UUV, fed by the engine each cycle.
    member_positions: dict[str, tuple[float, float]] = Field(default_factory=dict)
    #: Coarse position prior used to initialize the track.
    coarse_prior: tuple[float, float] = (0.0, 0.0)
    #: Observations supplied by the caller for this cycle.
    new_observations: tuple[BearingObservation, ...] = ()
    #: The observations actually ingested for this cycle (target-filtered).
    last_observations: tuple[BearingObservation, ...] = ()
    #: Serialized IMM-UIF filter; None until the first cycle initializes.
    filter_snapshot: FilterSnapshot | None = None
    #: Latest blended belief; None until the first cycle initializes.
    belief: TargetBelief | None = None
    #: Latest group quality.
    quality: GroupQuality | None = None
    #: (sim_time_s, instant) quality samples, restored into the calculator.
    quality_history: tuple[tuple[float, float], ...] = ()
    #: EWMA from the calculator, restored across cycles.
    quality_ewma: float | None = None
    plan_revision: int = 0
    pending_command: PlanCommand | None = None
    #: Report produced by the latest cycle (also kept in ``last_report``).
    report: GroupReport | None = None
    last_report: GroupReport | None = None
    emitted_events: tuple[RuntimeEvent, ...] = ()
    #: Guard reasons already turned into events (dedup across cycles).
    last_guard_reasons: tuple[str, ...] = ()
    #: NIS values of the last measurement update (cv model), for quality.
    last_nis_values: tuple[float, ...] = ()

    @classmethod
    def initial(
        cls,
        scenario_id: str,
        group_id: str,
        target_id: str,
        member_ids: tuple[str, ...],
        coarse_prior: tuple[float, float],
        member_positions: dict[str, tuple[float, float]] | None = None,
    ) -> GroupState:
        """Create a fresh group state for one target before its first cycle."""
        return cls(
            scenario_id=scenario_id,
            group_id=group_id,
            target_id=target_id,
            member_ids=tuple(member_ids),
            coarse_prior=coarse_prior,
            member_positions=dict(member_positions) if member_positions is not None else {},
        )
