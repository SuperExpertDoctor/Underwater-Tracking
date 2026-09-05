# src/underwater_tracking/groups/nodes.py
"""Deterministic graph nodes for one per-target group runtime.

The pipeline is strictly linear (see ``graph.py``): ingest observations,
guarantee an initialized IMM-UIF filter, predict/update the belief,
calculate group quality, apply any pending plan command, build the report,
and emit guard events.

All arithmetic is deterministic -- no randomness appears anywhere in the
graph logic, and every node is a pure function of its state (the IMM-UIF
estimator is rebuilt from its serialized snapshot on every cycle). numpy
arrays exist only inside node bodies; every value written back to state is
JSON-serializable (tuples, dicts, pydantic models).
"""

from __future__ import annotations

from itertools import groupby
from math import isfinite

import numpy as np

from underwater_tracking.domain.models import (
    BearingObservation,
    EventLevel,
    GroupReport,
    RuntimeEvent,
    TargetBelief,
)
from underwater_tracking.groups.state import FilterSnapshot, GroupState, ModelFilterState
from underwater_tracking.planning.fim import fim_metrics
from underwater_tracking.tracking.imm import (
    DEFAULT_COMMANDED_TURNS,
    DEFAULT_PROCESS_NOISE,
    DEFAULT_TRANSITION_MATRIX,
    MODEL_ORDER,
    ImmEstimator,
    build_default_imm,
)
from underwater_tracking.tracking.initialization import (
    InsufficientGeometryError,
    initialize_from_bearings,
)
from underwater_tracking.tracking.quality import QualityCalculator, QualityInputs
from underwater_tracking.tracking.uif import UnscentedInformationFilter

# Inflated prior covariance used when initialization is impossible.
#
# The position block is diffuse (630 m std) but deliberately not extreme:
# the reviewed UIF's circular statistics lose positive definiteness when a
# single update is asked to collapse a prior of ~800 m std or more, so the
# failure-path prior stays safely below that cliff. Velocity/turn variances
# are physical (UUVs move at a few m/s).
_INFLATED_PRIOR_COVARIANCE = np.diag([400_000.0, 400_000.0, 25.0, 25.0, 0.01])

# Velocity/turn-rate block applied on top of a triangulated position.
# Velocity variance is physical (UUVs move at a few m/s); a 100 m/s spread
# would re-introduce the circular-statistics instability on the next cycle.
_VELOCITY_VARIANCE = 25.0
_TURN_VARIANCE = 0.1

# NIS gate shared with the UIF, used to normalize NIS into [0, 1].
_NIS_GATE = 6.635

# Quality window/alpha matching the default TrackingConfig-backed calculator.
_QUALITY_WINDOW_S = 300
_QUALITY_EWMA_ALPHA = 0.2


def ingest_observations(state: GroupState) -> dict[str, object]:
    """Store this cycle's observations, dropping any aimed at another target.

    Observations whose ``target_id`` or ``scenario_id`` does not match the
    group are ignored, so a misrouted observation can never move a belief.
    """
    seen = set(state.processed_observation_ids)
    batch: list[BearingObservation] = []
    rejected: set[str] = set()
    state_time = state.belief.sim_time_s if state.belief is not None else 0
    for observation in sorted(
        state.new_observations, key=lambda item: (item.sim_time_s, item.observation_id)
    ):
        if observation.target_id != state.target_id or observation.scenario_id != state.scenario_id:
            continue
        if observation.observation_id in seen:
            rejected.add("duplicate_observation")
        elif observation.sim_time_s < state_time:
            rejected.add("out_of_sequence_observation")
        elif state.cycle_sim_time_s is not None and observation.sim_time_s > state.cycle_sim_time_s:
            rejected.add("future_observation")
        elif (
            observation.is_false_alarm
            or observation.detection_confidence <= 0
            or not isfinite(observation.azimuth_rad)
            or not isfinite(observation.variance_rad2)
            or (
                observation.observer_position_xy is not None
                and not all(isfinite(value) for value in observation.observer_position_xy)
            )
        ):
            rejected.add("invalid_observation")
        else:
            batch.append(observation)
            seen.add(observation.observation_id)
    return {
        "new_observations": (),
        "last_observations": tuple(batch),
        "observation_rejection_reasons": tuple(sorted(rejected)),
    }


def ensure_initialized(state: GroupState) -> dict[str, object]:
    """Initialize the filter on the first cycle; no-op afterwards.

    The success path triangulates the current observation batch against the
    coarse prior using simultaneous bearings only. A successful initialization
    consumes those bearings exactly once at their observation time. Otherwise
    the diffuse prior starts at zero. Zero-duration prediction is never run.
    """
    if state.filter_snapshot is not None:
        return {}
    first_time = min((obs.sim_time_s for obs in state.last_observations), default=0)
    used, origins, bearings, variances = _matched_observations(
        tuple(obs for obs in state.last_observations if obs.sim_time_s == first_time),
        state.member_positions,
    )
    mean = np.asarray([state.coarse_prior[0], state.coarse_prior[1], 0.0, 0.0, 0.0], dtype=float)
    covariance = _INFLATED_PRIOR_COVARIANCE.copy()
    source_ids: tuple[str, ...] = ()
    if len(used) >= 2:
        try:
            result = initialize_from_bearings(
                origins,
                bearings,
                variances,
                prior=np.asarray(state.coarse_prior, dtype=float),
            )
        except InsufficientGeometryError:
            pass
        else:
            mean[0], mean[1] = result.position_xy
            covariance = np.zeros((5, 5), dtype=float)
            covariance[:2, :2] = result.covariance_xy
            covariance[2, 2] = _VELOCITY_VARIANCE
            covariance[3, 3] = _VELOCITY_VARIANCE
            covariance[4, 4] = _TURN_VARIANCE
            source_ids = tuple(observation.observation_id for observation in used)
    estimator = build_default_imm(mean, covariance)
    belief = _belief_from_estimator(estimator, state, first_time if source_ids else 0, source_ids)
    if source_ids:
        belief = belief.model_copy(
            update={
                "track_revision": 1,
                "last_observed_at_s": first_time,
                "valid_until_s": first_time + state.observation_validity_s,
            }
        )
    return {
        "filter_snapshot": _capture_estimator(estimator),
        "belief": belief,
        "processed_observation_ids": source_ids,
        "bootstrap_observation_ids": source_ids,
    }


def predict_and_update(state: GroupState) -> dict[str, object]:
    """Predict the IMM forward and update it from the ingested bearings.

    Time advances through the explicit cycle timestamp or observation
    timestamps: ``dt`` is the gap between the newest available time and the
    previous belief time, whether or not an update runs. An empty observation
    batch therefore still predicts to the current engine clock.
    ``last_accepted_sim_time_s`` records the time of the last cycle that
    actually updated the filter; predict-only cycles leave it untouched so
    quality can age the track.

    Same-time observations are batched, triangulation inputs are consumed
    once, and a zero-duration prediction never injects artificial process
    noise. This does not claim to solve every numerical limitation of UIF.
    """
    snapshot = state.filter_snapshot
    belief = state.belief
    if snapshot is None or belief is None:
        raise RuntimeError("predict_and_update called before initialization")
    estimator = _restore_estimator(snapshot)
    observation_time_s = max(
        (observation.sim_time_s for observation in state.last_observations),
        default=belief.sim_time_s,
    )
    now_s = max(
        belief.sim_time_s,
        observation_time_s,
        state.cycle_sim_time_s if state.cycle_sim_time_s is not None else belief.sim_time_s,
    )
    cursor = belief.sim_time_s
    nis_values: list[float] = []
    accepted: list[BearingObservation] = []
    rejected = set(state.observation_rejection_reasons)
    processed = list(state.processed_observation_ids)
    pending = tuple(
        obs for obs in state.last_observations if obs.observation_id not in set(processed)
    )
    for observed_s, batch in groupby(
        sorted(pending, key=lambda item: item.sim_time_s), key=lambda item: item.sim_time_s
    ):
        used, origins, bearings, variances = _matched_observations(
            tuple(batch), state.member_positions
        )
        if not used:
            continue
        if observed_s > cursor:
            estimator.predict(float(observed_s - cursor))
            cursor = observed_s
        per_model = estimator.update(origins, bearings, variances)
        nis_values.extend(per_model[0])
        accepted.extend(
            obs
            for index, obs in enumerate(used)
            if any(isfinite(values[index]) and values[index] <= _NIS_GATE for values in per_model)
        )
        if any(
            not any(isfinite(values[index]) and values[index] <= _NIS_GATE for values in per_model)
            for index in range(len(used))
        ):
            rejected.add("innovation_gate_rejected")
        processed.extend(obs.observation_id for obs in used)
    if now_s > cursor:
        estimator.predict(float(now_s - cursor))
    current_ids = (*state.bootstrap_observation_ids, *(obs.observation_id for obs in accepted))
    source_ids = tuple(dict.fromkeys((*belief.source_observation_ids, *current_ids)))[-256:]
    last_observed = max((obs.sim_time_s for obs in accepted), default=belief.last_observed_at_s)
    updated = _belief_from_estimator(
        estimator,
        state,
        now_s,
        source_ids,
    )
    updated = updated.model_copy(
        update={
            "track_revision": belief.track_revision
            + bool(accepted and not state.bootstrap_observation_ids),
            "last_observed_at_s": last_observed,
            "valid_until_s": last_observed + state.observation_validity_s
            if last_observed is not None
            else None,
            "accepted_observation_ids_this_cycle": current_ids,
            "reason_codes": tuple(sorted(rejected))
            or (() if current_ids else ("no_new_observations",)),
        }
    )
    result: dict[str, object] = {
        "filter_snapshot": _capture_estimator(estimator),
        "belief": updated,
        "last_nis_values": tuple(nis_values),
        "processed_observation_ids": tuple(dict.fromkeys(processed))[-4096:],
        "bootstrap_observation_ids": (),
    }
    if current_ids:
        result["last_accepted_sim_time_s"] = last_observed
    return result


def calculate_quality(state: GroupState) -> dict[str, object]:
    """Compute group quality from the belief and the last update's NIS.

    The stateful calculator is restored from the persisted quality history
    and EWMA, so window mean and EWMA are exact across cycles and
    checkpoints. Hard guards fire immediately (instant quality pinned to
    0.0) and surface as ``hard_guard_reasons`` for the event node.

    Freshness ages the track honestly: ``age_s`` is the gap between the
    current belief time and ``last_accepted_sim_time_s``, so a track that
    stopped accepting observations decays (``q_fresh = exp(-age/window)``)
    instead of reporting maxed freshness forever. A track that never
    accepted anything ages from its time of birth (``sim_time_s == 0``).
    """
    belief = state.belief
    if belief is None:
        raise RuntimeError("calculate_quality requires an initialized belief")
    calculator = QualityCalculator(
        window_s=_QUALITY_WINDOW_S, ewma_alpha=_QUALITY_EWMA_ALPHA
    )
    _restore_calculator(calculator, state.quality_history, state.quality_ewma)
    nis_values = state.last_nis_values
    if nis_values:
        accepted = [value for value in nis_values if value <= _NIS_GATE]
        detection_rate = len(accepted) / len(nis_values)
        if accepted:
            normalized_nis = 1.0 - min(1.0, sum(accepted) / len(accepted) / _NIS_GATE)
        else:
            normalized_nis = 0.0
    else:
        detection_rate = 0.0
        normalized_nis = 0.0
    last_accepted_s = state.last_accepted_sim_time_s
    age_s = float(max(0, belief.sim_time_s - (last_accepted_s if last_accepted_s is not None else 0)))
    quality = calculator.update(
        float(belief.sim_time_s),
        QualityInputs(
            covariance_trace=_position_trace(belief),
            fim_min_eigenvalue=belief.fim_min_eigenvalue,
            fim_condition=belief.fim_condition,
            detection_rate=detection_rate,
            normalized_nis=normalized_nis,
            age_s=age_s,
        ),
    )
    return {
        "quality": quality,
        "quality_history": tuple(calculator._samples),
        "quality_ewma": calculator._ewma,
    }


def apply_plan_command(state: GroupState) -> dict[str, object]:
    """Apply a pending plan command and revision bump atomically.

    A command aimed at another target is dropped. Authoritative commands
    replace the complete roster and its positions, emitting deterministic
    add/remove/replacement events. Legacy replacement-only commands retain
    their historical ``member_failed`` events.
    """
    command = state.pending_command
    if command is None:
        return {}
    if command.target_id != state.target_id or command.scenario_id != state.scenario_id:
        return {"pending_command": None}
    members = list(state.member_ids)
    positions = dict(state.member_positions)
    events = list(state.emitted_events)
    if command.desired_member_ids is not None:
        desired = tuple(dict.fromkeys(command.desired_member_ids))
        current = set(state.member_ids)
        desired_set = set(desired)
        removed = sorted(current - desired_set)
        added = sorted(desired_set - current)
        supplied_positions = command.member_positions or {}
        positions = {
            member: supplied_positions[member]
            if member in supplied_positions
            else state.member_positions[member]
            for member in desired
            if member in supplied_positions or member in state.member_positions
        }

        def emit(event_type: str, entity_id: str, payload: dict[str, str] | None = None) -> None:
            events.append(
                RuntimeEvent(
                    event_id=f"{state.group_id}:{event_type}:{command.command_id}:{entity_id}",
                    scenario_id=state.scenario_id,
                    sim_time_s=command.sim_time_s,
                    event_type=event_type,
                    entity_id=entity_id,
                    level=EventLevel.TACTICAL,
                    payload=payload or {},
                )
            )

        replacements = min(len(removed), len(added))
        for removed_id, added_id in zip(removed[:replacements], added[:replacements], strict=True):
            emit("member_replaced", removed_id, {"replacement": added_id})
        for removed_id in removed[replacements:]:
            emit("member_removed", removed_id)
        for added_id in added[replacements:]:
            emit("member_added", added_id)
        return {
            "member_ids": desired,
            "member_positions": positions,
            "plan_revision": command.plan_revision,
            "pending_command": None,
            "emitted_events": _event_tail(events, state.event_history_limit),
        }

    applied: dict[str, str] = {}
    for index, member in enumerate(members):
        replacement = command.member_replacements.get(member)
        if replacement is not None:
            members[index] = replacement
            positions.pop(member, None)
            applied[member] = replacement
    deduped: list[str] = []
    for member in members:
        if member not in deduped:
            deduped.append(member)
    for failed, replacement in applied.items():
        events.append(
            RuntimeEvent(
                event_id=(
                    f"{state.group_id}:member_failed:{command.command_id}:{failed}"
                ),
                scenario_id=state.scenario_id,
                sim_time_s=command.sim_time_s,
                event_type="member_failed",
                entity_id=failed,
                level=EventLevel.TACTICAL,
                payload={"replacement": replacement},
            )
        )
    return {
        "member_ids": tuple(deduped),
        "member_positions": positions,
        "plan_revision": command.plan_revision,
        "pending_command": None,
        "emitted_events": _event_tail(events, state.event_history_limit),
    }


def build_report(state: GroupState) -> dict[str, object]:
    """Build the group report from the current belief, quality, and roster."""
    belief = state.belief
    quality = state.quality
    if belief is None or quality is None:
        raise RuntimeError("build_report requires an initialized belief and quality")
    report = GroupReport(
        group_id=state.group_id,
        target_id=state.target_id,
        sim_time_s=belief.sim_time_s,
        member_ids=state.member_ids,
        belief=belief,
        quality=quality,
        plan_revision=state.plan_revision,
        event_types=quality.hard_guard_reasons,
    )
    return {"report": report, "last_report": report}


def emit_events(state: GroupState) -> dict[str, object]:
    """Turn newly-appeared quality hard guards into runtime events."""
    quality = state.quality
    if quality is None:
        return {}
    guard_reasons = quality.hard_guard_reasons
    new_reasons = [reason for reason in guard_reasons if reason not in state.last_guard_reasons]
    if not new_reasons:
        return {"last_guard_reasons": guard_reasons}
    events = list(state.emitted_events)
    sim_time_s = state.belief.sim_time_s if state.belief is not None else 0
    for reason in new_reasons:
        events.append(
            RuntimeEvent(
                event_id=(
                    f"{state.group_id}:quality_guard:{reason}:{sim_time_s}"
                ),
                scenario_id=state.scenario_id,
                sim_time_s=sim_time_s,
                event_type=f"quality_guard:{reason}",
                entity_id=state.group_id,
                level=EventLevel.TACTICAL,
                payload={},
            )
        )
    return {
        "emitted_events": _event_tail(events, state.event_history_limit),
        "last_guard_reasons": guard_reasons,
    }


def _event_tail(
    events: list[RuntimeEvent], limit: int
) -> tuple[RuntimeEvent, ...]:
    """Keep group event state bounded while preserving chronological order."""
    return tuple(events[-limit:])


def _matched_observations(
    batch: tuple[BearingObservation, ...],
    positions: dict[str, tuple[float, float]],
) -> tuple[list[BearingObservation], np.ndarray, np.ndarray, np.ndarray]:
    """Pair observations with member positions, sorted by member and id.

    Observations from members without a known position are skipped. Sorting
    by ``(uuv_id, observation_id)`` makes the update order independent of
    the caller's input order.
    """
    used = [
        observation
        for observation in sorted(batch, key=lambda item: (item.uuv_id, item.observation_id))
        if observation.uuv_id in positions or observation.observer_position_xy is not None
    ]
    origins = np.asarray(
        [observation.observer_position_xy or positions[observation.uuv_id] for observation in used],
        dtype=float,
    )
    bearings = np.asarray([observation.azimuth_rad for observation in used], dtype=float)
    variances = np.asarray([observation.variance_rad2 for observation in used], dtype=float)
    return used, origins, bearings, variances


def _restore_estimator(snapshot: FilterSnapshot) -> ImmEstimator:
    """Rebuild the IMM estimator from its serialized snapshot."""
    filters = {
        name: UnscentedInformationFilter(
            mean=np.asarray(snapshot.filters[name].mean, dtype=float),
            covariance=np.asarray(snapshot.filters[name].covariance, dtype=float),
            process_noise=DEFAULT_PROCESS_NOISE,
        )
        for name in MODEL_ORDER
    }
    probabilities = np.asarray(
        [snapshot.model_probabilities[name] for name in MODEL_ORDER], dtype=float
    )
    return ImmEstimator(
        filters,
        DEFAULT_TRANSITION_MATRIX,
        probabilities,
        dict(zip(MODEL_ORDER, DEFAULT_COMMANDED_TURNS)),
    )


def _capture_estimator(estimator: ImmEstimator) -> FilterSnapshot:
    """Serialize the IMM estimator into a JSON-safe snapshot."""
    return FilterSnapshot(
        filters={
            name: ModelFilterState(
                mean=tuple(float(value) for value in model.mean),
                covariance=tuple(
                    tuple(float(value) for value in row) for row in model.covariance
                ),
            )
            for name, model in estimator.filters.items()
        },
        model_probabilities={
            name: float(estimator.model_probabilities[index])
            for index, name in enumerate(MODEL_ORDER)
        },
    )


def _belief_from_estimator(
    estimator: ImmEstimator,
    state: GroupState,
    sim_time_s: int,
    source_observation_ids: tuple[str, ...],
) -> TargetBelief:
    """Build the domain belief from the estimator's mixed output."""
    covariance = tuple(
        tuple(float(value) for value in row) for row in estimator.mixed_covariance
    )
    fim_min_eigenvalue, fim_condition = _fim_from_covariance(covariance)
    return TargetBelief(
        target_id=state.target_id,
        sim_time_s=sim_time_s,
        mean=tuple(float(value) for value in estimator.mixed_mean),
        covariance=covariance,
        model_probabilities={
            name: float(estimator.model_probabilities[index])
            for index, name in enumerate(MODEL_ORDER)
        },
        source_observation_ids=source_observation_ids,
        fim_min_eigenvalue=fim_min_eigenvalue,
        fim_condition=fim_condition,
    )


def _fim_from_covariance(covariance: tuple[tuple[float, ...], ...]) -> tuple[float, float]:
    """FIM summary of the 2x2 position block of a belief covariance.

    The reduction (minimum eigenvalue, condition number) is delegated to the
    planning layer's ``fim_metrics`` -- the single source of truth for FIM
    conventions -- so any future convention change propagates to the group
    runtime instead of silently diverging.
    """
    position_covariance = np.asarray(
        [[covariance[0][0], covariance[0][1]], [covariance[1][0], covariance[1][1]]],
        dtype=float,
    )
    metrics = fim_metrics(np.linalg.pinv(position_covariance))
    return metrics.min_eigenvalue, metrics.condition_number


def _position_trace(belief: TargetBelief) -> float:
    return float(sum(belief.covariance[index][index] for index in range(2)))


def _restore_calculator(
    calculator: QualityCalculator,
    history: tuple[tuple[float, float], ...],
    ewma: float | None,
) -> None:
    """Restore a fresh calculator from persisted history.

    ``QualityCalculator`` (a reviewed module) has no public restore API, so
    the samples deque and EWMA are repopulated directly. This is the only
    private-state access in the group runtime and is kept in one place.
    """
    calculator._samples.extend(history)
    if ewma is not None:
        calculator._ewma = ewma
