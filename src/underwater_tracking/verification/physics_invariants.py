"""In-process trajectory auditing for release acceptance.

The monitor deliberately returns aggregates and frame identifiers only.  The
caller supplies the internal truth projection while it is still in process;
coordinates and private depth never leave this module through its result
models.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import atan2, hypot
from typing import Literal

from pydantic import Field, model_validator

from underwater_tracking.domain.models import StrictModel
from underwater_tracking.simulation.kinematics import wrap_angle

EntityKind = Literal["carrier", "mother_ship", "uuv", "submarine"]


class EntityMotionLimits(StrictModel):
    max_speed_mps: float = Field(ge=0)
    max_acceleration_mps2: float = Field(ge=0)
    max_deceleration_mps2: float = Field(ge=0)
    max_turn_rate_rad_s: float = Field(ge=0)
    min_depth_m: float | None = Field(default=None, ge=0)
    max_depth_m: float | None = Field(default=None, ge=0)
    max_vertical_speed_mps: float | None = Field(default=None, ge=0)
    max_vertical_acceleration_mps2: float | None = Field(default=None, ge=0)
    max_pitch_rad: float | None = Field(default=None, ge=0)


class EntityMotionAudit(StrictModel):
    entity_id: str
    entity_kind: EntityKind
    observed_steps: int = Field(ge=0)
    max_speed_mps: float = Field(ge=0)
    max_acceleration_mps2: float = Field(ge=0)
    max_deceleration_mps2: float = Field(ge=0)
    max_turn_rate_rad_s: float = Field(ge=0)
    min_depth_m: float | None = None
    max_depth_m: float | None = None
    max_vertical_speed_mps: float | None = None
    max_vertical_acceleration_mps2: float | None = None
    max_pitch_rad: float | None = None
    teleport_count: int = Field(default=0, ge=0)
    boundary_violation_count: int = Field(default=0, ge=0)
    owner_colocation_violation_count: int = Field(default=0, ge=0)
    route_violation_count: int = Field(default=0, ge=0)
    formation_violation_count: int = Field(default=0, ge=0)
    resource_violation_count: int = Field(default=0, ge=0)
    limit_violation_count: int = Field(default=0, ge=0)
    violating_frame_ids: tuple[int, ...] = ()


class BattleEvidenceChain(StrictModel):
    target_detection_event_id: str
    adversary_decision_id: str
    adversary_provider_call_id: str
    adversary_provider_model: str
    adversary_source_event_ids: tuple[str, ...]
    resulting_public_observation_ids: tuple[str, ...]
    blue_estimate_ids: tuple[str, ...]
    motion_effect_event_id: str
    blue_epoch_id: str | None
    blue_plan_version: int | None


class BlueTrackingEvidenceChain(StrictModel):
    """One entity/mission-bound blue tracking lifecycle chain."""

    target_id: str
    carrier_id: str
    candidate_id: str
    uuv_ids: tuple[str, ...] = Field(min_length=1)
    dispatch_event_id: str
    deployment_event_ids: tuple[str, ...] = Field(min_length=1)
    active_ping_event_id: str
    estimate_event_ids: tuple[str, ...] = Field(min_length=1)
    handoff_event_id: str
    resource_event_id: str
    recovery_request_event_id: str
    recovered_event_id: str
    carrier_return_event_id: str
    plan_version: int = Field(ge=1)


class PredictionIntentEvidenceChain(StrictModel):
    target_id: str
    diff_id: str
    previous_prediction_id: str
    current_prediction_id: str
    absolute_rms_m: float = Field(ge=0)
    normalized_rms: float = Field(ge=0)
    absolute_floor_m: float = Field(gt=0)
    normalized_threshold: float = Field(gt=0)
    overlap_start_s: float = Field(ge=0)
    overlap_end_s: float = Field(ge=0)
    suspicion_event_id: str
    suspicion_sim_time_s: int = Field(ge=0)
    intent_llm_call_ids: tuple[str, ...] = Field(min_length=2)
    intent_provider_models: tuple[str, ...] = Field(min_length=1)
    confirmed_event_id: str
    confirmation_sim_time_s: int = Field(ge=0)
    resulting_plan_id: str
    resulting_plan_revision: int = Field(ge=1)
    blue_response_event_ids: tuple[str, ...] = Field(min_length=1)
    response_latency_s: int = Field(ge=0)


class FullBattleAcceptance(StrictModel):
    completed: bool
    final_sim_time_s: int = Field(ge=0)
    final_plan_version: int = Field(ge=0)
    final_run_phase: str = "unknown"
    wall_clock_start_utc: str | None = None
    wall_clock_end_utc: str | None = None
    first_plan_wall_s: float | None = None
    required_stage_ids: frozenset[str] = frozenset()
    stage_sim_times_s: dict[str, int] = Field(default_factory=dict)
    stage_plan_versions: dict[str, int] = Field(default_factory=dict)
    battle_evidence_chains: tuple[BattleEvidenceChain, ...] = ()
    blue_tracking_chains: tuple[BlueTrackingEvidenceChain, ...] = ()
    prediction_intent_chains: tuple[PredictionIntentEvidenceChain, ...] = ()
    motion_audits: tuple[EntityMotionAudit, ...] = ()
    motion_limits: dict[str, EntityMotionLimits] = Field(default_factory=dict)
    observed_physics_frame_count: int = Field(default=0, ge=0)
    expected_physics_frame_count: int | None = Field(default=None, ge=0)
    physics_frame_coverage: dict[str, object] = Field(default_factory=dict)
    browser_error_count: int = Field(default=0, ge=0)
    browser_error_details: tuple[str, ...] = ()
    failed_request_count: int = Field(default=0, ge=0)
    memory_event_count: int = Field(default=0, ge=0)
    api_p95_ms: float = Field(default=0.0, ge=0)
    output_bytes: int = Field(default=0, ge=0)
    shutdown_s: float = Field(default=0.0, ge=0)
    git_commit: str | None = None
    config_sha256: str | None = None
    screenshot_paths: tuple[str, ...] = ()
    violations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def completed_requires_zero_violations(self) -> "FullBattleAcceptance":
        if self.completed and self.violations:
            raise ValueError("completed acceptance cannot contain violations")
        return self


class _MotionSample(StrictModel):
    entity_id: str
    entity_kind: EntityKind
    frame_id: int = Field(ge=0)
    sim_time_s: float = Field(ge=0)
    position_xy: tuple[float, float]
    speed_mps: float = Field(ge=0)
    heading_rad: float
    depth_m: float | None = Field(default=None, ge=0)
    vertical_speed_mps: float | None = None
    lifecycle_state: str | None = None
    owner_id: str | None = None
    route_deviation_m: float | None = Field(default=None, ge=0)
    route_tolerance_m: float | None = Field(default=None, ge=0)
    formation_error_m: float | None = Field(default=None, ge=0)
    formation_tolerance_m: float | None = Field(default=None, ge=0)
    mileage_m: float | None = Field(default=None, ge=0)
    max_mileage_m: float | None = Field(default=None, gt=0)
    energy_fraction: float | None = Field(default=None, ge=0, le=1)
    min_energy_fraction: float | None = Field(default=None, ge=0, le=1)


class _AuditState:
    def __init__(self, entity_id: str, kind: EntityKind, limits: EntityMotionLimits) -> None:
        self.entity_id = entity_id
        self.kind = kind
        self.limits = limits
        self.observed_steps = 0
        self.max_speed_mps = 0.0
        self.max_acceleration_mps2 = 0.0
        self.max_deceleration_mps2 = 0.0
        self.max_turn_rate_rad_s = 0.0
        self.min_depth_m: float | None = None
        self.max_depth_m: float | None = None
        self.max_vertical_speed_mps: float | None = None
        self.max_vertical_acceleration_mps2: float | None = None
        self.max_pitch_rad: float | None = None
        self.teleport_count = 0
        self.boundary_violation_count = 0
        self.owner_colocation_violation_count = 0
        self.route_violation_count = 0
        self.formation_violation_count = 0
        self.resource_violation_count = 0
        self.limit_violation_count = 0
        self.violating_frame_ids: set[int] = set()
        self.previous: _MotionSample | None = None

    def result(self) -> EntityMotionAudit:
        return EntityMotionAudit(
            entity_id=self.entity_id,
            entity_kind=self.kind,
            observed_steps=self.observed_steps,
            max_speed_mps=self.max_speed_mps,
            max_acceleration_mps2=self.max_acceleration_mps2,
            max_deceleration_mps2=self.max_deceleration_mps2,
            max_turn_rate_rad_s=self.max_turn_rate_rad_s,
            min_depth_m=self.min_depth_m,
            max_depth_m=self.max_depth_m,
            max_vertical_speed_mps=self.max_vertical_speed_mps,
            max_vertical_acceleration_mps2=self.max_vertical_acceleration_mps2,
            max_pitch_rad=self.max_pitch_rad,
            teleport_count=self.teleport_count,
            boundary_violation_count=self.boundary_violation_count,
            owner_colocation_violation_count=self.owner_colocation_violation_count,
            route_violation_count=self.route_violation_count,
            formation_violation_count=self.formation_violation_count,
            resource_violation_count=self.resource_violation_count,
            limit_violation_count=self.limit_violation_count,
            violating_frame_ids=tuple(sorted(self.violating_frame_ids)),
        )


class PhysicsInvariantMonitor:
    """Accumulate physical invariants over a sequence of internal snapshots."""

    def __init__(
        self,
        limits_by_entity: Mapping[str, EntityMotionLimits] | None = None,
        *,
        bounds_by_entity: Mapping[str, tuple[float, float, float, float]] | None = None,
        tolerance: float = 1e-6,
        launch_tolerance_m: float = 250.0,
    ) -> None:
        if tolerance < 0.0 or launch_tolerance_m < 0.0:
            raise ValueError("monitor tolerances must be non-negative")
        self._states: dict[str, _AuditState] = {}
        self._expected_entity_ids: set[str] = set()
        self._bounds = dict(bounds_by_entity or {})
        self._tolerance = tolerance
        self._launch_tolerance_m = launch_tolerance_m
        self._observed_entity_ids: set[str] = set()
        self._frame_entity_ids: dict[int, set[str]] = {}
        self._frame_order: list[tuple[int, float]] = []
        self._seen_frame_ids: set[int] = set()
        self._duplicate_frame_ids: set[int] = set()
        self._duplicate_entity_frame_ids: set[tuple[str, int]] = set()
        self._frame_id_gaps: set[int] = set()
        self._nonmonotonic_frame_ids: set[int] = set()
        self._nonmonotonic_sim_time_frame_ids: set[int] = set()
        self._inconsistent_sample_frame_ids: set[int] = set()
        for entity_id, limits in (limits_by_entity or {}).items():
            self.register_entity(entity_id, _kind_for_id(entity_id), limits)

    def register_entity(
        self,
        entity_id: str,
        entity_kind: EntityKind,
        limits: EntityMotionLimits,
    ) -> None:
        if not entity_id:
            raise ValueError("entity_id must be non-empty")
        self._states[entity_id] = _AuditState(entity_id, entity_kind, limits)
        self._expected_entity_ids.add(entity_id)

    def observe(self, frame: object, *, events: Sequence[object] = ()) -> None:
        frame_id, sim_time_s = _extract_frame_metadata(frame)
        if frame_id in self._seen_frame_ids:
            self._duplicate_frame_ids.add(frame_id)
        self._seen_frame_ids.add(frame_id)
        if self._frame_order:
            previous_frame_id, previous_sim_time_s = self._frame_order[-1]
            if frame_id <= previous_frame_id:
                self._nonmonotonic_frame_ids.add(frame_id)
            if sim_time_s < previous_sim_time_s - self._tolerance:
                self._nonmonotonic_sim_time_frame_ids.add(frame_id)
        self._frame_order.append((frame_id, sim_time_s))
        self._frame_entity_ids.setdefault(frame_id, set())
        samples = _extract_samples(frame)
        samples_by_id = {sample.entity_id: sample for sample in samples}
        event_values = tuple(events)
        for sample in samples:
            if sample.frame_id != frame_id:
                self._inconsistent_sample_frame_ids.add(frame_id)
            frame_entities = self._frame_entity_ids.setdefault(sample.frame_id, set())
            if sample.entity_id in frame_entities:
                self._duplicate_entity_frame_ids.add((sample.entity_id, sample.frame_id))
            frame_entities.add(sample.entity_id)
            self._observed_entity_ids.add(sample.entity_id)
            state = self._states.get(sample.entity_id)
            if state is None:
                state = _AuditState(
                    sample.entity_id,
                    sample.entity_kind,
                    _default_limits(sample.entity_kind),
                )
                self._states[sample.entity_id] = state
            self._observe_cross_entity_constraints(state, sample, samples_by_id)
            self._observe_sample(state, sample, event_values)

    def _observe_cross_entity_constraints(
        self,
        state: _AuditState,
        current: _MotionSample,
        samples_by_id: Mapping[str, _MotionSample],
    ) -> None:
        if (
            current.entity_kind != "uuv"
            or current.lifecycle_state != "onboard"
            or not current.owner_id
        ):
            return
        owner = samples_by_id.get(current.owner_id)
        if owner is None:
            state.owner_colocation_violation_count += 1
            self._violation(state, current.frame_id, "owner-missing")
            return
        position_error_m = hypot(
            current.position_xy[0] - owner.position_xy[0],
            current.position_xy[1] - owner.position_xy[1],
        )
        heading_error_rad = abs(wrap_angle(current.heading_rad - owner.heading_rad))
        if position_error_m > self._tolerance or heading_error_rad > self._tolerance:
            state.owner_colocation_violation_count += 1
            self._violation(state, current.frame_id, "owner-colocation")

    def result(self, entity_id: str) -> EntityMotionAudit:
        state = self._states.get(entity_id)
        if state is None:
            raise KeyError(entity_id)
        return state.result()

    def results(self) -> tuple[EntityMotionAudit, ...]:
        return tuple(self._states[key].result() for key in sorted(self._states))

    def limits(self) -> dict[str, EntityMotionLimits]:
        """Return configured limits without exposing trajectory coordinates."""
        return {
            entity_id: state.limits.model_copy(deep=True)
            for entity_id, state in sorted(self._states.items())
        }

    def coverage(self, *, physics_step_s: int | None = None) -> dict[str, object]:
        """Return frame/entity coverage metadata without trajectory truth."""
        frame_ids = sorted(self._frame_entity_ids)
        if frame_ids:
            self._frame_id_gaps = set(
                range(frame_ids[0], frame_ids[-1] + 1)
            ) - set(frame_ids)
        missing_entity_frame_ids = {
            entity_id: tuple(
                frame_id
                for frame_id in frame_ids
                if entity_id not in self._frame_entity_ids[frame_id]
            )
            for entity_id in sorted(self._expected_entity_ids)
        }
        missing_entity_frame_ids = {
            entity_id: frame_ids
            for entity_id, frame_ids in missing_entity_frame_ids.items()
            if frame_ids
        }
        expected_frame_count = None
        if physics_step_s is not None and physics_step_s > 0 and frame_ids:
            expected_frame_count = frame_ids[-1] - frame_ids[0] + 1
        return {
            "expected_entity_ids": tuple(sorted(self._expected_entity_ids)),
            "expected_entity_count": len(self._expected_entity_ids),
            "observed_entity_ids": tuple(sorted(self._observed_entity_ids)),
            "observed_entity_count": len(self._observed_entity_ids),
            "observed_frame_count": len(frame_ids),
            "observed_frame_observation_count": len(self._frame_order),
            "first_frame_id": frame_ids[0] if frame_ids else None,
            "last_frame_id": frame_ids[-1] if frame_ids else None,
            "duplicate_frame_ids": tuple(sorted(self._duplicate_frame_ids)),
            "duplicate_entity_frame_ids": tuple(
                f"{entity_id}@{frame_id}"
                for entity_id, frame_id in sorted(self._duplicate_entity_frame_ids)
            ),
            "missing_entity_frame_ids": missing_entity_frame_ids,
            "frame_id_gaps": tuple(sorted(self._frame_id_gaps)),
            "nonmonotonic_frame_ids": tuple(sorted(self._nonmonotonic_frame_ids)),
            "nonmonotonic_sim_time_frame_ids": tuple(
                sorted(self._nonmonotonic_sim_time_frame_ids)
            ),
            "inconsistent_sample_frame_ids": tuple(
                sorted(self._inconsistent_sample_frame_ids)
            ),
            "physics_step_s": physics_step_s,
            "sequence_expected_frame_count": expected_frame_count,
        }

    def _observe_sample(
        self,
        state: _AuditState,
        current: _MotionSample,
        events: Sequence[object],
    ) -> None:
        limits = state.limits
        if (
            current.route_deviation_m is not None
            and current.route_tolerance_m is not None
            and current.route_deviation_m
            > current.route_tolerance_m + self._tolerance
        ):
            state.route_violation_count += 1
            self._violation(state, current.frame_id, "route")
        if (
            current.formation_error_m is not None
            and current.formation_tolerance_m is not None
            and current.formation_error_m
            > current.formation_tolerance_m + self._tolerance
        ):
            state.formation_violation_count += 1
            self._violation(state, current.frame_id, "formation")
        if (
            current.mileage_m is not None
            and current.max_mileage_m is not None
            and current.mileage_m > current.max_mileage_m + self._tolerance
        ):
            state.resource_violation_count += 1
            self._violation(state, current.frame_id, "mileage")
        if (
            current.energy_fraction is not None
            and current.min_energy_fraction is not None
            and current.energy_fraction
            < current.min_energy_fraction - self._tolerance
        ):
            state.resource_violation_count += 1
            self._violation(state, current.frame_id, "energy")
        state.max_speed_mps = max(state.max_speed_mps, current.speed_mps)
        if current.depth_m is not None:
            state.min_depth_m = (
                current.depth_m
                if state.min_depth_m is None
                else min(state.min_depth_m, current.depth_m)
            )
            state.max_depth_m = (
                current.depth_m
                if state.max_depth_m is None
                else max(state.max_depth_m, current.depth_m)
            )
            if (
                limits.min_depth_m is not None
                and current.depth_m < limits.min_depth_m - self._tolerance
            ) or (
                limits.max_depth_m is not None
                and current.depth_m > limits.max_depth_m + self._tolerance
            ):
                self._violation(state, current.frame_id, "depth")
        if limits.max_speed_mps is not None and current.speed_mps > limits.max_speed_mps + self._tolerance:
            self._violation(state, current.frame_id, "speed")
        if (
            current.vertical_speed_mps is not None
            and limits.max_vertical_speed_mps is not None
            and abs(current.vertical_speed_mps)
            > limits.max_vertical_speed_mps + self._tolerance
        ):
            self._violation(state, current.frame_id, "vertical-speed")
        bounds = self._bounds.get(current.entity_id)
        if bounds is not None:
            min_x, max_x, min_y, max_y = bounds
            x, y = current.position_xy
            if not (min_x - self._tolerance <= x <= max_x + self._tolerance) or not (
                min_y - self._tolerance <= y <= max_y + self._tolerance
            ):
                state.boundary_violation_count += 1
                self._violation(state, current.frame_id, "boundary")
        previous = state.previous
        if previous is None:
            state.previous = current
            return
        dt_s = current.sim_time_s - previous.sim_time_s
        if dt_s <= 0.0:
            state.previous = current
            return
        state.observed_steps += 1
        displacement_m = hypot(
            current.position_xy[0] - previous.position_xy[0],
            current.position_xy[1] - previous.position_xy[1],
        )
        observed_speed = displacement_m / dt_s
        transition = _transition_event(current.entity_id, events)
        onboard_transition = (
            state.kind == "uuv"
            and previous.lifecycle_state == "onboard"
            and current.lifecycle_state == "onboard"
        )
        lifecycle_handoff = (
            state.kind == "uuv"
            and transition
            and previous.lifecycle_state != current.lifecycle_state
        )
        if (
            observed_speed > limits.max_speed_mps + self._tolerance
            and not transition
            and not onboard_transition
        ):
            self._violation(state, current.frame_id, "displacement-speed")
        allowed_jump = max(
            self._launch_tolerance_m if transition else 0.0,
            limits.max_speed_mps * dt_s + self._tolerance,
        )
        if displacement_m > allowed_jump and not onboard_transition:
            state.teleport_count += 1
            self._violation(state, current.frame_id, "teleport")
        if onboard_transition or lifecycle_handoff:
            # The carrier owns the UUV's position, heading, and speed while
            # it is onboard.  A deployment/recovery event is the explicit
            # handoff boundary; do not attribute that parent-to-UUV transfer
            # derivative to either side.  Direct limits remain audited above,
            # and the next fully autonomous step is audited normally.
            state.previous = current
            return
        acceleration = (current.speed_mps - previous.speed_mps) / dt_s
        state.max_acceleration_mps2 = max(state.max_acceleration_mps2, max(0.0, acceleration))
        state.max_deceleration_mps2 = max(state.max_deceleration_mps2, max(0.0, -acceleration))
        state.max_turn_rate_rad_s = max(
            state.max_turn_rate_rad_s,
            abs(wrap_angle(current.heading_rad - previous.heading_rad)) / dt_s,
        )
        if state.max_acceleration_mps2 > limits.max_acceleration_mps2 + self._tolerance:
            self._violation(state, current.frame_id, "acceleration")
        if state.max_deceleration_mps2 > limits.max_deceleration_mps2 + self._tolerance:
            self._violation(state, current.frame_id, "deceleration")
        if state.max_turn_rate_rad_s > limits.max_turn_rate_rad_s + self._tolerance:
            self._violation(state, current.frame_id, "turn-rate")
        if current.depth_m is not None and previous.depth_m is not None:
            vertical_speed = (
                current.depth_m - previous.depth_m
            ) / dt_s
            state.max_vertical_speed_mps = max(
                state.max_vertical_speed_mps or 0.0,
                abs(vertical_speed),
            )
            if current.vertical_speed_mps is not None and previous.vertical_speed_mps is not None:
                vertical_acceleration = abs(
                    current.vertical_speed_mps - previous.vertical_speed_mps
                ) / dt_s
                state.max_vertical_acceleration_mps2 = max(
                    state.max_vertical_acceleration_mps2 or 0.0,
                    vertical_acceleration,
                )
            else:
                vertical_acceleration = 0.0
            state.max_pitch_rad = max(
                state.max_pitch_rad or 0.0,
                abs(atan2(vertical_speed, max(current.speed_mps, 1e-9))),
            )
            if (
                limits.max_vertical_speed_mps is not None
                and (state.max_vertical_speed_mps or 0.0)
                > limits.max_vertical_speed_mps + self._tolerance
            ):
                self._violation(state, current.frame_id, "vertical-speed")
            if (
                limits.max_vertical_acceleration_mps2 is not None
                and (state.max_vertical_acceleration_mps2 or 0.0)
                > limits.max_vertical_acceleration_mps2 + self._tolerance
            ):
                self._violation(state, current.frame_id, "vertical-acceleration")
            if (
                limits.max_pitch_rad is not None
                and (state.max_pitch_rad or 0.0) > limits.max_pitch_rad + self._tolerance
            ):
                self._violation(state, current.frame_id, "pitch")
        state.previous = current

    def _violation(self, state: _AuditState, frame_id: int, reason: str) -> None:
        state.limit_violation_count += 1
        state.violating_frame_ids.add(frame_id)


def _extract_samples(frame: object) -> tuple[_MotionSample, ...]:
    if isinstance(frame, Mapping):
        raw = frame.get("entities") or frame.get("entity_states") or frame.get("truth_entities")
        if raw is None:
            raw = frame.get("samples")
        frame_id = int(frame.get("frame_id", 0))
        sim_time_s = float(frame.get("sim_time_s", 0))
    elif isinstance(frame, Sequence) and not isinstance(frame, (str, bytes)):
        raw = frame
        frame_id = 0
        sim_time_s = 0.0
    else:
        raw = getattr(frame, "entities", None) or getattr(frame, "entity_states", None)
        frame_id = int(getattr(frame, "frame_id", 0))
        sim_time_s = float(getattr(frame, "sim_time_s", 0))
    if raw is None:
        return ()
    if isinstance(raw, Mapping):
        values = []
        for entity_id, value in raw.items():
            if isinstance(value, Mapping):
                values.append({"entity_id": entity_id, **value})
            else:
                values.append({"entity_id": entity_id, **_object_mapping(value)})
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = list(raw)
    else:
        return ()
    result: list[_MotionSample] = []
    for value in values:
        payload = dict(value) if isinstance(value, Mapping) else _object_mapping(value)
        payload.setdefault("frame_id", frame_id)
        payload.setdefault("sim_time_s", sim_time_s)
        payload.setdefault("entity_kind", _kind_for_id(str(payload.get("entity_id", ""))))
        if payload.get("lifecycle_state") is None and "deployment_state" in payload:
            payload["lifecycle_state"] = payload["deployment_state"]
        payload.pop("deployment_state", None)
        if "position_xy" not in payload:
            payload["position_xy"] = (
                payload.get("x", 0.0),
                payload.get("y", 0.0),
            )
        result.append(_MotionSample.model_validate(payload))
    return tuple(result)


def _extract_frame_metadata(frame: object) -> tuple[int, float]:
    if isinstance(frame, Mapping):
        return int(frame.get("frame_id", 0)), float(frame.get("sim_time_s", 0))
    if isinstance(frame, Sequence) and not isinstance(frame, (str, bytes)):
        return 0, 0.0
    return (
        int(getattr(frame, "frame_id", 0)),
        float(getattr(frame, "sim_time_s", 0)),
    )


def _object_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {
        name: getattr(value, name)
        for name in (
            "entity_id",
            "entity_kind",
            "frame_id",
            "sim_time_s",
            "position_xy",
            "speed_mps",
            "heading_rad",
            "depth_m",
            "vertical_speed_mps",
            "lifecycle_state",
            "deployment_state",
            "owner_id",
            "route_deviation_m",
            "route_tolerance_m",
            "formation_error_m",
            "formation_tolerance_m",
            "mileage_m",
            "max_mileage_m",
            "energy_fraction",
            "min_energy_fraction",
        )
        if hasattr(value, name)
    }


def _transition_event(entity_id: str, events: Sequence[object]) -> bool:
    for event in events:
        event_type = str(
            event.get("event_type", "")
            if isinstance(event, Mapping)
            else getattr(event, "event_type", "")
        )
        payload = (
            event.get("payload", {})
            if isinstance(event, Mapping)
            else getattr(event, "payload", {})
        )
        event_entity_id = str(
            event.get("entity_id", "")
            if isinstance(event, Mapping)
            else getattr(event, "entity_id", "")
        )
        if (
            (entity_id in event_type or event_entity_id == entity_id)
            and event_type in {
            "uuv_deployed",
            "uuv_recovered",
            "deployment_completed",
            "recovery_completed",
            }
        ):
            return True
        if isinstance(payload, Mapping) and str(payload.get("uuv_id")) == entity_id:
            if any(token in event_type for token in ("deploy", "recover", "rendezvous")):
                return True
    return False


def _kind_for_id(entity_id: str) -> EntityKind:
    if entity_id.startswith("target") or entity_id.startswith("submarine"):
        return "submarine"
    if entity_id.startswith("uuv"):
        return "uuv"
    if entity_id in {"carrier_01", "carrier"}:
        return "carrier"
    return "mother_ship"


def _default_limits(kind: EntityKind) -> EntityMotionLimits:
    if kind == "uuv":
        return EntityMotionLimits(
            max_speed_mps=4.0,
            max_acceleration_mps2=0.1,
            max_deceleration_mps2=0.1,
            max_turn_rate_rad_s=0.05235987755982988,
        )
    if kind == "submarine":
        return EntityMotionLimits(
            max_speed_mps=14.0,
            max_acceleration_mps2=0.08,
            max_deceleration_mps2=0.1,
            max_turn_rate_rad_s=0.010471975511965976,
            min_depth_m=0.0,
            max_depth_m=900.0,
            max_vertical_speed_mps=2.0,
            max_vertical_acceleration_mps2=0.2,
            max_pitch_rad=0.2617993877991494,
        )
    return EntityMotionLimits(
        max_speed_mps=8.0,
        max_acceleration_mps2=0.25,
        max_deceleration_mps2=0.25,
        max_turn_rate_rad_s=0.25,
    )


__all__ = [
    "BattleEvidenceChain",
    "EntityMotionAudit",
    "EntityMotionLimits",
    "FullBattleAcceptance",
    "PredictionIntentEvidenceChain",
    "PhysicsInvariantMonitor",
]
