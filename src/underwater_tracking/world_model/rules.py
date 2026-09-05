"""Deterministic future-event rules over IMM and B-spline products.

This module is intentionally small and inspectable.  It does not learn,
sample, call an LLM, or issue a waypoint.  It projects the public UUV state
against each B-spline sample, calculates a bearings-only geometry score, and
emits explainable event hypotheses grouped into the configured H1-H4 windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, hypot, pi, sqrt

from underwater_tracking.domain.models import EventLevel
from underwater_tracking.world_model.config import DEFAULT_WORLD_MODEL_CONFIG
from underwater_tracking.world_model.models import (
    DataStatus,
    EventType,
    HorizonCoverage,
    HorizonName,
    HorizonSpec,
    PredictedEvent,
    RuleEvidence,
    RuleWorldModelConfig,
    RuleWorldModelInput,
    UuvForecastInput,
    WorldModelForecast,
    ForecastProvenance,
)


@dataclass(frozen=True, slots=True)
class _FutureSample:
    time_s: float
    offset_s: float
    duration_s: float
    point_xy: tuple[float, float]
    corridor_radius_m: float
    speed_mps: float
    heading_rad: float
    heading_change_rad: float
    active_uuv_count: int
    low_energy_uuv_count: int
    geometry_od: float


class RuleEventPredictor:
    """Pure, truth-safe event predictor for the showcase world model."""

    def __init__(self, config: RuleWorldModelConfig = DEFAULT_WORLD_MODEL_CONFIG) -> None:
        self._config = config

    def predict(self, inputs: RuleWorldModelInput) -> WorldModelForecast:
        """Predict at most one earliest event per rule for one target."""

        provenance = {name: getattr(inputs, name) for name in ForecastProvenance.model_fields}
        provenance["source_prediction_id"] = inputs.trajectory.prediction_id
        reasons = list(inputs.source_reason_codes)
        status = inputs.source_status
        if (inputs.source_track_revision is None or inputs.prediction_revision is None
                or inputs.last_observed_at_s is None or inputs.generated_at_s is None
                or inputs.valid_until_s is None or not inputs.source_observation_ids):
            status = "unavailable"
            reasons.append("world_model_provenance_missing")
        elif inputs.generated_at_s > inputs.as_of_s or inputs.last_observed_at_s > inputs.as_of_s:
            status = "unavailable"
            reasons.append("world_model_source_time_invalid")
        elif inputs.as_of_s >= inputs.valid_until_s:
            status = "expired"
            reasons.append("world_model_source_expired")
        if status in {"expired", "unavailable"}:
            return WorldModelForecast(
                **provenance, scenario_id=inputs.scenario_id, target_id=inputs.target_id,
                as_of_s=inputs.as_of_s, source_observation_ids=inputs.source_observation_ids,
                data_status=DataStatus(status), trajectory_fallback_used=inputs.trajectory.fallback_used,
                imm_model_probabilities=inputs.belief.model_probabilities,
                horizons=tuple(HorizonCoverage(name=h.name, start_offset_s=h.start_offset_s,
                    end_offset_s=h.end_offset_s, sample_count=0, covered=False) for h in self._config.horizons),
                events=(), warnings=tuple(dict.fromkeys(reasons)),
            )
        samples = _build_samples(inputs, self._config)
        candidates = (
            self._turn_event(inputs, samples),
            self._sprint_event(inputs, samples),
            self._area_exit_event(inputs, samples),
            self._geometry_event(inputs, samples),
            self._coverage_gap_event(inputs, samples),
            self._track_loss_event(inputs, samples),
            self._decoy_event(inputs, samples),
            self._stop_event(inputs, samples),
        )
        events = tuple(
            sorted(
                (event for event in candidates if event is not None),
                key=lambda event: (
                    event.predicted_time_s,
                    event.event_type.value,
                    event.event_id,
                ),
            )
        )
        warnings: list[str] = list(inputs.source_reason_codes)
        if inputs.trajectory.fallback_used:
            warnings.append(
                inputs.trajectory.fallback_reason
                or "trajectory uses a short-history fallback instead of a B-spline fit"
            )
        if inputs.map_bounds_xy is None and inputs.task_region_bounds_xy is None:
            warnings.append("map bounds unavailable; area-exit prediction is disabled")
        if not inputs.uuvs:
            warnings.append(
                "UUV operational state unavailable; geometry and coverage rules are disabled"
            )
        elif any(not uuv.planned_times_s for uuv in inputs.uuvs):
            warnings.append(
                "one or more planned UUV tracks are unavailable; "
                "constant-velocity UUV projection is used"
            )
        requested_end_s = self._config.horizons[-1].end_offset_s
        available_end_s = inputs.trajectory.times_s[-1] - inputs.as_of_s
        if available_end_s + 1.0e-9 < requested_end_s:
            warnings.append(
                f"trajectory covers {available_end_s:.1f} s of the requested "
                f"{requested_end_s:.1f} s horizon"
            )
        horizons = tuple(
            HorizonCoverage(
                name=spec.name,
                start_offset_s=spec.start_offset_s,
                end_offset_s=spec.end_offset_s,
                sample_count=sum(
                    1 for sample in samples if _in_horizon(sample.offset_s, spec, self._config)
                ),
                covered=any(
                    _in_horizon(sample.offset_s, spec, self._config) for sample in samples
                ),
            )
            for spec in self._config.horizons
        )
        degraded = (
            status == "degraded" or inputs.trajectory.fallback_used
            or (inputs.map_bounds_xy is None and inputs.task_region_bounds_xy is None)
            or not inputs.uuvs
            or any(not uuv.planned_times_s for uuv in inputs.uuvs)
        )
        return WorldModelForecast(
            **provenance,
            scenario_id=inputs.scenario_id,
            target_id=inputs.target_id,
            as_of_s=inputs.as_of_s,
            source_observation_ids=inputs.source_observation_ids,
            source_observability_event_ids=inputs.source_observability_event_ids,
            data_status=DataStatus.DEGRADED if degraded else DataStatus.READY,
            trajectory_fallback_used=inputs.trajectory.fallback_used,
            imm_model_probabilities=inputs.belief.model_probabilities,
            horizons=horizons,
            events=events,
            warnings=tuple(warnings),
        )

    def _turn_event(
        self,
        inputs: RuleWorldModelInput,
        samples: tuple[_FutureSample, ...],
    ) -> PredictedEvent | None:
        thresholds = self._config.thresholds
        for sample in samples:
            magnitude = abs(sample.heading_change_rad)
            if magnitude < thresholds.turn_heading_change_rad:
                continue
            is_left = sample.heading_change_rad > 0.0
            model_name = "left_turn" if is_left else "right_turn"
            model_probability = _model_probability(inputs, model_name)
            if model_probability < thresholds.turn_model_probability_min:
                continue
            confidence = _clamp01(
                0.50 * model_probability
                + 0.50
                * min(1.0, magnitude / thresholds.turn_heading_change_strong_rad)
            )
            if confidence < thresholds.event_min_confidence:
                continue
            event_type = (
                EventType.TARGET_TURN_LEFT if is_left else EventType.TARGET_TURN_RIGHT
            )
            direction = "左" if is_left else "右"
            return self._event(
                inputs,
                sample,
                event_type,
                confidence,
                EventLevel.TACTICAL,
                "R-TURN-001",
                f"预测目标将在该时间段向{direction}转向",
                (
                    RuleEvidence(
                        key="imm_turn_probability",
                        source="imm",
                        value=model_probability,
                        threshold=thresholds.turn_model_probability_min,
                        unit="1",
                        description=f"IMM 的{direction}转模型概率",
                    ),
                    RuleEvidence(
                        key="predicted_heading_change",
                        source="bspline",
                        value=magnitude,
                        threshold=thresholds.turn_heading_change_rad,
                        unit="rad",
                        description="B-spline 相邻预测段的航向变化",
                    ),
                ),
            )
        return None

    def _sprint_event(
        self,
        inputs: RuleWorldModelInput,
        samples: tuple[_FutureSample, ...],
    ) -> PredictedEvent | None:
        thresholds = self._config.thresholds
        imm_speed = hypot(*inputs.belief.velocity_xy_mps)
        for sample in samples:
            assessed_speed = max(imm_speed, sample.speed_mps)
            if assessed_speed < thresholds.sprint_speed_threshold_mps:
                continue
            span = (
                thresholds.sprint_reference_speed_mps
                - thresholds.sprint_speed_threshold_mps
            )
            confidence = 0.55 + 0.45 * _clamp01(
                (assessed_speed - thresholds.sprint_speed_threshold_mps) / span
            )
            return self._event(
                inputs,
                sample,
                EventType.HIGH_SPEED_ESCAPE,
                confidence,
                EventLevel.TACTICAL,
                "R-SPRINT-001",
                "预测目标将保持或进入高速脱离状态",
                (
                    RuleEvidence(
                        key="imm_speed",
                        source="imm",
                        value=imm_speed,
                        threshold=thresholds.sprint_speed_threshold_mps,
                        unit="m/s",
                        description="IMM 当前混合状态给出的目标速度",
                    ),
                    RuleEvidence(
                        key="predicted_segment_speed",
                        source="bspline",
                        value=sample.speed_mps,
                        threshold=thresholds.sprint_speed_threshold_mps,
                        unit="m/s",
                        description="B-spline 预测段换算得到的速度",
                    ),
                ),
            )
        return None

    def _area_exit_event(
        self,
        inputs: RuleWorldModelInput,
        samples: tuple[_FutureSample, ...],
    ) -> PredictedEvent | None:
        bounds = inputs.task_region_bounds_xy or inputs.map_bounds_xy
        if bounds is None:
            return None
        for sample in samples:
            boundary_distance = _signed_boundary_distance(
                sample.point_xy, bounds
            )
            center_outside = boundary_distance < 0.0
            corridor_touches = sample.corridor_radius_m >= max(boundary_distance, 0.0)
            if not center_outside and not corridor_touches:
                continue
            if center_outside:
                confidence = 0.95
            else:
                penetration = sample.corridor_radius_m - boundary_distance
                confidence = 0.60 + 0.30 * _clamp01(
                    penetration / max(sample.corridor_radius_m, 1.0)
                )
            return self._event(
                inputs,
                sample,
                EventType.AREA_EXIT_RISK,
                confidence,
                EventLevel.STRATEGIC,
                "R-EXIT-001",
                "预测轨迹或其不确定范围将触及任务区边界",
                (
                    RuleEvidence(
                        key="distance_to_boundary",
                        source="task_region" if inputs.task_region_bounds_xy is not None else "map_bounds",
                        value=boundary_distance,
                        threshold=0.0,
                        unit="m",
                        description="预测中心点到最近任务区边界的有符号距离",
                    ),
                    RuleEvidence(
                        key="uncertainty_corridor_radius",
                        source="bspline",
                        value=sample.corridor_radius_m,
                        threshold=max(boundary_distance, 0.0),
                        unit="m",
                        description="B-spline 预测不确定走廊半径",
                    ),
                ),
            )
        return None

    def _geometry_event(
        self,
        inputs: RuleWorldModelInput,
        samples: tuple[_FutureSample, ...],
    ) -> PredictedEvent | None:
        if not inputs.uuvs:
            return None
        thresholds = self._config.thresholds
        for sample in samples:
            if sample.active_uuv_count < thresholds.min_tracking_uuvs:
                continue
            if sample.geometry_od >= thresholds.geometry_warning_od:
                continue
            confidence = 0.55 + 0.40 * _clamp01(
                1.0 - sample.geometry_od / thresholds.geometry_warning_od
            )
            return self._event(
                inputs,
                sample,
                EventType.GEOMETRY_DEGRADATION,
                confidence,
                EventLevel.TACTICAL,
                "R-GEOMETRY-001",
                "预测多 UUV 测向夹角将变差，定位能力下降",
                (
                    RuleEvidence(
                        key="future_geometry_od",
                        source="uuv_projection",
                        value=sample.geometry_od,
                        threshold=thresholds.geometry_warning_od,
                        unit="1",
                        description="预测位置上的方位观测几何指标",
                    ),
                    RuleEvidence(
                        key="future_active_uuv_count",
                        source="uuv_projection",
                        value=float(sample.active_uuv_count),
                        threshold=float(thresholds.min_tracking_uuvs),
                        unit="count",
                        description="预计仍能覆盖目标的 UUV 数量",
                    ),
                ),
            )
        return None

    def _coverage_gap_event(
        self,
        inputs: RuleWorldModelInput,
        samples: tuple[_FutureSample, ...],
    ) -> PredictedEvent | None:
        if not inputs.uuvs:
            return None
        thresholds = self._config.thresholds
        for sample in samples:
            if sample.active_uuv_count >= thresholds.min_tracking_uuvs:
                continue
            deficit = thresholds.min_tracking_uuvs - sample.active_uuv_count
            confidence = 0.65 + 0.25 * _clamp01(
                deficit / thresholds.min_tracking_uuvs
            )
            if sample.low_energy_uuv_count:
                confidence = min(1.0, confidence + 0.10)
            return self._event(
                inputs,
                sample,
                EventType.UUV_COVERAGE_GAP,
                confidence,
                EventLevel.TACTICAL,
                "R-COVERAGE-001",
                "预测可用 UUV 数量不足，编队将出现覆盖缺口",
                (
                    RuleEvidence(
                        key="future_active_uuv_count",
                        source="uuv_projection",
                        value=float(sample.active_uuv_count),
                        threshold=float(thresholds.min_tracking_uuvs),
                        unit="count",
                        description="预计仍健康、在线且处于声呐范围内的 UUV 数量",
                    ),
                    RuleEvidence(
                        key="low_energy_uuv_count",
                        source="uuv_projection",
                        value=float(sample.low_energy_uuv_count),
                        threshold=0.0,
                        unit="count",
                        description="低于展示规则电量门限的 UUV 数量",
                    ),
                ),
            )
        return None

    def _track_loss_event(
        self,
        inputs: RuleWorldModelInput,
        samples: tuple[_FutureSample, ...],
    ) -> PredictedEvent | None:
        if not inputs.uuvs:
            return None
        thresholds = self._config.thresholds
        for sample in samples:
            coverage_risk = _clamp01(
                (thresholds.min_tracking_uuvs - sample.active_uuv_count)
                / thresholds.min_tracking_uuvs
            )
            geometry_risk = _clamp01(
                1.0 - sample.geometry_od / thresholds.geometry_warning_od
            )
            corridor_risk = _clamp01(
                sample.corridor_radius_m / thresholds.track_loss_corridor_m
            )
            quality_risk = _clamp01(
                (thresholds.track_loss_quality_threshold - inputs.tracking.quality_ewma)
                / max(thresholds.track_loss_quality_threshold, 1.0e-9)
            )
            confidence = (
                0.45 * coverage_risk
                + 0.20 * geometry_risk
                + 0.20 * corridor_risk
                + 0.15 * quality_risk
            )
            hard_condition = (
                sample.active_uuv_count == 0
                or sample.geometry_od <= thresholds.geometry_critical_od
                or sample.corridor_radius_m >= thresholds.track_loss_corridor_m
            )
            if not hard_condition or confidence < thresholds.event_min_confidence:
                continue
            return self._event(
                inputs,
                sample,
                EventType.TRACK_LOSS_RISK,
                confidence,
                EventLevel.STRATEGIC,
                "R-TRACK-LOSS-001",
                "预测目标航迹可能失去可靠观测",
                (
                    RuleEvidence(
                        key="coverage_risk",
                        source="uuv_projection",
                        value=coverage_risk,
                        threshold=1.0,
                        unit="1",
                        description="未来 UUV 覆盖缺口的归一化风险",
                    ),
                    RuleEvidence(
                        key="geometry_risk",
                        source="uuv_projection",
                        value=geometry_risk,
                        threshold=1.0,
                        unit="1",
                        description="未来测向几何退化的归一化风险",
                    ),
                    RuleEvidence(
                        key="corridor_risk",
                        source="bspline",
                        value=corridor_risk,
                        threshold=1.0,
                        unit="1",
                        description="未来轨迹不确定走廊扩张风险",
                    ),
                    RuleEvidence(
                        key="quality_risk",
                        source="tracking_context",
                        value=quality_risk,
                        threshold=1.0,
                        unit="1",
                        description="当前跟踪质量不足带来的附加风险",
                    ),
                ),
            )
        return None

    def _decoy_event(
        self,
        inputs: RuleWorldModelInput,
        samples: tuple[_FutureSample, ...],
    ) -> PredictedEvent | None:
        context = inputs.tracking
        thresholds = self._config.thresholds
        confidence = 0.0
        evidence: list[RuleEvidence] = []
        observability_confidence = context.observability_hypotheses.get(
            "DECOY_OR_NEW_TARGET"
        )
        if observability_confidence is not None:
            confidence = max(confidence, float(observability_confidence))
            evidence.append(
                RuleEvidence(
                    key="observability_decoy_or_new_target",
                    source="observability",
                    value=float(observability_confidence),
                    threshold=thresholds.event_min_confidence,
                    unit="1",
                    description="可观测性反馈已形成诱饵或新目标证据假设",
                )
            )
        if (
            context.previous_contact_count is not None
            and context.current_contact_count > context.previous_contact_count
        ):
            increase = context.current_contact_count - context.previous_contact_count
            confidence += 0.60
            evidence.append(
                RuleEvidence(
                    key="contact_count_increase",
                    source="tracking_context",
                    value=float(increase),
                    threshold=1.0,
                    unit="count",
                    description="候选接触数量增加，只能形成诱饵或新目标假设",
                )
            )
        if (
            context.association_confidence is not None
            and context.previous_association_confidence is not None
        ):
            drop = context.previous_association_confidence - context.association_confidence
            if drop >= thresholds.association_confidence_drop:
                confidence += 0.25
                evidence.append(
                    RuleEvidence(
                        key="association_confidence_drop",
                        source="tracking_context",
                        value=drop,
                        threshold=thresholds.association_confidence_drop,
                        unit="1",
                        description="航迹关联置信度下降",
                    )
                )
        if (
            context.association_entropy is not None
            and context.previous_association_entropy is not None
        ):
            rise = context.association_entropy - context.previous_association_entropy
            if rise >= thresholds.association_entropy_rise:
                confidence += 0.15
                evidence.append(
                    RuleEvidence(
                        key="association_entropy_rise",
                        source="tracking_context",
                        value=rise,
                        threshold=thresholds.association_entropy_rise,
                        unit="1",
                        description="航迹关联不确定性上升",
                    )
                )
        confidence = _clamp01(confidence)
        if not samples or confidence < thresholds.event_min_confidence or not evidence:
            return None
        return self._event(
            inputs,
            samples[0],
            EventType.DECOY_OR_NEW_CONTACT_AMBIGUITY,
            confidence,
            EventLevel.TACTICAL,
            "R-DECOY-001",
            "预测短期内目标关联可能持续混乱；不能据此直接认定为诱饵",
            tuple(evidence),
        )

    def _stop_event(
        self,
        inputs: RuleWorldModelInput,
        samples: tuple[_FutureSample, ...],
    ) -> PredictedEvent | None:
        thresholds = self._config.thresholds
        low_speed_duration = 0.0
        for sample in samples:
            if sample.speed_mps <= thresholds.stop_speed_threshold_mps:
                low_speed_duration += sample.duration_s
            else:
                low_speed_duration = 0.0
            if low_speed_duration < thresholds.stop_confirmation_s:
                continue
            confidence = 0.65 + 0.30 * _clamp01(
                low_speed_duration / (2.0 * thresholds.stop_confirmation_s)
            )
            return self._event(
                inputs,
                sample,
                EventType.TARGET_ABNORMAL_STOP,
                confidence,
                EventLevel.TACTICAL,
                "R-STOP-001",
                "预测目标将持续低速或停止；二维轨迹不能据此判断沉没",
                (
                    RuleEvidence(
                        key="predicted_speed",
                        source="bspline",
                        value=sample.speed_mps,
                        threshold=thresholds.stop_speed_threshold_mps,
                        unit="m/s",
                        description="B-spline 预测段换算得到的低速状态",
                    ),
                    RuleEvidence(
                        key="predicted_low_speed_duration",
                        source="bspline",
                        value=low_speed_duration,
                        threshold=thresholds.stop_confirmation_s,
                        unit="s",
                        description="预测低速状态的连续持续时间",
                    ),
                ),
            )
        return None

    def _event(
        self,
        inputs: RuleWorldModelInput,
        sample: _FutureSample,
        event_type: EventType,
        confidence: float,
        level: EventLevel,
        rule_id: str,
        summary: str,
        evidence: tuple[RuleEvidence, ...],
    ) -> PredictedEvent | None:
        horizon = _horizon_for(sample.offset_s, self._config)
        if horizon is None:
            return None
        return PredictedEvent(
            event_id=(
                f"{inputs.scenario_id}:{inputs.target_id}:{event_type.value}:"
                f"{inputs.trajectory.prediction_id}:{inputs.source_plan_revision}:{inputs.owner_group_id}:"
                f"{sample.time_s:.3f}"
            ),
            **{name: (inputs.trajectory.prediction_id if name == "source_prediction_id" else getattr(inputs, name))
               for name in ForecastProvenance.model_fields},
            event_type=event_type,
            target_id=inputs.target_id,
            horizon=horizon,
            predicted_time_s=sample.time_s,
            time_to_event_s=sample.offset_s,
            predicted_position_xy=sample.point_xy,
            confidence=_clamp01(confidence),
            level=level,
            rule_id=rule_id,
            summary=summary,
            evidence=tuple(item.model_copy(update={
                "source": inputs.trajectory.prediction_regime,
                "description": item.description.replace("B-spline", inputs.trajectory.prediction_regime),
            }) if item.source == "bspline" else item for item in evidence),
        )


def predict_future_events(
    inputs: RuleWorldModelInput,
    config: RuleWorldModelConfig = DEFAULT_WORLD_MODEL_CONFIG,
) -> WorldModelForecast:
    """Convenience functional entry point for callers that do not retain state."""

    return RuleEventPredictor(config).predict(inputs)


def _build_samples(
    inputs: RuleWorldModelInput,
    config: RuleWorldModelConfig,
) -> tuple[_FutureSample, ...]:
    previous_time = float(inputs.as_of_s)
    previous_point = inputs.belief.position_xy
    velocity = inputs.belief.velocity_xy_mps
    previous_heading: float | None = (
        atan2(velocity[1], velocity[0]) if hypot(*velocity) > 1.0e-9 else None
    )
    samples: list[_FutureSample] = []
    for time_s, point_xy, corridor in zip(
        inputs.trajectory.times_s,
        inputs.trajectory.points_xy,
        inputs.trajectory.corridor_radius_m,
    ):
        duration_s = float(time_s - previous_time)
        delta_x = float(point_xy[0] - previous_point[0])
        delta_y = float(point_xy[1] - previous_point[1])
        distance = hypot(delta_x, delta_y)
        speed_mps = distance / duration_s
        heading = (
            atan2(delta_y, delta_x)
            if distance > 1.0e-9
            else previous_heading
            if previous_heading is not None
            else 0.0
        )
        heading_change = (
            _wrap_angle(heading - previous_heading)
            if previous_heading is not None and distance > 1.0e-9
            else 0.0
        )
        active_count, low_energy_count, geometry_od = _future_geometry(
            inputs,
            point_xy,
            float(time_s),
            config,
        )
        samples.append(
            _FutureSample(
                time_s=float(time_s),
                offset_s=float(time_s - inputs.as_of_s),
                duration_s=duration_s,
                point_xy=(float(point_xy[0]), float(point_xy[1])),
                corridor_radius_m=float(corridor),
                speed_mps=speed_mps,
                heading_rad=heading,
                heading_change_rad=heading_change,
                active_uuv_count=active_count,
                low_energy_uuv_count=low_energy_count,
                geometry_od=geometry_od,
            )
        )
        previous_time = float(time_s)
        previous_point = point_xy
        previous_heading = heading
    return tuple(samples)


def _future_geometry(
    inputs: RuleWorldModelInput,
    target_xy: tuple[float, float],
    time_s: float,
    config: RuleWorldModelConfig,
) -> tuple[int, int, float]:
    fim_00 = 0.0
    fim_01 = 0.0
    fim_11 = 0.0
    active_count = 0
    low_energy_count = 0
    thresholds = config.thresholds
    for uuv in inputs.uuvs:
        if uuv.energy_fraction <= thresholds.low_energy_fraction:
            low_energy_count += 1
        usable = (
            uuv.healthy
            and uuv.communication_ok
            and uuv.state_age_s <= thresholds.max_uuv_state_age_s
            and uuv.energy_fraction > thresholds.low_energy_fraction
        )
        if not usable:
            continue
        uuv_xy = _project_uuv(uuv, inputs.as_of_s, time_s)
        delta_x = target_xy[0] - uuv_xy[0]
        delta_y = target_xy[1] - uuv_xy[1]
        range_squared = delta_x * delta_x + delta_y * delta_y
        if range_squared <= 1.0 or sqrt(range_squared) > uuv.passive_range_m:
            continue
        gradient_x = -delta_y / range_squared
        gradient_y = delta_x / range_squared
        scale = 1.0 / uuv.bearing_variance_rad2
        fim_00 += gradient_x * gradient_x * scale
        fim_01 += gradient_x * gradient_y * scale
        fim_11 += gradient_y * gradient_y * scale
        active_count += 1
    trace = fim_00 + fim_11
    discriminant = sqrt(max(0.0, (fim_00 - fim_11) ** 2 + 4.0 * fim_01 * fim_01))
    lambda_min = max(0.0, 0.5 * (trace - discriminant))
    lambda_max = max(0.0, 0.5 * (trace + discriminant))
    singular = (
        active_count < 2
        or lambda_max <= 0.0
        or lambda_min <= thresholds.fim_rank_tolerance * max(lambda_max, 1.0)
    )
    geometry_od = 0.0 if singular else sqrt(lambda_min / lambda_max)
    return active_count, low_energy_count, geometry_od


def _project_uuv(
    uuv: UuvForecastInput,
    as_of_s: float,
    time_s: float,
) -> tuple[float, float]:
    future_pairs = [
        (float(sample_time), point)
        for sample_time, point in zip(uuv.planned_times_s, uuv.planned_points_xy)
        if sample_time > as_of_s
    ]
    if future_pairs:
        times = [float(as_of_s), *(pair[0] for pair in future_pairs)]
        points = [uuv.position_xy, *(pair[1] for pair in future_pairs)]
        if time_s >= times[-1]:
            return float(points[-1][0]), float(points[-1][1])
        for index in range(1, len(times)):
            if time_s <= times[index]:
                left_time = times[index - 1]
                right_time = times[index]
                fraction = (time_s - left_time) / (right_time - left_time)
                left = points[index - 1]
                right = points[index]
                return (
                    float(left[0] + fraction * (right[0] - left[0])),
                    float(left[1] + fraction * (right[1] - left[1])),
                )
    offset_s = max(0.0, time_s - as_of_s)
    return (
        float(uuv.position_xy[0] + uuv.velocity_xy_mps[0] * offset_s),
        float(uuv.position_xy[1] + uuv.velocity_xy_mps[1] * offset_s),
    )


def _signed_boundary_distance(
    point_xy: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> float:
    min_x, max_x, min_y, max_y = bounds
    x, y = point_xy
    if min_x <= x <= max_x and min_y <= y <= max_y:
        return min(x - min_x, max_x - x, y - min_y, max_y - y)
    outside_x = max(min_x - x, 0.0, x - max_x)
    outside_y = max(min_y - y, 0.0, y - max_y)
    return -hypot(outside_x, outside_y)


def _model_probability(inputs: RuleWorldModelInput, expected_name: str) -> float:
    direct = inputs.belief.model_probabilities.get(expected_name)
    if direct is not None:
        return float(direct)
    tokens = expected_name.split("_")
    return float(
        sum(
            probability
            for name, probability in inputs.belief.model_probabilities.items()
            if all(token in name.casefold() for token in tokens)
        )
    )


def _horizon_for(
    offset_s: float,
    config: RuleWorldModelConfig,
) -> HorizonName | None:
    for index, spec in enumerate(config.horizons):
        is_last = index == len(config.horizons) - 1
        if spec.start_offset_s <= offset_s < spec.end_offset_s or (
            is_last and offset_s == spec.end_offset_s
        ):
            return spec.name
    return None


def _in_horizon(
    offset_s: float,
    spec: HorizonSpec,
    config: RuleWorldModelConfig,
) -> bool:
    is_last = spec is config.horizons[-1]
    return spec.start_offset_s <= offset_s < spec.end_offset_s or (
        is_last and offset_s == spec.end_offset_s
    )


def _wrap_angle(value: float) -> float:
    return (value + pi) % (2.0 * pi) - pi


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
