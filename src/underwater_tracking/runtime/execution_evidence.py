"""Read-only execution evidence and operator-safe decision explanations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any, Literal

from underwater_tracking.domain.conversation_models import (
    HandoffDiagnosis,
    OperationalDiagnosis,
    RegionEntryDiagnosis,
)
from underwater_tracking.domain.execution_models import (
    EvidenceReference,
    EvidenceResolution,
    ExecutionContextRef,
    ExecutionContribution,
    ExecutionDecisionRecord,
    OperationalExecutionSnapshot,
    TaskGroupInstance,
)
from underwater_tracking.domain.models import RuntimeEvent, SituationSnapshot
from underwater_tracking.runtime.execution_health import classify_execution_health

OperationalQuestionKind = Literal[
    "target_not_moving",
    "still_active_scan",
    "missing_owner",
    "handoff_pending",
]


def build_operational_diagnosis(
    snapshot: OperationalExecutionSnapshot,
    *,
    situation: SituationSnapshot | None = None,
    frame_id: int | None = None,
    current_sim_time_s: int | float | None = None,
    hard_stale_s: float = 900.0,
) -> OperationalDiagnosis:
    """Build a bounded diagnosis from current public execution evidence."""

    if current_sim_time_s is None:
        current_sim_time_s = (
            situation.sim_time_s if situation is not None else snapshot.source_sim_time_s
        )
    current_sim_time = float(current_sim_time_s)
    if not isfinite(current_sim_time) or current_sim_time < 0.0:
        raise ValueError("current_sim_time_s must be finite and non-negative")
    sim_time_s = int(current_sim_time)
    resolved_frame_id = (
        frame_id
        if frame_id is not None
        else snapshot.frame_id
        if snapshot.frame_id is not None
        else snapshot.source_snapshot_revision
    )
    health = classify_execution_health(
        snapshot,
        sim_time_s=current_sim_time,
        hard_stale_s=hard_stale_s,
    )
    health_reasons = tuple(
        dict.fromkeys((*snapshot.degradation.reasons, *health.reason_codes))
    )
    planning_data_status = (
        "unavailable"
        if health.status == "failed"
        else "stale"
        if current_sim_time > snapshot.valid_until_s
        else "current"
    )
    planning_age = max(
        0,
        int(current_sim_time - float(snapshot.source_sim_time_s)),
    )
    last_observed = snapshot.target_track.last_observed_at_s
    target_age = (
        None
        if last_observed is None
        else max(0, int(current_sim_time - float(last_observed)))
    )
    active_group_count = sum(
        _enum_value(group.lifecycle) == "active_scan"
        for group in snapshot.task_groups
    )
    passive_group_count = sum(
        _enum_value(group.lifecycle)
        in {"passive_track", "dedicated_track", "dedicated_release_pending"}
        for group in snapshot.task_groups
    )
    probability_evidence = _region_probability_evidence(situation)
    regions = tuple(
        _build_region_entry_diagnosis(snapshot, region, probability_evidence)
        for region in snapshot.regions
    )
    return OperationalDiagnosis(
        scenario_id=snapshot.scenario_id,
        sim_time_s=sim_time_s,
        frame_id=int(resolved_frame_id),
        execution_revision=snapshot.execution_revision,
        execution_health_status=health.status,
        execution_health_reasons=health_reasons,
        planning_data_status=planning_data_status,
        planning_data_age_s=planning_age,
        target_estimate_status=snapshot.target_track.freshness_status,
        target_estimate_age_s=target_age,
        active_group_count=active_group_count,
        passive_group_count=passive_group_count,
        tracking_owner_group_id=snapshot.tracking_control.tracking_owner_group_id,
        region_entry_progress=regions,
        handoff_progress=_build_handoff_diagnosis(snapshot, situation),
    )


def classify_operational_question(question: str) -> OperationalQuestionKind | None:
    """Recognize only questions with deterministic execution answers."""

    normalized = question.casefold()
    if (
        "tracking owner" in normalized
        or "owner" in normalized
        or "\u8ddf\u8e2a\u6240\u6709\u8005" in normalized
        or ("\u6240\u6709\u8005" in normalized and "\u6ca1" in normalized)
    ):
        return "missing_owner"
    if (
        "handoff" in normalized
        or "hand off" in normalized
        or "\u4ea4\u63a5" in normalized
        or "\u63a5\u73ed" in normalized
    ):
        return "handoff_pending"
    if (
        "active scan" in normalized
        or "actively scanning" in normalized
        or "\u4e3b\u52a8\u626b\u63cf" in normalized
    ):
        return "still_active_scan"
    if (
        "target not moving" in normalized
        or "why is the target stationary" in normalized
        or "target stationary" in normalized
        or "\u76ee\u6807" in normalized
        and ("\u4e0d\u52a8" in normalized or "\u505c\u6b62" in normalized)
    ):
        return "target_not_moving"
    return None


class ExecutionEvidenceResolver:
    """Resolve execution evidence without changing runtime or plan state.

    Snapshot references are useful even when the source event has already
    rolled out of a bounded event query.  Repository records, when present,
    replace that structural reference with their public summary.  No source
    repository is written by this class.
    """

    def __init__(
        self,
        snapshot: OperationalExecutionSnapshot,
        *,
        events: object | None = None,
        ledger: object | None = None,
        plans: object | None = None,
        frame_id: int | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.events = events
        self.ledger = ledger
        self.plans = plans
        self.frame_id = (
            frame_id
            if frame_id is not None
            else (
                snapshot.frame_id
                if snapshot.frame_id is not None
                else snapshot.source_snapshot_revision
            )
        )
        self.context = ExecutionContextRef.from_snapshot(
            snapshot, frame_id=self.frame_id
        )
        self._snapshot_references = _snapshot_references(snapshot, self.frame_id)

    @property
    def known_evidence_ids(self) -> tuple[str, ...]:
        return tuple(self._snapshot_references)

    def __call__(self, evidence_id: str) -> EvidenceReference | None:
        return self.resolve_one(evidence_id)

    def resolve_one(self, evidence_id: str) -> EvidenceReference | None:
        """Resolve one ID from the event, decision, plan, or snapshot views."""

        if not isinstance(evidence_id, str) or not evidence_id.strip():
            return None
        event = _call_repository(self.events, "get", evidence_id)
        if event is not None and _same_scenario(event, self.snapshot.scenario_id):
            return _event_reference(event, self.snapshot, self.frame_id)
        decision = _call_repository(self.ledger, "get", evidence_id)
        if decision is not None and _same_scenario(decision, self.snapshot.scenario_id):
            return _decision_reference(decision, self.snapshot, self.frame_id)
        plan = _call_repository(self.plans, "get_plan", evidence_id)
        if plan is not None and _same_scenario(plan, self.snapshot.scenario_id):
            return _plan_reference(plan, self.snapshot, self.frame_id)
        return self._snapshot_references.get(evidence_id)

    def resolve(self, evidence_ids: Sequence[str]) -> EvidenceResolution:
        """Return resolved and unresolved IDs in deterministic request order."""

        requested = tuple(dict.fromkeys(item for item in evidence_ids if item))
        resolved: list[EvidenceReference] = []
        unresolved: list[str] = []
        for evidence_id in requested:
            reference = self.resolve_one(evidence_id)
            if reference is None:
                unresolved.append(evidence_id)
            else:
                resolved.append(reference)
        return EvidenceResolution(
            requested_evidence_ids=requested,
            resolved=tuple(resolved),
            unresolved_evidence=tuple(unresolved),
            execution_revision=self.context.execution_revision,
            frame_id=self.context.frame_id,
        )

    def explain(
        self,
        question: str = "为何这样制定方案？",
        *,
        evidence_ids: Sequence[str] = (),
    ) -> tuple[ExecutionDecisionRecord, EvidenceResolution]:
        """Build the bounded explanation used by the assistant question path."""

        requested = tuple(evidence_ids) or self.snapshot.evidence_ids
        resolution = self.resolve(requested)
        record = build_execution_decision_record(
            self.snapshot,
            frame_id=self.frame_id,
            resolved_evidence_ids=tuple(item.evidence_id for item in resolution.resolved),
            unresolved_evidence=resolution.unresolved_evidence,
        )
        return record, resolution


def build_execution_decision_record(
    snapshot: OperationalExecutionSnapshot,
    *,
    frame_id: int | None = None,
    resolved_evidence_ids: Sequence[str] = (),
    unresolved_evidence: Sequence[str] = (),
) -> ExecutionDecisionRecord:
    """Project execution state into an operator-safe, non-chain-of-thought record."""

    context = ExecutionContextRef.from_snapshot(snapshot, frame_id=frame_id)
    region_ids = tuple(region.region_id for region in snapshot.regions)
    group_ids = tuple(_group_id(group) for group in snapshot.task_groups)
    current_group = next(
        group
        for group in snapshot.task_groups
        if group.region_id == snapshot.current_region_id
    )
    other_groups = tuple(
        f"{_group_id(group)}负责{group.region_id}的预置、被动跟踪或交接"
        for group in snapshot.task_groups
        if _group_id(group) != _group_id(current_group)
    )
    probabilities = ", ".join(
        f"{name}={value:.2f}"
        for name, value in sorted(snapshot.prediction.model_probabilities.items())
    )
    unresolved_text = (
        "未解析证据=" + ", ".join(unresolved_evidence)
        if unresolved_evidence
        else "未解析证据=无"
    )
    rationale = (
        f"确定性算法链：目标={snapshot.target_id}；轨迹位置=({snapshot.target_track.position_xy[0]:.1f},"
        f" {snapshot.target_track.position_xy[1]:.1f})，速度=({snapshot.target_track.velocity_xy[0]:.1f},"
        f" {snapshot.target_track.velocity_xy[1]:.1f})；IMM={probabilities}。"
        f"意图={snapshot.intent.intent_label}（置信度={snapshot.intent.confidence:.2f}）。"
        "四区域依据当前全局轨迹的 IMM 预测中心线按连续时间窗切分，保持稳定槽位、"
        "交接重叠和不确定性余量；"
        f"当前区域={snapshot.current_region_id}由{_group_id(current_group)}承担，"
        f"task group={_group_id(current_group)}；"
        f"{_group_member_summary(current_group)}；"
        f"其他组职责={'；'.join(other_groups)}。"
        f"最近调整={_recent_adjustment(snapshot)}。{unresolved_text}"
    )
    algorithm_evidence = tuple(
        dict.fromkeys(
            (
                *snapshot.target_track.source_event_ids,
                *snapshot.prediction.source_observation_ids,
                *snapshot.intent.evidence_ids,
            )
        )
    )
    algorithm_contributions = (
        ExecutionContribution(
            contributor="algorithm",
            component="global_track",
            summary="确定性全局轨迹汇总已执行目标物理位置、速度和有界历史。",
            evidence_ids=tuple(snapshot.target_track.source_event_ids),
        ),
        ExecutionContribution(
            contributor="algorithm",
            component="imm_forecast",
            summary=f"IMM 对 CV、CT_LEFT、CT_RIGHT 分支做状态预测和概率加权（{probabilities}）。",
            evidence_ids=tuple(snapshot.prediction.source_observation_ids),
        ),
        ExecutionContribution(
            contributor="algorithm",
            component="intent_and_regions",
            summary=(
                f"规则意图为 {snapshot.intent.intent_label}，规范化器将预测走廊保持为四个连续区域槽位。"
            ),
            evidence_ids=algorithm_evidence,
        ),
    )
    llm_summary = (
        "LLM 仅提供受约束的时间窗、宽度、重叠、角色和优先级建议；几何、航点和资源约束仍由确定性规范化器决定。"
        if snapshot.plan_source == "llm_optimized"
        else "当前执行版本没有可解析的 LLM 几何或资源决定，不能推断额外的 LLM 贡献。"
    )
    human_summary = (
        f"人工反馈版本={snapshot.expert_request_version}，已进入当前执行快照。"
        if snapshot.expert_request_version
        else "当前执行版本没有已确认的人工反馈。"
    )
    evidence = tuple(dict.fromkeys(resolved_evidence_ids or snapshot.evidence_ids))
    return ExecutionDecisionRecord(
        decision_id=f"{snapshot.scenario_id}:execution-decision:{snapshot.execution_revision}",
        scenario_id=snapshot.scenario_id,
        execution_revision=context.execution_revision,
        frame_id=context.frame_id,
        target_id=snapshot.target_id,
        prediction_id=snapshot.prediction_id,
        intent_label=snapshot.intent.intent_label,
        current_region_id=snapshot.current_region_id,
        next_region_id=snapshot.next_region_id,
        region_ids=region_ids,
        task_group_ids=group_ids,
        evidence_ids=evidence,
        unresolved_evidence=tuple(dict.fromkeys(unresolved_evidence)),
        rationale=rationale,
        recent_adjustment=_recent_adjustment(snapshot),
        algorithm_contributions=algorithm_contributions,
        llm_contributions=(
            ExecutionContribution(
                contributor="llm",
                component="strategy_revision",
                summary=llm_summary,
                evidence_ids=tuple(snapshot.evidence_ids),
            ),
        ),
        human_contributions=(
            ExecutionContribution(
                contributor="human",
                component="operator_feedback",
                summary=human_summary,
                evidence_ids=tuple(snapshot.evidence_ids)
                if snapshot.expert_request_version
                else (),
            ),
        ),
    )


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value)).casefold()


def _region_probability_evidence(
    situation: SituationSnapshot | None,
) -> Mapping[str, Mapping[str, object]]:
    if situation is None:
        return {}
    value = getattr(situation, "region_probability_evidence", {})
    if not isinstance(value, Mapping):
        return {}
    return {
        str(region_id): data
        for region_id, data in value.items()
        if isinstance(data, Mapping)
    }


def _finite_unit(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if isfinite(numeric) and 0.0 <= numeric <= 1.0 else None


def _nonnegative_count(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return numeric if numeric >= 0 else default


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if not isinstance(value, Sequence):
        return ()
    return tuple(
        str(item)
        for item in value
        if isinstance(item, str) and item.strip()
    )


def _build_region_entry_diagnosis(
    snapshot: OperationalExecutionSnapshot,
    region: object,
    probability_evidence: Mapping[str, Mapping[str, object]],
) -> RegionEntryDiagnosis:
    region_id = str(getattr(region, "region_id"))
    data = probability_evidence.get(region_id, {})
    probability = _finite_unit(data.get("probability"))
    confirmations = _nonnegative_count(
        data.get("entry_confirmations", data.get("confirmations", 0))
    )
    required = max(
        1,
        _nonnegative_count(
            data.get("required_confirmations", 2),
            default=2,
        ),
    )
    coverage = _finite_unit(
        data.get("active_coverage_ratio", data.get("coverage_ratio"))
    )
    source_backed_ping_count = _nonnegative_count(
        data.get(
            "source_backed_ping_count",
            data.get("source_backed_pings", data.get("source_backed_ping_count_this_cycle", 0)),
        )
    )
    lifecycle = _enum_value(getattr(region, "status", "unknown"))
    group = next(
        (
            candidate
            for candidate in snapshot.task_groups
            if getattr(candidate, "region_id", None) == region_id
        ),
        None,
    )
    if lifecycle == "planned" and group is not None:
        lifecycle = _enum_value(getattr(group, "lifecycle", lifecycle))
    blocked = data.get("blocked_reason")
    blocked_reason = blocked.strip() if isinstance(blocked, str) and blocked.strip() else None
    if blocked_reason is None:
        reasons = _string_tuple(data.get("reason_codes"))
        if data.get("eligible_for_confirmation") is False:
            blocked_reason = reasons[0] if reasons else "entry_confirmation_not_eligible"
        elif probability is None:
            blocked_reason = reasons[0] if reasons else "entry_probability_unavailable"
        elif confirmations < required:
            blocked_reason = "entry_confirmation_pending"
    if not data and blocked_reason is None:
        blocked_reason = "region_entry_evidence_unavailable"
    return RegionEntryDiagnosis(
        region_id=region_id,
        probability=probability,
        confirmations=confirmations,
        required_confirmations=required,
        lifecycle=lifecycle,
        coverage_ratio=coverage,
        source_backed_ping_count=source_backed_ping_count,
        blocked_reason=blocked_reason,
    )


def _event_payloads(situation: SituationSnapshot | None) -> tuple[Mapping[str, object], ...]:
    if situation is None:
        return ()
    events = getattr(situation, "pending_events", ())
    if not isinstance(events, Sequence):
        return ()
    ordered = sorted(
        (event for event in events if isinstance(event, RuntimeEvent)),
        key=lambda event: (event.sim_time_s, event.event_id),
    )
    return tuple(
        event.payload
        for event in ordered
        if event.event_type
        in {
            "handoff_waiting_for_passive_observation",
            "handoff_blocked",
            "handoff_completed",
            "tracking_ownership_transferred",
        }
    )


def _build_handoff_diagnosis(
    snapshot: OperationalExecutionSnapshot,
    situation: SituationSnapshot | None,
) -> HandoffDiagnosis | None:
    control = snapshot.tracking_control
    owner_id = control.tracking_owner_group_id
    successor_id = control.pending_successor_group_id
    has_pending_region = any(
        _enum_value(region.status) == "handoff_pending" for region in snapshot.regions
    )
    payloads = _event_payloads(situation)
    if owner_id is None and successor_id is None and not has_pending_region and not payloads:
        return None
    latest = payloads[-1] if payloads else {}
    successor = next(
        (
            group
            for group in snapshot.task_groups
            if group.group_instance_id == successor_id
        ),
        None,
    )
    required = _string_tuple(latest.get("required_uuv_ids"))
    if not required and successor is not None:
        required = tuple(successor.member_uuv_ids)
    observed = _string_tuple(
        latest.get(
            "observed_uuv_ids",
            latest.get("passive_observed_uuv_ids", latest.get("observed_member_ids")),
        )
    )
    if not observed:
        accepted = latest.get("accepted_observations", ())
        if isinstance(accepted, Sequence):
            observed_values: list[str] = []
            for item in accepted:
                if isinstance(item, Mapping):
                    observer_id = item.get("observer_uuv_id")
                else:
                    observer_id = getattr(item, "observer_uuv_id", None)
                if isinstance(observer_id, str) and observer_id.strip():
                    observed_values.append(observer_id)
            observed = tuple(observed_values)
    blocked = latest.get("blocked_reason")
    blocked_reason = blocked.strip() if isinstance(blocked, str) and blocked.strip() else None
    if blocked_reason is None and required and set(required) - set(observed):
        blocked_reason = "waiting_for_passive_observation"
    return HandoffDiagnosis(
        owner_group_id=owner_id or "unassigned",
        successor_group_id=successor_id,
        required_uuv_ids=tuple(dict.fromkeys(required)),
        observed_uuv_ids=tuple(dict.fromkeys(observed)),
        blocked_reason=blocked_reason,
    )


def _region_progress_text(diagnosis: OperationalDiagnosis) -> str:
    lines: list[str] = []
    for region in diagnosis.region_entry_progress:
        probability = (
            "unavailable"
            if region.probability is None
            else f"{region.probability:.2f}"
        )
        coverage = (
            "unavailable"
            if region.coverage_ratio is None
            else f"{region.coverage_ratio:.2f}"
        )
        blocked = region.blocked_reason or "none"
        lines.append(
            f"{region.region_id}: probability={probability}, "
            f"confirmations={region.confirmations}/{region.required_confirmations}, "
            f"lifecycle={region.lifecycle}, coverage={coverage}, "
            f"source_backed_ping_count={region.source_backed_ping_count}, "
            f"blocked={blocked}"
        )
    return "; ".join(lines)


def _render_operational_answer(
    kind: OperationalQuestionKind,
    diagnosis: OperationalDiagnosis,
    snapshot: OperationalExecutionSnapshot,
) -> str:
    prefix = (
        f"frame_id={diagnosis.frame_id}, execution_revision={diagnosis.execution_revision}, "
        f"sim_time_s={diagnosis.sim_time_s}."
    )
    health = diagnosis.execution_health_status
    reasons = ", ".join(diagnosis.execution_health_reasons) or "none"
    if kind == "target_not_moving":
        velocity = snapshot.target_track.velocity_xy
        return (
            f"{prefix} The current public target estimate reports "
            f"velocity=({velocity[0]:.2f}, {velocity[1]:.2f}) and "
            f"estimate_status={diagnosis.target_estimate_status}, "
            f"estimate_age_s={diagnosis.target_estimate_age_s}. "
            f"Execution health={health}; reasons={reasons}. "
            "This frame does not provide private target state, so no stronger "
            "cause is claimed."
        )
    if kind == "still_active_scan":
        return (
            f"{prefix} Active groups={diagnosis.active_group_count}; "
            f"passive groups={diagnosis.passive_group_count}. "
            f"Region entry progress: {_region_progress_text(diagnosis)}. "
            "Active scanning remains the deterministic mode until the required "
            "entry confirmations are present. Missing public evidence is reported "
            "as blocked rather than inferred."
        )
    if kind == "missing_owner":
        owner = diagnosis.tracking_owner_group_id or "none"
        owner_sentence = (
            "No tracking owner is assigned in the current frame."
            if diagnosis.tracking_owner_group_id is None
            else f"tracking_owner_group_id={owner}."
        )
        return (
            f"{prefix} {owner_sentence} "
            f"tracking_owner_group_id={owner}; "
            f"active groups={diagnosis.active_group_count}; "
            f"passive groups={diagnosis.passive_group_count}; "
            f"execution_health={health}. "
            "The current frame contains no authority assigning a different owner."
        )
    handoff = diagnosis.handoff_progress
    if handoff is None:
        handoff_text = "no pending successor or handoff evidence is present"
    else:
        handoff_text = (
            f"owner={handoff.owner_group_id}, successor={handoff.successor_group_id or 'none'}, "
            f"required_uuv_ids={list(handoff.required_uuv_ids)}, "
            f"observed_uuv_ids={list(handoff.observed_uuv_ids)}, "
            f"blocked={handoff.blocked_reason or 'none'}"
        )
    return f"{prefix} Handoff status: {handoff_text}."


def answer_execution_question(
    snapshot: OperationalExecutionSnapshot,
    question: str,
    *,
    evidence_ids: Sequence[str] = (),
    resolver: ExecutionEvidenceResolver | None = None,
    frame_id: int | None = None,
    situation: SituationSnapshot | None = None,
    current_sim_time_s: int | float | None = None,
) -> dict[str, object]:
    """Return a JSON-ready execution explanation with explicit evidence gaps."""

    selected = resolver or ExecutionEvidenceResolver(snapshot, frame_id=frame_id)
    record, resolution = selected.explain(question, evidence_ids=evidence_ids)
    diagnosis = build_operational_diagnosis(
        snapshot,
        situation=situation,
        frame_id=frame_id,
        current_sim_time_s=current_sim_time_s,
    )
    question_kind = classify_operational_question(question)
    answer = (
        _render_operational_answer(question_kind, diagnosis, snapshot)
        if question_kind is not None
        else record.rationale
    )
    if resolution.unresolved_evidence:
        answer += " 无法基于未解析证据生成确定性理由。"
    answer += " 可解析证据 ID=" + ", ".join(
        item.evidence_id for item in resolution.resolved
    )
    return {
        "answer": answer,
        "evidence_ids": [item.evidence_id for item in resolution.resolved],
        "unresolved_evidence": list(resolution.unresolved_evidence),
        "execution_revision": selected.context.execution_revision,
        "frame_id": selected.context.frame_id,
        "decision_record": record.model_dump(mode="json"),
        "diagnosis": diagnosis.model_dump(mode="json"),
    }


def _snapshot_references(
    snapshot: OperationalExecutionSnapshot, frame_id: int
) -> dict[str, EvidenceReference]:
    references: dict[str, EvidenceReference] = {}

    def add(ids: Sequence[str], source_type: str, summary: str) -> None:
        for evidence_id in ids:
            if evidence_id and evidence_id not in references:
                references[evidence_id] = EvidenceReference(
                    evidence_id=evidence_id,
                    source_type=source_type,
                    scenario_id=snapshot.scenario_id,
                    summary=summary,
                    execution_revision=snapshot.execution_revision,
                    frame_id=frame_id,
                )

    add(snapshot.evidence_ids, "execution_snapshot", "当前执行快照提交证据")
    add(snapshot.target_track.source_event_ids, "global_track", "全局目标轨迹来源事件")
    add(snapshot.prediction.source_observation_ids, "imm_forecast", "IMM 预测来源观测")
    for branch in snapshot.prediction.model_branches:
        add(branch.source_observation_ids, "imm_branch", f"IMM {branch.model_name} 分支来源观测")
    add(snapshot.intent.evidence_ids, "deterministic_intent", "确定性意图判断证据")
    for region in snapshot.regions:
        add(region.evidence_ids, "execution_region", f"区域 {region.region_id} 规范化证据")
    for group in snapshot.task_groups:
        add(group.evidence_ids, "task_group", f"任务组 {_group_id(group)} 分配证据")
    return references


def _group_id(group: object) -> str:
    if isinstance(group, TaskGroupInstance):
        return group.group_instance_id
    return str(getattr(group, "task_group_id"))


def _group_member_summary(group: object) -> str:
    if isinstance(group, TaskGroupInstance):
        return (
            f"成员={','.join(group.member_uuv_ids)}，"
            f"生命周期={group.lifecycle.value}，传感器={group.sensor_mode.value}"
        )
    return (
        f"主动核验={group.active_verifier_uuv_id}、"
        f"被动跟踪={group.passive_tracker_uuv_id}"
    )


def _call_repository(repository: object | None, method_name: str, value: str) -> object | None:
    method = getattr(repository, method_name, None)
    if not callable(method):
        return None
    try:
        return method(value)
    except (LookupError, ValueError):
        return None


def _same_scenario(value: object, scenario_id: str) -> bool:
    return getattr(value, "scenario_id", None) == scenario_id


def _event_reference(
    event: object, snapshot: OperationalExecutionSnapshot, frame_id: int
) -> EvidenceReference:
    payload = getattr(event, "payload", {})
    summary = payload.get("summary") if isinstance(payload, Mapping) else None
    if not isinstance(summary, str) or not summary.strip():
        summary = f"{getattr(event, 'event_type', 'runtime_event')} at t={getattr(event, 'sim_time_s', '?')}"
    event_id = str(getattr(event, "event_id", ""))
    return EvidenceReference(
        evidence_id=event_id,
        source_type="runtime_event",
        scenario_id=snapshot.scenario_id,
        summary=summary,
        execution_revision=snapshot.execution_revision,
        frame_id=frame_id,
        source_event_id=event_id,
    )


def _decision_reference(
    decision: object, snapshot: OperationalExecutionSnapshot, frame_id: int
) -> EvidenceReference:
    decision_id = str(getattr(decision, "decision_id", ""))
    return EvidenceReference(
        evidence_id=decision_id,
        source_type="decision_record",
        scenario_id=snapshot.scenario_id,
        summary=f"执行前规划决策 {decision_id}",
        execution_revision=snapshot.execution_revision,
        frame_id=frame_id,
        source_decision_id=decision_id,
    )


def _plan_reference(
    plan: object, snapshot: OperationalExecutionSnapshot, frame_id: int
) -> EvidenceReference:
    plan_id = str(getattr(plan, "plan_id", ""))
    return EvidenceReference(
        evidence_id=plan_id,
        source_type="plan",
        scenario_id=snapshot.scenario_id,
        summary=f"执行审计计划 {plan_id}",
        execution_revision=snapshot.execution_revision,
        frame_id=frame_id,
    )


def _recent_adjustment(snapshot: OperationalExecutionSnapshot) -> str:
    if snapshot.base_execution_revision is None:
        return "初始执行版本，无上一版本调整。"
    return (
        f"从 execution_revision={snapshot.base_execution_revision} 更新到 "
        f"{snapshot.execution_revision}；物理目标轨迹来源快照为 "
        f"{snapshot.source_snapshot_revision}。"
    )


__all__ = [
    "ExecutionEvidenceResolver",
    "answer_execution_question",
    "build_execution_decision_record",
    "build_operational_diagnosis",
    "classify_operational_question",
]
