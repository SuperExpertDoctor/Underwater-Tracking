"""Read-only execution evidence and operator-safe decision explanations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from underwater_tracking.domain.execution_models import (
    EvidenceReference,
    EvidenceResolution,
    ExecutionContextRef,
    ExecutionContribution,
    ExecutionDecisionRecord,
    OperationalExecutionSnapshot,
)


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
    group_ids = tuple(group.task_group_id for group in snapshot.task_groups)
    current_group = next(
        group
        for group in snapshot.task_groups
        if group.region_id == snapshot.current_region_id
    )
    other_groups = tuple(
        f"{group.task_group_id}负责{group.region_id}的预置、被动跟踪或交接"
        for group in snapshot.task_groups
        if group.task_group_id != current_group.task_group_id
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
        f"当前区域={snapshot.current_region_id}由{current_group.task_group_id}承担，"
        f"task group={current_group.task_group_id}；"
        f"主动核验={current_group.active_verifier_uuv_id}、被动跟踪={current_group.passive_tracker_uuv_id}；"
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


def answer_execution_question(
    snapshot: OperationalExecutionSnapshot,
    question: str,
    *,
    evidence_ids: Sequence[str] = (),
    resolver: ExecutionEvidenceResolver | None = None,
    frame_id: int | None = None,
) -> dict[str, object]:
    """Return a JSON-ready execution explanation with explicit evidence gaps."""

    selected = resolver or ExecutionEvidenceResolver(snapshot, frame_id=frame_id)
    record, resolution = selected.explain(question, evidence_ids=evidence_ids)
    answer = record.rationale
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
        add(group.evidence_ids, "task_group", f"任务组 {group.task_group_id} 分配证据")
    return references


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
]
