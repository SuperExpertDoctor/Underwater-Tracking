import { type CSSProperties, type ReactNode, useState } from "react";
import {
  Activity,
  ChevronDown,
  CircleX,
  Link2,
  Radio,
  RotateCcw,
  Ship,
  Target,
  Waves,
} from "lucide-react";
import type {
  AdversaryDecisionView,
  AdversaryView,
  BrainView,
  BrainStatus,
  CommunicationStatus,
  OperationalFrame,
  OperationalStage,
  UUVStatus,
} from "../types/frames";
import CarrierStatusPanel from "./CarrierStatusPanel";
import "./SidebarPanels.css";
import { displayTargetName } from "../utils/presentation";

const STATUS_LABELS: Record<UUVStatus, string> = {
  available: "待命",
  tracking: "跟踪",
  returning: "返航",
  failed: "故障",
};

const STATUS_COLORS: Record<UUVStatus, string> = {
  available: "#687f92",
  tracking: "#18a9a0",
  returning: "#d88b16",
  failed: "#cc3f4d",
};

interface RightSidebarProps {
  frame: OperationalFrame | null;
  selectedUuvId: string | null;
  onSelectUuv: (id: string | null) => void;
  open: boolean;
  onClose: () => void;
  onSensorMode?: (
    uuvId: string,
    mode: "passive" | "active",
    targetId: string | null,
  ) => void;
  predictionPanel?: ReactNode;
  assistantPanel?: ReactNode;
  memoryPanel?: ReactNode;
  onRetryPlanning?: () => void;
  retryingPlanning?: boolean;
}

export default function RightSidebar({
  frame,
  selectedUuvId,
  onSelectUuv,
  open,
  onClose,
  onSensorMode,
  predictionPanel,
  assistantPanel,
  memoryPanel,
  onRetryPlanning,
  retryingPlanning = false,
}: RightSidebarProps) {
  const uuvs = frame?.uuvs ?? [];
  const brains = frame?.brains ?? [];
  const links = frame?.communication_links ?? [];
  const resources = new Map(
    (frame?.uuv_resources ?? []).map((resource) => [resource.uuv_id, resource]),
  );
  const targets = frame?.target_estimates ?? [];
  const groups = frame?.groups ?? [];
  const selected = uuvs.find((uuv) => uuv.uuv_id === selectedUuvId);
  const active = uuvs.filter((uuv) => uuv.status === "tracking").length;
  const failed = uuvs.filter((uuv) => uuv.status === "failed").length;
  const reserved = uuvs.filter((uuv) => uuv.reserved).length;
  const primaryQuality = targets[0]?.quality.quality_score;
  const intelligence = frame?.intelligence ?? [];
  const techIntelCount = intelligence.filter(
    (report) => report.source === "technical_reconnaissance",
  ).length;
  const adversary = frame?.adversary ?? frame?.adversaries?.[0] ?? null;
  const target = targets[0];
  const currentDecision =
    adversary?.current_decision ??
    frame?.adversary_decision ??
    adversaryDecisionFromSummary(adversary);
  const isModernOperationalFrame =
    frame != null && Object.prototype.hasOwnProperty.call(frame, "target_priors");
  const brainNodes: BrainView[] =
    brains.some((brain) => brain.role === "adversary") ||
    !adversary ||
    isModernOperationalFrame
      ? brains
      : [
          ...brains,
          {
            brain_id: "adversary-brain",
            role: "adversary",
            status: currentDecision ? "online" : "unknown",
            last_update_s: currentDecision?.sim_time_s ?? null,
            message: "目标潜艇反跟踪决策",
            connected_platform_ids: [],
          },
        ];
  const decisionHistory = uniqueDecisions([
    ...(adversary?.decision_history ?? []),
    ...(frame?.adversary_history ?? []),
    ...(frame?.adversaries ?? [])
      .map(adversaryDecisionFromSummary)
      .filter(isDecision),
    ...(currentDecision ? [currentDecision] : []),
  ]);
  const detectedPlatformIds =
    currentDecision?.detected_platform_ids ??
    target?.detected_platform_ids ??
    adversary?.detected_platform_ids ??
    brainNodes.find((brain) => brain.role === "adversary")
      ?.evidence_platform_ids ??
    [];
  const regionalEffects = Object.values(frame?.regional_plans ?? {}).flatMap(
    (plan) => plan.regions.map((region) => region.effect.coverage_ratio),
  );
  const coverage = regionalEffects.length
    ? `${Math.round((regionalEffects.reduce((total, value) => total + value, 0) / regionalEffects.length) * 100)}%`
    : "—";

  return (
    <aside className={`sidebar ${open ? "open" : ""}`} aria-label="编队态势">
      <div className="sidebar-header">
        <div>
          <span className="eyebrow">MISSION / LIVE ESTIMATE</span>
          <strong>编队态势</strong>
        </div>
        <button
          className="icon-btn mobile-only"
          onClick={onClose}
          aria-label="关闭编队状态"
          title="关闭"
        >
          <CircleX size={17} />
        </button>
      </div>
      {!frame ? (
        <div className="sidebar-empty">
          <Waves size={24} />
          <span>等待作业态势帧</span>
        </div>
      ) : (
        <>
          <OperationalStageMatrix
            stages={frame.operational_stage_flags ?? []}
          />
          <PlanningRunStatus
            frame={frame}
            onRetryPlanning={onRetryPlanning}
            retrying={retryingPlanning}
          />
          <CollapsiblePanel
            title="当前态势"
            subtitle={`${active} 艇跟踪`}
            className="command-center-panel current-situation-panel"
          >
            <section
              className="sidebar-section overview-grid"
              aria-label="任务概览"
            >
              <Metric
                label="仿真时间"
                value={formatSimTime(frame.sim_time_s)}
              />
              <Metric
                label="方案版本"
                value={`#${frame.plan_version}`}
                emphasized
              />
              <Metric label="跟踪中" value={`${active} 艇`} />
              <Metric label="目标估计" value={`${targets.length} 个`} />
              <Metric label="预测覆盖" value={coverage} />
            </section>
            <section
              className="sidebar-section status-strip"
              aria-label="系统状态"
            >
              <div>
                <Activity size={14} />
                <span>质量</span>
                <b>
                  {primaryQuality == null
                    ? "—"
                    : `${(primaryQuality * 100).toFixed(0)}%`}
                </b>
              </div>
              <div>
                <Radio size={14} />
                <span>主动声纳</span>
                <b>
                  {uuvs.filter((uuv) => uuv.sensor_mode === "active").length}
                </b>
              </div>
              <div>
                <Target size={14} />
                <span>故障艇</span>
                <b className={failed ? "danger-text" : ""}>{failed}</b>
              </div>
            </section>

            <section
              className="sidebar-section brain-section"
              aria-label="主从对手脑状态"
            >
              <div className="section-heading">
                <span>智能节点</span>
                <small>{`${brainNodes.filter(isOperationalBrainReady).length}/${brainNodes.length} 就绪`}</small>
              </div>
              <div className="brain-grid">
                {brainNodes.map((brain) => {
                  const cardContent = (
                    <>
                      <div className="brain-card-head">
                        <strong>{brainRoleLabel(brain.role)}</strong>
                        <span>{brainStatusLabel(brain.status)}</span>
                      </div>
                      <small>{brain.message}</small>
                      {brain.operation && <small>{brain.operation}</small>}
                      {(brain.evidence_platform_ids?.length ?? 0) > 0 && (
                        <small>
                          证据 {brain.evidence_platform_ids?.join(", ")}
                        </small>
                      )}
                      <em>
                        {brain.last_update_s == null
                          ? "未接入态势"
                          : `更新 ${formatSimTime(brain.last_update_s)}`}
                      </em>
                    </>
                  );
                  if (brain.role !== "adversary") {
                    return (
                      <div
                        className={`brain-card brain-${brain.status}`}
                        key={brain.brain_id}
                      >
                        {cardContent}
                      </div>
                    );
                  }
                  return (
                    <details
                      className={`brain-card brain-${brain.status} adversary-brain-card`}
                      key={brain.brain_id}
                    >
                      <summary aria-label="展开目标潜艇脑详情">
                        {cardContent}
                      </summary>
                      <TargetSubmarineBrain
                        targetId={target?.target_id ?? adversary?.target_id}
                        currentDecision={currentDecision}
                        detectedPlatformIds={detectedPlatformIds}
                        decisionHistory={decisionHistory}
                      />
                    </details>
                  );
                })}
              </div>
            </section>

            <section
              className="sidebar-section adaptive-context"
              aria-label="情报流"
            >
              <div className="section-heading">
                <span>情报流</span>
                <small>{`技侦 ${techIntelCount} / 情报 ${intelligence.length}`}</small>
              </div>
              {intelligence.length > 0 ? (
                <small className="adaptive-intel-latest">
                  最新 {formatSimTime(intelligence[0].issued_at_s)} ·{" "}
                  {displayTargetName(intelligence[0].target_id)} · 置信度{" "}
                  {(intelligence[0].confidence * 100).toFixed(0)}%
                </small>
              ) : (
                <span className="adaptive-muted">当前帧暂无新增情报</span>
              )}
            </section>

            <section
              className="sidebar-section uuv-section"
              aria-label="UUV 资源与底层控制状态"
            >
              <div className="section-heading">
                <span>UUV 资源</span>
                <small>
                  {reserved ? `${reserved} 艇已指派` : "未锁定资源"}
                </small>
              </div>
              <div className="uuv-list">
                {uuvs.map((uuv) => {
                  const color = STATUS_COLORS[uuv.status];
                  const energy = Math.round(uuv.energy_fraction * 100);
                  const selectedRow = uuv.uuv_id === selectedUuvId;
                  const targetId = trackedTargetId(frame, uuv.uuv_id);
                  const linkState = uuvCommunicationStatus(uuv);
                  return (
                    <button
                      key={uuv.uuv_id}
                      className={`uuv-row ${selectedRow ? "selected" : ""}`}
                      onClick={() =>
                        onSelectUuv(selectedRow ? null : uuv.uuv_id)
                      }
                      aria-pressed={selectedRow}
                    >
                      <span className="uuv-signal" style={{ color }}>
                        <span />
                      </span>
                      <span className="uuv-copy">
                        <strong>{uuv.uuv_id}</strong>
                        <small>
                          {STATUS_LABELS[uuv.status]} ·{" "}
                          {uuvGroupLabel(frame, uuv.uuv_id)}
                        </small>
                        <small>
                          归属 {resources.get(uuv.uuv_id)?.carrier_id ?? "未登记"} · 里程{" "}
                          {Math.round(resources.get(uuv.uuv_id)?.mileage_m ?? 0)} m · 健康{" "}
                          {resources.get(uuv.uuv_id)?.healthy === false ? "异常" : "正常"}
                        </small>
                        <span className="uuv-row-meta">
                          <span className={`link-dot link-${linkState}`}>
                            <Link2 size={10} />
                            {communicationStatusLabel(linkState)}
                          </span>
                          <span>
                            {targetId
                              ? `目标 ${displayTargetName(targetId)}`
                              : "未绑定目标"}
                          </span>
                        </span>
                      </span>
                      <span
                        className="energy-gauge"
                        style={
                          {
                            "--energy": `${energy}%`,
                            "--energy-color": color,
                          } as CSSProperties
                        }
                        aria-label={`剩余能量 ${energy}%`}
                      >
                        <b>{energy}</b>
                      </span>
                    </button>
                  );
                })}
                {!uuvs.length && (
                  <span className="adaptive-muted">当前帧未接入 UUV</span>
                )}
              </div>
              <div className="link-summary">
                <span>链路</span>
                <strong>{`${links.filter((link) => link.status === "connected").length} 通 / ${links.filter((link) => link.status === "disconnected").length} 断`}</strong>
              </div>
            </section>

            {currentDecision && false && (
              <section
                className="sidebar-section adversary-section"
                aria-label="目标潜艇反跟踪决策"
              >
                <div className="section-heading">
                  <span>目标潜艇脑</span>
                  <small>
                    {displayTargetName(
                      target?.target_id ?? adversary?.target_id,
                    )}
                  </small>
                </div>
                {currentDecision ? (
                  <>
                    <div className="adversary-intent-row">
                      <span className="adversary-status-dot" />
                      <strong>{currentDecision!.intent || "待决策"}</strong>
                      <span>{currentDecision!.maneuver || "保持航迹"}</span>
                      {currentDecision!.decision_status === "inconclusive" && (
                        <small className="adversary-estimate-badge">
                          目标侧估计 · 待对手脑确认
                        </small>
                      )}
                      {currentDecision!.confidence != null && (
                        <b>{Math.round(currentDecision!.confidence! * 100)}%</b>
                      )}
                    </div>
                    <div className="decision-facts">
                      <span>
                        分段 <b>{currentDecision!.segment || "当前水域"}</b>
                      </span>
                      <span>
                        触发{" "}
                        <b>
                          {currentDecision!.trigger_event_ids?.length ?? 0} 事件
                        </b>
                      </span>
                      <span>
                        暴露{" "}
                        <b
                          className={
                            detectedPlatformIds.length
                              ? "danger-text"
                              : "safe-text"
                          }
                        >
                          {detectedPlatformIds.length} 节点
                        </b>
                      </span>
                    </div>
                    <div className="decision-summary">
                      <small>LLM 决策摘要</small>
                      <p>
                        {currentDecision!.decision_summary ||
                          currentDecision!.rationale ||
                          "目标正在根据观测维护反跟踪方案。"}
                      </p>
                    </div>
                    {detectedPlatformIds.length > 0 && (
                      <div
                        className="detected-badges"
                        aria-label="目标已探测到的我方节点"
                      >
                        {detectedPlatformIds.slice(0, 8).map((id) => (
                          <span key={id}>已暴露 {id}</span>
                        ))}
                      </div>
                    )}
                    <div className="adversary-facts">
                      <span>
                        主动声纳风险{" "}
                        <b>{currentDecision!.active_ping_risk || "未报告"}</b>
                      </span>
                      <span>
                        通信纪律{" "}
                        <b>
                          {currentDecision!.communications_discipline ||
                            "未报告"}
                        </b>
                      </span>
                    </div>
                  </>
                ) : (
                  <div className="adversary-empty">
                    等待目标 LLM 根据观测生成反跟踪决策
                  </div>
                )}
                <div className="decision-history" aria-label="目标决策历史">
                  <div className="history-heading">
                    <span>反跟踪历史</span>
                    <small>{decisionHistory.length} 条</small>
                  </div>
                  {decisionHistory.slice(0, 5).map((decision) => (
                    <div
                      className="history-row"
                      key={
                        decision.decision_id ??
                        `${decision.target_id}-${decision.sim_time_s}`
                      }
                    >
                      <time>{formatSimTime(decision.sim_time_s)}</time>
                      <span>{decision.intent || "—"}</span>
                      <b>{decision.maneuver || "—"}</b>
                    </div>
                  ))}
                  {!decisionHistory.length && (
                    <small className="adaptive-muted">暂无动态调整记录</small>
                  )}
                </div>
              </section>
            )}

            <section
              className="sidebar-section carrier-section"
              aria-label="母舰与载荷"
            >
              <CarrierStatusPanel frame={frame} />
            </section>

            {selected && (
              <section
                className="sidebar-section selected-detail"
                aria-label={`${selected.uuv_id} 详情`}
              >
                <div className="section-heading">
                  <span>{selected.uuv_id} 详情</span>
                  <small>
                    {Math.round((selected.heading_rad * 180) / Math.PI)}° 航向
                  </small>
                </div>
                <dl>
                  <div>
                    <dt>坐标</dt>
                    <dd>
                      {selected.position.x.toFixed(0)},{" "}
                      {selected.position.y.toFixed(0)} m
                    </dd>
                  </div>
                  <div>
                    <dt>速度</dt>
                    <dd>{selected.speed_mps.toFixed(1)} m/s</dd>
                  </div>
                  <div>
                    <dt>编组</dt>
                    <dd>{selected.group_id ?? "—"}</dd>
                  </div>
                  <div>
                    <dt>传感器</dt>
                    <dd>
                      {selected.sensor_mode === "active"
                        ? "主动声纳"
                        : "被动声纳"}
                    </dd>
                  </div>
                  <div>
                    <dt>剩余续航</dt>
                    <dd>
                      {formatRange(
                        selected.remaining_range_m ??
                          selected.endurance_remaining_m,
                      )}
                    </dd>
                  </div>
                  <div>
                    <dt>通信链路</dt>
                    <dd className={`value-${uuvCommunicationStatus(selected)}`}>
                      {communicationStatusLabel(
                        uuvCommunicationStatus(selected),
                      )}
                    </dd>
                  </div>
                  <div>
                    <dt>负责目标</dt>
                    <dd>{trackedTargetId(frame, selected.uuv_id) ?? "—"}</dd>
                  </div>
                  <div>
                    <dt>人工锁定</dt>
                    <dd>{selected.reserved ? "是" : "否"}</dd>
                  </div>
                </dl>
                <label className="sensor-mode-control">
                  <span>人工声纳模式</span>
                  <select
                    value={selected.sensor_mode}
                    disabled={
                      !onSensorMode || selected.active_capable === false
                    }
                    onChange={(event) => {
                      const mode = event.target.value as "passive" | "active";
                      onSensorMode?.(
                        selected.uuv_id,
                        mode,
                        trackedTargetId(frame, selected.uuv_id),
                      );
                    }}
                    aria-label={`${selected.uuv_id} 人工声纳模式`}
                  >
                    <option value="passive">被动持续监听</option>
                    <option value="active">主动脉冲 + 被动持续</option>
                  </select>
                </label>
                <small className="sensor-mode-note">
                  被动声纳始终开启；主动模式仅增加选择性脉冲。
                </small>
              </section>
            )}

            <section
              className="sidebar-section compact-stats"
              aria-label="态势统计"
            >
              <div>
                <Ship size={14} />
                <span>编组</span>
                <b>{groups.length}</b>
              </div>
              <div>
                <Target size={14} />
                <span>估计</span>
                <b>{targets.length}</b>
              </div>
              <div>
                <Radio size={14} />
                <span>在线</span>
                <b>{uuvs.length - failed}</b>
              </div>
            </section>
          </CollapsiblePanel>
          <CollapsiblePanel
            title="预测与接力"
            subtitle="区域图谱与效果"
            className="command-center-panel prediction-panel"
          >
            {predictionPanel ?? (
              <p className="sidebar-panel-empty">等待区域预测与接力方案。</p>
            )}
          </CollapsiblePanel>
          <CollapsiblePanel
            title="智能助理"
            subtitle="干预与证据"
            className="command-center-panel assistant-panel"
            defaultOpen={false}
          >
            {assistantPanel ?? (
              <p className="sidebar-panel-empty">等待智能助理就绪。</p>
            )}
            {memoryPanel}
          </CollapsiblePanel>
        </>
      )}
    </aside>
  );
}

function PlanningRunStatus({
  frame,
  onRetryPlanning,
  retrying,
}: {
  frame: OperationalFrame;
  onRetryPlanning?: () => void;
  retrying: boolean;
}) {
  const phase = frame.run_phase ?? "running";
  const planning = frame.planning;
  const phaseLabel: Record<string, string> = {
    created: "已创建",
    bootstrap_planning: "启动规划中",
    awaiting_retry: "等待重试",
    running: "运行中",
    completed: "已完成",
    stopping: "正在停止",
    stopped: "已停止",
    failed: "运行失败",
  };
  const planningLabel: Record<string, string> = {
    idle: "未开始",
    queued: "排队中",
    running: "规划中",
    committed: "已提交",
    invalidated: "已失效",
    rejected: "已拒绝",
    failed: "失败",
    awaiting_retry: "等待重试",
    degraded: "降级",
  };
  return (
    <section
      className={`sidebar-section planning-run-status phase-${phase}`}
      aria-label="运行与规划状态"
    >
      <div className="section-heading">
        <span>运行与规划</span>
        <small>{phaseLabel[phase] ?? phase}</small>
      </div>
      <div className="planning-status-grid">
        <span>运行阶段</span>
        <strong>{phaseLabel[phase] ?? phase}</strong>
        <span>规划纪元</span>
        <strong>{planning ? planningLabel[planning.status] ?? planning.status : "未接入"}</strong>
        <span>方案版本</span>
        <strong>#{frame.plan_version}</strong>
        <span>数据修订</span>
        <strong>
          {planning?.base_physics_revision ?? "—"} → {planning?.current_physics_revision ?? "—"}
        </strong>
      </div>
      {planning?.epoch_id && (
        <small className="planning-epoch-id">epoch {planning.epoch_id}</small>
      )}
      {planning?.last_error && (
        <p className="planning-status-error" role="alert">
          {planning.last_error}
        </p>
      )}
      {phase === "awaiting_retry" && onRetryPlanning && (
        <button
          type="button"
          className="secondary-btn planning-retry-btn"
          onClick={onRetryPlanning}
          disabled={retrying}
        >
          <RotateCcw size={13} />
          {retrying ? "重试中" : "重试启动规划"}
        </button>
      )}
    </section>
  );
}

function TargetSubmarineBrain({
  targetId,
  currentDecision,
  detectedPlatformIds,
  decisionHistory,
}: {
  targetId: string | null | undefined;
  currentDecision: AdversaryDecisionView | null;
  detectedPlatformIds: string[];
  decisionHistory: AdversaryDecisionView[];
}) {
  return (
    <div className="target-submarine-brain" aria-label="目标潜艇脑">
      <div className="target-submarine-brain-heading">
        <span>目标潜艇脑</span>
        <small>{displayTargetName(targetId)}</small>
      </div>
      {currentDecision ? (
        <>
          <div className="adversary-intent-row">
            <span className="adversary-status-dot" />
            <strong>{currentDecision.intent || "待决策"}</strong>
            <span>{currentDecision.maneuver || "保持航迹"}</span>
            {currentDecision.decision_status === "inconclusive" && (
              <small className="adversary-estimate-badge">
                目标侧估计 · 待对手脑确认
              </small>
            )}
            {currentDecision.confidence != null && (
              <b>{Math.round(currentDecision.confidence * 100)}%</b>
            )}
          </div>
          <div className="decision-facts">
            <span>
              决策来源{" "}
              <b>
                {currentDecision.decision_source === "llm"
                  ? "目标脑意图"
                  : currentDecision.decision_source || "任务航线"}
              </b>
            </span>
            <span>
              确定性制导{" "}
              <b>
                {currentDecision.decision_source === "llm"
                  ? "意图解析"
                  : currentDecision.decision_source || "任务航线"}
              </b>
            </span>
            <span>
              制导速度{" "}
              <b>
                {currentDecision.guidance_speed_mps == null
                  ? "—"
                  : `${currentDecision.guidance_speed_mps.toFixed(1)} m/s`}
              </b>
            </span>
            <span>
              分段 <b>{currentDecision.segment || "当前水域"}</b>
            </span>
            <span>
              触发 <b>{currentDecision.trigger_event_ids?.length ?? 0} 事件</b>
            </span>
            <span>
              暴露{" "}
              <b
                className={
                  detectedPlatformIds.length ? "danger-text" : "safe-text"
                }
              >
                {detectedPlatformIds.length} 节点
              </b>
            </span>
          </div>
          <div className="decision-summary">
            <small>LLM 决策摘要</small>
            <p>
              {currentDecision.decision_summary ||
                currentDecision.rationale ||
                "目标正在根据观测维护反跟踪方案。"}
            </p>
          </div>
          {(currentDecision.escape_region_id || currentDecision.guidance_waypoint_xy) && (
            <div className="adversary-facts">
              {currentDecision.escape_region_id && (
                <span>
                  逃逸区 <b>{currentDecision.escape_region_id}</b>
                </span>
              )}
              {currentDecision.guidance_waypoint_xy && (
                <span>
                  制导航点{" "}
                  <b>
                    {currentDecision.guidance_waypoint_xy.x.toFixed(0)}, {" "}
                    {currentDecision.guidance_waypoint_xy.y.toFixed(0)}
                  </b>
                </span>
              )}
            </div>
          )}
          {detectedPlatformIds.length > 0 && (
            <div
              className="detected-badges"
              aria-label="目标已探测到的我方节点"
            >
              {detectedPlatformIds.slice(0, 8).map((id) => (
                <span key={id}>已暴露 {id}</span>
              ))}
            </div>
          )}
          <div className="adversary-facts">
            <span>
              主动声纳风险 <b>{currentDecision.active_ping_risk || "未报告"}</b>
            </span>
            <span>
              通信纪律{" "}
              <b>{currentDecision.communications_discipline || "未报告"}</b>
            </span>
          </div>
        </>
      ) : (
        <div className="adversary-empty">
          等待目标 LLM 根据观测生成反跟踪决策
        </div>
      )}
      <div className="decision-history" aria-label="目标决策历史">
        <div className="history-heading">
          <span>反跟踪历史</span>
          <small>{decisionHistory.length} 条</small>
        </div>
        {decisionHistory.slice(0, 5).map((decision) => (
          <div
            className="history-row"
            key={
              decision.decision_id ??
              `${decision.target_id}-${decision.sim_time_s}`
            }
          >
            <time>{formatSimTime(decision.sim_time_s)}</time>
            <span>{decision.intent || "—"}</span>
            <b>{decision.maneuver || "—"}</b>
          </div>
        ))}
        {!decisionHistory.length && (
          <small className="adaptive-muted">暂无动态调整记录</small>
        )}
      </div>
    </div>
  );
}

const OPERATIONAL_STAGES: Array<{ id: OperationalStage; label: string }> = [
  { id: "task_execution", label: "任务执行" },
  { id: "event_trigger", label: "事件触发" },
  { id: "human_feedback", label: "人工反馈" },
  { id: "dynamic_adjustment", label: "动态调整" },
];

function OperationalStageMatrix({ stages }: { stages: OperationalStage[] }) {
  return (
    <section className="operational-stage-matrix" aria-label="当前作业阶段">
      {OPERATIONAL_STAGES.map((item) => {
        const active = stages.includes(item.id);
        return (
          <div
            key={item.id}
            className={`operational-stage-cell ${active ? "active" : ""}`}
            aria-current={active ? "step" : undefined}
          >
            <span>{item.label}</span>
          </div>
        );
      })}
    </section>
  );
}

export function CollapsiblePanel({
  title,
  subtitle,
  children,
  defaultOpen = false,
  className = "",
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
  defaultOpen?: boolean;
  className?: string;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <details
      className={`sidebar-collapsible ${className}`}
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary>
        <span>{title}</span>
        {subtitle && <small>{subtitle}</small>}
        <ChevronDown size={14} aria-hidden="true" />
      </summary>
      <div className="sidebar-collapsible-content">{children}</div>
    </details>
  );
}

function Metric({
  label,
  value,
  emphasized,
}: {
  label: string;
  value: string;
  emphasized?: boolean;
}) {
  return (
    <div className={emphasized ? "metric emphasized" : "metric"}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export function formatSimTime(seconds: number): string {
  const hours = Math.floor(seconds / 3600)
    .toString()
    .padStart(2, "0");
  const minutes = Math.floor((seconds % 3600) / 60)
    .toString()
    .padStart(2, "0");
  const rest = Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${hours}:${minutes}:${rest}`;
}

export function formatRange(metres: number | null | undefined): string {
  if (metres == null || !Number.isFinite(metres)) return "—";
  return metres >= 1000
    ? `${(metres / 1000).toFixed(1)} km`
    : `${Math.round(metres)} m`;
}

export function uuvCommunicationStatus(
  uuv: OperationalFrame["uuvs"][number],
): CommunicationStatus {
  const explicit = uuv.communication_status ?? uuv.link_state;
  if (explicit) return explicit;
  if (uuv.master_connected === true) return "connected";
  if ((uuv.connected_peer_ids?.length ?? 0) > 0) return "degraded";
  return "unknown";
}

function communicationStatusLabel(status: CommunicationStatus): string {
  return status === "carrier"
    ? "母舰直连"
    : status === "relay"
      ? "协同链路"
      : status === "mesh"
        ? "水下组网"
        : status === "connected"
          ? "已连通"
          : status === "degraded"
            ? "经中继"
            : status === "disconnected"
              ? "已断开"
              : "未知";
}

function trackedTargetId(
  frame: OperationalFrame,
  uuvId: string,
): string | null {
  const uuv = frame.uuvs.find((candidate) => candidate.uuv_id === uuvId);
  if (!uuv) return null;
  if (uuv.tracked_target_id ?? uuv.tracked_target)
    return uuv.tracked_target_id ?? uuv.tracked_target ?? null;
  const groupId = uuv.group_id;
  return groupId
    ? (frame.groups.find((group) => group.group_id === groupId)?.target_id ??
        null)
    : null;
}

function uniqueDecisions(
  decisions: AdversaryDecisionView[],
): AdversaryDecisionView[] {
  const byIdentity = new Map<string, AdversaryDecisionView>();
  decisions.forEach((decision) => {
    byIdentity.set(
      decision.decision_id ??
        `${decision.target_id}-${decision.sim_time_s}-${decision.maneuver}`,
      decision,
    );
  });
  return [...byIdentity.values()].sort(
    (left, right) => right.sim_time_s - left.sim_time_s,
  );
}

function adversaryDecisionFromSummary(
  summary: AdversaryView | null,
): AdversaryDecisionView | null {
  if (!summary || summary.sim_time_s == null) return null;
  return {
    decision_id: summary.decision_id ?? undefined,
    target_id: summary.target_id,
    sim_time_s: summary.sim_time_s,
    intent: summary.intent ?? "待决策",
    maneuver: summary.maneuver ?? "保持航迹",
    segment: summary.segment,
    confidence: summary.confidence,
    rationale: summary.rationale ?? "目标正在根据观测维护反跟踪方案。",
    communications_discipline: summary.communications_discipline,
    trigger_event_ids: summary.trigger_event_ids ?? [],
    detected_platform_ids: summary.detected_platform_ids ?? [],
    speed_mps: summary.speed_mps,
    heading_rad: summary.heading_rad,
    decoy_count: summary.decoy_count,
    decision_status: summary.decision_status,
    escape_region_id: summary.escape_region_id,
    decision_source: summary.decision_source,
    guidance_id: summary.guidance_id,
    guidance_waypoint_xy: summary.guidance_waypoint_xy,
    guidance_speed_mps: summary.guidance_speed_mps,
    guidance_heading_rad: summary.guidance_heading_rad,
    guidance_valid_until_s: summary.guidance_valid_until_s,
    degraded_reason: summary.degraded_reason,
  };
}

function isDecision(
  decision: AdversaryDecisionView | null,
): decision is AdversaryDecisionView {
  return decision !== null;
}

function brainRoleLabel(role: "master" | "slave" | "adversary"): string {
  return role === "master" ? "主脑" : role === "slave" ? "从脑" : "对手脑";
}

function isOperationalBrainReady(brain: BrainView): boolean {
  return ["ready", "running", "succeeded", "online"].includes(brain.status);
}

function brainStatusLabel(status: BrainStatus): string {
  return status === "unconfigured"
    ? "未配置"
    : status === "ready"
      ? "待命"
      : status === "running"
        ? "运行中"
        : status === "succeeded" || status === "online"
          ? "在线"
          : status === "failed"
            ? "失败"
            : status === "paused"
              ? "暂停"
              : status === "degraded"
                ? "降级"
                : "未知";
}

function uuvGroupLabel(frame: OperationalFrame, uuvId: string): string {
  const execution = (frame.execution_groups ?? []).find((group) =>
    group.member_ids.includes(uuvId),
  );
  if (execution) {
    return execution.mode === "active_scan"
      ? `执行 ${execution.group_id}`
      : execution.mode === "passive_track"
        ? `被动 ${execution.group_id}`
        : `返航 ${execution.group_id}`;
  }
  const assignment = (frame.planned_assignments ?? []).find((candidate) =>
    candidate.uuv_ids.includes(uuvId),
  );
  if (assignment) return `计划 ${assignment.region_id}`;
  const uuv = frame.uuvs.find((candidate) => candidate.uuv_id === uuvId);
  return uuv?.deployment_state === "onboard"
    ? "计划分配"
    : uuv?.group_id ?? "未编组";
}
