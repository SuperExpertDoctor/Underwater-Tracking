import { type CSSProperties, type ReactNode } from "react";
import { Activity, CircleX, Link2, Radio, Ship, Target, Waves } from "lucide-react";
import type {
  AdversaryDecisionView,
  AdversaryView,
  CommunicationStatus,
  OperationalFrame,
  UUVStatus,
} from "../types/frames";

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
  onSensorMode?: (uuvId: string, mode: "passive" | "active", targetId: string | null) => void;
  children?: ReactNode;
}

export default function RightSidebar({
  frame,
  selectedUuvId,
  onSelectUuv,
  open,
  onClose,
  onSensorMode,
  children,
}: RightSidebarProps) {
  const uuvs = frame?.uuvs ?? [];
  const usvs = frame?.usvs ?? [];
  const brains = frame?.brains ?? [];
  const links = frame?.communication_links ?? [];
  const targets = frame?.target_estimates ?? [];
  const groups = frame?.groups ?? [];
  const selected = uuvs.find((uuv) => uuv.uuv_id === selectedUuvId);
  const active = uuvs.filter((uuv) => uuv.status === "tracking").length;
  const failed = uuvs.filter((uuv) => uuv.status === "failed").length;
  const reserved = uuvs.filter((uuv) => uuv.reserved).length;
  const primaryQuality = targets[0]?.quality.quality_score;
  const scheme = frame?.scheme ?? null;
  const intelligence = frame?.intelligence ?? [];
  const techIntelCount = intelligence.filter(
    (report) => report.source === "technical_reconnaissance",
  ).length;
  const qualityFloor = scheme
    ? Object.entries(scheme.minimum_quality).sort(([left], [right]) => left.localeCompare(right))[0]
    : undefined;
  const adversary = frame?.adversary ?? frame?.adversaries?.[0] ?? null;
  const target = targets[0];
  const currentDecision = adversary?.current_decision
    ?? frame?.adversary_decision
    ?? adversaryDecisionFromSummary(adversary);
  const decisionHistory = uniqueDecisions(
    [
      ...(adversary?.decision_history ?? []),
      ...(frame?.adversary_history ?? []),
      ...(frame?.adversaries ?? []).map(adversaryDecisionFromSummary).filter(isDecision),
      ...(currentDecision ? [currentDecision] : []),
    ],
  );
  const detectedPlatformIds = currentDecision?.detected_platform_ids
    ?? target?.detected_platform_ids
    ?? adversary?.detected_platform_ids
    ?? [];

  return (
    <aside className={`sidebar ${open ? "open" : ""}`} aria-label="编队态势">
      <div className="sidebar-header">
        <div>
          <span className="eyebrow">MISSION / LIVE ESTIMATE</span>
          <strong>编队态势</strong>
        </div>
        <button className="icon-btn mobile-only" onClick={onClose} aria-label="关闭编队状态" title="关闭">
          <CircleX size={17} />
        </button>
      </div>
      {!frame ? (
        <div className="sidebar-empty"><Waves size={24} /><span>等待作业态势帧</span></div>
      ) : (
        <>
          <section className="sidebar-section overview-grid" aria-label="任务概览">
            <Metric label="仿真时间" value={formatSimTime(frame.sim_time_s)} />
            <Metric label="方案版本" value={`#${frame.plan_version}`} emphasized />
            <Metric label="跟踪中" value={`${active} 艇`} />
            <Metric label="目标估计" value={`${targets.length} 个`} />
          </section>

          <section className="sidebar-section status-strip" aria-label="系统状态">
            <div><Activity size={14} /><span>质量</span><b>{primaryQuality == null ? "—" : `${(primaryQuality * 100).toFixed(0)}%`}</b></div>
            <div><Radio size={14} /><span>主动声纳</span><b>{uuvs.filter((uuv) => uuv.sensor_mode === "active").length}</b></div>
            <div><Target size={14} /><span>故障艇</span><b className={failed ? "danger-text" : ""}>{failed}</b></div>
          </section>

          <section className="sidebar-section brain-section" aria-label="主从对手脑状态">
            <div className="section-heading"><span>智能节点</span><small>{`${brains.filter((brain) => brain.status === "online").length}/${brains.length} 在线`}</small></div>
            <div className="brain-grid">
              {brains.map((brain) => (
                <div className={`brain-card brain-${brain.status}`} key={brain.brain_id}>
                  <div className="brain-card-head"><strong>{brainRoleLabel(brain.role)}</strong><span>{brainStatusLabel(brain.status)}</span></div>
                  <small>{brain.message}</small>
                  <em>{brain.last_update_s == null ? "未接入态势" : `更新 ${formatSimTime(brain.last_update_s)}`}</em>
                </div>
              ))}
            </div>
          </section>

          <section className="sidebar-section adaptive-context" aria-label="方案约束与情报">
            <div className="section-heading">
              <span>方案约束</span>
              <small>{scheme ? `有效至 ${formatSimTime(scheme.valid_until_s)}` : "未加载"}</small>
            </div>
            {scheme && qualityFloor ? (
              <strong className="adaptive-scheme-line">
                {`v${scheme.version} · ${qualityFloor[0]} 质量 ≥ ${(qualityFloor[1] * 100).toFixed(0)}%`}
              </strong>
            ) : (
              <span className="adaptive-muted">当前帧无有效作战方案</span>
            )}
            <div className="adaptive-intel-row">
              <span>情报流</span>
              <strong>{`技侦 ${techIntelCount} / 情报 ${intelligence.length}`}</strong>
            </div>
            {intelligence.length > 0 && (
              <small className="adaptive-intel-latest">
                最新 {formatSimTime(intelligence[0].issued_at_s)} · {intelligence[0].target_id} · 置信度 {(intelligence[0].confidence * 100).toFixed(0)}%
              </small>
            )}
          </section>

          <section className="sidebar-section uuv-section" aria-label="UUV 资源与底层控制状态">
            <div className="section-heading">
              <span>UUV 资源</span>
              <small>{reserved ? `${reserved} 艇已指派` : "未锁定资源"}</small>
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
                    onClick={() => onSelectUuv(selectedRow ? null : uuv.uuv_id)}
                    aria-pressed={selectedRow}
                  >
                    <span className="uuv-signal" style={{ color }}><span /></span>
                    <span className="uuv-copy">
                      <strong>{uuv.uuv_id}</strong>
                      <small>{STATUS_LABELS[uuv.status]} · {uuv.group_id ?? "未编组"}</small>
                      <span className="uuv-row-meta">
                        <span className={`link-dot link-${linkState}`}><Link2 size={10} />{communicationStatusLabel(linkState)}</span>
                        <span>{targetId ? `目标 ${targetId}` : "未绑定目标"}</span>
                      </span>
                    </span>
                    <span className="energy-gauge" style={{ "--energy": `${energy}%`, "--energy-color": color } as CSSProperties} aria-label={`剩余能量 ${energy}%`}>
                      <b>{energy}</b>
                    </span>
                  </button>
                );
              })}
              {!uuvs.length && <span className="adaptive-muted">当前帧未接入 UUV</span>}
            </div>
          </section>

          <section className="sidebar-section usv-section" aria-label="USV 水面节点">
            <div className="section-heading"><span>USV 水面节点</span><small>{`${usvs.filter((usv) => usv.connected).length}/${usvs.length} 有链路`}</small></div>
            <div className="usv-list">
              {usvs.map((usv) => (
                <div className="usv-row" key={usv.usv_id}>
                  <span className={`usv-signal ${usv.sensor_mode === "active" ? "active" : "passive"}`} />
                  <div className="usv-copy"><strong>{usv.usv_id}</strong><small>{usv.sensor_mode === "active" ? "主动声纳" : "被动声纳"} · {usv.relay_active ? "中继工作" : usv.connected ? "已接入" : "断开"}</small></div>
                  <span className="usv-range">通信 {formatRange(usvCommunicationRange(usv.communication_range_m, links, usv.usv_id))}</span>
                  <b>{Math.round(usv.energy_fraction * 100)}%</b>
                </div>
              ))}
              {!usvs.length && <small className="adaptive-muted">当前帧未接入 USV 水面节点</small>}
            </div>
            <div className="link-summary"><span>链路</span><strong>{`${links.filter((link) => link.status === "connected").length} 通 / ${links.filter((link) => link.status === "disconnected").length} 断`}</strong></div>
          </section>

          <section className="sidebar-section adversary-section" aria-label="目标潜艇反跟踪决策">
            <div className="section-heading">
              <span>目标潜艇脑</span>
              <small>{target?.target_id ?? adversary?.target_id ?? "未识别"}</small>
            </div>
            {currentDecision ? (
              <>
                <div className="adversary-intent-row">
                  <span className="adversary-status-dot" />
                  <strong>{currentDecision.intent || "待决策"}</strong>
                  <span>{currentDecision.maneuver || "保持航迹"}</span>
                  {currentDecision.confidence != null && <b>{Math.round(currentDecision.confidence * 100)}%</b>}
                </div>
                <div className="decision-facts">
                  <span>分段 <b>{currentDecision.segment || "当前水域"}</b></span>
                  <span>触发 <b>{currentDecision.trigger_event_ids?.length ?? 0} 事件</b></span>
                  <span>暴露 <b className={detectedPlatformIds.length ? "danger-text" : "safe-text"}>{detectedPlatformIds.length} 节点</b></span>
                </div>
                <div className="decision-summary">
                  <small>LLM 决策摘要</small>
                  <p>{currentDecision.decision_summary || currentDecision.rationale || "目标正在根据观测维护反跟踪方案。"}</p>
                </div>
                {detectedPlatformIds.length > 0 && (
                  <div className="detected-badges" aria-label="目标已探测到的我方节点">
                    {detectedPlatformIds.slice(0, 8).map((id) => <span key={id}>已暴露 {id}</span>)}
                  </div>
                )}
                <div className="adversary-facts">
                  <span>主动声纳风险 <b>{currentDecision.active_ping_risk || "未报告"}</b></span>
                  <span>通信纪律 <b>{currentDecision.communications_discipline || "未报告"}</b></span>
                </div>
              </>
            ) : (
              <div className="adversary-empty">等待目标 LLM 根据观测生成反跟踪决策</div>
            )}
            <div className="decision-history" aria-label="目标决策历史">
              <div className="history-heading"><span>反跟踪历史</span><small>{decisionHistory.length} 条</small></div>
              {decisionHistory.slice(0, 5).map((decision) => (
                <div className="history-row" key={decision.decision_id ?? `${decision.target_id}-${decision.sim_time_s}`}>
                  <time>{formatSimTime(decision.sim_time_s)}</time>
                  <span>{decision.intent || "—"}</span>
                  <b>{decision.maneuver || "—"}</b>
                </div>
              ))}
              {!decisionHistory.length && <small className="adaptive-muted">暂无动态调整记录</small>}
            </div>
          </section>

          {selected && (
            <section className="sidebar-section selected-detail" aria-label={`${selected.uuv_id} 详情`}>
              <div className="section-heading">
                <span>{selected.uuv_id} 详情</span>
                <small>{Math.round((selected.heading_rad * 180) / Math.PI)}° 航向</small>
              </div>
              <dl>
                <div><dt>坐标</dt><dd>{selected.position.x.toFixed(0)}, {selected.position.y.toFixed(0)} m</dd></div>
                <div><dt>速度</dt><dd>{selected.speed_mps.toFixed(1)} m/s</dd></div>
                <div><dt>编组</dt><dd>{selected.group_id ?? "—"}</dd></div>
                <div><dt>传感器</dt><dd>{selected.sensor_mode === "active" ? "主动声纳" : "被动声纳"}</dd></div>
                <div><dt>剩余续航</dt><dd>{formatRange(selected.remaining_range_m ?? selected.endurance_remaining_m)}</dd></div>
                <div><dt>通信链路</dt><dd className={`value-${uuvCommunicationStatus(selected)}`}>{communicationStatusLabel(uuvCommunicationStatus(selected))}</dd></div>
                <div><dt>负责目标</dt><dd>{trackedTargetId(frame, selected.uuv_id) ?? "—"}</dd></div>
                <div><dt>人工锁定</dt><dd>{selected.reserved ? "是" : "否"}</dd></div>
              </dl>
              <label className="sensor-mode-control">
                <span>人工声纳模式</span>
                <select
                  value={selected.sensor_mode}
                  disabled={!onSensorMode || selected.active_capable === false}
                  onChange={(event) => {
                    const mode = event.target.value as "passive" | "active";
                    onSensorMode?.(selected.uuv_id, mode, trackedTargetId(frame, selected.uuv_id));
                  }}
                  aria-label={`${selected.uuv_id} 人工声纳模式`}
                >
                  <option value="passive">被动持续监听</option>
                  <option value="active">主动脉冲 + 被动持续</option>
                </select>
              </label>
              <small className="sensor-mode-note">被动声纳始终开启；主动模式仅增加选择性脉冲。</small>
            </section>
          )}

          <section className="sidebar-section compact-stats" aria-label="态势统计">
            <div><Ship size={14} /><span>编组</span><b>{groups.length}</b></div>
            <div><Target size={14} /><span>估计</span><b>{targets.length}</b></div>
            <div><Radio size={14} /><span>在线</span><b>{uuvs.length - failed}</b></div>
          </section>
          {children && <div className="sidebar-assistant">{children}</div>}
        </>
      )}
    </aside>
  );
}

function Metric({ label, value, emphasized }: { label: string; value: string; emphasized?: boolean }) {
  return <div className={emphasized ? "metric emphasized" : "metric"}><span>{label}</span><strong>{value}</strong></div>;
}

export function formatSimTime(seconds: number): string {
  const hours = Math.floor(seconds / 3600).toString().padStart(2, "0");
  const minutes = Math.floor((seconds % 3600) / 60).toString().padStart(2, "0");
  const rest = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${hours}:${minutes}:${rest}`;
}

export function formatRange(metres: number | null | undefined): string {
  if (metres == null || !Number.isFinite(metres)) return "—";
  return metres >= 1000 ? `${(metres / 1000).toFixed(1)} km` : `${Math.round(metres)} m`;
}

export function uuvCommunicationStatus(uuv: OperationalFrame["uuvs"][number]): CommunicationStatus {
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
      ? "USV 中继"
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

function trackedTargetId(frame: OperationalFrame, uuvId: string): string | null {
  const uuv = frame.uuvs.find((candidate) => candidate.uuv_id === uuvId);
  if (!uuv) return null;
  if (uuv.tracked_target_id ?? uuv.tracked_target) return uuv.tracked_target_id ?? uuv.tracked_target ?? null;
  const groupId = uuv.group_id;
  return groupId ? frame.groups.find((group) => group.group_id === groupId)?.target_id ?? null : null;
}

function linkRange(links: OperationalFrame["communication_links"], usvId: string): number {
  const ranges = (links ?? [])
    .filter((link) => link.medium === "surface" && (link.source_id === usvId || link.target_id === usvId))
    .map((link) => link.limit_m)
    .filter((range): range is number => Number.isFinite(range));
  return Math.max(0, ...ranges);
}

function usvCommunicationRange(
  configuredRange: number | null | undefined,
  links: OperationalFrame["communication_links"],
  usvId: string,
): number {
  return configuredRange != null && Number.isFinite(configuredRange) && configuredRange > 1
    ? configuredRange
    : linkRange(links, usvId);
}

function uniqueDecisions(decisions: AdversaryDecisionView[]): AdversaryDecisionView[] {
  const byIdentity = new Map<string, AdversaryDecisionView>();
  decisions.forEach((decision) => {
    byIdentity.set(decision.decision_id ?? `${decision.target_id}-${decision.sim_time_s}-${decision.maneuver}`, decision);
  });
  return [...byIdentity.values()].sort((left, right) => right.sim_time_s - left.sim_time_s);
}

function adversaryDecisionFromSummary(summary: AdversaryView | null): AdversaryDecisionView | null {
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
  };
}

function isDecision(decision: AdversaryDecisionView | null): decision is AdversaryDecisionView {
  return decision !== null;
}

function brainRoleLabel(role: "master" | "slave" | "adversary"): string {
  return role === "master" ? "主脑" : role === "slave" ? "从脑" : "对手脑";
}

function brainStatusLabel(status: "online" | "paused" | "degraded" | "unknown"): string {
  return status === "online" ? "在线" : status === "paused" ? "暂停" : status === "degraded" ? "降级" : "未知";
}
