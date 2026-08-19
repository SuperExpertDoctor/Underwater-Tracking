import { useRef, useState } from "react";
import { Activity, BarChart3, ClipboardList, FileCheck2, GripHorizontal, Map, Route, X } from "lucide-react";
import type { EventView, LedgerView, MetricView, MissionEventView, OperationalFrame, PlanTimelineView, TimelineFactorView } from "../types/frames";
import SegmentOverlay from "./map/SegmentOverlay";
import RegionTimelinePanel from "./RegionTimelinePanel";
import { formatSimTime } from "./RightSidebar";

const TABS = [
  { label: "时间线", icon: Activity },
  { label: "方案", icon: Map },
  { label: "事件", icon: ClipboardList },
  { label: "决策台账", icon: FileCheck2 },
  { label: "指标", icon: BarChart3 },
  { label: "分段跟踪", icon: Route },
];

const EVENT_NAMES: Record<string, string> = {
  target_found: "发现目标", target_added: "发现目标", target_lost: "目标丢失", group_updated: "编组调整",
  plan_commit: "方案提交", plan_committed: "方案提交", plan_rejected: "方案拒绝", active_ping: "主动探测",
  route_replanned: "航路重规划", directive_applied: "专家指令已应用", question: "专家质询",
  quality_warning: "质量预警", quality_critical: "质量临界", uuv_failed: "UUV 故障",
  strategic_review: "战略复盘", battery_rotation: "电量轮换",
  operational_scheme_updated: "方案更新", intelligence_report_received: "情报接收",
  target_entered_region: "目标进入区域", target_exit_predicted: "目标将离区", handoff_completed: "交接完成",
  uuv_range_exhausted: "UUV 里程耗尽", uuv_energy_depleted: "UUV 能量耗尽", region_coverage_degraded: "区域覆盖降级",
  carrier_dispatch_completed: "载体部署完成", carrier_recovery_completed: "载体回收完成",
};

const LEVEL_ORDER: Record<EventView["level"], number> = { strategic: 0, tactical: 1, informational: 2 };

interface BottomDrawerProps {
  frame: OperationalFrame | null;
  events?: EventView[];
  visible: boolean;
  onToggle: () => void;
  onSelectEvidence?: (evidenceId: string) => void;
  highlightEvidenceId?: string | null;
  selectedRegionId?: string | null;
  onSelectRegion?: (regionId: string | null) => void;
}

export default function BottomDrawer({ frame, events = [], visible, onToggle, onSelectEvidence, highlightEvidenceId, selectedRegionId, onSelectRegion }: BottomDrawerProps) {
  const [activeTab, setActiveTab] = useState(0);
  const [height, setHeight] = useState(286);
  const drag = useRef<{ y: number; height: number } | null>(null);
  if (!visible) return null;
  const frameEvents = frame ? combinedEvents(frame, events) : events;

  return (
    <section className="bottom-drawer" style={{ height }} aria-label="任务详情">
      <button
        className="drawer-grip"
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          drag.current = { y: event.clientY, height };
        }}
        onPointerMove={(event) => {
          if (!drag.current) return;
          setHeight(Math.max(190, Math.min(window.innerHeight * 0.62, drag.current.height + drag.current.y - event.clientY)));
        }}
        onPointerUp={() => { drag.current = null; }}
        aria-label="调整面板高度"
        title="拖动调整高度"
      ><GripHorizontal size={18} /></button>
      <div className="drawer-tabs" role="tablist" aria-label="任务详情标签">
        {TABS.map(({ label, icon: Icon }, index) => (
          <button key={label} role="tab" aria-selected={index === activeTab} className={index === activeTab ? "active" : ""} onClick={() => setActiveTab(index)}>
            <Icon size={14} />{label}
          </button>
        ))}
        <button className="drawer-close" onClick={onToggle} aria-label="关闭任务详情" title="关闭"><X size={16} /></button>
      </div>
      <div className="drawer-content">
        {activeTab === 0 && <TimelineTab timeline={frame?.plan_timeline ?? []} events={frameEvents} onSelectEvidence={onSelectEvidence} highlightEvidenceId={highlightEvidenceId} />}
        {activeTab === 1 && <PlanTab frame={frame} />}
        {activeTab === 2 && <EventTab events={frameEvents} />}
        {activeTab === 3 && <LedgerTab ledger={frame?.ledger ?? []} onSelectEvidence={onSelectEvidence} />}
        {activeTab === 4 && <MetricsTab metrics={frame?.metrics ?? []} />}
        {activeTab === 5 && <RegionTimelinePanel frame={frame} selectedRegionId={selectedRegionId} onSelectRegion={onSelectRegion} />}
      </div>
    </section>
  );
}

function combinedEvents(frame: OperationalFrame, supplied: EventView[]): EventView[] {
  const byId = new globalThis.Map<string, EventView>();
  [...(frame.events ?? []), ...supplied, ...(frame.mission_events ?? []).map(missionEventToEvent)].forEach((event) => byId.set(event.event_id, event));
  return [...byId.values()];
}

function missionEventToEvent(event: MissionEventView): EventView {
  const payloadReason = event.payload.reason ?? event.payload.message ?? event.payload.degraded_reason;
  const message = typeof payloadReason === "string" ? payloadReason : Object.keys(event.payload).length ? JSON.stringify(event.payload) : "—";
  return {
    event_id: event.event_id,
    sim_time_s: event.sim_time_s,
    event_type: event.event_type,
    level: event.level,
    entity_id: event.entity_id,
    message,
  };
}

function TimelineTab({ timeline, events, onSelectEvidence, highlightEvidenceId }: { timeline: PlanTimelineView[]; events: EventView[]; onSelectEvidence?: (id: string) => void; highlightEvidenceId?: string | null }) {
  if (timeline.length) {
    return <div className="plan-timeline">
      {[...timeline].sort((left, right) => right.sim_time_s - left.sim_time_s).map((item) => (
        <article className="plan-timeline-row" key={item.adjustment_id}>
          <div className="timeline-factor-column">
            {item.factors.slice(0, 8).map((factor) => <TimelineFactor key={`${item.adjustment_id}-${factor.ref_id}`} factor={factor} onSelectEvidence={onSelectEvidence} />)}
          </div>
          <div className="plan-timeline-spine"><i /><time>{formatSimTime(item.sim_time_s)}</time></div>
          <div className="timeline-result-column">
            {item.plan ? <div className="timeline-result">
              <div className="timeline-result-head"><strong>v{item.plan.version} · {item.plan.plan_id}</strong><span>{item.plan.status}</span></div>
              <p>{item.plan.summary}</p>
              {item.plan.group_changes.slice(0, 4).map((change) => <small key={change}>{change}</small>)}
            </div> : <span className="timeline-no-plan">本轮未提交新方案</span>}
          </div>
        </article>
      ))}
    </div>;
  }
  const sorted = [...events].sort((left, right) => right.sim_time_s - left.sim_time_s || LEVEL_ORDER[left.level] - LEVEL_ORDER[right.level]);
  if (!sorted.length) return <EmptyState text="暂无任务事件" />;
  return <div className="timeline-list">
    {sorted.slice(0, 160).map((event) => (
      <button className={`timeline-item level-${event.level} ${event.event_id === highlightEvidenceId ? "highlighted" : ""}`} key={event.event_id} onClick={() => onSelectEvidence?.(event.event_id)}>
        <time>{formatSimTime(event.sim_time_s)}</time><i /><strong>{EVENT_NAMES[event.event_type] ?? event.event_type}</strong>
        <span>{(event.entity_id ?? event.message) || "—"}</span>
      </button>
    ))}
  </div>;
}

function TimelineFactor({ factor, onSelectEvidence }: { factor: TimelineFactorView; onSelectEvidence?: (id: string) => void }) {
  return <button className={`timeline-factor factor-${factor.kind}`} onClick={() => onSelectEvidence?.(factor.ref_id)} title={factor.detail || factor.ref_id}>
    <span>{factor.label}</span><small>{factor.ref_id}</small>
  </button>;
}

function PlanTab({ frame }: { frame: OperationalFrame | null }) {
  const plan = frame?.plans.find((candidate) => candidate.status === "active") ?? frame?.plans[0];
  if (!frame || !plan) return <EmptyState text="等待首个已提交方案" />;
  return <div className="plan-workspace">
    <div className="plan-header-row"><div><span className="eyebrow">CURRENT PLAN</span><h3>{plan.plan_id}</h3></div><span className={`status-pill plan-${plan.status}`}>v{plan.version} · {plan.status}</span></div>
    {plan.status === "degraded" && <div className="plan-degraded-banner" role="status">LLM 候选不可用，沿用上一版计划</div>}
    <p className="plan-reason">{plan.reason || `当前方案聚焦 ${plan.affected_targets.join("、") || "已知目标"}`}</p>
    <div className="plan-facts"><span>目标 <b>{plan.affected_targets.join("、") || "—"}</b></span><span>生效 <b>{formatSimTime(plan.valid_from_s)}</b></span><span>结束 <b>{plan.valid_until_s == null ? "持续" : formatSimTime(plan.valid_until_s)}</b></span></div>
    {plan.group_changes.length > 0 && <ul className="change-list">{plan.group_changes.map((change) => <li key={change}>{change}</li>)}</ul>}
    <SegmentOverlay plans={frame.plans} />
  </div>;
}

function EventTab({ events }: { events: EventView[] }) {
  if (!events.length) return <EmptyState text="当前帧没有事件" />;
  return <div className="table-wrap"><table className="region-table"><thead><tr><th>时间</th><th>级别</th><th>事件</th><th>实体</th><th>说明</th></tr></thead><tbody>
    {[...events].sort((a, b) => b.sim_time_s - a.sim_time_s).map((event) => <tr key={event.event_id}><td>{formatSimTime(event.sim_time_s)}</td><td><span className={`level-tag level-${event.level}`}>{event.level}</span></td><td>{EVENT_NAMES[event.event_type] ?? event.event_type}</td><td>{event.entity_id ?? "—"}</td><td>{event.message || "—"}</td></tr>)}
  </tbody></table></div>;
}

function LedgerTab({ ledger, onSelectEvidence }: { ledger: LedgerView[]; onSelectEvidence?: (id: string) => void }) {
  if (!ledger.length) return <EmptyState text="暂无决策台账" />;
  return <div className="ledger-list">{[...ledger].reverse().map((row) => <article className="ledger-row" key={row.decision_id}>
    <div className="ledger-row-head"><strong>{row.decision_id}</strong><span className={`status-pill outcome-${row.outcome}`}>{row.outcome}</span><time>{formatSimTime(row.sim_time_s)}</time></div>
    <div className="ledger-row-body"><span>方案 {row.final_plan_version == null ? "未提交" : `v${row.final_plan_version}`}</span><span>触发 {row.trigger_event_ids.length} 条</span><span className="evidence-links">证据 {row.evidence_ids.length > 0 ? row.evidence_ids.map((id) => <button key={id} onClick={() => onSelectEvidence?.(id)}>{id}</button>) : "—"}</span></div>
  </article>)}</div>;
}

function MetricsTab({ metrics }: { metrics: MetricView[] }) {
  if (!metrics.length) return <EmptyState text="暂无科学指标" />;
  return <div className="metric-grid">{metrics.map((metric) => <article className="metric-card" key={metric.metric_id}>
    <div><span>{metric.label}</span><b>{metric.value.toFixed(3)} <small>{metric.unit}</small></b></div>
    <Sparkline values={metric.series} />
    <footer><span className={`metric-status metric-${(metric.status ?? "OK").toLowerCase()}`}>{metric.status ?? "OK"}</span><span>窗口 {metric.window_s}s · 趋势 {metric.trend_per_sec == null ? "—" : metric.trend_per_sec.toExponential(1)}</span></footer>
  </article>)}</div>;
}

function Sparkline({ values }: { values: number[] }) {
  const safe = values.length ? values : [0];
  const max = Math.max(...safe); const min = Math.min(...safe); const span = max - min || 1;
  return <div className="sparkline" aria-hidden="true">{safe.slice(-28).map((value, index) => <i key={`${value}-${index}`} style={{ height: `${20 + ((value - min) / span) * 80}%` }} />)}</div>;
}

function EmptyState({ text }: { text: string }) { return <div className="tab-empty">{text}</div>; }
