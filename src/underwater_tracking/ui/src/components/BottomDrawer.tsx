import { useRef, useState } from "react";
import { Activity, BarChart3, ClipboardList, FileCheck2, GripHorizontal, Map, X } from "lucide-react";
import type { EventView, LedgerView, MetricView, OperationalFrame } from "../types/frames";
import SegmentOverlay from "./map/SegmentOverlay";
import { formatSimTime } from "./RightSidebar";

const TABS = [
  { label: "时间线", icon: Activity },
  { label: "方案", icon: Map },
  { label: "事件", icon: ClipboardList },
  { label: "决策台账", icon: FileCheck2 },
  { label: "指标", icon: BarChart3 },
];

const EVENT_NAMES: Record<string, string> = {
  target_found: "发现目标", target_added: "发现目标", target_lost: "目标丢失", group_updated: "编组调整",
  plan_commit: "方案提交", plan_committed: "方案提交", plan_rejected: "方案拒绝", active_ping: "主动探测",
  route_replanned: "航路重规划", directive_applied: "专家指令已应用", question: "专家质询",
  quality_warning: "质量预警", quality_critical: "质量临界", uuv_failed: "UUV 故障",
  strategic_review: "战略复盘", battery_rotation: "电量轮换",
  operational_scheme_updated: "方案更新", intelligence_report_received: "情报接收",
};

const LEVEL_ORDER: Record<EventView["level"], number> = { strategic: 0, tactical: 1, informational: 2 };

interface BottomDrawerProps {
  frame: OperationalFrame | null;
  events?: EventView[];
  visible: boolean;
  onToggle: () => void;
  onSelectEvidence?: (evidenceId: string) => void;
  highlightEvidenceId?: string | null;
}

export default function BottomDrawer({ frame, events = [], visible, onToggle, onSelectEvidence, highlightEvidenceId }: BottomDrawerProps) {
  const [activeTab, setActiveTab] = useState(0);
  const [height, setHeight] = useState(286);
  const drag = useRef<{ y: number; height: number } | null>(null);
  if (!visible) return null;

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
        {activeTab === 0 && <TimelineTab events={events.length ? events : frame?.events ?? []} onSelectEvidence={onSelectEvidence} highlightEvidenceId={highlightEvidenceId} />}
        {activeTab === 1 && <PlanTab frame={frame} />}
        {activeTab === 2 && <EventTab events={frame?.events ?? []} />}
        {activeTab === 3 && <LedgerTab ledger={frame?.ledger ?? []} onSelectEvidence={onSelectEvidence} />}
        {activeTab === 4 && <MetricsTab metrics={frame?.metrics ?? []} />}
      </div>
    </section>
  );
}

function TimelineTab({ events, onSelectEvidence, highlightEvidenceId }: { events: EventView[]; onSelectEvidence?: (id: string) => void; highlightEvidenceId?: string | null }) {
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

function PlanTab({ frame }: { frame: OperationalFrame | null }) {
  const plan = frame?.plans.find((candidate) => candidate.status === "active") ?? frame?.plans[0];
  if (!frame || !plan) return <EmptyState text="等待首个已提交方案" />;
  return <div className="plan-workspace">
    <div className="plan-header-row"><div><span className="eyebrow">CURRENT PLAN</span><h3>{plan.plan_id}</h3></div><span className={`status-pill plan-${plan.status}`}>v{plan.version} · {plan.status}</span></div>
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
    <footer><span>窗口 {metric.window_s}s</span><span>{metric.threshold == null ? "无阈值" : `阈值 ${metric.threshold}`}</span></footer>
  </article>)}</div>;
}

function Sparkline({ values }: { values: number[] }) {
  const safe = values.length ? values : [0];
  const max = Math.max(...safe); const min = Math.min(...safe); const span = max - min || 1;
  return <div className="sparkline" aria-hidden="true">{safe.slice(-28).map((value, index) => <i key={`${value}-${index}`} style={{ height: `${20 + ((value - min) / span) * 80}%` }} />)}</div>;
}

function EmptyState({ text }: { text: string }) { return <div className="tab-empty">{text}</div>; }
