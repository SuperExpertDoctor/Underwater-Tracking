import { useRef, useState } from "react";
import {
  Activity,
  Bot,
  ClipboardList,
  GripHorizontal,
  Map,
  SlidersHorizontal,
  X,
} from "lucide-react";
import type { EventView, OperationalFrame } from "../frameTypes";

const TABS = [
  { label: "时间线", icon: Activity },
  { label: "方案", icon: Map },
  { label: "事件", icon: ClipboardList },
  { label: "决策周期", icon: Bot },
  { label: "参数", icon: SlidersHorizontal },
];

const EVENT_NAMES: Record<string, string> = {
  target_found: "发现目标",
  target_lost: "目标丢失",
  group_updated: "编组调整",
  plan_committed: "方案提交",
  plan_rejected: "方案拒绝",
  uuv_returned: "UUV 返航",
  uuv_refueled: "能源补给完成",
  search_complete: "搜索完成",
  plan_decision: "方案决策",
  route_plan_failed: "航路失败",
  route_replanned: "航路重规划",
  environment_reset: "环境重置",
};

interface BottomDrawerProps {
  frame: OperationalFrame | null;
  events?: EventView[];
  planCycle: OperationalFrame["plan_cycle"];
  visible: boolean;
  onToggle: () => void;
}

/**
 * Bottom detail drawer (migrated from the reference project's
 * BottomDrawer; component boundary preserved).  The shell keeps the
 * tabbed drawer with the timeline of operator events; the plan, event,
 * ledger and metric tabs are built out in Task 8.
 */
export default function BottomDrawer({
  frame,
  events = [],
  planCycle,
  visible,
  onToggle,
}: BottomDrawerProps) {
  const [activeTab, setActiveTab] = useState(0);
  const [height, setHeight] = useState(220);
  const drag = useRef<{ y: number; height: number } | null>(null);

  if (!visible) return null;
  return (
    <section className="bottom-drawer" style={{ height }} aria-label="任务详情">
      <button
        className="drawer-grip"
        onPointerDown={(event) => { drag.current = { y: event.clientY, height }; }}
        onPointerMove={(event) => {
          if (!drag.current) return;
          setHeight(Math.max(180, Math.min(window.innerHeight * 0.58, drag.current.height + drag.current.y - event.clientY)));
        }}
        onPointerUp={() => { drag.current = null; }}
        aria-label="调整面板高度"
        title="拖动调整高度"
      >
        <GripHorizontal size={20} />
      </button>
      <div className="drawer-tabs" role="tablist">
        {TABS.map(({ label, icon: Icon }, index) => (
          <button
            key={label}
            role="tab"
            aria-selected={index === activeTab}
            className={index === activeTab ? "active" : ""}
            onClick={() => setActiveTab(index)}
          >
            <Icon size={15} />{label}
          </button>
        ))}
        <button className="drawer-close" onClick={onToggle} aria-label="关闭任务详情" title="关闭">
          <X size={16} />
        </button>
      </div>
      <div className="drawer-content">
        {activeTab === 0 && <TimelineTab events={events} />}
        {activeTab === 1 && <PlanTab frame={frame} />}
        {activeTab === 2 && <EventTab frame={frame} />}
        {activeTab === 3 && <DecisionTab planCycle={planCycle} />}
        {activeTab === 4 && <ParamsTab frame={frame} />}
      </div>
    </section>
  );
}

function TimelineTab({ events }: { events: EventView[] }) {
  if (!events.length) return <EmptyState text="暂无任务事件" />;
  return (
    <div className="timeline-list">
      {[...events].reverse().slice(0, 120).map((event, index) => (
        <div className={`timeline-item event-${event.type}`} key={`${event.time}-${event.type}-${index}`}>
          <time>{Number(event.time || 0).toFixed(0).padStart(3, "0")} min</time>
          <i />
          <strong>{EVENT_NAMES[event.type] || event.type}</strong>
          <span>{String(event.data?.group_id ?? event.data?.target_id ?? "")}</span>
        </div>
      ))}
    </div>
  );
}

function PlanTab({ frame }: { frame: OperationalFrame | null }) {
  if (!frame) return <EmptyState text="等待任务数据" />;
  return (
    <div className="table-wrap">
      <table className="region-table">
        <thead>
          <tr><th>方案版本</th><th>UUV</th><th>目标</th><th>编组</th><th>事件</th></tr>
        </thead>
        <tbody>
          <tr>
            <td><b>#{frame.plan_version}</b></td>
            <td>{frame.uuvs.length}</td>
            <td>{frame.targets.length}</td>
            <td>{frame.groups.length}</td>
            <td>{frame.events.length}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function EventTab({ frame }: { frame: OperationalFrame | null }) {
  if (!frame) return <EmptyState text="等待任务数据" />;
  return <EmptyState text={`本帧含 ${frame.events.length} 条事件；事件工作区将在后续任务中展开`} />;
}

function DecisionTab({ planCycle }: { planCycle: OperationalFrame["plan_cycle"] }) {
  if (!planCycle) return <EmptyState text="等待首次方案决策" />;
  return (
    <div className="llm-log">
      <div className="llm-log-head">
        <div>
          <span className={planCycle.success ? "success" : "failed"}>
            {planCycle.success ? "VALID" : "FAILED"}
          </span>
          <strong>{planCycle.model}</strong>
        </div>
        <small>{planCycle.attempts || 0} attempts · 方案 #{planCycle.plan_version}</small>
      </div>
      <div className="llm-sections">
        <details open>
          <summary>Response</summary>
          <pre>{planCycle.response || "无内容"}</pre>
        </details>
      </div>
    </div>
  );
}

function ParamsTab({ frame }: { frame: OperationalFrame | null }) {
  if (!frame) return <EmptyState text="等待任务数据" />;
  return (
    <div className="params-grid">
      <section><h3>帧</h3>
        <div><span>schema_version</span><b>{frame.schema_version}</b></div>
        <div><span>frame_id</span><b>{frame.frame_id}</b></div>
        <div><span>sim_time_s</span><b>{frame.sim_time_s}</b></div>
        <div><span>plan_version</span><b>{frame.plan_version}</b></div>
      </section>
    </div>
  );
}

function EmptyState({ text }: { text: string }) {
  return <div className="tab-empty">{text}</div>;
}
