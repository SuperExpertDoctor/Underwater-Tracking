import { type CSSProperties } from "react";
import { Bot, CircleX, Plane, Radar, Ship, Target, Waypoints } from "lucide-react";
import type { OperationalFrame, UUVState } from "../frameTypes";

const STATUS_LABELS: Record<UUVState, string> = {
  idle: "待命",
  transit: "转场",
  searching: "搜索",
  tracking: "跟踪",
  returning: "返航",
  refueling: "能源补给",
  holding: "保持",
};

const STATUS_COLORS: Record<UUVState, string> = {
  searching: "#0F766E",
  tracking: "#BE123C",
  returning: "#C2410C",
  refueling: "#0369A1",
  holding: "#A16207",
  idle: "#475569",
  transit: "#1D4ED8",
};

interface RightSidebarProps {
  frame: OperationalFrame | null;
  selectedUuvId: string | null;
  onSelectUuv: (id: string | null) => void;
  open: boolean;
  onClose: () => void;
  lastPlanCycle: OperationalFrame["plan_cycle"];
}

/**
 * Right command sidebar (migrated from the reference project's
 * RightSidebar; component boundary and data flow preserved).  Shows the
 * current snapshot: simulation time, plan version, coverage, estimated
 * targets, UUV units and the selected UUV's detail.  Full quality/metric
 * workspaces are added in Task 8.
 */
export default function RightSidebar({
  frame,
  selectedUuvId,
  onSelectUuv,
  open,
  onClose,
  lastPlanCycle,
}: RightSidebarProps) {
  const uuvs = frame?.uuvs || [];
  const targets = frame?.targets || [];
  const groups = frame?.groups || [];
  const selected = uuvs.find((uuv) => uuv.id === selectedUuvId);

  return (
    <aside className={`sidebar ${open ? "open" : ""}`} aria-label="编队态势">
      <div className="sidebar-header">
        <div><span className="eyebrow">MISSION STATE</span><strong>编队态势</strong></div>
        <button className="icon-btn mobile-only" onClick={onClose} aria-label="关闭编队状态" title="关闭">
          <CircleX size={17} />
        </button>
      </div>
      {!frame ? (
        <div className="sidebar-empty"><Radar size={24} /><span>等待任务数据</span></div>
      ) : (
        <>
          <section className="sidebar-section overview-grid" aria-label="任务概览">
            <Metric label="仿真时间" value={frame.timestamp || "--:--:--"} />
            <Metric label="方案版本" value={`#${frame.plan_version ?? 0}`} />
            <Metric
              label="海域覆盖"
              value={frame.coverage_pct != null ? `${frame.coverage_pct.toFixed(1)}%` : "--"}
              emphasized
            />
            <Metric label="目标估计" value={`${targets.length}`} />
          </section>

          <section className="sidebar-section uuv-section">
            <div className="section-heading">
              <span>UUV 单元</span>
              <small>{uuvs.filter((uuv) => uuv.status !== "idle").length} ACTIVE</small>
            </div>
            <div className="uuv-list">
              {uuvs.map((uuv) => {
                const color = STATUS_COLORS[uuv.status] || "#94A3B8";
                const energy = Math.max(0, Math.min(100, (uuv.energy_remaining_pct ?? 0) * 100));
                const isSelected = uuv.id === selectedUuvId;
                return (
                  <button
                    key={uuv.id}
                    className={`uuv-row ${isSelected ? "selected" : ""}`}
                    onClick={() => onSelectUuv(isSelected ? null : uuv.id)}
                    aria-pressed={isSelected}
                  >
                    <span className="uuv-plane" style={{ color }}><Plane size={16} /></span>
                    <span className="uuv-copy">
                      <strong>{uuv.id}</strong>
                      <small>{STATUS_LABELS[uuv.status] || uuv.status} · {uuv.group_id || "未编组"}</small>
                    </span>
                    <span className="energy-gauge" style={{ "--energy": `${energy}%`, "--energy-color": color } as CSSProperties}>
                      <b>{Math.round(energy)}</b>
                    </span>
                  </button>
                );
              })}
            </div>
          </section>

          {selected && (
            <section className="sidebar-section selected-detail">
              <div className="section-heading">
                <span>{selected.id} 详情</span>
                <small>{Math.round(selected.heading_deg || 0)}°</small>
              </div>
              <dl>
                <div><dt>坐标</dt><dd>{selected.position.map((value) => Number(value).toFixed(1)).join(", ")}</dd></div>
                <div><dt>剩余能量</dt><dd>{Math.round((selected.energy_remaining_pct ?? 0) * 100)}%</dd></div>
                <div><dt>编组</dt><dd>{selected.group_id || "-"}</dd></div>
                <div><dt>当前方案</dt><dd>{selected.active_plan_id || "-"}</dd></div>
              </dl>
            </section>
          )}

          <section className="sidebar-section compact-stats">
            <div><Waypoints size={15} /><span>方案</span><b>{frame.plan_version ?? 0}</b></div>
            <div><Target size={15} /><span>目标</span><b>{targets.length}</b></div>
            <div><Ship size={15} /><span>编组</span><b>{groups.length}</b></div>
          </section>

          <section className="sidebar-section llm-summary">
            <div className="section-heading">
              <span><Bot size={15} />决策周期</span>
              <small>{lastPlanCycle?.model || "PLAN-ENGINE"}</small>
            </div>
            {lastPlanCycle ? (
              <div className="llm-status-row">
                <span className={lastPlanCycle.success ? "success" : "failed"}>
                  {lastPlanCycle.success ? "校验通过" : "决策失败"}
                </span>
                <b>{lastPlanCycle.attempts || 0} 次请求</b>
              </div>
            ) : <p>等待首次方案决策</p>}
          </section>
        </>
      )}
    </aside>
  );
}

function Metric({ label, value, emphasized }: { label: string; value: string; emphasized?: boolean }) {
  return <div className={emphasized ? "metric emphasized" : "metric"}><span>{label}</span><strong>{value}</strong></div>;
}
