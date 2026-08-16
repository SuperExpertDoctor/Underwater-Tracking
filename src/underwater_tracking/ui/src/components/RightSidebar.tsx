import { type CSSProperties, type ReactNode } from "react";
import { Activity, CircleX, Radio, Ship, Target, Waves } from "lucide-react";
import type { OperationalFrame, UUVStatus } from "../types/frames";

const STATUS_LABELS: Record<UUVStatus, string> = {
  available: "待命",
  tracking: "跟踪",
  returning: "返航",
  failed: "故障",
};

const STATUS_COLORS: Record<UUVStatus, string> = {
  available: "#7e9bb8",
  tracking: "#52e3ef",
  returning: "#f6b94a",
  failed: "#ff6f7f",
};

interface RightSidebarProps {
  frame: OperationalFrame | null;
  selectedUuvId: string | null;
  onSelectUuv: (id: string | null) => void;
  open: boolean;
  onClose: () => void;
  children?: ReactNode;
}

export default function RightSidebar({
  frame,
  selectedUuvId,
  onSelectUuv,
  open,
  onClose,
  children,
}: RightSidebarProps) {
  const uuvs = frame?.uuvs ?? [];
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
    (report) => report.source === "technical_reconnaissance"
  ).length;
  const qualityFloor = scheme
    ? Object.entries(scheme.minimum_quality).sort(([left], [right]) => left.localeCompare(right))[0]
    : undefined;

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

          <section className="sidebar-section uuv-section">
            <div className="section-heading">
              <span>UUV 资源</span>
              <small>{reserved ? `${reserved} 艇已指派` : "未锁定资源"}</small>
            </div>
            <div className="uuv-list">
              {uuvs.map((uuv) => {
                const color = STATUS_COLORS[uuv.status];
                const energy = Math.round(uuv.energy_fraction * 100);
                const selectedRow = uuv.uuv_id === selectedUuvId;
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
                    </span>
                    <span
                      className="energy-gauge"
                      style={{ "--energy": `${energy}%`, "--energy-color": color } as CSSProperties}
                      aria-label={`剩余能量 ${energy}%`}
                    ><b>{energy}</b></span>
                  </button>
                );
              })}
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
                <div><dt>人工锁定</dt><dd>{selected.reserved ? "是" : "否"}</dd></div>
              </dl>
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
