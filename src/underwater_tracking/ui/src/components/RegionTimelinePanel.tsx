import { useEffect, useMemo, useState } from "react";
import type { OperationalFrame, RegionTimelineView } from "../types/frames";
import RegionTimelineRow from "./RegionTimelineRow";
import { formatOffset, offsetPercent, sortRegionTimeline, STATUS_LABELS, timelineWindow } from "./regionTimeline";

interface RegionTimelinePanelProps {
  frame: OperationalFrame | null;
}

function axisTicks(start: number, end: number): number[] {
  const span = end - start;
  const step = span <= 300 ? 60 : span <= 900 ? 120 : 300;
  const ticks: number[] = [];
  for (let value = Math.ceil(start / step) * step; value <= end; value += step) ticks.push(value);
  if (!ticks.includes(0)) ticks.push(0);
  return ticks.sort((left, right) => left - right);
}

function AssignmentList({ row }: { row: RegionTimelineView }) {
  const assignments = [...row.uuv_assignments, ...row.usv_assignments];
  return <div className="region-assignment-list">
    {assignments.map((assignment) => <span className={`region-assignment-chip ${assignment.platform_kind}`} key={`${assignment.platform_kind}-${assignment.platform_id}`}>
      {assignment.platform_id} · {assignment.role}
    </span>)}
  </div>;
}

function RegionDetail({ row }: { row: RegionTimelineView }) {
  return <section className="region-timeline-detail" aria-label="区域详情">
    <div className="region-detail-header"><strong>{row.region_id}</strong><span className={`region-status status-${row.status}`}>{STATUS_LABELS[row.status]}</span></div>
    <div className="region-detail-facts">
      <span>时间 <b>{formatOffset(row.start_offset_s)} → {formatOffset(row.end_offset_s)}</b></span>
      <span>中心 <b>({row.center.x.toFixed(0)}, {row.center.y.toFixed(0)}) m</b></span>
      <span>优先级 <b>{row.priority.toFixed(2)}</b></span>
      <span>计划 <b>v{row.plan_revision}</b></span>
    </div>
    <AssignmentList row={row} />
    {(row.handoff_from || row.handoff_to) && <p className="region-handoff-detail">接力：{row.handoff_from ?? "起始"} → {row.handoff_to ?? "结束"}</p>}
    {row.degraded_reasons.length > 0 && <p className="region-degraded-detail">原因：{row.degraded_reasons.join("、")}</p>}
  </section>;
}

export default function RegionTimelinePanel({ frame }: RegionTimelinePanelProps) {
  const rows = useMemo(() => sortRegionTimeline(frame?.region_timeline ?? []), [frame?.region_timeline]);
  const window = useMemo(() => timelineWindow(rows), [rows]);
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);
  const selected = rows.find((row) => row.region_id === selectedRegionId) ?? rows[0] ?? null;

  useEffect(() => {
    if (selectedRegionId && !rows.some((row) => row.region_id === selectedRegionId)) setSelectedRegionId(null);
  }, [rows, selectedRegionId]);

  if (!rows.length) return <section className="region-timeline-panel" aria-label="区域分段跟踪甘特图"><div className="region-timeline-empty">当前暂无区域任务</div></section>;

  const ticks = axisTicks(window.start, window.end);
  return <section className="region-timeline-panel" aria-label="区域分段跟踪甘特图">
    <header className="region-timeline-header"><div><span className="eyebrow">REGION HANDOFF</span><h3>分段跟踪</h3></div><span className="region-timeline-origin">T+0 · {frame?.sim_time_s ?? 0}s</span></header>
    <div className="region-timeline-scroll">
      <div className="region-timeline-axis"><span className="region-timeline-axis-label">区域 / 状态</span><span className="region-timeline-axis-track">{ticks.map((tick) => <i key={tick} style={{ left: `${offsetPercent(tick, window.start, window.end)}%` }}>{formatOffset(tick)}</i>)}</span></div>
      <div className="region-timeline-current" style={{ left: `calc(118px + (100% - 118px) * ${offsetPercent(0, window.start, window.end) / 100})` }} aria-hidden="true" />
      <div className="region-timeline-rows">{rows.map((row) => <RegionTimelineRow key={row.region_id} row={row} window={window} selected={row.region_id === selected?.region_id} onSelect={() => setSelectedRegionId(row.region_id)} />)}</div>
    </div>
    {selected && <RegionDetail row={selected} />}
    <footer className="region-timeline-legend"><span><i className="status-active" />当前覆盖</span><span><i className="status-degraded" />降级</span><span><i className="status-handed_off" />交接</span><span><i className="status-uncovered" />未覆盖</span></footer>
  </section>;
}
