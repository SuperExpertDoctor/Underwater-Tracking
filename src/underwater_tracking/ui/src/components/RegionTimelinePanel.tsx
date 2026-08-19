import { useEffect, useMemo, useState } from "react";
import type { OperationalFrame, RegionalMissionView, RegionAssignmentView, RegionTimelineView } from "../types/frames";
import RegionTimelineRow from "./RegionTimelineRow";
import { formatOffset, offsetPercent, sortRegionTimeline, STATUS_LABELS, timelineWindow } from "./regionTimeline";

interface RegionTimelinePanelProps {
  frame: OperationalFrame | null;
  selectedRegionId?: string | null;
  onSelectRegion?: (regionId: string | null) => void;
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

function missionStatus(mission: RegionalMissionView): RegionTimelineView["status"] {
  switch (mission.lifecycle) {
    case "ACTIVE_SCAN":
    case "PASSIVE_TRACK": return "active";
    case "HANDOFF_PENDING": return "handed_off";
    case "DEGRADED": return "degraded";
    case "UNCOVERED": return "uncovered";
    default: return "planned";
  }
}

function missionCenter(mission: RegionalMissionView) {
  if (!mission.geometry.length) return { x: 0, y: 0 };
  const sum = mission.geometry.reduce((total, point) => ({ x: total.x + point.x, y: total.y + point.y }), { x: 0, y: 0 });
  return { x: sum.x / mission.geometry.length, y: sum.y / mission.geometry.length };
}

function missionBounds(mission: RegionalMissionView) {
  const points = mission.geometry.length ? mission.geometry : [{ x: 0, y: 0 }];
  return points.reduce((bounds, point) => ({
    min_x: Math.min(bounds.min_x, point.x),
    min_y: Math.min(bounds.min_y, point.y),
    max_x: Math.max(bounds.max_x, point.x),
    max_y: Math.max(bounds.max_y, point.y),
  }), { min_x: points[0].x, min_y: points[0].y, max_x: points[0].x, max_y: points[0].y });
}

function missionAssignments(mission: RegionalMissionView, start: number, end: number): RegionAssignmentView[] {
  const assignments: RegionAssignmentView[] = [];
  mission.active_scan_uuv_ids.forEach((platformId) => assignments.push({
    platform_id: platformId,
    platform_kind: "uuv",
    role: "主动扫描",
    start_offset_s: start,
    end_offset_s: end,
    sonar_mode: "active",
  }));
  mission.passive_track_uuv_ids.forEach((platformId) => assignments.push({
    platform_id: platformId,
    platform_kind: "uuv",
    role: "被动跟踪",
    start_offset_s: start,
    end_offset_s: end,
    sonar_mode: "passive",
  }));
  mission.reserve_uuv_ids.forEach((platformId) => assignments.push({
    platform_id: platformId,
    platform_kind: "uuv",
    role: "交接储备",
    start_offset_s: start,
    end_offset_s: end,
    sonar_mode: "passive",
  }));
  return assignments;
}

export function regionalMissionsToTimeline(frame: OperationalFrame): RegionTimelineView[] {
  const simTime = frame.sim_time_s ?? 0;
  return (frame.regional_missions ?? []).map((mission) => {
    const start = mission.entry_s - simTime;
    const end = mission.exit_s - simTime;
    return {
      region_id: mission.region_id,
      target_id: mission.target_id,
      center: missionCenter(mission),
      bounds: missionBounds(mission),
      start_offset_s: start,
      end_offset_s: end,
      status: missionStatus(mission),
      coverage_mode: "required",
      priority: mission.coverage,
      occupancy_likelihood: mission.tracking_quality,
      uuv_assignments: missionAssignments(mission, start, end),
      usv_assignments: [],
      communication_links: [],
      handoff_from: mission.handoff_from,
      handoff_to: mission.handoff_to,
      evidence_ids: [],
      degraded_reasons: mission.degraded_reasons,
      plan_revision: mission.plan_revision,
    };
  });
}

export default function RegionTimelinePanel({ frame, selectedRegionId: controlledRegionId, onSelectRegion }: RegionTimelinePanelProps) {
  const rows = useMemo(() => {
    if (!frame) return [];
    const timeline = frame.region_timeline?.length ? frame.region_timeline : regionalMissionsToTimeline(frame);
    return sortRegionTimeline(timeline);
  }, [frame]);
  const window = useMemo(() => timelineWindow(rows), [rows]);
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);
  const regionSelectionIsControlled = controlledRegionId !== undefined;
  const activeRegionId = regionSelectionIsControlled ? controlledRegionId : selectedRegionId;
  const selected = rows.find((row) => row.region_id === activeRegionId) ?? (regionSelectionIsControlled ? null : rows[0] ?? null);

  useEffect(() => {
    if (!regionSelectionIsControlled && activeRegionId && !rows.some((row) => row.region_id === activeRegionId)) setSelectedRegionId(null);
  }, [activeRegionId, regionSelectionIsControlled, rows]);

  if (!rows.length) return <section className="region-timeline-panel" aria-label="区域分段跟踪甘特图"><div className="region-timeline-empty">当前暂无区域任务</div></section>;

  const ticks = axisTicks(window.start, window.end);
  return <section className="region-timeline-panel" aria-label="区域分段跟踪甘特图">
    <header className="region-timeline-header"><div><span className="eyebrow">REGION HANDOFF</span><h3>分段跟踪</h3></div><span className="region-timeline-origin">T+0 · {frame?.sim_time_s ?? 0}s</span></header>
    <div className="region-timeline-scroll">
      <div className="region-timeline-axis"><span className="region-timeline-axis-label">区域 / 状态</span><span className="region-timeline-axis-track">{ticks.map((tick) => <i key={tick} style={{ left: `${offsetPercent(tick, window.start, window.end)}%` }}>{formatOffset(tick)}</i>)}</span></div>
      <div className="region-timeline-current" style={{ left: `calc(118px + (100% - 118px) * ${offsetPercent(0, window.start, window.end) / 100})` }} aria-hidden="true" />
      <div className="region-timeline-rows">{rows.map((row) => <RegionTimelineRow key={row.region_id} row={row} window={window} selected={row.region_id === selected?.region_id} onSelect={() => {
        const nextRegionId = activeRegionId === row.region_id ? null : row.region_id;
        if (!regionSelectionIsControlled) setSelectedRegionId(nextRegionId);
        onSelectRegion?.(nextRegionId);
      }} />)}</div>
    </div>
    {selected && <RegionDetail row={selected} />}
    <footer className="region-timeline-legend"><span><i className="status-active" />当前覆盖</span><span><i className="status-degraded" />降级</span><span><i className="status-handed_off" />交接</span><span><i className="status-uncovered" />未覆盖</span></footer>
  </section>;
}
