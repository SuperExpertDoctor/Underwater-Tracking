import type { RegionTimelineView } from "../types/frames";

export interface TimelineWindow {
  start: number;
  end: number;
}

export function sortRegionTimeline(rows: RegionTimelineView[]): RegionTimelineView[] {
  return [...rows].sort((left, right) => left.start_offset_s - right.start_offset_s || left.region_id.localeCompare(right.region_id));
}

export function timelineWindow(rows: RegionTimelineView[], horizon = 600): TimelineWindow {
  const maxEnd = rows.reduce((max, row) => Math.max(max, row.end_offset_s), horizon);
  const minStart = rows.reduce((min, row) => Math.min(min, row.start_offset_s), 0);
  return { start: Math.min(0, minStart), end: Math.max(horizon, maxEnd) };
}

export function offsetPercent(offset: number, start: number, end: number): number {
  if (end <= start) return 0;
  return Math.max(0, Math.min(100, ((offset - start) / (end - start)) * 100));
}

export function formatOffset(offset: number): string {
  if (offset === 0) return "T+0";
  return offset > 0 ? `T+${Math.round(offset)}s` : `T${Math.round(offset)}s`;
}

export const STATUS_LABELS: Record<RegionTimelineView["status"], string> = {
  planned: "计划",
  active: "当前覆盖",
  handed_off: "已交接",
  degraded: "降级",
  uncovered: "未覆盖",
};
