import type {
  OperationalFrame,
  RegionTimelineView,
} from "../types/frames";
import {
  groupInstanceId,
  groupsByRegionSlot,
} from "../state/executionSelectors";

export interface TimelineWindow {
  start: number;
  end: number;
}

export function sortRegionTimeline(rows: RegionTimelineView[]): RegionTimelineView[] {
  return [...rows].sort((left, right) => left.start_offset_s - right.start_offset_s || left.region_id.localeCompare(right.region_id));
}

function polygonBounds(points: { x: number; y: number }[]) {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  return {
    min_x: Math.min(...xs),
    min_y: Math.min(...ys),
    max_x: Math.max(...xs),
    max_y: Math.max(...ys),
  };
}

function polygonCenter(points: { x: number; y: number }[]) {
  const total = points.reduce(
    (sum, point) => ({ x: sum.x + point.x, y: sum.y + point.y }),
    { x: 0, y: 0 },
  );
  return { x: total.x / points.length, y: total.y / points.length };
}

function executionStatus(
  status: NonNullable<OperationalFrame["execution"]>["regions"][number]["status"],
): RegionTimelineView["status"] {
  if (status === "active" || status === "passive") return "active";
  if (status === "handoff_pending" || status === "handoff_completed") return "handed_off";
  if (status === "degraded") return "degraded";
  if (status === "uncovered") return "uncovered";
  return "planned";
}

function executionTimelineRows(frame: OperationalFrame): RegionTimelineView[] {
  const execution = frame.execution;
  if (!execution) return [];
  const groupsByRegion = groupsByRegionSlot(frame);
  return [...execution.regions]
    .sort((left, right) => left.slot_index - right.slot_index)
    .map((region) => {
      const groups = groupsByRegion.get(region.region_id) ?? [];
      const runtimeGroups = groups
        .filter((group) => group.lifecycle !== "disappeared")
        .sort((left, right) =>
          left.deployment_revision - right.deployment_revision
          || groupInstanceId(left).localeCompare(groupInstanceId(right)));
      const geometry = region.geometry;
      const assignments = runtimeGroups.flatMap((runtimeGroup) =>
        runtimeGroup.member_uuv_ids.map((platform_id) => {
          const active = runtimeGroup.sensor_mode === "active";
          return {
            platform_id,
            platform_kind: "uuv" as const,
            role: active ? "active_verifier" : "passive_tracker",
            start_offset_s: region.start_s - frame.sim_time_s,
            end_offset_s: region.end_s - frame.sim_time_s,
            sonar_mode: active ? "active" as const : "passive" as const,
          };
        }),
      );
      const displayGroup = runtimeGroups.find(
        (candidate) => candidate.lifecycle !== "exiting",
      ) ?? runtimeGroups[0];
      return {
        region_id: region.region_id,
        target_id: region.target_id,
        center: polygonCenter(geometry),
        bounds: polygonBounds(geometry),
        start_offset_s: region.start_s - frame.sim_time_s,
        end_offset_s: region.end_s - frame.sim_time_s,
        status: executionStatus(region.status),
        coverage_mode: "required" as const,
        priority: 1,
        occupancy_likelihood: 1,
        uuv_assignments: assignments,
        communication_links: [],
        handoff_from: region.predecessor_region_id,
        handoff_to: region.successor_region_id,
        evidence_ids: region.evidence_ids,
        degraded_reasons: execution.degradation_reasons,
        plan_revision: execution.execution_revision,
        task_group_ids: runtimeGroups.map(groupInstanceId),
        slot_index: region.slot_index,
        geometry_revision: region.geometry_revision,
        task_group_id: displayGroup
          ? groupInstanceId(displayGroup)
          : null,
      };
    });
}

/** Select the authoritative runtime execution timeline. */
export function timelineRowsForFrame(frame: OperationalFrame): RegionTimelineView[] {
  return frame.execution ? executionTimelineRows(frame) : [];
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
