import type { Point2D, RegionalPlanView, RegionTaskView, RegionTimelineView } from "../../types/frames";
import { displayRegionPoints, sharedRegionDisplaySide } from "./geometry";

export type RegionOverlayState = "active" | "handoff" | "degraded" | "uncovered" | "planned";

export interface RegionOverlayEntry {
  region: RegionTaskView;
  label: string;
  probability: number | null;
  priority: number | null;
  predictionId: string;
  executionRevision: number;
  state: RegionOverlayState;
  stateSource: "region_timeline" | "region_effect";
  handoff: boolean;
}

export interface RegionOverlayProps {
  plans: RegionalPlanView[];
  timeline?: RegionTimelineView[];
  selectedRegionId?: string | null;
  currentRegionId?: string | null;
  nextRegionId?: string | null;
  onSelectRegion?: (regionId: string | null) => void;
  project: (point: Point2D) => Point2D;
  width?: number;
  height?: number;
  interactive?: boolean;
  showHandoffs?: boolean;
}

const STATE_LABELS: Record<RegionOverlayState, string> = {
  active: "当前覆盖",
  handoff: "接力",
  degraded: "降级",
  uncovered: "未覆盖",
  planned: "计划",
};

const STATE_STYLE: Record<RegionOverlayState, { fill: string; stroke: string }> = {
  active: { fill: "rgba(33, 208, 195, 0.09)", stroke: "rgba(81, 232, 219, 0.94)" },
  handoff: { fill: "rgba(247, 189, 69, 0.08)", stroke: "rgba(255, 207, 97, 0.94)" },
  degraded: { fill: "rgba(255, 120, 130, 0.09)", stroke: "rgba(255, 137, 146, 0.94)" },
  uncovered: { fill: "rgba(173, 190, 205, 0.04)", stroke: "rgba(173, 190, 205, 0.70)" },
  planned: { fill: "rgba(196, 180, 255, 0.055)", stroke: "rgba(196, 180, 255, 0.70)" },
};

function shortRegionLabel(region: RegionTaskView, ordinal: number): string {
  const match = region.display_name.match(/(?:region|区域)[_\s-]?(\d+)$/i);
  return `R${String(match ? Number(match[1]) : ordinal + 1).padStart(2, "0")}`;
}

function shortTaskGroupLabel(taskGroupId: string): string {
  const match = taskGroupId.match(/(?:task[-_ ]?group|group|tg)[:_-]?(\d+)$/i);
  return match
    ? `TG-${String(Number(match[1])).padStart(2, "0")}`
    : taskGroupId;
}

function overlayState(region: RegionTaskView, timeline: RegionTimelineView | undefined): RegionOverlayState {
  const status = timeline?.status ?? region.effect.status;
  if (status === "handed_off" || status === "handoff_ready") return "handoff";
  if (status === "active") return "active";
  if (status === "degraded") return "degraded";
  if (status === "uncovered") return "uncovered";
  return "planned";
}

export function regionOverlayEntries(plans: RegionalPlanView[], timeline: RegionTimelineView[] = []): RegionOverlayEntry[] {
  const timelineById = new Map(timeline.map((row) => [row.region_id, row]));
  return plans.flatMap((plan) => [...plan.regions]
    .sort((left, right) => left.start_time_s - right.start_time_s || left.region_id.localeCompare(right.region_id))
    .map((region, ordinal) => {
      const timelineRow = timelineById.get(region.region_id);
      const state = overlayState(region, timelineRow);
      return {
        region,
        label: shortRegionLabel(region, ordinal),
        probability: timelineRow?.occupancy_likelihood ?? null,
        priority: timelineRow?.priority ?? null,
        predictionId: plan.prediction_id,
        executionRevision: plan.revision,
        state,
        stateSource: timelineRow ? "region_timeline" : "region_effect",
        handoff: state === "handoff" || Boolean(timelineRow?.handoff_from || timelineRow?.handoff_to),
      };
    }));
}

function centroid(points: Point2D[]): Point2D {
  const total = points.reduce((sum, point) => ({ x: sum.x + point.x, y: sum.y + point.y }), { x: 0, y: 0 });
  return { x: total.x / points.length, y: total.y / points.length };
}

function regionScreenRect(points: Point2D[]) {
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const x = Math.min(...xs);
  const y = Math.min(...ys);
  return {
    x,
    y,
    width: Math.max(0, Math.max(...xs) - x),
    height: Math.max(0, Math.max(...ys) - y),
  };
}

export default function RegionOverlay({
  plans,
  timeline = [],
  selectedRegionId = null,
  currentRegionId = null,
  nextRegionId = null,
  onSelectRegion,
  project,
  width,
  height,
  interactive = true,
  showHandoffs = true,
}: RegionOverlayProps) {
  const entries = regionOverlayEntries(plans, timeline).filter((entry) => displayRegionPoints(entry.region).length >= 3);
  if (!entries.length) return null;
  const displaySide = sharedRegionDisplaySide(entries.map((entry) => entry.region));
  const entriesById = new Map(entries.map((entry) => [entry.region.region_id, entry]));
  const flowLinks = showHandoffs ? entries.flatMap((entry) => entry.region.successor_region_ids.flatMap((successorId) => {
    const successor = entriesById.get(successorId);
    if (!successor) return [];
    return [{
      id: `${entry.region.region_id}:${successorId}`,
      start: centroid(displayRegionPoints(entry.region, displaySide).map(project)),
      end: centroid(displayRegionPoints(successor.region, displaySide).map(project)),
    }];
  })) : [];
  return <svg
    className="region-map-overlay"
    aria-label="预测区域覆盖层"
    width={width}
    height={height}
    style={{ position: "absolute", inset: 0, pointerEvents: interactive ? "auto" : "none", overflow: "visible" }}
  >
    <defs>
      <marker id="region-task-flow-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 0 L 8 4 L 0 8 z" fill="#f7bd45" />
      </marker>
    </defs>
    {flowLinks.map((link) => <line
      key={link.id}
      className="region-task-flow"
      x1={link.start.x}
      y1={link.start.y}
      x2={link.end.x}
      y2={link.end.y}
      stroke="rgba(247, 189, 69, 0.9)"
      strokeWidth="1.8"
      strokeDasharray="6 4"
      markerEnd="url(#region-task-flow-arrow)"
      pointerEvents="none"
    />)}
    {entries.map((entry, entryIndex) => {
      const style = STATE_STYLE[entry.state];
      const points = displayRegionPoints(entry.region, displaySide).map(project);
      const rect = regionScreenRect(points);
      const selected = entry.region.region_id === selectedRegionId;
      const current = entry.region.region_id === currentRegionId;
      const next = entry.region.region_id === nextRegionId;
      const detailed = selected || current || next;
      const showDetails = detailed && rect.width >= 120 && rect.height >= 72;
      const groupLabel = entry.region.group_id
        ? `${shortTaskGroupLabel(entry.region.group_id)} · ${entry.region.assigned_uuv_ids.length} UUV`
        : null;
      const probability = entry.probability === null ? "—" : `${Math.round(entry.probability * 100)}%`;
      const priority = entry.priority === null ? "—" : entry.priority.toFixed(2);
      const anchorRight = entryIndex % 2 === 1;
      const anchorBottom = entryIndex >= 2;
      const labelX = anchorRight ? rect.x + rect.width - 8 : rect.x + 8;
      const labelY = anchorBottom ? rect.y + rect.height - 10 : rect.y + 16;
      const lineDirection = anchorBottom ? -1 : 1;
      const textAnchor = anchorRight ? "end" : "start";
      const accessibleLabel = `${entry.label}，概率 ${probability}，优先级 ${priority}，${STATE_LABELS[entry.state]}`;
      const select = () => onSelectRegion?.(selected ? null : entry.region.region_id);
      return <g
        key={entry.region.region_id}
        data-execution-region-id={entry.region.region_id}
        data-task-group-id={entry.region.group_id ?? undefined}
        data-prediction-id={entry.predictionId}
        data-execution-revision={entry.executionRevision}
        data-region-state={entry.state}
        data-current-region={current ? "true" : undefined}
        data-next-region={next ? "true" : undefined}
        role={interactive ? "button" : undefined}
        tabIndex={interactive ? 0 : undefined}
        aria-label={interactive ? accessibleLabel : undefined}
        aria-pressed={interactive ? selected : undefined}
        onClick={interactive ? select : undefined}
        onKeyDown={interactive ? (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            select();
          }
        } : undefined}
      >
        <rect
          data-region-shape="square"
          x={rect.x}
          y={rect.y}
          width={rect.width}
          height={rect.height}
          fill={style.fill}
          stroke={selected || current ? "#f8fdff" : next ? "#f7bd45" : style.stroke}
          strokeWidth={selected ? 2.4 : current ? 2 : next ? 1.7 : 1.25}
          strokeDasharray={entry.state === "uncovered" ? "4 4" : undefined}
        />
        <text
          x={labelX}
          y={labelY}
          textAnchor={textAnchor}
          fill="#f8fdff"
          stroke="rgba(4, 24, 49, 0.88)"
          strokeWidth="3"
          paintOrder="stroke"
          fontSize={showDetails ? "11" : "10"}
          fontWeight="700"
          pointerEvents="none"
        >{showDetails ? `${entry.label} · ${STATE_LABELS[entry.state]}` : entry.label}</text>
        {showDetails && groupLabel && <text
          className="region-task-group-label"
          x={labelX}
          y={labelY + 14 * lineDirection}
          textAnchor={textAnchor}
          fill={current ? "#f8fdff" : style.stroke}
          stroke="rgba(4, 24, 49, 0.82)"
          strokeWidth="2.5"
          paintOrder="stroke"
          fontSize="9"
          fontWeight={current ? "700" : "500"}
          pointerEvents="none"
        >{groupLabel}</text>}
        {showDetails && <text
          x={labelX}
          y={labelY + 28 * lineDirection}
          textAnchor={textAnchor}
          fill={style.stroke}
          stroke="rgba(4, 24, 49, 0.82)"
          strokeWidth="2.5"
          paintOrder="stroke"
          fontSize="9"
          pointerEvents="none"
        >{`${probability} · P${priority}`}</text>}
        <title>{accessibleLabel}</title>
      </g>;
    })}
  </svg>;
}
