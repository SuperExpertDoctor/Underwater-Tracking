import type {
  Point2D,
  RegionalMissionView,
  RegionalPlanView,
  RegionTaskView,
  RegionTimelineView,
} from "../../types/frames";

export type RegionOverlayState = "active" | "handoff" | "degraded" | "uncovered" | "planned";

export interface RegionOverlayEntry {
  region: RegionTaskView;
  label: string;
  probability: number | null;
  priority: number | null;
  state: RegionOverlayState;
  stateSource: "region_timeline" | "region_effect";
  handoff: boolean;
}

export interface RegionOverlayProps {
  plans: RegionalPlanView[];
  missions?: RegionalMissionView[];
  timeline?: RegionTimelineView[];
  selectedRegionId?: string | null;
  onSelectRegion?: (regionId: string | null) => void;
  project: (point: Point2D) => Point2D;
  width?: number;
  height?: number;
  interactive?: boolean;
}

const STATE_LABELS: Record<RegionOverlayState, string> = {
  active: "当前覆盖",
  handoff: "接力",
  degraded: "降级",
  uncovered: "未覆盖",
  planned: "计划",
};

const STATE_STYLE: Record<RegionOverlayState, { fill: string; stroke: string }> = {
  active: { fill: "rgba(33, 208, 195, 0.10)", stroke: "rgba(33, 208, 195, 0.92)" },
  handoff: { fill: "rgba(247, 189, 69, 0.10)", stroke: "rgba(247, 189, 69, 0.92)" },
  degraded: { fill: "rgba(255, 120, 130, 0.10)", stroke: "rgba(255, 120, 130, 0.92)" },
  uncovered: { fill: "rgba(173, 190, 205, 0.06)", stroke: "rgba(173, 190, 205, 0.78)" },
  planned: { fill: "rgba(196, 180, 255, 0.08)", stroke: "rgba(196, 180, 255, 0.76)" },
};

const MISSION_REGION_FILL = "rgba(245, 194, 64, 0.66)";

const MISSION_STATE_LABELS: Record<RegionalMissionView["lifecycle"], string> = {
  PLANNED: "计划",
  CARRIER_DEPLOYING: "航母部署",
  ACTIVE_SCAN: "主动扫描",
  PASSIVE_TRACK: "被动跟踪",
  HANDOFF_PENDING: "等待交接",
  TRACKING_COMPLETED: "跟踪完成",
  CARRIER_RECOVERY: "航母回收",
  RECOVERED: "已回收",
  DEGRADED: "降级",
  UNCOVERED: "未覆盖",
};

function shortRegionLabel(region: RegionTaskView, ordinal: number): string {
  const match = region.display_name.match(/(?:region|区域)[_\s-]?(\d+)$/i);
  return `R${String(match ? Number(match[1]) : ordinal + 1).padStart(2, "0")}`;
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
        state,
        stateSource: timelineRow ? "region_timeline" : "region_effect",
        handoff: state === "handoff" || Boolean(timelineRow?.handoff_from || timelineRow?.handoff_to),
      };
    }));
}

export interface MissionOverlayEntry {
  mission: RegionalMissionView;
  label: string;
  state: RegionOverlayState;
}

function missionOverlayState(mission: RegionalMissionView): RegionOverlayState {
  switch (mission.lifecycle) {
    case "ACTIVE_SCAN":
    case "PASSIVE_TRACK": return "active";
    case "HANDOFF_PENDING": return "handoff";
    case "DEGRADED": return "degraded";
    case "UNCOVERED": return "uncovered";
    default: return "planned";
  }
}

export function missionOverlayEntries(missions: RegionalMissionView[] = []): MissionOverlayEntry[] {
  return [...missions]
    .filter((mission) => mission.geometry.length >= 3)
    .sort((left, right) => left.entry_s - right.entry_s || left.region_id.localeCompare(right.region_id))
    .map((mission, ordinal) => ({
      mission,
      label: `R${String(ordinal + 1).padStart(2, "0")}`,
      state: missionOverlayState(mission),
    }));
}

function centroid(points: Point2D[]): Point2D {
  const total = points.reduce((sum, point) => ({ x: sum.x + point.x, y: sum.y + point.y }), { x: 0, y: 0 });
  return { x: total.x / points.length, y: total.y / points.length };
}

export default function RegionOverlay({
  plans,
  missions = [],
  timeline = [],
  selectedRegionId = null,
  onSelectRegion,
  project,
  width,
  height,
  interactive = true,
}: RegionOverlayProps) {
  const missionEntries = missionOverlayEntries(missions);
  const missionIds = new Set(missionEntries.map((entry) => entry.mission.region_id));
  const entries = regionOverlayEntries(plans, timeline).filter((entry) => entry.region.geometry.length >= 3 && !missionIds.has(entry.region.region_id));
  if (!entries.length && !missionEntries.length) return null;
  return <svg
    className="region-map-overlay"
    aria-label="预测区域覆盖层"
    width={width}
    height={height}
    style={{ position: "absolute", inset: 0, pointerEvents: interactive ? "auto" : "none", overflow: "visible" }}
  >
    {entries.map((entry) => {
      const style = STATE_STYLE[entry.state];
      const points = entry.region.geometry.map(project);
      const center = centroid(points);
      const selected = entry.region.region_id === selectedRegionId;
      const probability = entry.probability === null ? "—" : `${Math.round(entry.probability * 100)}%`;
      const priority = entry.priority === null ? "—" : entry.priority.toFixed(2);
      const accessibleLabel = `${entry.label}，概率 ${probability}，优先级 ${priority}，${STATE_LABELS[entry.state]}`;
      const select = () => onSelectRegion?.(selected ? null : entry.region.region_id);
      return <g
        key={entry.region.region_id}
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
        <polygon points={points.map((point) => `${point.x},${point.y}`).join(" ")} fill={style.fill} stroke={selected ? "#f8fdff" : style.stroke} strokeWidth={selected ? 2.4 : 1.25} strokeDasharray={entry.state === "uncovered" ? "4 4" : undefined} />
        <text x={center.x} y={center.y - 5} textAnchor="middle" fill="#f8fdff" fontSize="9" fontWeight="700" pointerEvents="none">{entry.label}</text>
        <text x={center.x} y={center.y + 7} textAnchor="middle" fill={style.stroke} fontSize="7" pointerEvents="none">{`${probability} / ${priority}`}</text>
        <text x={center.x} y={center.y + 18} textAnchor="middle" fill={style.stroke} fontSize="7" pointerEvents="none">{STATE_LABELS[entry.state]}</text>
        <title>{accessibleLabel}</title>
      </g>;
    })}
    {missionEntries.map((entry) => {
      const style = STATE_STYLE[entry.state];
      const points = entry.mission.geometry.map(project);
      const center = centroid(points);
      const selected = entry.mission.region_id === selectedRegionId;
      const coverage = `${Math.round(entry.mission.coverage * 100)}%`;
      const quality = `${Math.round(entry.mission.tracking_quality * 100)}%`;
      const lifecycle = MISSION_STATE_LABELS[entry.mission.lifecycle];
      const accessibleLabel = `${entry.label}，覆盖 ${coverage}，跟踪质量 ${quality}，${lifecycle}`;
      const select = () => onSelectRegion?.(selected ? null : entry.mission.region_id);
      return <g
        key={`mission-${entry.mission.region_id}`}
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
        <polygon points={points.map((point) => `${point.x},${point.y}`).join(" ")} fill={MISSION_REGION_FILL} stroke={selected ? "#f8fdff" : style.stroke} strokeWidth={selected ? 2.4 : 1.25} strokeDasharray={entry.state === "uncovered" ? "4 4" : undefined} />
        <text x={center.x} y={center.y - 5} textAnchor="middle" fill="#f8fdff" fontSize="9" fontWeight="700" pointerEvents="none">{entry.label}</text>
        <text x={center.x} y={center.y + 7} textAnchor="middle" fill={style.stroke} fontSize="7" pointerEvents="none">{`${coverage} / ${quality}`}</text>
        <text x={center.x} y={center.y + 18} textAnchor="middle" fill={style.stroke} fontSize="7" pointerEvents="none">{lifecycle}</text>
        <title>{accessibleLabel}</title>
      </g>;
    })}
  </svg>;
}
