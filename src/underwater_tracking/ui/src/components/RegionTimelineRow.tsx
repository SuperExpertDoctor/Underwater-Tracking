import type { RegionTimelineView } from "../types/frames";
import { offsetPercent, STATUS_LABELS, type TimelineWindow } from "./regionTimeline";

interface RegionTimelineRowProps {
  row: RegionTimelineView;
  window: TimelineWindow;
  selected: boolean;
  onSelect: () => void;
}

function barWidth(row: RegionTimelineView, window: TimelineWindow): { left: string; width: string } {
  const left = offsetPercent(row.start_offset_s, window.start, window.end);
  const right = offsetPercent(row.end_offset_s, window.start, window.end);
  return { left: `${left}%`, width: `${Math.max(1.5, right - left)}%` };
}

export default function RegionTimelineRow({ row, window, selected, onSelect }: RegionTimelineRowProps) {
  const style = barWidth(row, window);
  const labels = [
    ...row.uuv_assignments.map((assignment) => `${assignment.platform_id} · ${assignment.role}`),
  ];
  return (
    <button
      type="button"
      className={`region-timeline-row ${selected ? "selected" : ""}`}
      aria-label={`${row.region_id} ${STATUS_LABELS[row.status]}`}
      aria-pressed={selected}
      data-task-group-id={row.task_group_id ?? undefined}
      data-task-group-ids={row.task_group_ids?.join(",")}
      data-region-slot={row.slot_index}
      data-geometry-revision={row.geometry_revision}
      onClick={onSelect}
    >
      <span className="region-timeline-row-label">
        <strong>{row.region_id}</strong>
        {row.task_group_id && <em className="region-timeline-task-group">{row.task_group_id}</em>}
        <small>{STATUS_LABELS[row.status]} · {Math.round(row.occupancy_likelihood * 100)}%</small>
      </span>
      <span className="region-timeline-track">
        <span className={`region-timeline-bar status-${row.status}`} style={style}>
          {labels.slice(0, 2).join(" / ") || "待分配"}
        </span>
        {row.handoff_to && <i className="region-handoff-node" style={{ left: `calc(${style.left} + ${style.width})` }} aria-label={`交接至 ${row.handoff_to}`} />}
      </span>
    </button>
  );
}
