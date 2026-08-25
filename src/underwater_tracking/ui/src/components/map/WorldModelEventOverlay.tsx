import type { Point2D, TargetEstimateView } from "../../types/frames";

interface WorldModelEventOverlayProps {
  targets: TargetEstimateView[];
  project: (point: Point2D) => Point2D;
  width: number;
  height: number;
}

const LEVEL_COLOR: Record<string, string> = {
  critical: "#ff5f6d",
  strategic: "#ff7882",
  tactical: "#f7bd45",
  informational: "#34d2e0",
};

export default function WorldModelEventOverlay({
  targets,
  project,
  width,
  height,
}: WorldModelEventOverlayProps) {
  const events = targets.flatMap((target) =>
    (target.world_model?.events ?? []).map((event) => ({
      targetId: target.target_id,
      event,
    })),
  );
  if (events.length === 0) return null;

  return (
    <svg
      className="world-model-event-overlay"
      aria-label="世界模型未来事件位置"
      width={width}
      height={height}
      style={{ position: "absolute", inset: 0, pointerEvents: "none", overflow: "visible" }}
    >
      {events.map(({ targetId, event }) => {
        const point = project(event.predicted_position);
        const color = LEVEL_COLOR[event.level] ?? LEVEL_COLOR.tactical;
        const radius = 6;
        const diamond = [
          `${point.x},${point.y - radius}`,
          `${point.x + radius},${point.y}`,
          `${point.x},${point.y + radius}`,
          `${point.x - radius},${point.y}`,
        ].join(" ");
        return (
          <g
            data-event-type={event.event_type}
            data-horizon={event.horizon}
            key={event.event_id}
          >
            <polygon
              points={diamond}
              fill="rgba(8, 20, 39, 0.90)"
              stroke={color}
              strokeWidth="2"
            />
            <text
              x={point.x + 9}
              y={point.y + 3}
              fill={color}
              fontSize="9"
              fontWeight="700"
              paintOrder="stroke"
              stroke="rgba(8, 20, 39, 0.96)"
              strokeWidth="3"
            >
              {event.horizon}
            </text>
            <title>{`${targetId} · ${event.summary} · 规则置信度 ${Math.round(event.confidence * 100)}%`}</title>
          </g>
        );
      })}
    </svg>
  );
}
