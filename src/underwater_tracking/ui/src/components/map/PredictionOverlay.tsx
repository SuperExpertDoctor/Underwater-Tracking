import type { Point2D, PredictionCorridorView } from "../../types/frames";
import { corridorPolygon } from "./geometry";

interface PredictionOverlayEntry {
  targetId: string;
  prediction: PredictionCorridorView;
}

interface PredictionOverlayProps {
  predictions: PredictionOverlayEntry[];
  project: (point: Point2D) => Point2D;
  width: number;
  height: number;
}

export function confidenceAdjustedRadii(
  prediction: PredictionCorridorView,
): number[] {
  if (
    !prediction.point_confidence?.length
    || prediction.point_confidence.length !== prediction.centerline_xy.length
  ) {
    return [...prediction.radius_m];
  }
  return prediction.centerline_xy.map((_, index) => {
    const radius = Math.max(0, prediction.radius_m[index] ?? prediction.radius_m.at(-1) ?? 0);
    const confidence = Math.max(0, Math.min(1, prediction.point_confidence?.[index] ?? 1));
    return Number((radius * (1 + (1 - confidence) * 0.5)).toFixed(6));
  });
}

function pointsAttribute(points: Point2D[]): string {
  return points.map((point) => `${point.x},${point.y}`).join(" ");
}

export default function PredictionOverlay({
  predictions,
  project,
  width,
  height,
}: PredictionOverlayProps) {
  const visible = predictions.filter(
    ({ prediction }) => prediction.centerline_xy.length >= 2,
  );
  if (!visible.length) return null;
  return (
    <svg
      className="imm-prediction-overlay"
      aria-label="IMM 预测置信轨迹"
      width={width}
      height={height}
      style={{ position: "absolute", inset: 0, pointerEvents: "none", overflow: "visible" }}
    >
      {visible.map(({ targetId, prediction }) => {
        const centerline = prediction.centerline_xy.map(project);
        const band = corridorPolygon(
          prediction.centerline_xy,
          confidenceAdjustedRadii(prediction),
        ).map(project);
        return (
          <g key={targetId} data-target-id={targetId}>
            <polygon
              className="imm-confidence-band"
              points={pointsAttribute(band)}
              fill="rgba(52, 210, 224, 0.20)"
              stroke="rgba(117, 238, 242, 0.88)"
              strokeWidth="1.5"
            />
            <polyline
              points={pointsAttribute(centerline)}
              fill="none"
              stroke="rgba(4, 24, 49, 0.92)"
              strokeWidth="5"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            <polyline
              className="imm-prediction-centerline"
              points={pointsAttribute(centerline)}
              fill="none"
              stroke="rgba(255, 211, 107, 0.98)"
              strokeWidth="2.3"
              strokeDasharray="8 5"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            {centerline.map((point, index) => {
              const confidence = Math.max(
                0,
                Math.min(1, prediction.point_confidence?.[index] ?? 1),
              );
              return (
                <circle
                  key={`${targetId}:${index}`}
                  className="imm-prediction-point"
                  cx={point.x}
                  cy={point.y}
                  r={2.2 + confidence * 1.8}
                  fill="rgba(255, 224, 139, 0.98)"
                  stroke="rgba(4, 24, 49, 0.92)"
                  strokeWidth="1.2"
                  data-confidence={confidence.toFixed(3)}
                >
                  <title>{`${targetId} IMM ${(confidence * 100).toFixed(0)}%`}</title>
                </circle>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}
