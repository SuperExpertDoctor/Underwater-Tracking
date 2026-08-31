import type {
  Point2D,
  PredictionCorridorView,
  PredictionHealthStatus,
} from "../../types/frames";
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

export function displayRadii(prediction: PredictionCorridorView): number[] {
  return [...(prediction.imm_radius_m?.length ? prediction.imm_radius_m : prediction.radius_m)];
}

function displayImmCenterline(prediction: PredictionCorridorView): Point2D[] {
  return prediction.imm_centerline_xy?.length
    ? prediction.imm_centerline_xy
    : prediction.centerline_xy;
}

function displayBsplineCenterline(prediction: PredictionCorridorView): Point2D[] {
  const centerline = prediction.bspline_centerline_xy;
  return centerline && centerline.length >= 2
    ? centerline
    : displayImmCenterline(prediction);
}

const HEALTH_LABELS: Record<PredictionHealthStatus, string> = {
  valid: "VALID",
  degraded: "DEGRADED",
  unavailable: "UNAVAILABLE",
  legacy_unknown: "LEGACY UNKNOWN",
};

function pointsAttribute(points: Point2D[]): string {
  return points.map((point) => `${point.x},${point.y}`).join(" ");
}

function confidenceAt(prediction: PredictionCorridorView, index: number): number {
  return Math.max(0, Math.min(1, prediction.point_confidence?.[index] ?? 1));
}

function healthOf(prediction: PredictionCorridorView) {
  return prediction.health ?? {
    status: "legacy_unknown" as const,
    regime: "legacy_unknown" as const,
    reason_codes: ["legacy_health_missing"],
    source_track_age_s: 0,
    clipped_point_fraction: 0,
    maximum_radius_m: Math.max(...prediction.radius_m, 0),
    raw_prediction_id: null,
  };
}

export default function PredictionOverlay({
  predictions,
  project,
  width,
  height,
}: PredictionOverlayProps) {
  const visible = predictions.filter(
    ({ prediction }) => healthOf(prediction).status !== "unavailable",
  );
  const unavailable = predictions.filter(
    ({ prediction }) => healthOf(prediction).status === "unavailable",
  );
  if (!visible.length && !unavailable.length) return null;
  return (
    <svg
      className="imm-prediction-overlay"
      aria-label="IMM 预测置信轨迹"
      width={width}
      height={height}
      style={{ position: "absolute", inset: 0, pointerEvents: "none", overflow: "visible" }}
    >
      {visible.map(({ targetId, prediction }) => {
        const immCenterlineSource = displayImmCenterline(prediction);
        if (immCenterlineSource.length < 2) return null;
        const centerline = immCenterlineSource.map(project);
        const bsplineCenterline = displayBsplineCenterline(prediction).map(project);
        const band = corridorPolygon(
          immCenterlineSource,
          displayRadii(prediction),
        ).map(project);
        const status = healthOf(prediction).status;
        const isDegraded = status === "degraded";
        const isLegacy = status === "legacy_unknown";
        const stroke = isDegraded
          ? "rgba(247, 189, 69, 0.92)"
          : isLegacy
            ? "rgba(173, 190, 205, 0.68)"
            : "rgba(117, 238, 242, 0.88)";
        return (
          <g
            key={targetId}
            data-target-id={targetId}
            data-health-status={status}
            data-prediction-id={prediction.prediction_id}
            data-prediction-revision={prediction.prediction_revision}
          >
            <defs>
              <pattern
                id={`prediction-degraded-${targetId}`}
                width="8"
                height="8"
                patternUnits="userSpaceOnUse"
                patternTransform="rotate(35)"
              >
                <line x1="0" y1="0" x2="0" y2="8" stroke={stroke} strokeWidth="2" />
              </pattern>
            </defs>
            <polygon
              data-testid="prediction-corridor"
              data-prediction-source="imm"
              className={`imm-confidence-band prediction-health-${status}`}
              points={pointsAttribute(band)}
              fill={isDegraded ? `url(#prediction-degraded-${targetId})` : isLegacy ? "rgba(173, 190, 205, 0.08)" : "rgba(52, 210, 224, 0.20)"}
              stroke={stroke}
              strokeWidth="1.5"
              strokeDasharray={isLegacy ? "3 5" : undefined}
            />
            <polyline
              className="bspline-prediction-centerline-shadow"
              points={pointsAttribute(bsplineCenterline)}
              fill="none"
              stroke="rgba(4, 24, 49, 0.92)"
              strokeWidth="5"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            <polyline
              className="bspline-prediction-centerline imm-prediction-centerline"
              points={pointsAttribute(bsplineCenterline)}
              fill="none"
              stroke={stroke}
              strokeWidth="2.3"
              strokeDasharray="6 6"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            {centerline.map((point, index) => {
              const confidence = confidenceAt(prediction, index);
              return (
                <circle
                  key={`${targetId}:${index}`}
                  className="imm-prediction-point"
                  cx={point.x}
                  cy={point.y}
                  r={isLegacy ? 2.2 : 2.2 + confidence * 1.8}
                  fill={stroke}
                  fillOpacity={isLegacy ? 0.65 : 0.35 + confidence * 0.65}
                  stroke="rgba(4, 24, 49, 0.92)"
                  strokeWidth="1.2"
                  data-confidence={confidence.toFixed(3)}
                >
                  <title>{`${targetId} ${HEALTH_LABELS[status]} ${(confidence * 100).toFixed(0)}%`}</title>
                </circle>
              );
            })}
            <text
              className="prediction-health-status"
              x={centerline[0].x + 8}
              y={centerline[0].y - 10}
              fill={stroke}
              fontSize="9"
              fontWeight="700"
            >
              {HEALTH_LABELS[status]}
            </text>
          </g>
        );
      })}
      {unavailable.map(({ targetId, prediction }) => (
        <g
          key={`${targetId}:unavailable`}
          data-target-id={targetId}
          data-health-status="unavailable"
          data-prediction-id={prediction.prediction_id}
          data-prediction-revision={prediction.prediction_revision}
          className="prediction-unavailable"
        >
          <text className="prediction-health-status" x="12" y="20" fill="rgba(255, 120, 130, 0.92)" fontSize="9" fontWeight="700">
            {`${targetId} ${HEALTH_LABELS[healthOf(prediction).status]}`}
          </text>
        </g>
      ))}
    </svg>
  );
}
