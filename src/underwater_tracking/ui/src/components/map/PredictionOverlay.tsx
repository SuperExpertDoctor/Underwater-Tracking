import type {
  Point2D,
  PredictionCorridorView,
  PredictionHealthStatus,
} from "../../types/frames";
import { MAP_DISPLAY_CONFIG } from "../../../configs/map_display";
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

export interface DisplayPredictionPoint {
  point: Point2D;
  sourceIndex: number;
}

/**
 * Keep prediction samples legible at low zoom without changing the line or
 * the source data. The endpoints are always retained; intermediate markers
 * are selected by their projected screen-space distance.
 */
export function decimatePredictionPoints(
  projectedPoints: Point2D[],
  minimumSpacingPx: number = MAP_DISPLAY_CONFIG.predictionSampleSpacingPx,
): DisplayPredictionPoint[] {
  if (projectedPoints.length <= 2) {
    return projectedPoints.map((point, sourceIndex) => ({ point, sourceIndex }));
  }

  const minimumSpacing = Math.max(0, minimumSpacingPx);
  const lastIndex = projectedPoints.length - 1;
  const displayed: DisplayPredictionPoint[] = [{
    point: projectedPoints[0],
    sourceIndex: 0,
  }];
  let lastDisplayed = projectedPoints[0];

  for (let sourceIndex = 1; sourceIndex < lastIndex; sourceIndex += 1) {
    const point = projectedPoints[sourceIndex];
    if (Math.hypot(point.x - lastDisplayed.x, point.y - lastDisplayed.y) < minimumSpacing) {
      continue;
    }
    displayed.push({ point, sourceIndex });
    lastDisplayed = point;
  }

  displayed.push({ point: projectedPoints[lastIndex], sourceIndex: lastIndex });
  return displayed;
}

/**
 * Return presentation-only IMM radii for the map corridor.
 *
 * The backend radii remain authoritative for tracking, planning, and audit.
 * A live degraded prediction can legitimately publish a very large fallback
 * radius, though, and drawing that value literally turns the corridor into a
 * rectangle that dominates the map.  The UI therefore uses a restrained,
 * monotonic taper: it starts at the current-radius side and widens toward the
 * end of the forecast, with a display cap for pathological values.
 */
export function displayCorridorRadii(prediction: PredictionCorridorView): number[] {
  const source = displayRadii(prediction).map((radius) =>
    Number.isFinite(radius) && radius > 0 ? radius : 0,
  );
  if (!source.length) return source;

  const rawMaximum = Math.max(...source);
  if (!(rawMaximum > 0)) return source;

  const displayEnd = Math.min(rawMaximum, IMM_DISPLAY_RADIUS_CAP_M);
  const firstRadius = Math.min(source[0] ?? displayEnd, displayEnd);
  const displayStart = Math.min(
    displayEnd,
    firstRadius * IMM_DISPLAY_START_RATIO,
  );

  return source.map((_radius, index) => {
    const progress = source.length <= 1 ? 1 : index / (source.length - 1);
    // A pathological source radius above the display cap must not pin every
    // sample to the cap; the linear taper is what gives that fallback
    // corridor its trapezoid shape.  The backend values remain untouched.
    return displayStart + (displayEnd - displayStart) * progress;
  });
}

/** Maximum half-width used by the presentation layer, in metres. */
export const IMM_DISPLAY_RADIUS_CAP_M = 1_200;
/** The corridor begins at 55% of the current-side radius and opens forward. */
export const IMM_DISPLAY_START_RATIO = 0.55;

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
        const centerline = decimatePredictionPoints(immCenterlineSource.map(project));
        const bsplineCenterline = displayBsplineCenterline(prediction).map(project);
        const band = corridorPolygon(
          immCenterlineSource,
          displayCorridorRadii(prediction),
        ).map(project);
        const status = healthOf(prediction).status;
        const isDegraded = status === "degraded";
        const isLegacy = status === "legacy_unknown";
        const bandStroke = isDegraded
          ? "rgba(247, 189, 69, 0.74)"
          : isLegacy
            ? "rgba(173, 190, 205, 0.58)"
            : "rgba(81, 216, 226, 0.68)";
        const splineStroke = isDegraded
          ? "rgba(255, 181, 71, 0.98)"
          : isLegacy
            ? "rgba(205, 214, 224, 0.82)"
            : "rgba(255, 218, 106, 0.98)";
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
                width="22"
                height="22"
                patternUnits="userSpaceOnUse"
                patternTransform="rotate(35)"
              >
                <line
                  x1="0"
                  y1="0"
                  x2="0"
                  y2="22"
                  stroke="rgba(247, 189, 69, 0.16)"
                  strokeWidth="1.2"
                />
              </pattern>
            </defs>
            <polygon
              data-testid="prediction-corridor"
              data-prediction-source="imm"
              className={`imm-confidence-band prediction-health-${status}`}
              points={pointsAttribute(band)}
              fill={isDegraded ? `url(#prediction-degraded-${targetId})` : isLegacy ? "rgba(173, 190, 205, 0.08)" : "rgba(52, 210, 224, 0.20)"}
              stroke={bandStroke}
              strokeWidth="1.2"
              strokeDasharray={isLegacy ? "3 5" : isDegraded ? "7 6" : undefined}
            />
            <polyline
              className="bspline-prediction-centerline-shadow"
              points={pointsAttribute(bsplineCenterline)}
              fill="none"
              stroke="rgba(4, 24, 49, 0.92)"
              strokeWidth="4.5"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            <polyline
              className="bspline-prediction-centerline imm-prediction-centerline"
              points={pointsAttribute(bsplineCenterline)}
              fill="none"
              stroke={splineStroke}
              strokeWidth="2.2"
              strokeDasharray="8 6"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            {centerline.map(({ point, sourceIndex }) => {
              const confidence = confidenceAt(prediction, sourceIndex);
              return (
                <circle
                  key={`${targetId}:${sourceIndex}`}
                  className="imm-prediction-point"
                  cx={point.x}
                  cy={point.y}
                  r={isLegacy ? 2.2 : 2.2 + confidence * 1.8}
                  fill={bandStroke}
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
              x={bsplineCenterline[bsplineCenterline.length - 1].x + 8}
              y={bsplineCenterline[bsplineCenterline.length - 1].y - 10}
              fill={bandStroke}
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
