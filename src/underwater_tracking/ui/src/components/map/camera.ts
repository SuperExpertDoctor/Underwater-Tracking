import type {
  ExecutionRegionView,
  MapBounds,
  OperationalFrame,
  Point2D,
  TargetEstimateView,
} from "../../types/frames";
import { boundsForPoints } from "./geometry";

export interface CameraViewport {
  width: number;
  height: number;
}

export interface SemanticCamera {
  worldBounds: MapBounds;
  scale: number;
  targetDetectionDiameterPx: number;
  minimumRegionDimensionPx: number;
  twoKilometerSegmentPx: number;
  readabilityWarnings: string[];
}

export interface LabelCandidate {
  id: string;
  anchor: Point2D;
  width: number;
  height: number;
  priority: number;
}

export interface LabelPlacement extends LabelCandidate {
  x: number;
  y: number;
  suppressed: boolean;
}

const FIT_PADDING = 0.08;
const TARGET_DIAMETER_MIN_PX = 160;
const REGION_DIMENSION_MIN_PX = 48;
const TWO_KILOMETER_MIN_PX = 120;
const DEFAULT_DETECTION_RANGE_M = 5_000;
const LABEL_OFFSETS: Point2D[] = [
  { x: 10, y: -10 },
  { x: 10, y: 12 },
  { x: -10, y: -10 },
  { x: -10, y: 12 },
  { x: 0, y: -18 },
  { x: 0, y: 20 },
];

function finiteBounds(bounds: MapBounds): MapBounds {
  const min_x = Number.isFinite(bounds.min_x) ? bounds.min_x : 0;
  const min_y = Number.isFinite(bounds.min_y) ? bounds.min_y : 0;
  const max_x = Number.isFinite(bounds.max_x) ? bounds.max_x : min_x + 1;
  const max_y = Number.isFinite(bounds.max_y) ? bounds.max_y : min_y + 1;
  return {
    min_x: Math.min(min_x, max_x),
    min_y: Math.min(min_y, max_y),
    max_x: Math.max(min_x, max_x),
    max_y: Math.max(min_y, max_y),
  };
}

function clampPoint(point: Point2D, bounds: MapBounds): Point2D {
  return {
    x: Math.max(bounds.min_x, Math.min(bounds.max_x, point.x)),
    y: Math.max(bounds.min_y, Math.min(bounds.max_y, point.y)),
  };
}

function containsPoint(bounds: MapBounds, point: Point2D): boolean {
  return point.x >= bounds.min_x
    && point.x <= bounds.max_x
    && point.y >= bounds.min_y
    && point.y <= bounds.max_y;
}

function containsAll(bounds: MapBounds, points: Point2D[]): boolean {
  return points.every((point) => containsPoint(bounds, point));
}

function currentTarget(frame: OperationalFrame): TargetEstimateView | null {
  if (frame.execution) {
    return frame.target_estimates.find(
      (target) => target.target_id === frame.execution?.target_id,
    ) ?? null;
  }
  return frame.target_estimates[0] ?? null;
}

function detectionRange(frame: OperationalFrame, target: TargetEstimateView): number {
  const configured = frame.adversary?.detection_range_m ?? target.detection_range_m;
  return configured != null && Number.isFinite(configured) && configured > 1
    ? configured
    : DEFAULT_DETECTION_RANGE_M;
}

function addClamped(points: Point2D[], point: Point2D, bounds: MapBounds): void {
  if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) return;
  points.push(clampPoint(point, bounds));
}

function executionRegions(frame: OperationalFrame): ExecutionRegionView[] {
  const execution = frame.execution;
  if (!execution || !["current", "degraded"].includes(execution.health_status)) return [];
  return execution.regions;
}

function assignedUuvs(frame: OperationalFrame) {
  const memberIds = new Set(frame.execution?.task_groups.flatMap((group) => group.member_uuv_ids) ?? []);
  return frame.uuvs.filter((uuv) => uuv.physically_exposed && (memberIds.size === 0 || memberIds.has(uuv.uuv_id)));
}

/**
 * Collects only operator-safe geometry. Every point is clamped at this
 * boundary so a malformed live value cannot enlarge the camera window.
 */
export function semanticCameraCandidates(frame: OperationalFrame): Point2D[] {
  const map = finiteBounds(frame.map_bounds);
  const points: Point2D[] = [];
  const target = currentTarget(frame);
  if (target) {
    addClamped(points, target.mean, map);
    const prediction = target.prediction;
    if (prediction && (prediction.health?.status === "valid" || prediction.health?.status === "degraded")) {
      prediction.centerline_xy.forEach((point) => addClamped(points, point, map));
    }
  }

  executionRegions(frame).forEach((region) => {
    region.geometry.forEach((point) => addClamped(points, point, map));
  });

  const assignedIds = new Set(frame.execution?.task_groups.flatMap((group) => group.member_uuv_ids) ?? []);
  assignedUuvs(frame)
    .filter((uuv) => assignedIds.size === 0 || assignedIds.has(uuv.uuv_id))
    .forEach((uuv) => addClamped(points, uuv.position, map));

  if (target) {
    const radius = detectionRange(frame, target);
    [
      { x: target.mean.x - radius, y: target.mean.y },
      { x: target.mean.x + radius, y: target.mean.y },
      { x: target.mean.x, y: target.mean.y - radius },
      { x: target.mean.x, y: target.mean.y + radius },
    ].forEach((point) => addClamped(points, point, map));
  }
  return points;
}

function width(bounds: MapBounds): number {
  return Math.max(0, bounds.max_x - bounds.min_x);
}

function height(bounds: MapBounds): number {
  return Math.max(0, bounds.max_y - bounds.min_y);
}

function centeredBounds(center: Point2D, requestedWidth: number, requestedHeight: number, map: MapBounds): MapBounds {
  const mapWidth = width(map);
  const mapHeight = height(map);
  const actualWidth = Math.min(mapWidth, Math.max(0, requestedWidth));
  const actualHeight = Math.min(mapHeight, Math.max(0, requestedHeight));
  const min_x = Math.max(map.min_x, Math.min(map.max_x - actualWidth, center.x - actualWidth / 2));
  const min_y = Math.max(map.min_y, Math.min(map.max_y - actualHeight, center.y - actualHeight / 2));
  return {
    min_x,
    max_x: min_x + actualWidth,
    min_y,
    max_y: min_y + actualHeight,
  };
}

function aspectFitBounds(
  candidateBounds: MapBounds,
  viewport: CameraViewport,
  map: MapBounds,
  candidates: Point2D[],
): MapBounds {
  const viewportAspect = Math.max(1e-6, viewport.width) / Math.max(1e-6, viewport.height);
  const mapWidth = width(map);
  const mapHeight = height(map);
  let requestedWidth = Math.max(width(candidateBounds), 1);
  let requestedHeight = Math.max(height(candidateBounds), 1);
  if (requestedWidth / requestedHeight > viewportAspect) requestedHeight = requestedWidth / viewportAspect;
  else requestedWidth = requestedHeight * viewportAspect;
  if (requestedWidth > mapWidth) {
    requestedWidth = mapWidth;
    requestedHeight = requestedWidth / viewportAspect;
  }
  if (requestedHeight > mapHeight) {
    requestedHeight = mapHeight;
    requestedWidth = requestedHeight * viewportAspect;
  }
  const center = {
    x: (candidateBounds.min_x + candidateBounds.max_x) / 2,
    y: (candidateBounds.min_y + candidateBounds.max_y) / 2,
  };
  const fitted = centeredBounds(center, requestedWidth, requestedHeight, map);
  return containsAll(fitted, candidates) ? fitted : map;
}

function fitScale(bounds: MapBounds, viewport: CameraViewport): number {
  return Math.min(
    Math.max(0, viewport.width) / Math.max(width(bounds), Number.EPSILON),
    Math.max(0, viewport.height) / Math.max(height(bounds), Number.EPSILON),
  );
}

function regionMinimumDimension(frame: OperationalFrame): number {
  const dimensions = executionRegions(frame)
    .map((region) => {
      const bounds = boundsForPoints(region.geometry);
      return bounds ? Math.min(width(bounds), height(bounds)) : 0;
    })
    .filter((value) => value > 0);
  return dimensions.length ? Math.min(...dimensions) : 0;
}

function readableBounds(
  bounds: MapBounds,
  frame: OperationalFrame,
  viewport: CameraViewport,
  candidates: Point2D[],
): { bounds: MapBounds; scale: number; minimumsConstrained: boolean } {
  const target = currentTarget(frame);
  const radius = target ? detectionRange(frame, target) : 0;
  const regionDimension = regionMinimumDimension(frame);
  const requiredScale = Math.max(
    radius > 0 ? TARGET_DIAMETER_MIN_PX / (radius * 2) : 0,
    regionDimension > 0 ? REGION_DIMENSION_MIN_PX / regionDimension : 0,
    TWO_KILOMETER_MIN_PX / 2_000,
  );
  const currentScale = fitScale(bounds, viewport);
  if (currentScale >= requiredScale || currentScale <= 0) {
    return { bounds, scale: currentScale, minimumsConstrained: false };
  }

  const factor = currentScale / requiredScale;
  const map = finiteBounds(frame.map_bounds);
  const tightened = centeredBounds(
    { x: (bounds.min_x + bounds.max_x) / 2, y: (bounds.min_y + bounds.max_y) / 2 },
    width(bounds) * factor,
    height(bounds) * factor,
    map,
  );
  if (!containsAll(tightened, candidates)) {
    return { bounds, scale: currentScale, minimumsConstrained: true };
  }
  return {
    bounds: tightened,
    scale: fitScale(tightened, viewport),
    minimumsConstrained: false,
  };
}

export function semanticCameraForFrame(
  frame: OperationalFrame,
  viewport: CameraViewport,
): SemanticCamera {
  const map = finiteBounds(frame.map_bounds);
  const candidates = semanticCameraCandidates(frame);
  const candidateBounds = boundsForPoints(candidates, FIT_PADDING) ?? map;
  const clampedCandidateBounds = centeredBounds(
    {
      x: (candidateBounds.min_x + candidateBounds.max_x) / 2,
      y: (candidateBounds.min_y + candidateBounds.max_y) / 2,
    },
    width(candidateBounds),
    height(candidateBounds),
    map,
  );
  const aspectBounds = aspectFitBounds(
    clampedCandidateBounds,
    viewport,
    map,
    candidates,
  );
  const fitted = readableBounds(aspectBounds, frame, viewport, candidates);
  const target = currentTarget(frame);
  const radius = target ? detectionRange(frame, target) : 0;
  const scale = fitted.scale;
  const minimumRegionDimensionPx = regionMinimumDimension(frame) * scale;
  const targetDetectionDiameterPx = radius * 2 * scale;
  const twoKilometerSegmentPx = 2_000 * scale;
  const warnings: string[] = [];
  if (fitted.minimumsConstrained) {
    warnings.push("semantic candidate span prevents all minimum readability constraints");
  }
  if (target && target.prediction?.health?.status === "unavailable") warnings.push("prediction geometry unavailable");
  if (targetDetectionDiameterPx < TARGET_DIAMETER_MIN_PX) warnings.push("target detection range cannot reach 160px within map bounds");
  if (minimumRegionDimensionPx > 0 && minimumRegionDimensionPx < REGION_DIMENSION_MIN_PX) warnings.push("region geometry cannot reach 48px within map bounds");
  if (twoKilometerSegmentPx < TWO_KILOMETER_MIN_PX) warnings.push("2km scale cannot reach 120px within map bounds");
  if (width(fitted.bounds) >= width(map) || height(fitted.bounds) >= height(map)) warnings.push("camera is clamped by map bounds");
  return {
    worldBounds: fitted.bounds,
    scale,
    targetDetectionDiameterPx,
    minimumRegionDimensionPx,
    twoKilometerSegmentPx,
    readabilityWarnings: warnings,
  };
}

function rectanglesOverlap(left: LabelPlacement, right: LabelPlacement): boolean {
  return left.x < right.x + right.width
    && left.x + left.width > right.x
    && left.y < right.y + right.height
    && left.y + left.height > right.y;
}

/** Place labels deterministically and suppress only lower-priority collisions. */
export function stableLabelPlacements(candidates: LabelCandidate[]): LabelPlacement[] {
  const placed: LabelPlacement[] = [];
  [...candidates]
    .sort((left, right) => left.priority - right.priority || left.id.localeCompare(right.id))
    .forEach((candidate) => {
      const found = LABEL_OFFSETS
        .map((offset) => ({ ...candidate, x: candidate.anchor.x + offset.x, y: candidate.anchor.y + offset.y, suppressed: false }))
        .find((placement) => !placed.some((other) => !other.suppressed && rectanglesOverlap(placement, other)));
      if (found) {
        placed.push(found);
      } else {
        placed.push({ ...candidate, x: candidate.anchor.x, y: candidate.anchor.y, suppressed: true });
      }
    });
  return placed;
}
