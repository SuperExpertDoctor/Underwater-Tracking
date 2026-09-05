import {
  useEffect,
  useRef,
  useState,
  type MouseEvent,
  type PointerEvent,
  type WheelEvent,
} from "react";
import { LocateFixed, RadioTower } from "lucide-react";
import type {
  CarrierView,
  CovarianceEllipse,
  ExecutionRegionView,
  ExecutionView,
  MapBounds,
  OperationalFrame,
  Point2D,
  RegionalMissionView,
  RegionalPlanView,
  RegionTaskView,
  TaskGroupInstanceView,
  TargetEstimateView,
  UUVView,
} from "../types/frames";
import type { ViewConfig } from "../types/viewConfig";
import {
  boundsForPoints,
  clipRayToBounds,
  corridorPolygon,
  displayRegionPoints,
  pointInPolygon,
  screenToWorld,
  spriteHitAreaContains,
  worldToScreen,
  type ViewState,
} from "./map/geometry";
import {
  coverImageRect,
  loadSceneAssets,
  type SceneAssets,
} from "./map/sceneAssets";
import {
  semanticCameraForFrame,
  stableLabelPlacements,
  type LabelCandidate,
} from "./map/camera";
import RegionOverlay from "./map/RegionOverlay";
import PredictionOverlay from "./map/PredictionOverlay";
import WorldModelEventOverlay from "./map/WorldModelEventOverlay";
import { displayTargetName } from "../utils/presentation";
import { timelineRowsForFrame } from "./regionTimeline";
import { MAP_DISPLAY_CONFIG } from "../../configs/map_display";
import {
  groupForUuv,
  groupInstanceId,
  groupsByRegionSlot,
  executionCounts,
  ownerGroup,
  visibleExecutionUuvs,
} from "../state/executionSelectors";

export type TrailMode = "tail" | "full" | "comet";

interface CanvasMapProps {
  frame: OperationalFrame | null;
  selectedUuvId: string | null;
  onSelectUuv: (id: string | null) => void;
  selectedRegionId?: string | null;
  onSelectRegion?: (id: string | null) => void;
  showGrid: boolean;
  showPredictedRegions: boolean;
  showRegionHandoffs: boolean;
  showDetectionRange: boolean;
  trailMode: TrailMode;
  viewConfig: ViewConfig;
}

const COLORS = {
  ink: "#f8fdff",
  muted: "#c3d9e4",
  cyan: "#21d0c3",
  cyanSoft: "rgba(33, 208, 195, 0.20)",
  amber: "#f7bd45",
  red: "#ff7882",
  green: "#66e0ad",
  violet: "#c4b4ff",
  grid: "rgba(225, 245, 248, 0.15)",
};

const EMPTY_SCENE_ASSETS: SceneAssets = {
  background: null,
  aircraftCarrier: null,
  warship: null,
  uuv: null,
  submarine: null,
};

/** The carrier image points east; the warship image points north. */
export const CARRIER_ASSET_HEADING_OFFSET = 0;
export const WARSHIP_ASSET_HEADING_OFFSET = Math.PI / 2;

const UUV_HIT_TOLERANCE_PX = 6;
const MINIMUM_TARGET_ONLY_CAMERA_SPAN_M = 1000;
export const GRID_DIVISIONS = 24;
export const UUV_SENSOR_FOOTPRINT_SPAN_RAD = MAP_DISPLAY_CONFIG.uuvSensorSpanRad;
export const TARGET_MARKER_SIZE_RANGE_PX = { min: 24, max: 32 } as const;
export const UUV_MARKER_SIZE_RANGE_PX = { min: 22, max: 30 } as const;
export const SUBMARINE_ASSET_HEADING_OFFSET = Math.PI;
/** Multiplier applied to the current zoom when a predicted task region is selected. */
export const REGION_FOCUS_ZOOM_FACTOR = 2;
const MAX_MAP_ZOOM = 8;

export const CANVAS_LAYER_ORDER = [
  "map/grid",
  "regions/handoffs",
  "prediction corridor",
  "prediction centerline/samples",
  "target detection circle",
  "UUV sonar fans",
  "labels",
  "selection/errors",
] as const;

export type RegionLayerStatus =
  | "planned"
  | "active"
  | "handoff"
  | "degraded"
  | "uncovered";

export function regionLayerStyle(status: RegionLayerStatus) {
  const styles: Record<RegionLayerStatus, { fill: string; stroke: string; dash: number[] }> = {
    planned: { fill: "rgba(196, 180, 255, 0.10)", stroke: "rgba(196, 180, 255, 0.84)", dash: [7, 5] },
    active: { fill: "rgba(33, 208, 195, 0.14)", stroke: "rgba(33, 208, 195, 0.96)", dash: [] },
    handoff: { fill: "rgba(247, 189, 69, 0.14)", stroke: "rgba(247, 189, 69, 0.96)", dash: [4, 4] },
    degraded: { fill: "rgba(255, 120, 130, 0.14)", stroke: "rgba(255, 120, 130, 0.96)", dash: [2, 5] },
    uncovered: { fill: "rgba(173, 190, 205, 0.06)", stroke: "rgba(173, 190, 205, 0.78)", dash: [3, 6] },
  };
  return styles[status];
}

export function sensorLayerStyle(mode: UUVView["sensor_mode"]) {
  return mode === "active"
    ? { stroke: "rgba(247, 189, 69, 0.88)", fill: "rgba(247, 189, 69, 0.14)" }
    : { stroke: "rgba(33, 208, 195, 0.82)", fill: "rgba(33, 208, 195, 0.11)" };
}

export const TARGET_DETECTION_STYLE = {
  stroke: COLORS.red,
  fill: "rgba(255, 120, 130, 0.065)",
  lineDash: [4, 7],
} as const;
export const DETECTION_LABEL_LAYER = "labels" as const;

export interface DetectionZoneLabels {
  radiusM: number;
  detectedCount: number;
  rangeText: string;
  detectedText: string | null;
}

interface PaintedDetectionLayer {
  target_id: string;
  center: Point2D;
  radius_px: number;
  stroke_style: string;
  line_dash: number[];
}

interface PaintedSonarLayer {
  uuv_id: string;
  deployment_key: string;
  target_id: string | null;
  task_group_id: string | null;
  role: "active_verifier" | "passive_tracker" | null;
  sensor_mode: UUVView["sensor_mode"];
  center: Point2D;
  radius_px: number;
  start_angle_rad: number;
  end_angle_rad: number;
  stroke_style: string;
  fill_style: string;
}

interface PaintedVisualLayerContract {
  detection: PaintedDetectionLayer[];
  sonar: PaintedSonarLayer[];
}

interface PlatformMarkerRing {
  color: string;
  lineWidth: number;
  radiusPadding: number;
  highlightColor: string | null;
  highlightPadding: number;
}

export function carrierAssetRotation(headingRad: number): number {
  return -headingRad + CARRIER_ASSET_HEADING_OFFSET;
}

export function warshipAssetRotation(headingRad: number): number {
  return -headingRad + WARSHIP_ASSET_HEADING_OFFSET;
}

export function submarineAssetRotation(headingRad: number): number {
  return -headingRad + SUBMARINE_ASSET_HEADING_OFFSET;
}

export function markerRingStyle(
  color: string,
  selected: boolean,
): PlatformMarkerRing {
  return {
    color,
    lineWidth: selected ? 1.75 : 1.5,
    radiusPadding: 3,
    highlightColor: selected ? COLORS.ink : null,
    highlightPadding: selected ? 4 : 0,
  };
}

export function communicationRangeForUuv(
  frame: OperationalFrame,
  uuvId: string,
): number {
  return Math.max(
    0,
    ...(frame.communication_links ?? [])
      .filter(
        (link) =>
          (link.source_id === uuvId || link.target_id === uuvId),
      )
      .map((link) => link.limit_m)
      .filter((limit): limit is number => Number.isFinite(limit)),
  );
}

export function isWaterborneUuv(uuv: UUVView): boolean {
  return uuv.physically_exposed;
}

export function waterborneUuvs(frame: OperationalFrame): UUVView[] {
  return (frame.uuvs ?? []).filter(isWaterborneUuv);
}

/** Return only the eight spatial members of the authoritative execution groups. */
export function spatialExecutionUuvs(frame: OperationalFrame): UUVView[] {
  return visibleExecutionUuvs(frame);
}

export function executionTargetEstimates(frame: OperationalFrame): TargetEstimateView[] {
  if (!frame.execution) return frame.target_estimates;
  return frame.target_estimates.filter(
    (target) => target.target_id === frame.execution?.target_id,
  );
}

function mapCarriers(frame: OperationalFrame): CarrierView[] {
  void frame;
  return [];
}

export function targetDetectionRange(frame: OperationalFrame): number {
  return frame.execution?.tracking_policy.target_detection_radius_m ?? 0;
}

function executionGroupRole(
  group: TaskGroupInstanceView | undefined,
  uuvId: string,
): PaintedSonarLayer["role"] {
  if (!group || !group.member_uuv_ids.includes(uuvId)) return null;
  return group.sensor_mode === "active"
    ? "active_verifier"
    : group.sensor_mode === "passive"
      ? "passive_tracker"
      : null;
}

/**
 * Bound only the uncertainty ellipse shown by the UI. The source estimate is
 * intentionally left untouched so this cannot affect tracking or planning.
 */
export function displayCovarianceEllipse(
  ellipse: CovarianceEllipse,
  scale = 1,
): CovarianceEllipse {
  const maxSemimajorM = Math.max(
    1,
    MAP_DISPLAY_CONFIG.estimateEllipseMaxSemimajorM,
  );
  const maxAspectRatio = Math.max(
    1,
    MAP_DISPLAY_CONFIG.estimateEllipseMaxAspectRatio,
  );
  const axisScale = Math.min(1, maxSemimajorM / ellipse.semimajor_m);
  const semimajorM = ellipse.semimajor_m * axisScale;
  const semiminorM = Math.min(
    semimajorM,
    Math.max(
      ellipse.semiminor_m * axisScale,
      semimajorM / maxAspectRatio,
      scale > 0
        ? MAP_DISPLAY_CONFIG.estimateEllipseMinSemiminorPx / scale
        : 0,
    ),
  );
  if (
    axisScale === 1 &&
    semiminorM === ellipse.semiminor_m
  ) return ellipse;
  return {
    ...ellipse,
    semimajor_m: semimajorM,
    semiminor_m: Math.max(0.001, semiminorM),
  };
}

export interface UuvSensorFootprint {
  radiusM: number;
  centerAngleRad: number;
  spanAngleRad: number;
  strokeStyle: string;
  fillStyle: string;
}

export interface UuvDetectionFootprint extends UuvSensorFootprint {
  mode: UUVView["sensor_mode"];
}

export function uuvSensorFootprint(
  uuv: UUVView,
  frame: OperationalFrame,
): UuvSensorFootprint | null {
  const policy = frame.execution?.tracking_policy;
  if (!policy) return null;
  const sensorHeadingRad = uuv.sensor_heading_rad ?? uuv.heading_rad;
  const centerAngleRad = sensorHeadingRad === 0 ? 0 : -sensorHeadingRad;
  const policyRadius = uuv.sensor_mode === "active"
    ? policy.uuv_active_detection_radius_m
    : policy.uuv_passive_detection_radius_m;
  return {
    radiusM: policyRadius,
    centerAngleRad,
    spanAngleRad: UUV_SENSOR_FOOTPRINT_SPAN_RAD,
    strokeStyle: sensorLayerStyle(uuv.sensor_mode).stroke,
    fillStyle: sensorLayerStyle(uuv.sensor_mode).fill,
  };
}

export function uuvDetectionFootprint(
  uuv: UUVView,
  frame: OperationalFrame,
): UuvDetectionFootprint | null {
  const footprint = uuvSensorFootprint(uuv, frame);
  return footprint ? { ...footprint, mode: uuv.sensor_mode } : null;
}

export function detectionZoneLabels(
  frame: OperationalFrame,
  target: TargetEstimateView,
): DetectionZoneLabels {
  const radiusM = targetDetectionRange(
    frame,
  );
  const detectedCount = detectedPlatformIds(frame, target).length;
  return {
    radiusM,
    detectedCount,
    rangeText: formatRange(radiusM),
    detectedText: detectedCount ? `${detectedCount} DETECTED` : null,
  };
}

export function shouldDrawDetectionRange(enabled: boolean): boolean {
  return enabled;
}

function executionEffectStatus(
  lifecycle: RegionalMissionView["lifecycle"],
): RegionTaskView["effect"]["status"] {
  if (lifecycle === "ACTIVE_SCAN" || lifecycle === "PASSIVE_TRACK") return "active";
  if (lifecycle === "HANDOFF_PENDING") return "handoff_ready";
  if (lifecycle === "DEGRADED") return "degraded";
  if (lifecycle === "UNCOVERED") return "uncovered";
  return "planned";
}

function executionRegionTaskView(
  region: ExecutionRegionView,
  groups: TaskGroupInstanceView[],
  execution: ExecutionView,
): RegionTaskView {
  const lifecycle = executionLifecycle(region.status);
  const runtimeGroups = groups
    .filter((group) => group.lifecycle !== "disappeared");
  const preferredGroup = runtimeGroups.find(
    (group) => groupInstanceId(group) === execution.tracking_control?.tracking_owner_group_id,
  ) ?? runtimeGroups.find((group) => group.lifecycle !== "exiting")
    ?? runtimeGroups[0];
  const activeIds = runtimeGroups.flatMap((group) =>
    group.sensor_mode === "active" ? [...group.member_uuv_ids] : [],
  );
  const passiveIds = runtimeGroups.flatMap((group) =>
    group.sensor_mode === "passive" ? [...group.member_uuv_ids] : [],
  );
  const assignedUuvIds = [...activeIds, ...passiveIds];
  const effectStatus = executionEffectStatus(lifecycle);
  return {
    region_id: region.region_id,
    display_name: region.region_id,
    target_id: region.target_id,
    geometry: region.geometry,
    top_left_xy: region.top_left_xy,
    bottom_right_xy: region.bottom_right_xy,
    start_time_s: region.start_s,
    end_time_s: region.end_s,
    predecessor_region_ids: region.predecessor_region_id
      ? [region.predecessor_region_id]
      : [],
    successor_region_ids: region.successor_region_id
      ? [region.successor_region_id]
      : [],
    assigned_uuv_ids: [...new Set(assignedUuvIds)],
    task_group_ids: runtimeGroups.map(groupInstanceId),
    authoritative_geometry: true,
    tracking_mode: "heuristic_uuv",
    uuv_roles: [
      ...activeIds.map(() => "active_verifier" as const),
      ...passiveIds.map(() => "passive_tracker" as const),
    ],
    group_id: preferredGroup
      ? groupInstanceId(preferredGroup)
      : null,
    status: region.status,
    revision: region.execution_revision,
    effect: {
      status: effectStatus,
      coverage_ratio: effectStatus === "uncovered" ? 0 : 1,
      quality_score: effectStatus === "degraded" ? 0 : 1,
      handoff_progress: region.status === "handoff_pending" ? 1 : 0,
      quality_source: "region_telemetry",
      hard_guard_reasons: [],
      expert_feedback_ids: [],
    },
  };
}

function executionLifecycle(
  status: ExecutionRegionView["status"],
): RegionalMissionView["lifecycle"] {
  const lifecycleByStatus: Record<
    ExecutionRegionView["status"],
    RegionalMissionView["lifecycle"]
  > = {
    planned: "PLANNED",
    prepositioning: "CARRIER_DEPLOYING",
    active: "ACTIVE_SCAN",
    passive: "PASSIVE_TRACK",
    handoff_pending: "HANDOFF_PENDING",
    handoff_completed: "TRACKING_COMPLETED",
    monitoring_complete: "TRACKING_COMPLETED",
    degraded: "DEGRADED",
    uncovered: "UNCOVERED",
  };
  return lifecycleByStatus[status];
}

function executionRegionalPlan(frame: OperationalFrame): RegionalPlanView[] {
  const execution = frame.execution;
  if (!execution) return [];
  const groupsByRegion = groupsByRegionSlot(frame);
  const regions = [...execution.regions]
    .sort((left, right) => left.slot_index - right.slot_index)
    .map((region) =>
      executionRegionTaskView(
        region,
        groupsByRegion.get(region.region_id) ?? [],
        execution,
      ),
    );
  return [
    {
      target_id: execution.target_id,
      prediction_id: execution.regions[0]?.prediction_id ?? "execution",
      revision: execution.execution_revision,
      cell_size_m: 1,
      evidence_ids: execution.evidence_ids,
      current_handoff_region_id: execution.current_region_id,
      next_handoff_region_id: execution.next_region_id,
      regions,
    },
  ];
}

/** Return the authoritative runtime regional plan. */
export function displayRegionalPlans(frame: OperationalFrame): RegionalPlanView[] {
  return frame.execution ? executionRegionalPlan(frame) : [];
}

export function cameraBoundsForFrame(
  frame: OperationalFrame,
  viewConfig: ViewConfig,
  showDetectionRange: boolean,
  showPredictedRegions = true,
): MapBounds {
  const includeDetectionRange =
    showDetectionRange || viewConfig.focusMode === "full_area";
  let hasPredictionCenterline = false;
  let hasVisibleRegionalCells = false;
  const points: Point2D[] =
    viewConfig.focusMode === "full_area"
      ? [
          { x: frame.map_bounds.min_x, y: frame.map_bounds.min_y },
          { x: frame.map_bounds.min_x, y: frame.map_bounds.max_y },
          { x: frame.map_bounds.max_x, y: frame.map_bounds.min_y },
          { x: frame.map_bounds.max_x, y: frame.map_bounds.max_y },
        ]
      : [];
  executionTargetEstimates(frame).forEach((target) => {
    points.push(target.mean);
    if (showPredictedRegions) {
      points.push(
        ...(target.world_model?.events.map((event) => event.predicted_position) ?? []),
      );
    }
    const prediction = target.prediction;
    if (prediction) {
      const immCenterline = prediction.imm_centerline_xy?.length
        ? prediction.imm_centerline_xy
        : prediction.centerline_xy;
      const immRadii = prediction.imm_radius_m?.length
        ? prediction.imm_radius_m
        : prediction.radius_m;
      hasPredictionCenterline ||= immCenterline.length >= 2;
      points.push(
        ...corridorPolygon(immCenterline, immRadii),
      );
    }
    if (includeDetectionRange) {
      const radius = targetDetectionRange(frame);
      points.push(
        { x: target.mean.x - radius, y: target.mean.y },
        { x: target.mean.x + radius, y: target.mean.y },
        { x: target.mean.x, y: target.mean.y - radius },
        { x: target.mean.x, y: target.mean.y + radius },
      );
    }
  });
  mapCarriers(frame).forEach((carrier) => points.push(carrier.position));
  spatialExecutionUuvs(frame).forEach((uuv) => {
    points.push(uuv.position);
    const policy = frame.execution?.tracking_policy;
    const radius = uuv.sensor_mode === "active"
      ? policy?.uuv_active_detection_radius_m
      : policy?.uuv_passive_detection_radius_m;
    if (radius !== undefined) {
      points.push(
        { x: uuv.position.x - radius, y: uuv.position.y },
        { x: uuv.position.x + radius, y: uuv.position.y },
        { x: uuv.position.x, y: uuv.position.y - radius },
        { x: uuv.position.x, y: uuv.position.y + radius },
      );
    }
  });
  if (showPredictedRegions) {
    const displayRegions = displayRegionalPlans(frame).flatMap((plan) => plan.regions);
    displayRegions.forEach((region) => {
      const geometry = displayRegionPoints(region);
      hasVisibleRegionalCells ||= geometry.length >= 3;
      points.push(...geometry);
    });
  }
  const bounds =
    boundsForPoints(
      points,
      viewConfig.focusMode === "full_area" ? 0 : viewConfig.predictionPadding,
    ) ?? frame.map_bounds;
  const hasOnlyTargetMean =
    viewConfig.focusMode !== "full_area" &&
    !includeDetectionRange &&
    executionTargetEstimates(frame).length === 1 &&
    !hasPredictionCenterline &&
    !hasVisibleRegionalCells;
  return hasOnlyTargetMean
    ? expandBoundsToMinimumSpan(bounds, MINIMUM_TARGET_ONLY_CAMERA_SPAN_M)
    : bounds;
}

function expandBoundsToMinimumSpan(
  bounds: MapBounds,
  minimumSpan: number,
): MapBounds {
  const centerX = (bounds.min_x + bounds.max_x) / 2;
  const centerY = (bounds.min_y + bounds.max_y) / 2;
  const width = Math.max(minimumSpan, bounds.max_x - bounds.min_x);
  const height = Math.max(minimumSpan, bounds.max_y - bounds.min_y);
  return {
    min_x: centerX - width / 2,
    max_x: centerX + width / 2,
    min_y: centerY - height / 2,
    max_y: centerY + height / 2,
  };
}

export function clampedMarkerPixels(
  value: number,
  min: number,
  max: number,
): number {
  return Math.max(min, Math.min(max, value));
}

export function regionLabelForZoom(
  region: RegionTaskView,
  zoom: number,
): string {
  if (zoom < 1.15) return "区域";
  const ordinal = region.display_name.match(
    /(?:region|区域)[_\s-]?(\d+)$/i,
  )?.[1];
  return ordinal ? `R${ordinal.padStart(2, "0")}` : region.region_id;
}

export function hitTestRegion(
  point: Point2D,
  regions: RegionTaskView[],
): RegionTaskView | null {
  return (
    regions.find((region) => pointInPolygon(point, displayRegionPoints(region))) ?? null
  );
}

export function detectedPlatformIds(
  frame: OperationalFrame,
  target: TargetEstimateView,
): string[] {
  const visibleUuvs = spatialExecutionUuvs(frame);
  const visibleIds = new Set(visibleUuvs.map((uuv) => uuv.uuv_id));
  const explicit =
    target.detected_platform_ids ?? frame.adversary?.detected_platform_ids;
  if (explicit) {
    return [...new Set(explicit.filter((platformId) => visibleIds.has(platformId)))];
  }
  const radius = targetDetectionRange(frame);
  const platforms = visibleUuvs.map((uuv) => ({
    id: uuv.uuv_id,
    position: uuv.position,
  }));
  return platforms
    .filter((platform) => distance(platform.position, target.mean) <= radius)
    .map((platform) => platform.id);
}

/** The selected UUV and its peer UUVs are the only platforms ring-highlighted on the map. */
export function highlightedUuvIds(
  frame: OperationalFrame,
  selectedUuvId: string | null,
): Set<string> {
  if (!selectedUuvId) return new Set();
  const visibleUuvs = spatialExecutionUuvs(frame);
  const selected = visibleUuvs.find((uuv) => uuv.uuv_id === selectedUuvId);
  if (!selected) return new Set();
  const executionGroup = groupForUuv(frame, selected.uuv_id);
  return new Set(executionGroup?.member_uuv_ids ?? []);
}

/** Return the waterborne UUVs responsible for the region executing now. */
export function currentTaskUuvIds(frame: OperationalFrame): Set<string> {
  const visibleIds = new Set(spatialExecutionUuvs(frame).map((uuv) => uuv.uuv_id));
  const currentGroups = frame.execution?.task_groups.filter(
    (group) => group.region_id === frame.execution?.current_region_id,
  ) ?? [];
  return new Set(
    currentGroups
      .filter((group) => group.lifecycle !== "disappeared")
      .flatMap((group) => group.member_uuv_ids)
      .filter((id) => visibleIds.has(id)),
  );
}

export function uuvSpriteAppearance(
  uuv: UUVView,
  image: HTMLImageElement | null,
  _scale: number,
  selected: boolean,
  markerPixels = 30,
) {
  const stateColor =
    uuv.status === "unavailable"
      ? COLORS.red
      : uuv.sensor_mode === "active"
        ? COLORS.amber
        : COLORS.cyan;
  return {
    size: screenSpriteSize(
      image,
      markerPixels,
      UUV_MARKER_SIZE_RANGE_PX.min,
      UUV_MARKER_SIZE_RANGE_PX.max,
    ),
    rotation: -uuv.heading_rad,
    cueColors: [
      stateColor,
      ...(uuv.reserved ? [COLORS.violet] : []),
      ...(selected ? [COLORS.ink] : []),
    ],
    markerRing: markerRingStyle(stateColor, selected),
  };
}

export default function CanvasMap({
  frame,
  selectedUuvId,
  onSelectUuv,
  selectedRegionId: controlledRegionId,
  onSelectRegion,
  showGrid,
  showPredictedRegions,
  showRegionHandoffs,
  showDetectionRange,
  trailMode,
  viewConfig,
}: CanvasMapProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const frameRef = useRef(frame);
  const lastPaintedFrameRef = useRef<OperationalFrame | null>(null);
  const lastPaintedMapBoundsRef = useRef<MapBounds | null>(null);
  const lastPaintedViewRef = useRef<ViewState>({ zoom: 1, pan: { x: 0, y: 0 } });
  const lastPaintedVisualLayersRef = useRef<PaintedVisualLayerContract | null>(null);
  const paintSequenceRef = useRef(0);
  const viewRef = useRef<ViewState>({ zoom: 1, pan: { x: 0, y: 0 } });
  const semanticBoundsRef = useRef<MapBounds | null>(null);
  const lastFittedPredictionRevisionRef = useRef<number | null>(null);
  const lastFittedDetectionRangeRef = useRef<boolean | null>(null);
  const lastExecutionUsableRef = useRef(false);
  const userCameraDirtyRef = useRef(false);
  const sizeRef = useRef({ width: 1, height: 1, dpr: 1 });
  const dragRef = useRef<{ x: number; y: number; pan: Point2D } | null>(null);
  const redrawRef = useRef<number | null>(null);
  const drawOptionsRef = useRef({
    showGrid,
    showPredictedRegions,
    showRegionHandoffs,
    showDetectionRange,
    trailMode,
    selectedUuvId,
    viewConfig,
    mapBounds: null as MapBounds | null,
  });
  const assetsRef = useRef<SceneAssets>(EMPTY_SCENE_ASSETS);
  const [hovered, setHovered] = useState(false);
  const [sceneAssetsReady, setSceneAssetsReady] = useState(false);
  const [internalSelectedRegionId, setInternalSelectedRegionId] = useState<
    string | null
  >(null);
  const [mapVersion, setMapVersion] = useState(0);
  const [paintSequence, setPaintSequence] = useState(0);
  const regionSelectionIsControlled = controlledRegionId !== undefined;
  const selectedRegionId = regionSelectionIsControlled
    ? controlledRegionId
    : internalSelectedRegionId;
  const allRegions = frame
    ? displayRegionalPlans(frame).flatMap((plan) => plan.regions)
    : [];
  const selectedRegion =
    allRegions.find((region) => region.region_id === selectedRegionId) ?? null;
  const frameExecutionCounts = frame
    ? executionCounts(frame)
    : {
        visibleUuvs: 0,
        enteringGroups: 0,
        exitingGroups: 0,
        activeScanGroups: 0,
        passiveTrackGroups: 0,
      };
  const trackingOwner = frame ? ownerGroup(frame) : undefined;
  const trackingOwnerGroupId = trackingOwner ? groupInstanceId(trackingOwner) : null;
  const paintedFrame = lastPaintedFrameRef.current;
  const paintedVisualLayers = paintedFrame
    ? lastPaintedVisualLayersRef.current
    : null;
  const paintedTaskUuvIds = paintedFrame
    ? currentTaskUuvIds(paintedFrame)
    : new Set<string>();
  const paintedCurrentTaskGroup = paintedFrame?.execution?.task_groups.find(
    (group) => group.region_id === paintedFrame.execution?.current_region_id,
  );
  const taskUuvTelemetry = paintedFrame
    ? [...paintedTaskUuvIds].sort().flatMap((uuvId) => {
      const uuv = spatialExecutionUuvs(paintedFrame).find((candidate) => candidate.uuv_id === uuvId);
      const telemetryGroup = groupForUuv(paintedFrame, uuvId) ?? paintedCurrentTaskGroup;
      return uuv
        ? [{
          uuv_id: uuv.uuv_id,
          deployment_key: deploymentAwareUuvKey(uuv),
          physically_exposed: uuv.physically_exposed,
          sensor_mode: uuv.sensor_mode,
          task_group_id: telemetryGroup ? groupInstanceId(telemetryGroup) : null,
          role: executionGroupRole(
            telemetryGroup,
            uuv.uuv_id,
          ),
          tracked_target_id: uuv.tracked_target_id ?? uuv.tracked_target ?? null,
          position: uuv.position,
          heading_rad: uuv.heading_rad,
          sensor_heading_rad: uuv.sensor_heading_rad ?? null,
          active_range_m: uuv.active_range_m ?? null,
          passive_range_m: uuv.passive_range_m ?? null,
        }]
        : [];
    })
    : [];
  const paintedTarget = paintedFrame
    ? executionTargetEstimates(paintedFrame)[0] ?? null
    : null;
  const paintedPrediction = paintedTarget?.prediction ?? null;
  const paintedMapBounds = paintedFrame
    ? lastPaintedMapBoundsRef.current ?? paintedFrame.map_bounds
    : null;
  const paintedView = paintedFrame ? lastPaintedViewRef.current : viewRef.current;
  const visibleBounds = frame
    ? semanticBoundsRef.current ?? cameraBoundsForFrame(
        frame,
        viewConfig,
        showDetectionRange,
        showPredictedRegions,
      )
    : null;
  const scaleBar = visibleBounds
    ? mapScaleForView(
        visibleBounds,
        sizeRef.current.width,
        sizeRef.current.height,
        viewRef.current.zoom,
      )
    : null;

  frameRef.current = frame;
  drawOptionsRef.current = {
    showGrid,
    showPredictedRegions,
    showRegionHandoffs,
    showDetectionRange,
    trailMode,
    selectedUuvId,
    viewConfig,
    mapBounds: visibleBounds,
  };

  const requestDraw = () => {
    if (redrawRef.current !== null) return;
    redrawRef.current = window.requestAnimationFrame(() => {
      redrawRef.current = null;
      const frameToPaint = frameRef.current;
      const viewToPaint: ViewState = {
        zoom: viewRef.current.zoom,
        pan: { ...viewRef.current.pan },
      };
      const optionsToPaint = drawOptionsRef.current;
      const paintedLayers = drawMap(
        canvasRef.current,
        frameToPaint,
        viewToPaint,
        sizeRef.current,
        optionsToPaint,
        assetsRef.current,
      );
      if (!paintedLayers) return;
      lastPaintedFrameRef.current = frameToPaint;
      lastPaintedMapBoundsRef.current = optionsToPaint.mapBounds
        ? { ...optionsToPaint.mapBounds }
        : frameToPaint
          ? { ...frameToPaint.map_bounds }
          : null;
      lastPaintedViewRef.current = viewToPaint;
      lastPaintedVisualLayersRef.current = paintedLayers;
      paintSequenceRef.current += 1;
      setPaintSequence(paintSequenceRef.current);
    });
  };

  const fitSemanticCamera = (force = false) => {
    const frameValue = frameRef.current;
    if (!frameValue) {
      semanticBoundsRef.current = null;
      lastFittedPredictionRevisionRef.current = null;
      lastFittedDetectionRangeRef.current = null;
      lastExecutionUsableRef.current = false;
      return;
    }
    const target = frameValue.execution
      ? frameValue.target_estimates.find(
          (estimate) => estimate.target_id === frameValue.execution?.target_id,
        )
      : frameValue.target_estimates[0];
    const predictionRevision =
      frameValue.execution?.prediction_revision
      ?? target?.prediction?.prediction_revision
      ?? null;
    const executionUsable = frameValue.execution?.health_status === "current"
      || frameValue.execution?.health_status === "degraded";
    const revisionChanged =
      predictionRevision !== lastFittedPredictionRevisionRef.current;
    const executionBecameUsable = executionUsable && !lastExecutionUsableRef.current;
    if (
      force
      || (!userCameraDirtyRef.current && (
        semanticBoundsRef.current === null
        || revisionChanged
        || executionBecameUsable
        || showDetectionRange !== lastFittedDetectionRangeRef.current
      ))
    ) {
      semanticBoundsRef.current = semanticCameraForFrame(
        frameValue,
        {
          width: sizeRef.current.width,
          height: sizeRef.current.height,
        },
        showDetectionRange,
      ).worldBounds;
      viewRef.current = { zoom: 1, pan: { x: 0, y: 0 } };
      userCameraDirtyRef.current = false;
    }
    lastFittedPredictionRevisionRef.current = predictionRevision;
    lastFittedDetectionRangeRef.current = showDetectionRange;
    lastExecutionUsableRef.current = executionUsable;
    requestDraw();
    setMapVersion((value) => value + 1);
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return undefined;
    const updateSize = () => {
      const width = Math.max(1, container.clientWidth);
      const height = Math.max(1, container.clientHeight);
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      sizeRef.current = { width, height, dpr };
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      if (!userCameraDirtyRef.current && frameRef.current) fitSemanticCamera();
      requestDraw();
      setMapVersion((value) => value + 1);
    };
    updateSize();
    if (typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver(updateSize);
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    fitSemanticCamera();
    requestDraw();
  }, [
    frame,
    showGrid,
    showPredictedRegions,
    showRegionHandoffs,
    showDetectionRange,
    trailMode,
    selectedUuvId,
    viewConfig,
  ]);

  useEffect(() => {
    if (controlledRegionId === null) setInternalSelectedRegionId(null);
  }, [controlledRegionId]);

  useEffect(() => {
    if (!showPredictedRegions) {
      setInternalSelectedRegionId(null);
      onSelectRegion?.(null);
    }
  }, [onSelectRegion, showPredictedRegions]);

  useEffect(() => {
    let disposed = false;
    void loadSceneAssets().then((assets) => {
      if (disposed) return;
      assetsRef.current = assets;
      setSceneAssetsReady(true);
      requestDraw();
    });
    return () => {
      disposed = true;
    };
  }, []);

  useEffect(
    () => () => {
      if (redrawRef.current !== null) {
        window.cancelAnimationFrame(redrawRef.current);
        redrawRef.current = null;
      }
    },
    [],
  );

  const handlePointerDown = (event: PointerEvent<HTMLCanvasElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      x: event.clientX,
      y: event.clientY,
      pan: { ...viewRef.current.pan },
    };
  };

  const handlePointerMove = (event: PointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    if (!drag) return;
    viewRef.current = {
      ...viewRef.current,
      pan: {
        x: drag.pan.x + event.clientX - drag.x,
        y: drag.pan.y + event.clientY - drag.y,
      },
    };
    userCameraDirtyRef.current = true;
    requestDraw();
    setMapVersion((value) => value + 1);
  };

  const handlePointerUp = (event: PointerEvent<HTMLCanvasElement>) => {
    if (dragRef.current) {
      event.currentTarget.releasePointerCapture(event.pointerId);
      dragRef.current = null;
    }
  };

  const handleWheel = (event: WheelEvent<HTMLCanvasElement>) => {
    const bounds = drawOptionsRef.current.mapBounds;
    if (!bounds) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const cursor = {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top,
    };
    const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
    const { zoom, pan } = zoomAroundCursorForCanvas(
      bounds,
      sizeRef.current,
      viewRef.current,
      cursor,
      factor,
    );
    viewRef.current = { zoom, pan };
    userCameraDirtyRef.current = true;
    requestDraw();
    setMapVersion((value) => value + 1);
  };

  const handleClick = (event: MouseEvent<HTMLCanvasElement>) => {
    if (dragRef.current || !frameRef.current) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const point = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    const frameValue = frameRef.current;
    const bounds = drawOptionsRef.current.mapBounds ?? frameValue.map_bounds;
    const scale =
      fittedScaleForMap(bounds, sizeRef.current.width, sizeRef.current.height) *
      viewRef.current.zoom;
    const nearest = spatialExecutionUuvs(frameValue)
      .map((uuv) => ({
        id: uuv.uuv_id,
        distance: distance(
          point,
          worldToScreen(
            uuv.position,
            bounds,
            sizeRef.current.width,
            sizeRef.current.height,
            viewRef.current,
          ),
        ),
        selected: spriteHitAreaContains(
          point,
          worldToScreen(
            uuv.position,
            bounds,
            sizeRef.current.width,
            sizeRef.current.height,
            viewRef.current,
          ),
          uuvSpriteAppearance(
            uuv,
            assetsRef.current.uuv,
            scale,
            false,
            viewConfig.uuvMarkerPixels,
          ).size,
          uuv.heading_rad,
          UUV_HIT_TOLERANCE_PX,
        ),
      }))
      .filter((candidate) => candidate.selected)
      .sort((a, b) => a.distance - b.distance)[0];
    if (nearest) {
      onSelectUuv(nearest.id === selectedUuvId ? null : nearest.id);
      return;
    }
    const markerHit = executionTargetEstimates(frameValue).map((target) =>
        spriteHitAreaContains(
          point,
          worldToScreen(
            target.mean,
            bounds,
            sizeRef.current.width,
            sizeRef.current.height,
            viewRef.current,
          ),
          screenSpriteSize(
            assetsRef.current.submarine,
            viewConfig.targetMarkerPixels,
            TARGET_MARKER_SIZE_RANGE_PX.min,
            TARGET_MARKER_SIZE_RANGE_PX.max,
          ),
          submarineAssetRotation(
            target.heading_rad ?? target.covariance_ellipse?.rotation_rad ?? 0,
          ),
          UUV_HIT_TOLERANCE_PX,
        ),
      ).some(Boolean);
    if (markerHit) return;
    if (!showPredictedRegions) return;
    const regions = displayRegionalPlans(frameValue).flatMap(
      (plan) => plan.regions,
    );
    const region = hitTestRegion(
      screenToWorld(
        point,
        bounds,
        sizeRef.current.width,
        sizeRef.current.height,
        viewRef.current,
      ),
      regions,
    );
    const nextRegionId = region?.region_id ?? null;
    if (!regionSelectionIsControlled) setInternalSelectedRegionId(nextRegionId);
    onSelectRegion?.(nextRegionId);
  };

  const handleDoubleClick = (event: MouseEvent<HTMLCanvasElement>) => {
    const frameValue = frameRef.current;
    if (!frameValue || !showPredictedRegions) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const point = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    const regions = displayRegionalPlans(frameValue).flatMap((plan) => plan.regions);
    const region = hitTestRegion(
      screenToWorld(
        point,
        drawOptionsRef.current.mapBounds ?? frameValue.map_bounds,
        sizeRef.current.width,
        sizeRef.current.height,
        viewRef.current,
      ),
      regions,
    );
    if (!region) return;
    viewRef.current = focusRegionForCanvas(
      drawOptionsRef.current.mapBounds ?? frameValue.map_bounds,
      sizeRef.current,
      region,
      nextRegionFocusZoom(viewRef.current.zoom),
    );
    userCameraDirtyRef.current = true;
    if (!regionSelectionIsControlled) setInternalSelectedRegionId(region.region_id);
    onSelectRegion?.(region.region_id);
    requestDraw();
    setMapVersion((value) => value + 1);
  };

  const fitAll = () => {
    userCameraDirtyRef.current = false;
    fitSemanticCamera(true);
  };

  return (
    <div
      className="canvas-area"
      ref={containerRef}
      data-show-grid={Boolean(showGrid && !frame?.execution)}
      data-show-predicted-regions={showPredictedRegions}
      data-show-region-handoffs={showRegionHandoffs}
      data-show-detection-range={showDetectionRange}
      data-trail-mode={trailMode}
      data-focus-mode={viewConfig.focusMode}
      data-map-version={mapVersion}
      data-camera-dirty={userCameraDirtyRef.current}
      data-camera-pan={JSON.stringify(viewRef.current.pan)}
      data-camera-zoom={viewRef.current.zoom}
      data-visible-bounds={visibleBounds ? JSON.stringify(visibleBounds) : undefined}
      data-rendered-frame-id={paintedFrame?.frame_id}
      data-rendered-sim-time-s={paintedFrame?.sim_time_s}
      data-rendered-execution-revision={paintedFrame?.execution?.execution_revision}
      data-rendered-prediction-id={paintedPrediction?.prediction_id}
      data-rendered-prediction-revision={paintedPrediction?.prediction_revision}
      data-rendered-target-id={paintedTarget?.target_id}
      data-last-painted-frame-id={paintedFrame?.frame_id}
      data-last-painted-sim-time-s={paintedFrame?.sim_time_s}
      data-last-painted-execution-revision={paintedFrame?.execution?.execution_revision}
      data-last-painted-prediction-id={paintedPrediction?.prediction_id}
      data-last-painted-prediction-revision={paintedPrediction?.prediction_revision}
      data-last-painted-target-id={paintedTarget?.target_id}
      data-last-painted-paint-sequence={paintSequence}
      data-last-painted-visible-bounds={paintedMapBounds ? JSON.stringify(paintedMapBounds) : undefined}
      data-last-painted-camera-pan={paintedFrame ? JSON.stringify(paintedView.pan) : undefined}
      data-last-painted-camera-zoom={paintedFrame ? paintedView.zoom : undefined}
      data-last-painted-plan-version={paintedFrame?.plan_version}
      data-last-painted-execution-region-count={paintedFrame?.execution?.regions.length}
      data-last-painted-visual-layers={
        paintedVisualLayers ? JSON.stringify(paintedVisualLayers) : undefined
      }
    >
      <canvas
        ref={canvasRef}
        tabIndex={0}
        onClick={handleClick}
        onDoubleClick={handleDoubleClick}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onWheel={handleWheel}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        data-carrier-count={frame ? mapCarriers(frame).length : 0}
        data-scene-assets-ready={sceneAssetsReady}
        data-waterborne-uuv-count={frame ? spatialExecutionUuvs(frame).length : 0}
        data-execution-uuv-count={frame ? spatialExecutionUuvs(frame).length : 0}
        data-visible-uuv-count={frameExecutionCounts.visibleUuvs}
        data-entering-group-count={frameExecutionCounts.enteringGroups}
        data-exiting-group-count={frameExecutionCounts.exitingGroups}
        data-active-scan-group-count={frameExecutionCounts.activeScanGroups}
        data-passive-track-group-count={frameExecutionCounts.passiveTrackGroups}
        data-tracking-owner-group-id={trackingOwnerGroupId ?? undefined}
        data-execution-region-count={frame?.execution?.regions.length ?? 0}
        data-task-group-count={frame?.execution?.task_groups.length ?? 0}
        data-region-count={frame?.execution?.regions.length ?? 0}
        data-task-group-size={frame?.execution?.tracking_policy?.task_group_size ?? undefined}
        data-region-side-m={frame?.execution?.tracking_policy?.task_region_side_m ?? undefined}
        data-target-radius-m={frame?.execution?.tracking_policy?.target_detection_radius_m ?? undefined}
        data-uuv-radius-m={frame?.execution?.tracking_policy
          ? Math.max(
            frame.execution.tracking_policy.uuv_active_detection_radius_m,
            frame.execution.tracking_policy.uuv_passive_detection_radius_m,
          )
          : undefined}
        data-tracking-mode={frame?.execution?.tracking_control?.mode ?? undefined}
        data-target-estimate-count={frame ? executionTargetEstimates(frame).length : 0}
        data-world-model-event-count={
          frame
            ? executionTargetEstimates(frame).reduce(
            (count, target) => count + (target.world_model?.events.length ?? 0),
            0,
              )
            : 0
        }
        data-plan-version={frame?.plan_version ?? 0}
        data-current-task-uuv-ids={[...paintedTaskUuvIds].sort().join(",")}
        data-current-task-uuv-telemetry={JSON.stringify(taskUuvTelemetry)}
        data-rendered-frame-id={paintedFrame?.frame_id}
        data-rendered-sim-time-s={paintedFrame?.sim_time_s}
        data-rendered-execution-revision={paintedFrame?.execution?.execution_revision}
        data-rendered-prediction-id={paintedPrediction?.prediction_id}
        data-rendered-prediction-revision={paintedPrediction?.prediction_revision}
        data-rendered-target-id={paintedTarget?.target_id}
        data-last-painted-frame-id={paintedFrame?.frame_id}
        data-last-painted-sim-time-s={paintedFrame?.sim_time_s}
        data-last-painted-execution-revision={paintedFrame?.execution?.execution_revision}
        data-last-painted-prediction-id={paintedPrediction?.prediction_id}
        data-last-painted-prediction-revision={paintedPrediction?.prediction_revision}
        data-last-painted-target-id={paintedTarget?.target_id}
        data-last-painted-paint-sequence={paintSequence}
        data-last-painted-visible-bounds={paintedMapBounds ? JSON.stringify(paintedMapBounds) : undefined}
        data-last-painted-camera-pan={paintedFrame ? JSON.stringify(paintedView.pan) : undefined}
        data-last-painted-camera-zoom={paintedFrame ? paintedView.zoom : undefined}
        data-last-painted-plan-version={paintedFrame?.plan_version}
        data-last-painted-execution-region-count={paintedFrame?.execution?.regions.length}
        data-last-painted-visual-layers={
          paintedVisualLayers ? JSON.stringify(paintedVisualLayers) : undefined
        }
        style={{
          cursor: dragRef.current
            ? "grabbing"
            : hovered
              ? "crosshair"
              : "default",
        }}
        aria-label="水下跟踪态势地图，支持拖动、滚轮缩放、区域双击聚焦与 UUV、区域选择"
      />
      {showPredictedRegions && paintedFrame && paintedMapBounds && (
        <RegionOverlay
          plans={displayRegionalPlans(paintedFrame)}
          timeline={timelineRowsForFrame(paintedFrame)}
          selectedRegionId={selectedRegionId}
          currentRegionId={paintedFrame.execution?.current_region_id}
          nextRegionId={paintedFrame.execution?.next_region_id}
          onSelectRegion={onSelectRegion}
          width={sizeRef.current.width}
          height={sizeRef.current.height}
          interactive={false}
          showHandoffs={showRegionHandoffs}
          project={(point) =>
            worldToScreen(
              point,
              paintedMapBounds,
              sizeRef.current.width,
              sizeRef.current.height,
              paintedView,
            )
          }
        />
      )}
      {showPredictedRegions && paintedFrame && paintedMapBounds && (
        <PredictionOverlay
          predictions={executionTargetEstimates(paintedFrame).flatMap((target) =>
            target.prediction
              ? [{ targetId: target.target_id, prediction: target.prediction }]
              : [],
          )}
          width={sizeRef.current.width}
          height={sizeRef.current.height}
          project={(point) =>
            worldToScreen(
              point,
              paintedMapBounds,
              sizeRef.current.width,
              sizeRef.current.height,
              paintedView,
            )
          }
        />
      )}
      {showPredictedRegions && paintedFrame && paintedMapBounds && (
        <WorldModelEventOverlay
          targets={executionTargetEstimates(paintedFrame)}
          width={sizeRef.current.width}
          height={sizeRef.current.height}
          project={(point) =>
            worldToScreen(
              point,
              paintedMapBounds,
              sizeRef.current.width,
              sizeRef.current.height,
              paintedView,
            )
          }
        />
      )}
      {!frame && (
        <div className="map-empty" role="status">
          <RadioTower size={22} />
          <strong>等待作业态势</strong>
          <span>实时估计帧或回放帧接入后将在此显示。</span>
        </div>
      )}
      <div className="map-tools" aria-label="地图工具">
        <button
          type="button"
          onClick={fitAll}
          title="适配当前焦点"
          aria-label="适配当前焦点"
        >
          <LocateFixed size={15} />
        </button>
        <span>{viewRef.current.zoom.toFixed(1)}×</span>
      </div>
      {showPredictedRegions && selectedRegion && (
        <div className="map-region-selection" role="status">
          <strong>区域 {selectedRegion.display_name}</strong>
          <span>
            {displayTargetName(selectedRegion.target_id)} ·{" "}
            {selectedRegion.effect.status}
          </span>
        </div>
      )}
      {scaleBar && (
        <div className="map-scale" aria-label={`地图比例尺 ${scaleBar.label}`}>
          <i style={{ width: `${scaleBar.widthPx}px` }} />
          {scaleBar.label}
        </div>
      )}
    </div>
  );
}

function zoomAroundCursorForCanvas(
  bounds: OperationalFrame["map_bounds"],
  size: { width: number; height: number },
  view: ViewState,
  cursor: Point2D,
  factor: number,
): ViewState {
  const before = screenToWorld(cursor, bounds, size.width, size.height, view);
  const zoom = Math.max(0.25, Math.min(8, view.zoom * factor));
  const after = worldToScreen(before, bounds, size.width, size.height, {
    zoom,
    pan: { x: 0, y: 0 },
  });
  return { zoom, pan: { x: cursor.x - after.x, y: cursor.y - after.y } };
}

/**
 * Returns the local map transform that places a region centre in the middle
 * of the canvas at the requested operator focus level. World coordinates remain
 * unchanged; regions, sprites, trails, and hit tests share this transform.
 */
export function focusRegionForCanvas(
  bounds: OperationalFrame["map_bounds"],
  size: { width: number; height: number },
  region: Pick<RegionTaskView, "geometry" | "top_left_xy" | "bottom_right_xy">,
  zoom: number,
): ViewState {
  const geometry = displayRegionPoints(region);
  const xs = geometry.map((point) => point.x);
  const ys = geometry.map((point) => point.y);
  const center = {
    x: (Math.min(...xs) + Math.max(...xs)) / 2,
    y: (Math.min(...ys) + Math.max(...ys)) / 2,
  };
  const focused = { zoom, pan: { x: 0, y: 0 } };
  const centerOnCanvas = worldToScreen(
    center,
    bounds,
    size.width,
    size.height,
    focused,
  );
  return {
    zoom,
    pan: {
      x: size.width / 2 - centerOnCanvas.x,
      y: size.height / 2 - centerOnCanvas.y,
    },
  };
}

export function nextRegionFocusZoom(currentZoom: number): number {
  return Math.min(MAX_MAP_ZOOM, currentZoom * REGION_FOCUS_ZOOM_FACTOR);
}

function drawMap(
  canvas: HTMLCanvasElement | null,
  frame: OperationalFrame | null,
  view: ViewState,
  size: { width: number; height: number; dpr: number },
  options: {
    showGrid: boolean;
    showPredictedRegions: boolean;
    showRegionHandoffs: boolean;
    showDetectionRange: boolean;
    trailMode: TrailMode;
    selectedUuvId: string | null;
    viewConfig: ViewConfig;
    mapBounds: MapBounds | null;
  },
  assets: SceneAssets,
): PaintedVisualLayerContract | null {
  const context = canvas?.getContext("2d");
  if (!context) return null;
  const { width, height, dpr } = size;
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);
  drawSceneBackground(context, assets.background, width, height);
  context.fillStyle = "rgba(5, 32, 73, 0.46)";
  context.fillRect(0, 0, width, height);
  if (!frame) return { detection: [], sonar: [] };
  const bounds = options.mapBounds ?? frame.map_bounds;
  const transform = (point: Point2D) =>
    worldToScreen(point, bounds, width, height, view);
  const scale = fittedScaleForMap(bounds, width, height) * view.zoom;
  const visibleUuvs = spatialExecutionUuvs(frame);
  const taskUuvIds = currentTaskUuvIds(frame);
  const highlighted = options.selectedUuvId
    ? highlightedUuvIds(frame, options.selectedUuvId)
    : taskUuvIds;
  const sonarUuvs = visibleUuvs;
  drawMapAndGrid(context, frame, bounds, transform, options);
  const detection = shouldDrawDetectionRange(options.showDetectionRange)
    ? drawTargetDetectionZones(context, frame, transform, scale)
    : [];
  const sonar = drawUuvSonarFields(
    context,
    frame,
    sonarUuvs,
    transform,
    scale,
    highlighted,
  );
  drawLabels(
    context,
    frame,
    assets,
    transform,
    scale,
    options,
    highlighted,
    visibleUuvs,
    { width, height },
  );
  drawSelectionAndErrors(context, frame, transform, highlighted, visibleUuvs);
  return { detection, sonar };
}

function drawMapAndGrid(
  context: CanvasRenderingContext2D,
  frame: OperationalFrame,
  bounds: MapBounds,
  transform: (point: Point2D) => Point2D,
  options: Pick<Parameters<typeof drawMap>[4], "showGrid" | "viewConfig">,
) {
  if (options.showGrid && !frame.execution)
    drawGrid(context, bounds, transform, options.viewConfig.gridDivisions);
}

function drawUuvSonarFields(
  context: CanvasRenderingContext2D,
  frame: OperationalFrame,
  visibleUuvs: UUVView[],
  transform: (point: Point2D) => Point2D,
  scale: number,
  highlighted: Set<string>,
): PaintedSonarLayer[] {
  const painted = drawUuvSensorFootprints(
    context,
    frame,
    visibleUuvs,
    transform,
    scale,
    highlighted,
  );
  if (highlighted.size) drawBearings(context, frame, transform, highlighted);
  return painted;
}

function drawLabels(
  context: CanvasRenderingContext2D,
  frame: OperationalFrame,
  assets: SceneAssets,
  transform: (point: Point2D) => Point2D,
  scale: number,
  options: {
    selectedUuvId: string | null;
    viewConfig: ViewConfig;
    trailMode: TrailMode;
    showDetectionRange: boolean;
  },
  highlighted: Set<string>,
  visibleUuvs: UUVView[],
  viewport: { width: number; height: number },
) {
  drawUuvTrails(context, transform, options.trailMode, highlighted, visibleUuvs);
  drawEstimates(context, frame, transform, scale);
  mapCarriers(frame).forEach((carrier) => {
    const image = carrier.role === "carrier" ? assets.aircraftCarrier : assets.warship;
    drawCarrier(context, carrier, image, transform, scale);
  });
  drawTargetSprites(context, frame, assets.submarine, transform, options.viewConfig.targetMarkerPixels);
  drawUuvSprites(context, assets.uuv, transform, scale, options.selectedUuvId, highlighted, options.viewConfig.uuvMarkerPixels, visibleUuvs);
  drawStableLabels(context, frame, transform, options, visibleUuvs, viewport);
}

interface CanvasLabelCandidate extends LabelCandidate {
  text: string;
  color: string;
  font: string;
  textOffsetX?: number;
}

function canvasLabel(
  id: string,
  text: string,
  anchor: Point2D,
  priority: number,
  color: string,
  font: string,
): CanvasLabelCandidate {
  return {
    id,
    text,
    anchor,
    priority,
    color,
    font,
    width: Math.max(18, text.length * 7),
    height: 14,
    textOffsetX: 0,
  };
}

export function stableLabelCandidatesForFrame(
  frame: OperationalFrame,
  transform: (point: Point2D) => Point2D,
  options: { selectedUuvId: string | null; showDetectionRange: boolean },
  visibleUuvs: UUVView[],
): CanvasLabelCandidate[] {
  const candidates: CanvasLabelCandidate[] = [];
  const selectedUuv = visibleUuvs.find((uuv) => uuv.uuv_id === options.selectedUuvId);
  if (selectedUuv) {
    candidates.push(canvasLabel(
      `selected:${selectedUuv.uuv_id}`,
      selectedUuv.uuv_id,
      transform(selectedUuv.position),
      0,
      COLORS.ink,
      "600 10px 'IBM Plex Mono', monospace",
    ));
  }
  executionTargetEstimates(frame).forEach((target) => {
    const detection = options.showDetectionRange ? detectionZoneLabels(frame, target) : null;
    const text = detection
      ? [
          displayTargetName(target.target_id),
          detection.rangeText,
          detection.detectedText,
        ]
          .filter((part): part is string => part !== null)
          .join(" ")
      : displayTargetName(target.target_id);
    candidates.push(canvasLabel(
      `target:${target.target_id}`,
      text,
      transform(target.mean),
      1,
      COLORS.ink,
      "600 11px 'IBM Plex Mono', monospace",
    ));
  });

  const activeIds = currentTaskUuvIds(frame);
  const labelledUuvs = activeIds.size || options.selectedUuvId
    ? visibleUuvs.filter(
      (uuv) => activeIds.has(uuv.uuv_id) || uuv.uuv_id === options.selectedUuvId,
    )
    : visibleUuvs;
  [...labelledUuvs]
    .sort((left, right) => {
      const leftRank = activeIds.has(left.uuv_id) ? 0 : 1;
      const rightRank = activeIds.has(right.uuv_id) ? 0 : 1;
      return leftRank - rightRank || left.uuv_id.localeCompare(right.uuv_id);
    })
    .forEach((uuv) => {
      if (uuv.uuv_id === options.selectedUuvId) return;
      candidates.push(canvasLabel(
        `uuv:${deploymentAwareUuvKey(uuv)}`,
        `${uuv.uuv_id} ${uuv.sensor_mode === "active" ? "ACT" : "PAS"}`,
        transform(uuv.position),
        activeIds.has(uuv.uuv_id) ? 4 : 5,
        COLORS.muted,
        "10px 'IBM Plex Mono', monospace",
      ));
    });

  mapCarriers(frame).forEach((carrier) => {
    candidates.push(canvasLabel(
      `carrier:${carrier.carrier_id}`,
      `${carrier.role === "carrier" ? "CARRIER" : "MOTHER SHIP"} ${carrier.carrier_id}`,
      transform(carrier.position),
      6,
      COLORS.ink,
      "600 10px 'IBM Plex Mono', monospace",
    ));
  });

  return candidates;
}

function measuredCanvasLabelCandidates(
  context: CanvasRenderingContext2D,
  candidates: CanvasLabelCandidate[],
): CanvasLabelCandidate[] {
  context.save();
  try {
    context.textAlign = "left";
    context.textBaseline = "top";
    return candidates.map((candidate) => {
      context.font = candidate.font;
      const metrics = context.measureText(candidate.text);
      const advanceWidth = Number.isFinite(metrics.width)
        ? metrics.width
        : candidate.width;
      const actualLeft = Number.isFinite(metrics.actualBoundingBoxLeft)
        ? metrics.actualBoundingBoxLeft
        : 0;
      const actualRight = Number.isFinite(metrics.actualBoundingBoxRight)
        ? metrics.actualBoundingBoxRight
        : advanceWidth;
      const actualHeight = Number.isFinite(metrics.actualBoundingBoxAscent)
        && Number.isFinite(metrics.actualBoundingBoxDescent)
        ? metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent
        : candidate.height;
      return {
        ...candidate,
        width: Math.max(
          candidate.width,
          Math.ceil(advanceWidth),
          Math.ceil(Math.max(0, actualRight - actualLeft)),
        ),
        height: Math.max(candidate.height, Math.ceil(Math.max(0, actualHeight))),
        textOffsetX: Math.max(0, Math.ceil(-actualLeft)),
      };
    });
  } finally {
    context.restore();
  }
}

export function drawStableLabels(
  context: CanvasRenderingContext2D,
  frame: OperationalFrame,
  transform: (point: Point2D) => Point2D,
  options: { selectedUuvId: string | null; showDetectionRange: boolean },
  visibleUuvs: UUVView[],
  viewport: { width: number; height: number },
) {
  const candidates = measuredCanvasLabelCandidates(context, stableLabelCandidatesForFrame(
    frame,
    transform,
    options,
    visibleUuvs,
  ));
  stableLabelPlacements(candidates, viewport).forEach((placement) => {
    if (placement.suppressed) return;
    const candidate = candidates.find((item) => item.id === placement.id);
    if (!candidate) return;
    context.save();
    context.fillStyle = candidate.color;
    context.font = candidate.font;
    context.textAlign = "left";
    context.textBaseline = "top";
    context.fillText(
      candidate.text,
      placement.x + (candidate.textOffsetX ?? 0),
      placement.y,
    );
    context.restore();
  });
}

function drawSelectionAndErrors(
  context: CanvasRenderingContext2D,
  frame: OperationalFrame,
  transform: (point: Point2D) => Point2D,
  highlighted: Set<string>,
  visibleUuvs: UUVView[],
) {
  if (!frame.execution) drawRecoveryLinks(context, frame, transform, visibleUuvs);
  if (highlighted.size) drawSelectedGroupLinks(context, frame, transform, highlighted, visibleUuvs);
}

function fittedScaleForMap(
  bounds: OperationalFrame["map_bounds"],
  width: number,
  height: number,
) {
  return Math.min(
    width / (bounds.max_x - bounds.min_x),
    height / (bounds.max_y - bounds.min_y),
  );
}

export interface MapScale {
  distanceM: number;
  widthPx: number;
  label: string;
}

export function mapScaleForView(
  bounds: OperationalFrame["map_bounds"],
  width: number,
  height: number,
  zoom: number,
  targetWidthPx = 96,
): MapScale {
  const pixelsPerMetre =
    fittedScaleForMap(bounds, Math.max(1, width), Math.max(1, height)) *
    Math.max(0.25, zoom);
  if (!Number.isFinite(pixelsPerMetre) || pixelsPerMetre <= 0) {
    return { distanceM: 0, widthPx: 0, label: "0 m" };
  }
  const rawDistanceM = Math.max(1, targetWidthPx) / pixelsPerMetre;
  const magnitude = 10 ** Math.floor(Math.log10(rawDistanceM));
  const normalized = rawDistanceM / magnitude;
  const multiple = normalized >= 5 ? 5 : normalized >= 2 ? 2 : 1;
  const distanceM = multiple * magnitude;
  return {
    distanceM,
    widthPx: Math.max(1, distanceM * pixelsPerMetre),
    label: formatRange(distanceM),
  };
}

function drawGrid(
  context: CanvasRenderingContext2D,
  bounds: MapBounds,
  transform: (point: Point2D) => Point2D,
  divisions: number,
) {
  const step = gridStep(bounds, divisions);
  context.strokeStyle = COLORS.grid;
  context.lineWidth = 1;
  for (
    let x = Math.ceil(bounds.min_x / step) * step;
    x <= bounds.max_x;
    x += step
  ) {
    const start = transform({ x, y: bounds.min_y });
    const end = transform({ x, y: bounds.max_y });
    context.beginPath();
    context.moveTo(start.x, start.y);
    context.lineTo(end.x, end.y);
    context.stroke();
  }
  for (
    let y = Math.ceil(bounds.min_y / step) * step;
    y <= bounds.max_y;
    y += step
  ) {
    const start = transform({ x: bounds.min_x, y });
    const end = transform({ x: bounds.max_x, y });
    context.beginPath();
    context.moveTo(start.x, start.y);
    context.lineTo(end.x, end.y);
    context.stroke();
  }
}

function gridStep(bounds: MapBounds, divisions = GRID_DIVISIONS): number {
  const span = Math.max(
    bounds.max_x - bounds.min_x,
    bounds.max_y - bounds.min_y,
    1,
  );
  const raw = span / Math.max(1, divisions);
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const normalized = raw / magnitude;
  const multiple = normalized >= 5 ? 5 : normalized >= 2 ? 2 : 1;
  return multiple * magnitude;
}

function drawSelectedGroupLinks(
  context: CanvasRenderingContext2D,
  frame: OperationalFrame,
  transform: (point: Point2D) => Point2D,
  highlightedUuvIds: Set<string>,
  visibleUuvs: UUVView[],
) {
  const positions = new Map<string, Point2D>();
  const carriers = mapCarriers(frame);
  const carrierIds = new Set(carriers.map((carrier) => carrier.carrier_id));
  carriers.forEach((carrier) =>
    positions.set(carrier.carrier_id, carrier.position),
  );
  visibleUuvs.forEach((uuv) => positions.set(uuv.uuv_id, uuv.position));
  const relatedRelayIds = new Set<string>();
  (frame.communication_links ?? []).forEach((link) => {
    if (!link.relay || link.status !== "connected") return;
    if (highlightedUuvIds.has(link.source_id))
      relatedRelayIds.add(link.target_id);
    if (highlightedUuvIds.has(link.target_id))
      relatedRelayIds.add(link.source_id);
  });
  (frame.communication_links ?? []).forEach((link) => {
    if (!link.relay || link.status !== "connected") return;
    const isGroupLink =
      highlightedUuvIds.has(link.source_id) ||
      highlightedUuvIds.has(link.target_id);
    const isRelayBackhaul =
      (carrierIds.has(link.source_id) || carrierIds.has(link.target_id)) &&
      (relatedRelayIds.has(link.source_id) || relatedRelayIds.has(link.target_id));
    if (!isGroupLink && !isRelayBackhaul) return;
    const source = positions.get(link.source_id);
    const target = positions.get(link.target_id);
    if (!source || !target) return;
    const start = transform(source);
    const end = transform(target);
    context.save();
    context.strokeStyle = "rgba(98, 230, 167, 0.72)";
    context.lineWidth = 1.5;
    context.setLineDash(link.medium === "acoustic" ? [4, 4] : []);
    context.beginPath();
    context.moveTo(start.x, start.y);
    context.lineTo(end.x, end.y);
    context.stroke();
    context.restore();
  });
}

function drawTargetDetectionZones(
  context: CanvasRenderingContext2D,
  frame: OperationalFrame,
  transform: (point: Point2D) => Point2D,
  scale: number,
): PaintedDetectionLayer[] {
  const painted: PaintedDetectionLayer[] = [];
  executionTargetEstimates(frame).forEach((target) => {
    const radius = targetDetectionRange(frame);
    const center = transform(target.mean);
    context.save();
    context.strokeStyle = TARGET_DETECTION_STYLE.stroke;
    context.fillStyle = TARGET_DETECTION_STYLE.fill;
    context.lineWidth = 1.5;
    context.setLineDash(TARGET_DETECTION_STYLE.lineDash);
    context.beginPath();
    context.arc(center.x, center.y, radius * scale, 0, Math.PI * 2);
    context.fill();
    context.stroke();
    context.setLineDash([]);
    context.restore();
    painted.push({
      target_id: target.target_id,
      center,
      radius_px: radius * scale,
      stroke_style: TARGET_DETECTION_STYLE.stroke,
      line_dash: [...TARGET_DETECTION_STYLE.lineDash],
    });
  });
  return painted;
}

function drawUuvSensorFootprints(
  context: CanvasRenderingContext2D,
  frame: OperationalFrame,
  visibleUuvs: UUVView[],
  transform: (point: Point2D) => Point2D,
  scale: number,
  highlighted: Set<string>,
): PaintedSonarLayer[] {
  const execution = frame.execution;
  const painted: PaintedSonarLayer[] = [];
  visibleUuvs.forEach((uuv) => {
    const footprint = uuvSensorFootprint(uuv, frame);
    if (!footprint) return;
    const center = transform(uuv.position);
    const radius = footprint.radiusM * scale;
    const startAngle = footprint.centerAngleRad - footprint.spanAngleRad / 2;
    const endAngle = footprint.centerAngleRad + footprint.spanAngleRad / 2;
    const taskGroup = groupForUuv(frame, uuv.uuv_id);
    const emphasis = highlighted.has(uuv.uuv_id) ? 1 : 0.24;
    context.save();
    context.globalAlpha = uuvDisplayOpacity(uuv) * emphasis;
    context.strokeStyle = footprint.strokeStyle;
    context.fillStyle = footprint.fillStyle;
    context.lineWidth = uuv.sensor_mode === "active" ? 1.55 : 1.25;
    context.beginPath();
    context.moveTo(center.x, center.y);
    context.arc(center.x, center.y, radius, startAngle, endAngle);
    context.closePath();
    context.fill();
    context.stroke();
    context.restore();
    painted.push({
      uuv_id: uuv.uuv_id,
      deployment_key: deploymentAwareUuvKey(uuv),
      target_id: execution?.target_id ?? null,
      task_group_id: taskGroup ? groupInstanceId(taskGroup) : null,
      role: executionGroupRole(taskGroup, uuv.uuv_id),
      sensor_mode: uuv.sensor_mode,
      center,
      radius_px: radius,
      start_angle_rad: startAngle,
      end_angle_rad: endAngle,
      stroke_style: footprint.strokeStyle,
      fill_style: footprint.fillStyle,
    });
  });
  return painted;
}

function drawBearings(
  context: CanvasRenderingContext2D,
  frame: OperationalFrame,
  transform: (point: Point2D) => Point2D,
  highlightedUuvIds: Set<string>,
) {
  frame.bearing_rays
    .filter((ray) => highlightedUuvIds.has(ray.uuv_id))
    .forEach((ray) => {
      const endpoint = clipRayToBounds(
        ray.origin,
        ray.azimuth_rad,
        frame.map_bounds,
      );
      const start = transform(ray.origin);
      const end = transform(endpoint);
      context.strokeStyle = "rgba(246, 185, 74, 0.32)";
      context.lineWidth = 1;
      context.setLineDash([2, 6]);
      context.beginPath();
      context.moveTo(start.x, start.y);
      context.lineTo(end.x, end.y);
      context.stroke();
      context.setLineDash([]);
    });
}

/** Draw short, fading UUV movement history without restoring full route clutter. */
function drawUuvTrails(
  context: CanvasRenderingContext2D,
  transform: (point: Point2D) => Point2D,
  trailMode: TrailMode,
  highlightedUuvIds: Set<string>,
  visibleUuvs: UUVView[],
) {
  visibleUuvs.forEach((uuv) => {
    const breadcrumb = uuv.breadcrumb ?? [];
    if (breadcrumb.length < 2) return;
    const points =
      trailMode === "full"
        ? breadcrumb
        : breadcrumb.slice(-(trailMode === "comet" ? 12 : 8));
    if (points.length < 2) return;
    const isHighlighted = highlightedUuvIds.has(uuv.uuv_id);
    context.save();
    context.globalAlpha = uuvDisplayOpacity(uuv);
    context.lineWidth = isHighlighted ? 1.7 : 1.15;
    context.lineCap = "round";
    points.slice(1).forEach((point, index) => {
      const progress = (index + 1) / (points.length - 1);
      const alpha =
        (isHighlighted ? 0.18 : 0.08) +
        progress * (isHighlighted ? 0.52 : 0.28);
      const start = transform(points[index]);
      const end = transform(point);
      context.strokeStyle = isHighlighted
        ? `rgba(196, 180, 255, ${alpha})`
        : `rgba(33, 208, 195, ${alpha})`;
      context.beginPath();
      context.moveTo(start.x, start.y);
      context.lineTo(end.x, end.y);
      context.stroke();
    });
    context.restore();
  });
}

function drawEstimates(
  context: CanvasRenderingContext2D,
  frame: OperationalFrame,
  transform: (point: Point2D) => Point2D,
  scale: number,
) {
  executionTargetEstimates(frame).forEach((target) => {
    if (!target.covariance_ellipse) return;
    const center = transform(target.mean);
    const ellipse = displayCovarianceEllipse(target.covariance_ellipse, scale);
    context.save();
    context.translate(center.x, center.y);
    context.rotate(-ellipse.rotation_rad);
    context.strokeStyle =
      target.classification === "decoy" ? COLORS.amber : COLORS.red;
    context.fillStyle =
      target.classification === "decoy"
        ? "rgba(246, 185, 74, 0.08)"
        : "rgba(255, 111, 127, 0.075)";
    context.lineWidth = 1.25;
    context.beginPath();
    context.ellipse(
      0,
      0,
      Math.max(4, ellipse.semimajor_m * scale),
      Math.max(3, ellipse.semiminor_m * scale),
      0,
      0,
      Math.PI * 2,
    );
    context.fill();
    context.stroke();
    context.restore();
  });
}

function drawSceneBackground(
  context: CanvasRenderingContext2D,
  image: HTMLImageElement | null,
  width: number,
  height: number,
) {
  context.fillStyle = "#071421";
  context.fillRect(0, 0, width, height);
  if (!image || !image.naturalWidth || !image.naturalHeight) return;
  const rect = coverImageRect(
    image.naturalWidth,
    image.naturalHeight,
    width,
    height,
  );
  context.drawImage(image, rect.x, rect.y, rect.width, rect.height);
}

function drawCarrier(
  context: CanvasRenderingContext2D,
  carrier: OperationalFrame["carrier"],
  image: HTMLImageElement | null,
  transform: (point: Point2D) => Point2D,
  scale: number,
) {
  if (!carrier) return;
  const point = transform(carrier.position);
  const size = clampedSpriteSize(image, scale, 78, 0.44, 2.7);
  context.save();
  context.translate(point.x, point.y);
  context.rotate(
    image
      ? carrier.role === "carrier"
        ? carrierAssetRotation(carrier.heading_rad)
        : warshipAssetRotation(carrier.heading_rad)
      : -carrier.heading_rad,
  );
  if (image) {
    drawCenteredImage(context, image, size);
  } else {
    context.fillStyle = COLORS.ink;
    context.strokeStyle = COLORS.cyan;
    context.lineWidth = 1.5;
    context.beginPath();
    context.moveTo(size.width / 2, 0);
    context.lineTo(-size.width / 2, -size.height / 4);
    context.lineTo(-size.width / 3, 0);
    context.lineTo(-size.width / 2, size.height / 4);
    context.closePath();
    context.fill();
    context.stroke();
  }
  context.restore();
}

function carriersForFrame(frame: OperationalFrame): CarrierView[] {
  if (frame.carriers?.length) return frame.carriers;
  return frame.carrier ? [frame.carrier] : [];
}

function drawRecoveryLinks(
  context: CanvasRenderingContext2D,
  frame: OperationalFrame,
  transform: (point: Point2D) => Point2D,
  visibleUuvs: UUVView[],
) {
  const carriers = carriersForFrame(frame);
  if (!carriers.length) return;
  const carrierByUuv = new Map<string, CarrierView>();
  carriers.forEach((carrier) => {
    [
      ...carrier.onboard_uuv_ids,
      ...carrier.deployed_uuv_ids,
      ...carrier.returning_uuv_ids,
    ].forEach((uuvId) => carrierByUuv.set(uuvId, carrier));
  });
  visibleUuvs.forEach((uuv) => {
    const carrier = carrierByUuv.get(uuv.uuv_id) ?? carriers[0];
    if (
      uuv.deployment_state !== "returning" &&
      !carrier.returning_uuv_ids.includes(uuv.uuv_id)
    )
      return;
    const start = transform(carrier.position);
    const end = transform(uuv.position);
    context.strokeStyle = "rgba(98, 230, 167, 0.68)";
    context.lineWidth = 1.5;
    context.setLineDash([6, 5]);
    context.beginPath();
    context.moveTo(start.x, start.y);
    context.lineTo(end.x, end.y);
    context.stroke();
    context.setLineDash([]);
  });
}

function drawTargetSprites(
  context: CanvasRenderingContext2D,
  frame: OperationalFrame,
  image: HTMLImageElement | null,
  transform: (point: Point2D) => Point2D,
  markerPixels: number,
) {
  executionTargetEstimates(frame).forEach((target) => {
    const center = transform(target.mean);
    const heading =
      target.heading_rad ?? target.covariance_ellipse?.rotation_rad ?? 0;
    const size = screenSpriteSize(
      image,
      markerPixels,
      TARGET_MARKER_SIZE_RANGE_PX.min,
      TARGET_MARKER_SIZE_RANGE_PX.max,
    );
    if (image) {
      context.save();
      context.translate(center.x, center.y);
      context.rotate(submarineAssetRotation(heading));
      drawCenteredImage(context, image, size);
      context.restore();
    } else {
      context.fillStyle =
        target.classification === "decoy" ? COLORS.amber : COLORS.red;
      context.beginPath();
      context.arc(center.x, center.y, 4, 0, Math.PI * 2);
      context.fill();
    }
    // Labels are rendered in one deterministic pass after all markers.
  });
}

function drawUuvSprites(
  context: CanvasRenderingContext2D,
  image: HTMLImageElement | null,
  transform: (point: Point2D) => Point2D,
  scale: number,
  selectedId: string | null,
  highlightedIds: Set<string>,
  markerPixels: number,
  visibleUuvs: UUVView[],
) {
  visibleUuvs.forEach((uuv) => {
    const point = transform(uuv.position);
    const appearance = uuvSpriteAppearance(
      uuv,
      image,
      scale,
      selectedId === uuv.uuv_id,
      markerPixels,
    );
    context.save();
    context.globalAlpha = uuvDisplayOpacity(uuv);
    context.translate(point.x, point.y);
    context.rotate(appearance.rotation);
    if (image) {
      drawCenteredImage(context, image, appearance.size);
    } else {
      context.fillStyle = appearance.cueColors[0];
      context.strokeStyle = "rgba(219, 234, 254, 0.72)";
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(9, 0);
      context.lineTo(-6, -5);
      context.lineTo(-3, 0);
      context.lineTo(-6, 5);
      context.closePath();
      context.fill();
      context.stroke();
    }
    context.restore();
    if (highlightedIds.has(uuv.uuv_id)) {
      drawGroupHighlightRing(
        context,
        point,
        appearance.size,
        uuv.uuv_id === selectedId,
      );
    }
    return;
  });
}

export function uuvDisplayOpacity(uuv: UUVView): number {
  const opacity = uuv.display_opacity ?? 1;
  return Math.max(0, Math.min(1, opacity));
}

export function deploymentAwareUuvKey(uuv: UUVView): string {
  const deploymentIdentity = uuv.group_instance_id ?? uuv.deployment_revision;
  return deploymentIdentity == null
    ? uuv.uuv_id
    : `${uuv.uuv_id}:${deploymentIdentity}`;
}

function drawGroupHighlightRing(
  context: CanvasRenderingContext2D,
  point: Point2D,
  size: { width: number; height: number },
  selected: boolean,
) {
  const radius = Math.max(size.width, size.height) / 2 + 6;
  context.save();
  context.strokeStyle = selected ? COLORS.ink : COLORS.violet;
  context.lineWidth = selected ? 2.2 : 1.5;
  context.setLineDash(selected ? [] : [3, 3]);
  context.beginPath();
  context.arc(point.x, point.y, radius, 0, Math.PI * 2);
  context.stroke();
  if (selected) {
    context.strokeStyle = COLORS.cyan;
    context.lineWidth = 1.2;
    context.beginPath();
    context.arc(point.x, point.y, radius + 4, 0, Math.PI * 2);
    context.stroke();
  }
  context.restore();
}

function clampedSpriteSize(
  image: HTMLImageElement | null,
  scale: number,
  markerPixels: number,
  minFactor: number,
  maxFactor: number,
) {
  const size = clampedMarkerPixels(
    markerPixels * scale,
    markerPixels * minFactor,
    markerPixels * maxFactor,
  );
  const ratio =
    image && image.naturalWidth && image.naturalHeight
      ? image.naturalWidth / image.naturalHeight
      : 1;
  return { width: size * ratio, height: size };
}

/** Keeps platform markers within an absolute screen-pixel range at every zoom. */
export function screenSpriteSize(
  image: HTMLImageElement | null,
  markerPixels: number,
  minimumPixels: number,
  maximumPixels: number,
) {
  const longestSide = clampedMarkerPixels(
    markerPixels,
    minimumPixels,
    maximumPixels,
  );
  const ratio =
    image && image.naturalWidth && image.naturalHeight
      ? image.naturalWidth / image.naturalHeight
      : 1;
  return ratio >= 1
    ? { width: longestSide, height: longestSide / ratio }
    : { width: longestSide * ratio, height: longestSide };
}

function drawCenteredImage(
  context: CanvasRenderingContext2D,
  image: HTMLImageElement,
  size: { width: number; height: number },
) {
  context.drawImage(
    image,
    -size.width / 2,
    -size.height / 2,
    size.width,
    size.height,
  );
}

function distance(a: Point2D, b: Point2D) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function formatRange(metres: number): string {
  if (metres < 1000) return `${Math.round(metres)} m`;
  const kilometres = Number((metres / 1000).toFixed(1));
  return `${kilometres} km`;
}
