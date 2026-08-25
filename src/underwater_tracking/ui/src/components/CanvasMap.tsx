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
  MapBounds,
  OperationalFrame,
  Point2D,
  RegionTaskView,
  TargetPriorView,
  TargetEstimateView,
  UUVView,
} from "../types/frames";
import type { ViewConfig } from "../types/viewConfig";
import {
  boundsForPoints,
  clipRayToBounds,
  corridorPolygon,
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
import RegionOverlay from "./map/RegionOverlay";
import { displayTargetName } from "../utils/presentation";

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
  carrier: null,
  uuv: null,
  submarine: null,
};

/**
 * The carrier PNG is drawn bow-up (screen north), while the vector fallback
 * points right at heading 0. Rotate the asset by this offset to share the
 * world convention: heading 0 is right/east and pi/2 is up/north.
 */
export const CARRIER_ASSET_HEADING_OFFSET = Math.PI / 2;

const UUV_HIT_TOLERANCE_PX = 6;
const MINIMUM_TARGET_ONLY_CAMERA_SPAN_M = 1000;
export const GRID_DIVISIONS = 16;
export const DEFAULT_SUBMARINE_DETECTION_RANGE_M = 1200;
export const SUBMARINE_ASSET_HEADING_OFFSET = Math.PI;

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

export function submarineAssetRotation(headingRad: number): number {
  return -headingRad + SUBMARINE_ASSET_HEADING_OFFSET;
}

export function targetPriorLabel(prior: TargetPriorView): string {
  return `待确认目标 ${displayTargetName(prior.target_id)} · ${(prior.confidence * 100).toFixed(0)}%`;
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

export function targetDetectionRange(
  target: TargetEstimateView,
  detectionRange?: number | null,
): number {
  const explicit = detectionRange ?? target.detection_range_m;
  return explicit != null && Number.isFinite(explicit) && explicit > 1
    ? explicit
    : DEFAULT_SUBMARINE_DETECTION_RANGE_M;
}

export function shouldDrawDetectionRange(enabled: boolean): boolean {
  return enabled;
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
  frame.target_estimates.forEach((target) => {
    points.push(target.mean);
    const prediction = target.prediction;
    if (prediction) {
      hasPredictionCenterline ||= prediction.centerline_xy.length >= 2;
      points.push(
        ...corridorPolygon(prediction.centerline_xy, prediction.radius_m),
      );
    }
    if (includeDetectionRange) {
      const radius = targetDetectionRange(
        target,
        frame.adversary?.detection_range_m,
      );
      points.push(
        { x: target.mean.x - radius, y: target.mean.y },
        { x: target.mean.x + radius, y: target.mean.y },
        { x: target.mean.x, y: target.mean.y - radius },
        { x: target.mean.x, y: target.mean.y + radius },
      );
    }
  });
  (frame.target_priors ?? []).forEach((prior) => {
    points.push(prior.center);
    const ellipse = prior.covariance_ellipse;
    const radius = Math.max(ellipse.semimajor_m, ellipse.semiminor_m);
    points.push(
      { x: prior.center.x - radius, y: prior.center.y },
      { x: prior.center.x + radius, y: prior.center.y },
      { x: prior.center.x, y: prior.center.y - radius },
      { x: prior.center.x, y: prior.center.y + radius },
    );
  });
  carriersForFrame(frame).forEach((carrier) => points.push(carrier.position));
  waterborneUuvs(frame).forEach((uuv) => points.push(uuv.position));
  if (showPredictedRegions) {
    Object.values(frame.regional_plans ?? {}).forEach((plan) =>
      plan.regions.forEach((region) => {
        hasVisibleRegionalCells ||= region.geometry.length >= 3;
        points.push(...region.geometry);
      }),
    );
  }
  const bounds =
    boundsForPoints(
      points,
      viewConfig.focusMode === "full_area" ? 0 : viewConfig.predictionPadding,
    ) ?? frame.map_bounds;
  const hasOnlyTargetMean =
    viewConfig.focusMode !== "full_area" &&
    !includeDetectionRange &&
    frame.target_estimates.length === 1 &&
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
    regions.find((region) => pointInPolygon(point, region.geometry)) ?? null
  );
}

export function detectedPlatformIds(
  frame: OperationalFrame,
  target: TargetEstimateView,
  detectionRange?: number | null,
): string[] {
  const visibleUuvs = waterborneUuvs(frame);
  const visibleIds = new Set(visibleUuvs.map((uuv) => uuv.uuv_id));
  const explicit =
    target.detected_platform_ids ?? frame.adversary?.detected_platform_ids;
  if (explicit) {
    return [...new Set(explicit.filter((platformId) => visibleIds.has(platformId)))];
  }
  const radius = targetDetectionRange(target, detectionRange);
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
  const visibleUuvs = waterborneUuvs(frame);
  const selected = visibleUuvs.find((uuv) => uuv.uuv_id === selectedUuvId);
  if (!selected) return new Set();
  const group = selected.group_id
    ? frame.groups.find((candidate) => candidate.group_id === selected.group_id)
    : null;
  return new Set(
    group?.member_ids.length
      ? group.member_ids
      : selected.group_id
        ? visibleUuvs
            .filter((uuv) => uuv.group_id === selected.group_id)
            .map((uuv) => uuv.uuv_id)
        : [selected.uuv_id],
  );
}

export function uuvSpriteAppearance(
  uuv: UUVView,
  image: HTMLImageElement | null,
  scale: number,
  selected: boolean,
  markerPixels = 30,
) {
  const stateColor =
    uuv.status === "failed"
      ? COLORS.red
      : uuv.sensor_mode === "active"
        ? COLORS.amber
        : COLORS.cyan;
  return {
    size: clampedSpriteSize(image, scale, markerPixels, 0.55, 1.8),
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
  const viewRef = useRef<ViewState>({ zoom: 1, pan: { x: 0, y: 0 } });
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
  });
  const assetsRef = useRef<SceneAssets>(EMPTY_SCENE_ASSETS);
  const [hovered, setHovered] = useState(false);
  const [internalSelectedRegionId, setInternalSelectedRegionId] = useState<
    string | null
  >(null);
  const [mapVersion, setMapVersion] = useState(0);
  const regionSelectionIsControlled = controlledRegionId !== undefined;
  const selectedRegionId = regionSelectionIsControlled
    ? controlledRegionId
    : internalSelectedRegionId;
  const allRegions = Object.values(frame?.regional_plans ?? {}).flatMap(
    (plan) => plan.regions,
  );
  const selectedRegion =
    allRegions.find((region) => region.region_id === selectedRegionId) ?? null;
  const visibleBounds = frame
    ? cameraBoundsForFrame(
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
  };

  const requestDraw = () => {
    if (redrawRef.current !== null) return;
    redrawRef.current = window.requestAnimationFrame(() => {
      redrawRef.current = null;
      drawMap(
        canvasRef.current,
        frameRef.current,
        viewRef.current,
        sizeRef.current,
        drawOptionsRef.current,
        assetsRef.current,
      );
    });
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
    const currentFrame = frameRef.current;
    const bounds = currentFrame
      ? cameraBoundsForFrame(
          currentFrame,
          drawOptionsRef.current.viewConfig,
          drawOptionsRef.current.showDetectionRange,
          drawOptionsRef.current.showPredictedRegions,
        )
      : null;
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
    requestDraw();
    setMapVersion((value) => value + 1);
  };

  const handleClick = (event: MouseEvent<HTMLCanvasElement>) => {
    if (dragRef.current || !frameRef.current) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const point = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    const frameValue = frameRef.current;
    const bounds = cameraBoundsForFrame(
      frameValue,
      viewConfig,
      showDetectionRange,
      showPredictedRegions,
    );
    const scale =
      fittedScaleForMap(bounds, sizeRef.current.width, sizeRef.current.height) *
      viewRef.current.zoom;
    const nearest = waterborneUuvs(frameValue)
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
    const markerHit = frameValue.target_estimates.map((target) =>
        spriteHitAreaContains(
          point,
          worldToScreen(
            target.mean,
            bounds,
            sizeRef.current.width,
            sizeRef.current.height,
            viewRef.current,
          ),
          clampedSpriteSize(
            assetsRef.current.submarine,
            scale,
            viewConfig.targetMarkerPixels,
            0.6,
            1.8,
          ),
          submarineAssetRotation(
            target.heading_rad ?? target.covariance_ellipse.rotation_rad,
          ),
          UUV_HIT_TOLERANCE_PX,
        ),
      ).some(Boolean);
    if (markerHit) return;
    if (!showPredictedRegions) return;
    const regions = Object.values(frameValue.regional_plans ?? {}).flatMap(
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

  const fitAll = () => {
    viewRef.current = { zoom: 1, pan: { x: 0, y: 0 } };
    requestDraw();
    setMapVersion((value) => value + 1);
  };

  return (
    <div
      className="canvas-area"
      ref={containerRef}
      data-show-grid={showGrid}
      data-show-predicted-regions={showPredictedRegions}
      data-show-region-handoffs={showRegionHandoffs}
      data-show-detection-range={showDetectionRange}
      data-trail-mode={trailMode}
      data-focus-mode={viewConfig.focusMode}
      data-map-version={mapVersion}
    >
      <canvas
        ref={canvasRef}
        tabIndex={0}
        onClick={handleClick}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onWheel={handleWheel}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        data-carrier-count={frame ? carriersForFrame(frame).length : 0}
        data-waterborne-uuv-count={frame ? waterborneUuvs(frame).length : 0}
        data-target-estimate-count={frame?.target_estimates.length ?? 0}
        data-plan-version={frame?.plan_version ?? 0}
        style={{
          cursor: dragRef.current
            ? "grabbing"
            : hovered
              ? "crosshair"
              : "default",
        }}
        aria-label="水下跟踪态势地图，支持拖动、滚轮缩放、UUV 与区域选择"
      />
      {showPredictedRegions && frame && (
        <RegionOverlay
          plans={Object.values(frame.regional_plans ?? {})}
          timeline={frame.region_timeline}
          selectedRegionId={selectedRegionId}
          onSelectRegion={onSelectRegion}
          width={sizeRef.current.width}
          height={sizeRef.current.height}
          interactive={false}
          project={(point) =>
            worldToScreen(
              point,
              cameraBoundsForFrame(
                frame,
                viewConfig,
                showDetectionRange,
                showPredictedRegions,
              ),
              sizeRef.current.width,
              sizeRef.current.height,
              viewRef.current,
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
  },
  assets: SceneAssets,
) {
  const context = canvas?.getContext("2d");
  if (!context) return;
  const { width, height, dpr } = size;
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);
  drawSceneBackground(context, assets.background, width, height);
  context.fillStyle = "rgba(5, 32, 73, 0.46)";
  context.fillRect(0, 0, width, height);
  if (!frame) return;
  const bounds = cameraBoundsForFrame(
    frame,
    options.viewConfig,
    options.showDetectionRange,
    options.showPredictedRegions,
  );
  const transform = (point: Point2D) =>
    worldToScreen(point, bounds, width, height, view);
  const scale = fittedScaleForMap(bounds, width, height) * view.zoom;
  const visibleUuvs = waterborneUuvs(frame);
  if (options.showGrid)
    drawGrid(context, bounds, transform, options.viewConfig.gridDivisions);
  if (options.showPredictedRegions) {
    drawPredictions(context, frame, transform);
  }
  if (options.showRegionHandoffs)
    drawRegionalHandoffs(context, frame, transform);
  drawTargetPriors(context, frame, transform, scale);
  if (shouldDrawDetectionRange(options.showDetectionRange)) {
    drawTargetDetectionZones(
      context,
      frame,
      transform,
      scale * options.viewConfig.radarScale,
    );
  }
  const highlighted = highlightedUuvIds(frame, options.selectedUuvId);
  if (highlighted.size) {
    drawSelectedGroupLinks(context, frame, transform, highlighted, visibleUuvs);
    drawBearings(context, frame, transform, highlighted);
  }
  drawUuvTrails(context, transform, options.trailMode, highlighted, visibleUuvs);
  drawEstimates(context, frame, transform, scale);
  carriersForFrame(frame).forEach((carrier) =>
    drawCarrier(context, carrier, assets.carrier, transform, scale),
  );
  drawRecoveryLinks(context, frame, transform, visibleUuvs);
  drawTargetSprites(
    context,
    frame,
    assets.submarine,
    transform,
    scale,
    options.viewConfig.targetMarkerPixels,
  );
  drawUuvSprites(
    context,
    assets.uuv,
    transform,
    scale,
    options.selectedUuvId,
    highlighted,
    options.viewConfig.uuvMarkerPixels,
    visibleUuvs,
  );
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

function drawRegionalHandoffs(
  context: CanvasRenderingContext2D,
  frame: OperationalFrame,
  transform: (point: Point2D) => Point2D,
) {
  Object.values(frame.regional_plans ?? {}).forEach((regionalPlan) => {
    const byId = new Map(
      regionalPlan.regions.map((region) => [region.region_id, region]),
    );
    regionalPlan.regions.forEach((region) => {
      const source = region.geometry.reduce(
        (center, point) => ({ x: center.x + point.x, y: center.y + point.y }),
        { x: 0, y: 0 },
      );
      source.x /= region.geometry.length;
      source.y /= region.geometry.length;
      region.successor_region_ids.forEach((successorId) => {
        const successor = byId.get(successorId);
        if (!successor) return;
        const target = successor.geometry.reduce(
          (center, point) => ({ x: center.x + point.x, y: center.y + point.y }),
          { x: 0, y: 0 },
        );
        target.x /= successor.geometry.length;
        target.y /= successor.geometry.length;
        const start = transform(source);
        const end = transform(target);
        const angle = Math.atan2(end.y - start.y, end.x - start.x);
        const head = 6;
        context.save();
        context.strokeStyle = "rgba(247, 189, 69, 0.76)";
        context.fillStyle = "rgba(247, 189, 69, 0.76)";
        context.lineWidth = 1.25;
        context.setLineDash([4, 4]);
        context.beginPath();
        context.moveTo(start.x, start.y);
        context.lineTo(end.x, end.y);
        context.stroke();
        context.setLineDash([]);
        context.beginPath();
        context.moveTo(end.x, end.y);
        context.lineTo(
          end.x - head * Math.cos(angle - Math.PI / 6),
          end.y - head * Math.sin(angle - Math.PI / 6),
        );
        context.lineTo(
          end.x - head * Math.cos(angle + Math.PI / 6),
          end.y - head * Math.sin(angle + Math.PI / 6),
        );
        context.closePath();
        context.fill();
        context.restore();
      });
    });
  });
}

function drawPredictions(
  context: CanvasRenderingContext2D,
  frame: OperationalFrame,
  transform: (point: Point2D) => Point2D,
) {
  frame.target_estimates.forEach((target) => {
    const prediction = target.prediction;
    if (!prediction || prediction.centerline_xy.length < 2) return;
    const polygon = corridorPolygon(
      prediction.centerline_xy,
      prediction.radius_m,
    ).map(transform);
    context.fillStyle = "rgba(178, 156, 255, 0.06)";
    context.strokeStyle = "rgba(178, 156, 255, 0.40)";
    path(context, polygon, true);
    context.fill();
    context.stroke();
    context.setLineDash([5, 5]);
    context.strokeStyle = "rgba(178, 156, 255, 0.64)";
    path(context, prediction.centerline_xy.map(transform), false);
    context.stroke();
    context.setLineDash([]);
  });
}

function drawSelectedGroupLinks(
  context: CanvasRenderingContext2D,
  frame: OperationalFrame,
  transform: (point: Point2D) => Point2D,
  highlightedUuvIds: Set<string>,
  visibleUuvs: UUVView[],
) {
  const positions = new Map<string, Point2D>();
  const carriers = carriersForFrame(frame);
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
) {
  frame.target_estimates.forEach((target) => {
    const radius = targetDetectionRange(
      target,
      frame.adversary?.detection_range_m,
    );
    const center = transform(target.mean);
    const detected = detectedPlatformIds(frame, target, radius);
    context.save();
    context.strokeStyle = "rgba(255, 120, 130, 0.78)";
    context.fillStyle = "rgba(255, 120, 130, 0.065)";
    context.lineWidth = 1.5;
    context.setLineDash([4, 7]);
    context.beginPath();
    context.arc(center.x, center.y, radius * scale, 0, Math.PI * 2);
    context.fill();
    context.stroke();
    context.setLineDash([]);
    context.fillStyle = COLORS.ink;
    context.font = "600 9px 'IBM Plex Mono', monospace";
    context.fillText(
      `${displayTargetName(target.target_id)} 探测圈 ${formatRange(radius)}`,
      center.x + 8,
      center.y + radius * scale - 8,
    );
    drawDetectedBadges(context, center, detected);
    context.restore();
  });
}

function drawDetectedBadges(
  context: CanvasRenderingContext2D,
  center: Point2D,
  ids: string[],
) {
  ids.slice(0, 6).forEach((id, index) => {
    const label = `已暴露 ${id}`;
    const x = center.x + 12;
    const y = center.y + 18 + index * 16;
    context.font = "600 8px 'IBM Plex Mono', monospace";
    const width = context.measureText(label).width + 10;
    context.fillStyle = "rgba(255, 246, 235, 0.94)";
    context.strokeStyle = "rgba(255, 120, 130, 0.78)";
    context.lineWidth = 1;
    context.beginPath();
    context.roundRect(x, y - 10, width, 14, 3);
    context.fill();
    context.stroke();
    context.fillStyle = "#9c2d3a";
    context.fillText(label, x + 5, y);
  });
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
  frame.target_estimates.forEach((target) => {
    const center = transform(target.mean);
    const ellipse = target.covariance_ellipse;
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

function drawTargetPriors(
  context: CanvasRenderingContext2D,
  frame: OperationalFrame,
  transform: (point: Point2D) => Point2D,
  scale: number,
) {
  (frame.target_priors ?? []).forEach((prior) => {
    drawPriorEllipse(context, prior, transform, scale);
  });
}

function drawPriorEllipse(
  context: CanvasRenderingContext2D,
  prior: TargetPriorView,
  transform: (point: Point2D) => Point2D,
  scale: number,
) {
  const center = transform(prior.center);
  const ellipse = prior.covariance_ellipse;
  context.save();
  context.translate(center.x, center.y);
  context.rotate(-ellipse.rotation_rad);
  context.strokeStyle = "rgba(196, 180, 255, 0.76)";
  context.fillStyle = `rgba(196, 180, 255, ${0.035 + prior.confidence * 0.08})`;
  context.lineWidth = 1.2;
  context.setLineDash([6, 5]);
  context.beginPath();
  context.ellipse(
    0,
    0,
    Math.max(5, ellipse.semimajor_m * scale),
    Math.max(4, ellipse.semiminor_m * scale),
    0,
    0,
    Math.PI * 2,
  );
  context.fill();
  context.stroke();
  context.setLineDash([]);
  context.strokeStyle = COLORS.amber;
  context.fillStyle = "rgba(247, 189, 69, 0.16)";
  context.lineWidth = 1.6;
  context.beginPath();
  context.moveTo(0, -8);
  context.lineTo(8, 0);
  context.lineTo(0, 8);
  context.lineTo(-8, 0);
  context.closePath();
  context.fill();
  context.stroke();
  context.restore();
  context.fillStyle = COLORS.ink;
  context.font = "700 11px 'IBM Plex Mono', monospace";
  context.fillText(
    targetPriorLabel(prior),
    center.x + 10,
    center.y - 10,
  );
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
    image ? carrierAssetRotation(carrier.heading_rad) : -carrier.heading_rad,
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
  context.fillStyle = COLORS.ink;
  context.font = "600 10px 'IBM Plex Mono', monospace";
  context.fillText(
    `${carrier.role === "mother_ship" ? "母舰" : "航母"} ${carrier.carrier_id}`,
    point.x + size.width / 2 + 4,
    point.y - 5,
  );
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
  scale: number,
  markerPixels: number,
) {
  frame.target_estimates.forEach((target) => {
    const center = transform(target.mean);
    const heading =
      target.heading_rad ?? target.covariance_ellipse.rotation_rad;
    const size = clampedSpriteSize(image, scale, markerPixels, 0.6, 1.8);
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
    context.fillStyle = COLORS.ink;
    context.font = "600 11px 'IBM Plex Mono', monospace";
    context.fillText(
      displayTargetName(target.target_id),
      center.x + 8,
      center.y - 8,
    );
    context.fillStyle = COLORS.muted;
    context.font = "10px 'IBM Plex Mono', monospace";
    const role =
      target.classification === "submarine"
        ? "SUB"
        : target.classification.toUpperCase();
    context.fillText(
      `${role} · ${target.intent.label} ${(target.quality.quality_score * 100).toFixed(0)}%`,
      center.x + 8,
      center.y + 7,
    );
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
    context.fillStyle = COLORS.muted;
    context.font = "10px 'IBM Plex Mono', monospace";
    context.fillText(uuv.uuv_id, point.x + 10, point.y + 4);
    context.font = "9px 'IBM Plex Mono', monospace";
    context.fillText(
      `${uuv.sensor_mode === "active" ? "ACT" : "PAS"} · ${uuv.status.toUpperCase()}`,
      point.x + 10,
      point.y + 16,
    );
    if (highlightedIds.has(uuv.uuv_id)) {
      drawGroupHighlightRing(
        context,
        point,
        appearance.size,
        uuv.uuv_id === selectedId,
      );
    }
  });
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

function path(
  context: CanvasRenderingContext2D,
  points: Point2D[],
  close: boolean,
) {
  if (points.length === 0) return;
  context.beginPath();
  context.moveTo(points[0].x, points[0].y);
  points.slice(1).forEach((point) => context.lineTo(point.x, point.y));
  if (close) context.closePath();
}

function distance(a: Point2D, b: Point2D) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function formatRange(metres: number): string {
  if (metres < 1000) return `${Math.round(metres)} m`;
  const kilometres = Number((metres / 1000).toFixed(1));
  return `${kilometres} km`;
}
