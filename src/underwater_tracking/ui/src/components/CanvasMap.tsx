import { useEffect, useRef, useState, type MouseEvent, type PointerEvent, type WheelEvent } from "react";
import { LocateFixed, RadioTower } from "lucide-react";
import type { MapBounds, OperationalFrame, Point2D, RegionTaskView, TargetEstimateView, UUVView } from "../types/frames";
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
import { coverImageRect, loadSceneAssets, type SceneAssets } from "./map/sceneAssets";

export type TrailMode = "tail" | "full" | "comet";

interface CanvasMapProps {
  frame: OperationalFrame | null;
  selectedUuvId: string | null;
  onSelectUuv: (id: string | null) => void;
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
  grid: "rgba(225, 245, 248, 0.26)",
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
export const GRID_DIVISIONS = 16;
export const DEFAULT_SUBMARINE_DETECTION_RANGE_M = 1800;
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

export function markerRingStyle(color: string, selected: boolean): PlatformMarkerRing {
  return {
    color,
    lineWidth: selected ? 1.75 : 1.5,
    radiusPadding: 3,
    highlightColor: selected ? COLORS.ink : null,
    highlightPadding: selected ? 4 : 0,
  };
}

export function communicationRangeForUsv(frame: OperationalFrame, usvId: string): number {
  const usv = frame.usvs?.find((candidate) => candidate.usv_id === usvId);
  if (usv?.communication_range_m != null && Number.isFinite(usv.communication_range_m) && usv.communication_range_m > 1) {
    return Math.max(0, usv.communication_range_m);
  }
  return Math.max(
    0,
    ...(frame.communication_links ?? [])
      .filter((link) => link.medium === "surface" && (link.source_id === usvId || link.target_id === usvId))
      .map((link) => link.limit_m)
      .filter((limit): limit is number => Number.isFinite(limit)),
  );
}

export function targetDetectionRange(target: TargetEstimateView, detectionRange?: number | null): number {
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
  const includeDetectionRange = showDetectionRange || viewConfig.focusMode === "full_area";
  const points: Point2D[] = viewConfig.focusMode === "full_area"
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
    if (prediction) points.push(...corridorPolygon(prediction.centerline_xy, prediction.radius_m));
    if (includeDetectionRange) {
      const radius = targetDetectionRange(target, frame.adversary?.detection_range_m);
      points.push(
        { x: target.mean.x - radius, y: target.mean.y },
        { x: target.mean.x + radius, y: target.mean.y },
        { x: target.mean.x, y: target.mean.y - radius },
        { x: target.mean.x, y: target.mean.y + radius },
      );
    }
  });
  if (showPredictedRegions) {
    Object.values(frame.regional_plans ?? {}).forEach((plan) => plan.regions.forEach((region) => points.push(...region.geometry)));
  }
  return boundsForPoints(points, viewConfig.focusMode === "full_area" ? 0 : viewConfig.predictionPadding) ?? frame.map_bounds;
}

export function clampedMarkerPixels(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function regionLabelForZoom(region: RegionTaskView, zoom: number): string {
  if (zoom < 1.15) return "区域";
  const ordinal = region.display_name.match(/(?:region|区域)[_\s-]?(\d+)$/i)?.[1];
  return ordinal ? `R${ordinal.padStart(2, "0")}` : region.region_id;
}

export function hitTestRegion(point: Point2D, regions: RegionTaskView[]): RegionTaskView | null {
  return regions.find((region) => pointInPolygon(point, region.geometry)) ?? null;
}

export function detectedPlatformIds(
  frame: OperationalFrame,
  target: TargetEstimateView,
  detectionRange?: number | null,
): string[] {
  const explicit = target.detected_platform_ids ?? frame.adversary?.detected_platform_ids;
  if (explicit) return [...new Set(explicit)];
  const radius = targetDetectionRange(target, detectionRange);
  const platforms = [
    ...frame.uuvs.map((uuv) => ({ id: uuv.uuv_id, position: uuv.position })),
    ...(frame.usvs ?? []).map((usv) => ({ id: usv.usv_id, position: usv.position })),
  ];
  return platforms.filter((platform) => distance(platform.position, target.mean) <= radius).map((platform) => platform.id);
}

export function uuvSpriteAppearance(
  uuv: UUVView,
  image: HTMLImageElement | null,
  scale: number,
  selected: boolean,
  markerPixels = 30,
) {
  const stateColor = uuv.status === "failed"
    ? COLORS.red
    : uuv.sensor_mode === "active"
      ? COLORS.amber
      : COLORS.cyan;
  return {
    size: clampedSpriteSize(image, scale, markerPixels, 0.55, 1.8),
    rotation: -uuv.heading_rad,
    cueColors: [stateColor, ...(uuv.reserved ? [COLORS.violet] : []), ...(selected ? [COLORS.ink] : [])],
    markerRing: markerRingStyle(stateColor, selected),
  };
}

export function usvSpriteAppearance(
  usv: { sensor_mode: "active" | "passive" },
  image: HTMLImageElement | null,
  scale: number,
  markerPixels = 38,
) {
  const color = usv.sensor_mode === "active" ? COLORS.amber : COLORS.green;
  return {
    size: clampedSpriteSize(image, scale * 0.68, markerPixels, 0.55, 1.8),
    color,
    markerRing: markerRingStyle(color, false),
  };
}

export default function CanvasMap({
  frame,
  selectedUuvId,
  onSelectUuv,
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
  const drawOptionsRef = useRef({ showGrid, showPredictedRegions, showRegionHandoffs, showDetectionRange, trailMode, selectedUuvId, viewConfig });
  const assetsRef = useRef<SceneAssets>(EMPTY_SCENE_ASSETS);
  const [hovered, setHovered] = useState(false);

  frameRef.current = frame;
  drawOptionsRef.current = { showGrid, showPredictedRegions, showRegionHandoffs, showDetectionRange, trailMode, selectedUuvId, viewConfig };

  const requestDraw = () => {
    if (redrawRef.current !== null) return;
    redrawRef.current = window.requestAnimationFrame(() => {
      redrawRef.current = null;
      drawMap(canvasRef.current, frameRef.current, viewRef.current, sizeRef.current, drawOptionsRef.current, assetsRef.current);
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
    };
    updateSize();
    if (typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver(updateSize);
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    requestDraw();
  }, [frame, showGrid, showPredictedRegions, showRegionHandoffs, showDetectionRange, trailMode, selectedUuvId, viewConfig]);

  useEffect(() => {
    let disposed = false;
    void loadSceneAssets().then((assets) => {
      if (disposed) return;
      assetsRef.current = assets;
      requestDraw();
    });
    return () => { disposed = true; };
  }, []);

  useEffect(() => () => {
    if (redrawRef.current !== null) {
      window.cancelAnimationFrame(redrawRef.current);
      redrawRef.current = null;
    }
  }, []);

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
  };

  const handlePointerUp = (event: PointerEvent<HTMLCanvasElement>) => {
    if (dragRef.current) {
      event.currentTarget.releasePointerCapture(event.pointerId);
      dragRef.current = null;
    }
  };

  const handleWheel = (event: WheelEvent<HTMLCanvasElement>) => {
    event.preventDefault();
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
    const cursor = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
    const { zoom, pan } = zoomAroundCursorForCanvas(bounds, sizeRef.current, viewRef.current, cursor, factor);
    viewRef.current = { zoom, pan };
    requestDraw();
  };

  const handleClick = (event: MouseEvent<HTMLCanvasElement>) => {
    if (dragRef.current || !frameRef.current) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const point = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    const frameValue = frameRef.current;
    const bounds = cameraBoundsForFrame(frameValue, viewConfig, showDetectionRange, showPredictedRegions);
    const scale = fittedScaleForMap(bounds, sizeRef.current.width, sizeRef.current.height) * viewRef.current.zoom;
    const nearest = frameValue.uuvs
      .map((uuv) => ({
        id: uuv.uuv_id,
        distance: distance(point, worldToScreen(uuv.position, bounds, sizeRef.current.width, sizeRef.current.height, viewRef.current)),
        selected: spriteHitAreaContains(
          point,
        worldToScreen(uuv.position, bounds, sizeRef.current.width, sizeRef.current.height, viewRef.current),
          uuvSpriteAppearance(uuv, assetsRef.current.uuv, scale, false, viewConfig.uuvMarkerPixels).size,
          uuv.heading_rad,
          UUV_HIT_TOLERANCE_PX,
        ),
      }))
      .filter((candidate) => candidate.selected)
      .sort((a, b) => a.distance - b.distance)[0];
    if (nearest) {
      onSelectUuv(nearest.id === selectedUuvId ? null : nearest.id);
    }
  };

  const fitAll = () => {
    viewRef.current = { zoom: 1, pan: { x: 0, y: 0 } };
    requestDraw();
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
        style={{ cursor: dragRef.current ? "grabbing" : hovered ? "crosshair" : "default" }}
        aria-label="水下跟踪态势地图，支持拖动、滚轮缩放和 UUV 选择"
      />
      {!frame && (
        <div className="map-empty" role="status">
          <RadioTower size={22} />
          <strong>等待作业态势</strong>
          <span>实时估计帧或回放帧接入后将在此显示。</span>
        </div>
      )}
      <div className="map-tools" aria-label="地图工具">
        <button type="button" onClick={fitAll} title="适配当前焦点" aria-label="适配当前焦点">
          <LocateFixed size={15} />
        </button>
        <span>{viewRef.current.zoom.toFixed(1)}×</span>
      </div>
      <div className="map-scale" aria-hidden="true"><i /> 1 km</div>
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
  const after = worldToScreen(before, bounds, size.width, size.height, { zoom, pan: { x: 0, y: 0 } });
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
  const bounds = cameraBoundsForFrame(frame, options.viewConfig, options.showDetectionRange, options.showPredictedRegions);
  const transform = (point: Point2D) => worldToScreen(point, bounds, width, height, view);
  const scale = fittedScaleForMap(bounds, width, height) * view.zoom;
  if (options.showGrid) drawGrid(context, bounds, transform, options.viewConfig.gridDivisions);
  if (options.showPredictedRegions) {
    drawRegionalCells(context, frame, transform, view.zoom);
    drawPredictions(context, frame, transform);
  }
  if (options.showRegionHandoffs) drawRegionalHandoffs(context, frame, transform);
  drawCommunicationRanges(context, frame, transform, scale);
  if (shouldDrawDetectionRange(options.showDetectionRange)) {
    drawTargetDetectionZones(context, frame, transform, scale * options.viewConfig.radarScale);
  }
  drawPlatformLinks(context, frame, transform);
  drawCarrierSupport(context, frame, transform, scale);
  drawRoutes(context, frame, transform);
  drawBreadcrumbs(context, frame, transform, options.trailMode);
  drawBearings(context, frame, transform);
  drawEstimates(context, frame, transform, scale);
  drawCarrier(context, frame.carrier, assets.carrier, transform, scale);
  drawUsvSprites(context, frame, assets.carrier, transform, scale, options.viewConfig.usvMarkerPixels);
  drawRecoveryLinks(context, frame, transform);
  drawTargetSprites(context, frame, assets.submarine, transform, scale, options.viewConfig.targetMarkerPixels);
  drawUuvSprites(context, frame, assets.uuv, transform, scale, options.selectedUuvId, options.viewConfig.uuvMarkerPixels);
}

function fittedScaleForMap(bounds: OperationalFrame["map_bounds"], width: number, height: number) {
  return Math.min(width / (bounds.max_x - bounds.min_x), height / (bounds.max_y - bounds.min_y));
}

function drawGrid(context: CanvasRenderingContext2D, bounds: MapBounds, transform: (point: Point2D) => Point2D, divisions: number) {
  const step = gridStep(bounds, divisions);
  context.strokeStyle = COLORS.grid;
  context.lineWidth = 1;
  for (let x = Math.ceil(bounds.min_x / step) * step; x <= bounds.max_x; x += step) {
    const start = transform({ x, y: bounds.min_y });
    const end = transform({ x, y: bounds.max_y });
    context.beginPath(); context.moveTo(start.x, start.y); context.lineTo(end.x, end.y); context.stroke();
  }
  for (let y = Math.ceil(bounds.min_y / step) * step; y <= bounds.max_y; y += step) {
    const start = transform({ x: bounds.min_x, y });
    const end = transform({ x: bounds.max_x, y });
    context.beginPath(); context.moveTo(start.x, start.y); context.lineTo(end.x, end.y); context.stroke();
  }
}

function gridStep(bounds: MapBounds, divisions = GRID_DIVISIONS): number {
  const span = Math.max(bounds.max_x - bounds.min_x, bounds.max_y - bounds.min_y, 1);
  const raw = span / Math.max(1, divisions);
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const normalized = raw / magnitude;
  const multiple = normalized >= 5 ? 5 : normalized >= 2 ? 2 : 1;
  return multiple * magnitude;
}

function regionStyle(status: string): { fill: string; stroke: string; lineWidth: number } {
  if (status === "active") {
    return { fill: "rgba(33, 208, 195, 0.18)", stroke: "rgba(33, 208, 195, 0.92)", lineWidth: 1.8 };
  }
  if (status === "degraded") {
    return { fill: "rgba(255, 120, 130, 0.15)", stroke: "rgba(255, 120, 130, 0.88)", lineWidth: 1.4 };
  }
  if (status === "uncovered") {
    return { fill: "rgba(173, 190, 205, 0.08)", stroke: "rgba(173, 190, 205, 0.64)", lineWidth: 1 };
  }
  return { fill: "rgba(196, 180, 255, 0.12)", stroke: "rgba(196, 180, 255, 0.62)", lineWidth: 1 };
}

function drawRegionalCells(
  context: CanvasRenderingContext2D,
  frame: OperationalFrame,
  transform: (point: Point2D) => Point2D,
  zoom: number,
) {
  Object.values(frame.regional_plans ?? {}).forEach((regionalPlan) => {
    const centers: Point2D[] = [];
    regionalPlan.regions.forEach((region) => {
      if (region.geometry.length < 3) return;
      const polygon = region.geometry.map(transform);
      const style = regionStyle(region.effect.status);
      context.save();
      context.fillStyle = style.fill;
      context.strokeStyle = style.stroke;
      context.lineWidth = style.lineWidth;
      path(context, polygon, true);
      context.fill();
      context.stroke();
      const label = polygon.reduce(
        (center, point) => ({ x: center.x + point.x, y: center.y + point.y }),
        { x: 0, y: 0 },
      );
      label.x /= polygon.length;
      label.y /= polygon.length;
      centers.push(label);
      if (zoom >= 1.15) {
        context.fillStyle = COLORS.ink;
        context.font = "600 8px 'IBM Plex Mono', monospace";
        context.fillText(regionLabelForZoom(region, zoom), label.x + 3, label.y - 3);
      }
      context.restore();
    });
    if (zoom < 1.15 && centers.length > 0) {
      const label = centers.reduce((total, center) => ({ x: total.x + center.x, y: total.y + center.y }), { x: 0, y: 0 });
      label.x /= centers.length;
      label.y /= centers.length;
      context.fillStyle = COLORS.ink;
      context.font = "600 9px 'IBM Plex Mono', monospace";
      context.fillText(`区域 ${centers.length}`, label.x + 3, label.y - 3);
    }
  });
}

function drawRegionalHandoffs(
  context: CanvasRenderingContext2D,
  frame: OperationalFrame,
  transform: (point: Point2D) => Point2D,
) {
  Object.values(frame.regional_plans ?? {}).forEach((regionalPlan) => {
    const byId = new Map(regionalPlan.regions.map((region) => [region.region_id, region]));
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
        context.lineTo(end.x - head * Math.cos(angle - Math.PI / 6), end.y - head * Math.sin(angle - Math.PI / 6));
        context.lineTo(end.x - head * Math.cos(angle + Math.PI / 6), end.y - head * Math.sin(angle + Math.PI / 6));
        context.closePath();
        context.fill();
        context.restore();
      });
    });
  });
}

function drawPredictions(context: CanvasRenderingContext2D, frame: OperationalFrame, transform: (point: Point2D) => Point2D) {
  frame.target_estimates.forEach((target) => {
    const prediction = target.prediction;
    if (!prediction || prediction.centerline_xy.length < 2) return;
    const polygon = corridorPolygon(prediction.centerline_xy, prediction.radius_m).map(transform);
    context.fillStyle = "rgba(178, 156, 255, 0.10)";
    context.strokeStyle = "rgba(178, 156, 255, 0.55)";
    path(context, polygon, true); context.fill(); context.stroke();
    context.setLineDash([5, 5]);
    context.strokeStyle = "rgba(178, 156, 255, 0.78)";
    path(context, prediction.centerline_xy.map(transform), false); context.stroke();
    context.setLineDash([]);
  });
}

function drawPlatformLinks(
  context: CanvasRenderingContext2D,
  frame: OperationalFrame,
  transform: (point: Point2D) => Point2D,
) {
  const positions = new Map<string, Point2D>();
  if (frame.carrier) positions.set(frame.carrier.carrier_id, frame.carrier.position);
  (frame.usvs ?? []).forEach((usv) => positions.set(usv.usv_id, usv.position));
  frame.uuvs.forEach((uuv) => positions.set(uuv.uuv_id, uuv.position));
  (frame.communication_links ?? []).forEach((link) => {
    const source = positions.get(link.source_id);
    const target = positions.get(link.target_id);
    if (!source || !target) return;
    const start = transform(source);
    const end = transform(target);
    const connected = link.status === "connected";
    context.strokeStyle = connected
      ? link.relay ? "rgba(82, 227, 239, 0.68)" : "rgba(98, 230, 167, 0.55)"
      : "rgba(126, 155, 184, 0.2)";
    context.lineWidth = connected ? 1.5 : 1;
    context.setLineDash(connected ? [] : [3, 5]);
    context.beginPath(); context.moveTo(start.x, start.y); context.lineTo(end.x, end.y); context.stroke();
    context.setLineDash([]);
  });
}

function drawCommunicationRanges(
  context: CanvasRenderingContext2D,
  frame: OperationalFrame,
  transform: (point: Point2D) => Point2D,
  scale: number,
) {
  (frame.usvs ?? []).forEach((usv) => {
    const radius = communicationRangeForUsv(frame, usv.usv_id);
    if (radius <= 0) return;
    const center = transform(usv.position);
    context.save();
    context.strokeStyle = usv.connected ? "rgba(33, 208, 195, 0.82)" : "rgba(247, 189, 69, 0.62)";
    context.fillStyle = usv.connected ? "rgba(33, 208, 195, 0.055)" : "rgba(247, 189, 69, 0.045)";
    context.lineWidth = 1.25;
    context.setLineDash([6, 5]);
    context.beginPath();
    context.arc(center.x, center.y, radius * scale, 0, Math.PI * 2);
    context.fill();
    context.stroke();
    context.setLineDash([]);
    context.fillStyle = COLORS.ink;
    context.font = "600 9px 'IBM Plex Mono', monospace";
    context.fillText(`${usv.usv_id} 通信 ${formatRange(radius)}`, center.x + 8, center.y - radius * scale + 14);
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
    const radius = targetDetectionRange(target, frame.adversary?.detection_range_m);
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
    context.fillText(`${target.target_id} 探测圈 ${formatRange(radius)}`, center.x + 8, center.y + radius * scale - 8);
    drawDetectedBadges(context, center, detected);
    context.restore();
  });
}

function drawDetectedBadges(context: CanvasRenderingContext2D, center: Point2D, ids: string[]) {
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

function drawCarrierSupport(
  context: CanvasRenderingContext2D,
  frame: OperationalFrame,
  transform: (point: Point2D) => Point2D,
  scale: number,
) {
  if (!frame.carrier?.support_radius_m) return;
  const center = transform(frame.carrier.position);
  context.strokeStyle = "rgba(246, 185, 74, 0.42)";
  context.fillStyle = "rgba(246, 185, 74, 0.045)";
  context.lineWidth = 1;
  context.setLineDash([7, 6]);
  context.beginPath(); context.arc(center.x, center.y, frame.carrier.support_radius_m * scale, 0, Math.PI * 2); context.fill(); context.stroke();
  context.setLineDash([]);
}

function drawRoutes(context: CanvasRenderingContext2D, frame: OperationalFrame, transform: (point: Point2D) => Point2D) {
  const targets = new Map(frame.target_estimates.map((target) => [target.target_id, target.mean]));
  frame.uuvs.forEach((uuv) => {
    if (!uuv.current_waypoint) return;
    const start = transform(uuv.position);
    const end = transform(uuv.current_waypoint);
    context.strokeStyle = "rgba(82, 227, 239, 0.56)";
    context.lineWidth = 1;
    context.setLineDash([3, 5]);
    context.beginPath(); context.moveTo(start.x, start.y); context.lineTo(end.x, end.y); context.stroke();
    context.setLineDash([]);
    context.fillStyle = COLORS.cyan;
    context.beginPath(); context.arc(end.x, end.y, 3, 0, Math.PI * 2); context.fill();
  });
  frame.plans.forEach((plan) => {
    if (plan.status !== "active") return;
    plan.affected_targets.forEach((targetId) => {
      const mean = targets.get(targetId);
      if (!mean) return;
      const center = transform(mean);
      context.strokeStyle = "rgba(98, 230, 167, 0.22)";
      context.beginPath(); context.arc(center.x, center.y, 16, 0, Math.PI * 2); context.stroke();
    });
  });
}

function drawBreadcrumbs(context: CanvasRenderingContext2D, frame: OperationalFrame, transform: (point: Point2D) => Point2D, mode: TrailMode) {
  frame.uuvs.forEach((uuv) => {
    const points = mode === "full" ? uuv.breadcrumb : uuv.breadcrumb.slice(-12);
    if (points.length < 2) return;
    points.forEach((point, index) => {
      const current = transform(point);
      const previous = transform(points[Math.max(0, index - 1)]);
      context.strokeStyle = `rgba(82, 227, 239, ${0.08 + (index / points.length) * 0.34})`;
      context.lineWidth = 1.5;
      context.beginPath(); context.moveTo(previous.x, previous.y); context.lineTo(current.x, current.y); context.stroke();
    });
  });
}

function drawBearings(context: CanvasRenderingContext2D, frame: OperationalFrame, transform: (point: Point2D) => Point2D) {
  frame.bearing_rays.forEach((ray) => {
    const endpoint = clipRayToBounds(ray.origin, ray.azimuth_rad, frame.map_bounds);
    const start = transform(ray.origin); const end = transform(endpoint);
    context.strokeStyle = "rgba(246, 185, 74, 0.32)";
    context.lineWidth = 1;
    context.setLineDash([2, 6]);
    context.beginPath(); context.moveTo(start.x, start.y); context.lineTo(end.x, end.y); context.stroke();
    context.setLineDash([]);
  });
}

function drawEstimates(context: CanvasRenderingContext2D, frame: OperationalFrame, transform: (point: Point2D) => Point2D, scale: number) {
  frame.target_estimates.forEach((target) => {
    const center = transform(target.mean);
    const ellipse = target.covariance_ellipse;
    context.save(); context.translate(center.x, center.y); context.rotate(-ellipse.rotation_rad);
    context.strokeStyle = target.classification === "decoy" ? COLORS.amber : COLORS.red;
    context.fillStyle = target.classification === "decoy" ? "rgba(246, 185, 74, 0.12)" : "rgba(255, 111, 127, 0.12)";
    context.lineWidth = 1.5;
    context.beginPath(); context.ellipse(0, 0, Math.max(4, ellipse.semimajor_m * scale), Math.max(3, ellipse.semiminor_m * scale), 0, 0, Math.PI * 2); context.fill(); context.stroke();
    context.restore();
  });
}

function drawSceneBackground(context: CanvasRenderingContext2D, image: HTMLImageElement | null, width: number, height: number) {
  context.fillStyle = "#071421";
  context.fillRect(0, 0, width, height);
  if (!image || !image.naturalWidth || !image.naturalHeight) return;
  const rect = coverImageRect(image.naturalWidth, image.naturalHeight, width, height);
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
  context.save(); context.translate(point.x, point.y); context.rotate(image ? carrierAssetRotation(carrier.heading_rad) : -carrier.heading_rad);
  if (image) {
    drawCenteredImage(context, image, size);
  } else {
    context.fillStyle = COLORS.ink; context.strokeStyle = COLORS.cyan; context.lineWidth = 1.5;
    context.beginPath(); context.moveTo(size.width / 2, 0); context.lineTo(-size.width / 2, -size.height / 4); context.lineTo(-size.width / 3, 0); context.lineTo(-size.width / 2, size.height / 4); context.closePath(); context.fill(); context.stroke();
  }
  context.restore();
  context.fillStyle = COLORS.ink; context.font = "600 10px 'IBM Plex Mono', monospace";
  context.fillText(carrier.carrier_id, point.x + size.width / 2 + 4, point.y - 5);
}

function drawUsvSprites(
  context: CanvasRenderingContext2D,
  frame: OperationalFrame,
  image: HTMLImageElement | null,
  transform: (point: Point2D) => Point2D,
  scale: number,
  markerPixels: number,
) {
  (frame.usvs ?? []).forEach((usv) => {
    const point = transform(usv.position);
    const appearance = usvSpriteAppearance(usv, image, scale, markerPixels);
    const { size, color } = appearance;
    drawPlatformMarkerRing(context, point, size, appearance.markerRing);
    context.save();
    context.translate(point.x, point.y);
    context.rotate(image ? carrierAssetRotation(usv.heading_rad) : -usv.heading_rad);
    if (image) {
      drawCenteredImage(context, image, size);
    } else {
      context.fillStyle = "rgba(8, 37, 54, 0.96)";
      context.strokeStyle = color;
      context.lineWidth = 1.5;
      context.beginPath(); context.moveTo(size.width / 2, 0); context.lineTo(-size.width / 2, -size.height / 3); context.lineTo(-size.width / 3, 0); context.lineTo(-size.width / 2, size.height / 3); context.closePath(); context.fill(); context.stroke();
    }
    context.restore();
    context.fillStyle = COLORS.ink; context.font = "600 10px 'IBM Plex Mono', monospace";
    context.fillText(usv.usv_id, point.x + size.width / 2 + 4, point.y - 5);
    context.fillStyle = COLORS.muted; context.font = "9px 'IBM Plex Mono', monospace";
    context.fillText(`${usv.sensor_mode === "active" ? "ACT" : "PAS"} · ${usv.relay_active ? "RELAY" : usv.connected ? "LINK" : "OFF"}`, point.x + size.width / 2 + 4, point.y + 8);
  });
}

function drawRecoveryLinks(context: CanvasRenderingContext2D, frame: OperationalFrame, transform: (point: Point2D) => Point2D) {
  if (!frame.carrier) return;
  const returningIds = new Set(frame.carrier.returning_uuv_ids);
  frame.uuvs.forEach((uuv) => {
    if (uuv.deployment_state !== "returning" && !returningIds.has(uuv.uuv_id)) return;
    const start = transform(frame.carrier!.position); const end = transform(uuv.position);
    context.strokeStyle = "rgba(98, 230, 167, 0.68)"; context.lineWidth = 1.5; context.setLineDash([6, 5]);
    context.beginPath(); context.moveTo(start.x, start.y); context.lineTo(end.x, end.y); context.stroke(); context.setLineDash([]);
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
    const heading = target.heading_rad ?? target.covariance_ellipse.rotation_rad;
    const size = clampedSpriteSize(image, scale, markerPixels, 0.6, 1.8);
    drawPlatformMarkerRing(context, center, size, markerRingStyle(target.classification === "decoy" ? COLORS.amber : COLORS.red, false));
    if (image) {
      context.save(); context.translate(center.x, center.y); context.rotate(submarineAssetRotation(heading)); drawCenteredImage(context, image, size); context.restore();
    } else {
      context.fillStyle = target.classification === "decoy" ? COLORS.amber : COLORS.red;
      context.beginPath(); context.arc(center.x, center.y, 4, 0, Math.PI * 2); context.fill();
    }
    context.fillStyle = COLORS.ink; context.font = "600 11px 'IBM Plex Mono', monospace";
    context.fillText(target.target_id, center.x + 8, center.y - 8);
    context.fillStyle = COLORS.muted; context.font = "10px 'IBM Plex Mono', monospace";
    const role = target.classification === "submarine" ? "SUB" : target.classification.toUpperCase();
    context.fillText(`${role} · ${target.intent.label} ${(target.quality.quality_score * 100).toFixed(0)}%`, center.x + 8, center.y + 7);
  });
}

function drawUuvSprites(
  context: CanvasRenderingContext2D,
  frame: OperationalFrame,
  image: HTMLImageElement | null,
  transform: (point: Point2D) => Point2D,
  scale: number,
  selectedId: string | null,
  markerPixels: number,
) {
  frame.uuvs.forEach((uuv) => {
    const point = transform(uuv.position);
    const appearance = uuvSpriteAppearance(uuv, image, scale, selectedId === uuv.uuv_id, markerPixels);
    drawPlatformMarkerRing(context, point, appearance.size, appearance.markerRing);
    drawUuvStateCues(context, point, appearance.cueColors.slice(1, appearance.markerRing.highlightColor ? -1 : undefined), appearance.size);
    context.save(); context.translate(point.x, point.y); context.rotate(appearance.rotation);
    if (image) {
      drawCenteredImage(context, image, appearance.size);
    } else {
      context.fillStyle = appearance.cueColors[0]; context.strokeStyle = "rgba(219, 234, 254, 0.72)";
      context.lineWidth = 1;
      context.beginPath(); context.moveTo(9, 0); context.lineTo(-6, -5); context.lineTo(-3, 0); context.lineTo(-6, 5); context.closePath(); context.fill(); context.stroke();
    }
    context.restore();
    context.fillStyle = COLORS.muted; context.font = "10px 'IBM Plex Mono', monospace";
    context.fillText(uuv.uuv_id, point.x + 10, point.y + 4);
    context.font = "9px 'IBM Plex Mono', monospace";
    context.fillText(`${uuv.sensor_mode === "active" ? "ACT" : "PAS"} · ${uuv.status.toUpperCase()}`, point.x + 10, point.y + 16);
  });
}

function drawUuvStateCues(
  context: CanvasRenderingContext2D,
  point: Point2D,
  colors: string[],
  size: { width: number; height: number },
) {
  const baseRadius = Math.max(size.width, size.height) / 2 + 2;
  colors.forEach((color, index) => {
    context.strokeStyle = color;
    context.lineWidth = index === colors.length - 1 && color === COLORS.ink ? 1.5 : 2;
    context.beginPath(); context.arc(point.x, point.y, baseRadius + (index + 1) * 3, 0, Math.PI * 2); context.stroke();
  });
}

function drawPlatformMarkerRing(
  context: CanvasRenderingContext2D,
  point: Point2D,
  size: { width: number; height: number },
  appearance: PlatformMarkerRing,
) {
  const radius = Math.max(size.width, size.height) / 2 + appearance.radiusPadding;
  context.save();
  context.strokeStyle = appearance.color;
  context.lineWidth = appearance.lineWidth;
  context.beginPath();
  context.arc(point.x, point.y, radius, 0, Math.PI * 2);
  context.stroke();
  if (appearance.highlightColor) {
    context.strokeStyle = appearance.highlightColor;
    context.lineWidth = 1.5;
    context.beginPath();
    context.arc(point.x, point.y, radius + appearance.highlightPadding, 0, Math.PI * 2);
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
  const size = clampedMarkerPixels(markerPixels * scale, markerPixels * minFactor, markerPixels * maxFactor);
  const ratio = image && image.naturalWidth && image.naturalHeight ? image.naturalWidth / image.naturalHeight : 1;
  return { width: size * ratio, height: size };
}

function drawCenteredImage(context: CanvasRenderingContext2D, image: HTMLImageElement, size: { width: number; height: number }) {
  context.drawImage(image, -size.width / 2, -size.height / 2, size.width, size.height);
}

function path(context: CanvasRenderingContext2D, points: Point2D[], close: boolean) {
  if (points.length === 0) return;
  context.beginPath(); context.moveTo(points[0].x, points[0].y);
  points.slice(1).forEach((point) => context.lineTo(point.x, point.y));
  if (close) context.closePath();
}

function distance(a: Point2D, b: Point2D) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function formatRange(metres: number): string {
  return metres >= 1000 ? `${(metres / 1000).toFixed(1)} km` : `${Math.round(metres)} m`;
}
