import { useEffect, useRef, useState, type MouseEvent, type PointerEvent, type WheelEvent } from "react";
import { LocateFixed, RadioTower } from "lucide-react";
import type { OperationalFrame, Point2D } from "../types/frames";
import {
  clipRayToBounds,
  corridorPolygon,
  screenToWorld,
  worldToScreen,
  type ViewState,
} from "./map/geometry";

export type TrailMode = "tail" | "full" | "comet";

interface CanvasMapProps {
  frame: OperationalFrame | null;
  selectedUuvId: string | null;
  onSelectUuv: (id: string | null) => void;
  showGrid: boolean;
  trailMode: TrailMode;
}

const COLORS = {
  ink: "#dbeafe",
  muted: "#7e9bb8",
  cyan: "#52e3ef",
  cyanSoft: "rgba(82, 227, 239, 0.18)",
  amber: "#f6b94a",
  red: "#ff6f7f",
  green: "#62e6a7",
  violet: "#b29cff",
  grid: "rgba(109, 157, 192, 0.13)",
};

export default function CanvasMap({
  frame,
  selectedUuvId,
  onSelectUuv,
  showGrid,
  trailMode,
}: CanvasMapProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const frameRef = useRef(frame);
  const viewRef = useRef<ViewState>({ zoom: 1, pan: { x: 0, y: 0 } });
  const sizeRef = useRef({ width: 1, height: 1, dpr: 1 });
  const dragRef = useRef<{ x: number; y: number; pan: Point2D } | null>(null);
  const redrawRef = useRef<number | null>(null);
  const drawOptionsRef = useRef({ showGrid, trailMode, selectedUuvId });
  const [hovered, setHovered] = useState(false);

  frameRef.current = frame;
  drawOptionsRef.current = { showGrid, trailMode, selectedUuvId };

  const requestDraw = () => {
    if (redrawRef.current !== null) return;
    redrawRef.current = window.requestAnimationFrame(() => {
      redrawRef.current = null;
      drawMap(canvasRef.current, frameRef.current, viewRef.current, sizeRef.current, drawOptionsRef.current);
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
  }, [frame, showGrid, trailMode, selectedUuvId]);

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
    const bounds = frameRef.current?.map_bounds;
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
    const nearest = frameValue.uuvs
      .map((uuv) => ({
        id: uuv.uuv_id,
        distance: distance(point, worldToScreen(uuv.position, frameValue.map_bounds, sizeRef.current.width, sizeRef.current.height, viewRef.current)),
      }))
      .sort((a, b) => a.distance - b.distance)[0];
    if (nearest && nearest.distance <= 18) {
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
      data-trail-mode={trailMode}
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
        <button type="button" onClick={fitAll} title="适配全图" aria-label="适配全图">
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
  options: { showGrid: boolean; trailMode: TrailMode; selectedUuvId: string | null },
) {
  const context = canvas?.getContext("2d");
  if (!context) return;
  const { width, height, dpr } = size;
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#071421";
  context.fillRect(0, 0, width, height);
  if (!frame) return;
  const transform = (point: Point2D) => worldToScreen(point, frame.map_bounds, width, height, view);
  const scale = fittedScaleForMap(frame.map_bounds, width, height) * view.zoom;
  if (options.showGrid) drawGrid(context, frame, transform);
  drawPredictions(context, frame, transform);
  drawRoutes(context, frame, transform);
  drawBreadcrumbs(context, frame, transform, options.trailMode);
  drawBearings(context, frame, transform);
  drawEstimates(context, frame, transform, scale);
  drawUuvs(context, frame, transform, options.selectedUuvId);
}

function fittedScaleForMap(bounds: OperationalFrame["map_bounds"], width: number, height: number) {
  return Math.min(width / (bounds.max_x - bounds.min_x), height / (bounds.max_y - bounds.min_y));
}

function drawGrid(context: CanvasRenderingContext2D, frame: OperationalFrame, transform: (point: Point2D) => Point2D) {
  const step = gridStep(frame.map_bounds);
  context.strokeStyle = COLORS.grid;
  context.lineWidth = 1;
  for (let x = Math.ceil(frame.map_bounds.min_x / step) * step; x <= frame.map_bounds.max_x; x += step) {
    const start = transform({ x, y: frame.map_bounds.min_y });
    const end = transform({ x, y: frame.map_bounds.max_y });
    context.beginPath(); context.moveTo(start.x, start.y); context.lineTo(end.x, end.y); context.stroke();
  }
  for (let y = Math.ceil(frame.map_bounds.min_y / step) * step; y <= frame.map_bounds.max_y; y += step) {
    const start = transform({ x: frame.map_bounds.min_x, y });
    const end = transform({ x: frame.map_bounds.max_x, y });
    context.beginPath(); context.moveTo(start.x, start.y); context.lineTo(end.x, end.y); context.stroke();
  }
}

function gridStep(bounds: OperationalFrame["map_bounds"]): number {
  const span = Math.max(bounds.max_x - bounds.min_x, bounds.max_y - bounds.min_y, 1);
  const raw = span / 8;
  const magnitude = 10 ** Math.floor(Math.log10(raw));
  const normalized = raw / magnitude;
  const multiple = normalized >= 5 ? 5 : normalized >= 2 ? 2 : 1;
  return multiple * magnitude;
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
    context.fillStyle = COLORS.ink; context.font = "600 11px 'IBM Plex Mono', monospace";
    context.fillText(target.target_id, center.x + 8, center.y - 8);
    context.fillStyle = COLORS.muted; context.font = "10px 'IBM Plex Mono', monospace";
    context.fillText(`${target.intent.label} ${(target.quality.quality_score * 100).toFixed(0)}%`, center.x + 8, center.y + 7);
    context.fillStyle = target.classification === "decoy" ? COLORS.amber : COLORS.red;
    context.beginPath(); context.arc(center.x, center.y, 4, 0, Math.PI * 2); context.fill();
  });
}

function drawUuvs(context: CanvasRenderingContext2D, frame: OperationalFrame, transform: (point: Point2D) => Point2D, selectedId: string | null) {
  frame.uuvs.forEach((uuv) => {
    const point = transform(uuv.position);
    const color = uuv.status === "failed" ? COLORS.red : uuv.sensor_mode === "active" ? COLORS.amber : COLORS.cyan;
    context.save(); context.translate(point.x, point.y); context.rotate(-uuv.heading_rad);
    context.fillStyle = color; context.strokeStyle = uuv.reserved ? COLORS.amber : "rgba(219, 234, 254, 0.72)";
    context.lineWidth = uuv.reserved ? 2 : 1;
    context.beginPath(); context.moveTo(9, 0); context.lineTo(-6, -5); context.lineTo(-3, 0); context.lineTo(-6, 5); context.closePath(); context.fill(); context.stroke();
    context.restore();
    if (selectedId === uuv.uuv_id) {
      context.strokeStyle = COLORS.ink; context.lineWidth = 1;
      context.beginPath(); context.arc(point.x, point.y, 14, 0, Math.PI * 2); context.stroke();
    }
    context.fillStyle = COLORS.muted; context.font = "10px 'IBM Plex Mono', monospace";
    context.fillText(uuv.uuv_id, point.x + 10, point.y + 4);
  });
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
