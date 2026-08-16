import type { CovarianceEllipse, MapBounds, Point2D } from "../../types/frames";

export interface ViewState {
  zoom: number;
  /** Screen-pixel translation applied after the fitted world transform. */
  pan: Point2D;
}

export interface SpriteDimensions {
  width: number;
  height: number;
}

function fittedScale(bounds: MapBounds, width: number, height: number): number {
  return Math.min(
    width / Math.max(bounds.max_x - bounds.min_x, Number.EPSILON),
    height / Math.max(bounds.max_y - bounds.min_y, Number.EPSILON),
  );
}

function fittedOffset(bounds: MapBounds, width: number, height: number, zoom: number): Point2D {
  const scale = fittedScale(bounds, width, height) * zoom;
  return {
    x: (width - (bounds.max_x - bounds.min_x) * scale) / 2,
    y: (height - (bounds.max_y - bounds.min_y) * scale) / 2,
  };
}

export function worldToScreen(
  point: Point2D,
  bounds: MapBounds,
  width: number,
  height: number,
  view: ViewState,
): Point2D {
  const scale = fittedScale(bounds, width, height) * view.zoom;
  const offset = fittedOffset(bounds, width, height, view.zoom);
  return {
    x: offset.x + (point.x - bounds.min_x) * scale + view.pan.x,
    y: height - offset.y - (point.y - bounds.min_y) * scale + view.pan.y,
  };
}

export function recoverySegment(
  carrier: Point2D,
  uuv: Point2D,
  bounds: MapBounds,
  width: number,
  height: number,
  view: ViewState,
) {
  return {
    start: worldToScreen(carrier, bounds, width, height, view),
    end: worldToScreen(uuv, bounds, width, height, view),
  };
}

/**
 * Tests a screen point against the same rotated rectangle used for a sprite.
 * The tolerance makes narrow sprites practical to select without turning the
 * whole map into a hit target.
 */
export function spriteHitAreaContains(
  point: Point2D,
  center: Point2D,
  size: SpriteDimensions,
  headingRad: number,
  tolerance = 6,
): boolean {
  const dx = point.x - center.x;
  const dy = point.y - center.y;
  const cos = Math.cos(headingRad);
  const sin = Math.sin(headingRad);
  const localX = cos * dx - sin * dy;
  const localY = sin * dx + cos * dy;
  return Math.abs(localX) <= size.width / 2 + tolerance
    && Math.abs(localY) <= size.height / 2 + tolerance;
}

export function screenToWorld(
  point: Point2D,
  bounds: MapBounds,
  width: number,
  height: number,
  view: ViewState,
): Point2D {
  const scale = fittedScale(bounds, width, height) * view.zoom;
  const offset = fittedOffset(bounds, width, height, view.zoom);
  return {
    x: (point.x - view.pan.x - offset.x) / scale + bounds.min_x,
    y: (height - point.y + view.pan.y - offset.y) / scale + bounds.min_y,
  };
}

export function zoomAroundCursor(
  bounds: MapBounds,
  width: number,
  height: number,
  view: ViewState,
  cursor: Point2D,
  factor: number,
): ViewState {
  const nextZoom = Math.max(0.25, Math.min(8, view.zoom * factor));
  const anchor = screenToWorld(cursor, bounds, width, height, view);
  const scale = fittedScale(bounds, width, height) * nextZoom;
  const offset = fittedOffset(bounds, width, height, nextZoom);
  return {
    zoom: nextZoom,
    pan: {
      x: cursor.x - offset.x - (anchor.x - bounds.min_x) * scale,
      y: cursor.y - (height - offset.y - (anchor.y - bounds.min_y) * scale),
    },
  };
}

export function covarianceAxes(ellipse: CovarianceEllipse): {
  major: number;
  minor: number;
  rotation: number;
} {
  return {
    major: Math.max(ellipse.semimajor_m, ellipse.semiminor_m),
    minor: Math.min(ellipse.semimajor_m, ellipse.semiminor_m),
    rotation: ellipse.rotation_rad,
  };
}

export function clipRayToBounds(origin: Point2D, angle: number, bounds: MapBounds): Point2D {
  const dx = Math.cos(angle);
  const dy = Math.sin(angle);
  const candidates: number[] = [];
  if (Math.abs(dx) > 1e-12) {
    candidates.push((bounds.min_x - origin.x) / dx, (bounds.max_x - origin.x) / dx);
  }
  if (Math.abs(dy) > 1e-12) {
    candidates.push((bounds.min_y - origin.y) / dy, (bounds.max_y - origin.y) / dy);
  }
  const distance = Math.min(...candidates.filter((value) => value >= 0));
  return { x: origin.x + dx * distance, y: origin.y + dy * distance };
}

export function corridorPolygon(centerline: Point2D[], radii: number[]): Point2D[] {
  if (centerline.length === 0) return [];
  const right: Point2D[] = [];
  const left: Point2D[] = [];
  centerline.forEach((point, index) => {
    const previous = centerline[Math.max(0, index - 1)];
    const next = centerline[Math.min(centerline.length - 1, index + 1)];
    const dx = next.x - previous.x;
    const dy = next.y - previous.y;
    const length = Math.hypot(dx, dy) || 1;
    const radius = Math.max(0, radii[index] ?? radii.at(-1) ?? 0);
    const nx = dy / length;
    const ny = -dx / length;
    right.push({ x: point.x + nx * radius, y: point.y + ny * radius });
    left.push({ x: point.x - nx * radius, y: point.y - ny * radius });
  });
  return [...right, ...left.reverse(), right[0]];
}
