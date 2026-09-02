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

export interface DisplayRegionGeometry {
  geometry: Point2D[];
  top_left_xy?: Point2D | null;
  bottom_right_xy?: Point2D | null;
}

/** Return the side length of the square (or square-compatible bbox) used for display. */
export function regionDisplaySide(region: DisplayRegionGeometry): number {
  if (region.top_left_xy && region.bottom_right_xy) {
    return Math.max(
      Math.abs(region.bottom_right_xy.x - region.top_left_xy.x),
      Math.abs(region.top_left_xy.y - region.bottom_right_xy.y),
    );
  }
  const bounds = boundsForPoints(region.geometry);
  return bounds
    ? Math.max(bounds.max_x - bounds.min_x, bounds.max_y - bounds.min_y)
    : 0;
}

/** Use one common display side so a four-region mission reads as a uniform grid. */
export function sharedRegionDisplaySide(
  regions: DisplayRegionGeometry[],
): number | null {
  const sides = regions
    .map(regionDisplaySide)
    .filter((side) => Number.isFinite(side) && side > 0);
  return sides.length ? Math.max(...sides) : null;
}

/**
 * Returns the exact operator-facing region geometry used by every UI layer.
 * Live frames currently expose square display corners alongside the internal
 * support polygon; legacy frames fall back to their published geometry.
 * When a shared side is provided, the square is expanded around its centre so
 * all regions in one mission use the same display footprint.
 */
export function displayRegionPoints(
  region: DisplayRegionGeometry,
  sharedSide?: number | null,
): Point2D[] {
  if (region.top_left_xy && region.bottom_right_xy) {
    const { top_left_xy: topLeft, bottom_right_xy: bottomRight } = region;
    const currentSide = regionDisplaySide(region);
    const side = Math.max(currentSide, sharedSide ?? 0);
    const center = {
      x: (topLeft.x + bottomRight.x) / 2,
      y: (topLeft.y + bottomRight.y) / 2,
    };
    const halfSide = side / 2;
    return [
      { x: center.x - halfSide, y: center.y - halfSide },
      { x: center.x + halfSide, y: center.y - halfSide },
      { x: center.x + halfSide, y: center.y + halfSide },
      { x: center.x - halfSide, y: center.y + halfSide },
    ];
  }
  if (sharedSide && sharedSide > 0) {
    const bounds = boundsForPoints(region.geometry);
    if (bounds) {
      const side = Math.max(
        sharedSide,
        bounds.max_x - bounds.min_x,
        bounds.max_y - bounds.min_y,
      );
      const center = {
        x: (bounds.min_x + bounds.max_x) / 2,
        y: (bounds.min_y + bounds.max_y) / 2,
      };
      const halfSide = side / 2;
      return [
        { x: center.x - halfSide, y: center.y - halfSide },
        { x: center.x + halfSide, y: center.y - halfSide },
        { x: center.x + halfSide, y: center.y + halfSide },
        { x: center.x - halfSide, y: center.y + halfSide },
      ];
    }
  }
  return region.geometry;
}

export function boundsForPoints(points: Point2D[], padding = 0): MapBounds | null {
  if (points.length === 0) return null;
  const min_x = Math.min(...points.map((point) => point.x));
  const max_x = Math.max(...points.map((point) => point.x));
  const min_y = Math.min(...points.map((point) => point.y));
  const max_y = Math.max(...points.map((point) => point.y));
  const horizontalPadding = padding > 0 ? Math.max(1, (max_x - min_x) * padding) : 0;
  const verticalPadding = padding > 0 ? Math.max(1, (max_y - min_y) * padding) : 0;
  return {
    min_x: min_x - horizontalPadding,
    max_x: max_x + horizontalPadding,
    min_y: min_y - verticalPadding,
    max_y: max_y + verticalPadding,
  };
}

export function pointInPolygon(point: Point2D, polygon: Point2D[]): boolean {
  if (polygon.length < 3) return false;
  let inside = false;
  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
    const current = polygon[index];
    const prior = polygon[previous];
    const crosses = (current.y > point.y) !== (prior.y > point.y)
      && point.x < ((prior.x - current.x) * (point.y - current.y)) / (prior.y - current.y) + current.x;
    if (crosses) inside = !inside;
  }
  return inside;
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
