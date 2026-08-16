import { describe, expect, it } from "vitest";
import type { MapBounds, Point2D } from "../../types/frames";
import {
  clipRayToBounds,
  covarianceAxes,
  corridorPolygon,
  screenToWorld,
  worldToScreen,
  zoomAroundCursor,
} from "./geometry";

const bounds: MapBounds = { min_x: 0, min_y: 0, max_x: 100, max_y: 50 };

describe("tactical map geometry", () => {
  it("maps world coordinates to screen and back deterministically", () => {
    const view = { zoom: 1, pan: { x: 0, y: 0 } };
    const screen = worldToScreen({ x: 50, y: 25 }, bounds, 800, 400, view);
    expect(screen).toEqual({ x: 400, y: 200 });
    expect(screenToWorld(screen, bounds, 800, 400, view)).toEqual({ x: 50, y: 25 });
  });

  it("keeps the world point under a cursor fixed while zooming", () => {
    const cursor = { x: 220, y: 140 };
    const before = screenToWorld(cursor, bounds, 800, 400, { zoom: 1, pan: { x: 0, y: 0 } });
    const next = zoomAroundCursor(bounds, 800, 400, { zoom: 1, pan: { x: 0, y: 0 } }, cursor, 2);
    expect(worldToScreen(before, bounds, 800, 400, next)).toEqual(cursor);
  });

  it("returns ordered covariance axes and rotation", () => {
    const result = covarianceAxes({ semimajor_m: 30, semiminor_m: 10, rotation_rad: 0.4 });
    expect(result).toEqual({ major: 30, minor: 10, rotation: 0.4 });
  });

  it("clips a bearing ray to the map boundary", () => {
    expect(clipRayToBounds({ x: 50, y: 25 }, 0, bounds)).toEqual({ x: 100, y: 25 });
    expect(clipRayToBounds({ x: 50, y: 25 }, Math.PI / 2, bounds)).toEqual({ x: 50, y: 50 });
  });

  it("builds a closed prediction corridor polygon", () => {
    const centerline: Point2D[] = [{ x: 10, y: 10 }, { x: 20, y: 10 }];
    const polygon = corridorPolygon(centerline, [2, 2]);
    expect(polygon.length).toBe(5);
    expect(polygon[0]).toEqual({ x: 10, y: 8 });
    expect(polygon.at(-1)).toEqual(polygon[0]);
  });
});
