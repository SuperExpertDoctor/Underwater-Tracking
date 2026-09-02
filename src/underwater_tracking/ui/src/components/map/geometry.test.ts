import { describe, expect, it } from "vitest";
import type { MapBounds, Point2D } from "../../types/frames";
import {
  clipRayToBounds,
  covarianceAxes,
  corridorPolygon,
  displayRegionPoints,
  regionDisplaySide,
  recoverySegment,
  sharedRegionDisplaySide,
  screenToWorld,
  spriteHitAreaContains,
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

  it("uses published square corners consistently across display layers", () => {
    expect(displayRegionPoints({
      geometry: [
        { x: 10, y: 10 },
        { x: 30, y: 10 },
        { x: 20, y: 20 },
      ],
      top_left_xy: { x: 5, y: 35 },
      bottom_right_xy: { x: 35, y: 5 },
    })).toEqual([
      { x: 5, y: 5 },
      { x: 35, y: 5 },
      { x: 35, y: 35 },
      { x: 5, y: 35 },
    ]);
  });

  it("normalizes a mission's region display squares to one common side", () => {
    const regions = [
      {
        geometry: [],
        top_left_xy: { x: 0, y: 1_600 },
        bottom_right_xy: { x: 1_600, y: 0 },
      },
      {
        geometry: [],
        top_left_xy: { x: 2_000, y: 1_200 },
        bottom_right_xy: { x: 3_200, y: 0 },
      },
    ];
    const side = sharedRegionDisplaySide(regions);
    expect(side).toBe(1_600);
    expect(regionDisplaySide(regions[1])).toBe(1_200);
    expect(displayRegionPoints(regions[1], side)).toEqual([
      { x: 1_800, y: -200 },
      { x: 3_400, y: -200 },
      { x: 3_400, y: 1_400 },
      { x: 1_800, y: 1_400 },
    ]);
  });

  it("returns a recovery segment in the current zoomed and panned view", () => {
    expect(recoverySegment(
      { x: 0, y: 0 },
      { x: 100, y: 50 },
      bounds,
      800,
      600,
      { zoom: 2, pan: { x: 20, y: -30 } },
    )).toEqual({
      start: { x: -380, y: 670 },
      end: { x: 1220, y: -130 },
    });
  });

  it("selects points through the rendered sprite edge and tolerance", () => {
    const center = { x: 100, y: 100 };
    const size = { width: 40, height: 20 };

    expect(spriteHitAreaContains({ x: 124, y: 100 }, center, size, 0, 4)).toBe(true);
    expect(spriteHitAreaContains({ x: 125, y: 100 }, center, size, 0, 4)).toBe(false);
    expect(spriteHitAreaContains({ x: 100, y: 124 }, center, size, Math.PI / 2, 4)).toBe(true);
    expect(spriteHitAreaContains({ x: 124, y: 100 }, center, size, Math.PI / 2, 4)).toBe(false);
  });
});
