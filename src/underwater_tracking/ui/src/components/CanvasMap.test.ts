import { describe, expect, it } from "vitest";
import type { UUVView } from "../types/frames";
import {
  CARRIER_ASSET_HEADING_OFFSET,
  DEFAULT_SUBMARINE_DETECTION_RANGE_M,
  communicationRangeForUsv,
  carrierAssetRotation,
  detectedPlatformIds,
  GRID_DIVISIONS,
  shouldDrawDetectionRange,
  submarineAssetRotation,
  targetDetectionRange,
  usvSpriteAppearance,
  uuvSpriteAppearance,
} from "./CanvasMap";
import type { OperationalFrame, TargetEstimateView, USVView } from "../types/frames";

const uuv: UUVView = {
  uuv_id: "uuv_01",
  status: "available",
  deployment_state: "deployed",
  position: { x: 0, y: 0 },
  heading_rad: 0,
  speed_mps: 1,
  energy_fraction: 1,
  group_id: null,
  current_waypoint: null,
  breadcrumb: [],
  sensor_mode: "passive",
  reserved: false,
};

describe("CanvasMap sprite semantics", () => {
  it("aligns the upward-facing carrier asset with the vector heading convention", () => {
    expect(CARRIER_ASSET_HEADING_OFFSET).toBeCloseTo(Math.PI / 2);
    expect(carrierAssetRotation(0)).toBeCloseTo(Math.PI / 2);
    expect(carrierAssetRotation(Math.PI / 2)).toBeCloseTo(0);
  });

  it("keeps active, failed, reserved, and selected cues when a UUV image is loaded", () => {
    const image = { naturalWidth: 1536, naturalHeight: 1024 } as HTMLImageElement;

    expect(uuvSpriteAppearance({ ...uuv, sensor_mode: "active" }, image, 1, false).cueColors)
      .toContain("#f7bd45");
    expect(uuvSpriteAppearance({ ...uuv, status: "failed" }, image, 1, false).cueColors)
      .toContain("#ff7882");
    expect(uuvSpriteAppearance({ ...uuv, reserved: true }, image, 1, false).cueColors)
      .toContain("#c4b4ff");
    expect(uuvSpriteAppearance(uuv, image, 1, true).cueColors)
      .toContain("#f8fdff");
  });

  it("keeps a marker ring visible for unselected UUVs and USVs", () => {
    const image = { naturalWidth: 1536, naturalHeight: 1024 } as HTMLImageElement;
    const uuvAppearance = uuvSpriteAppearance(uuv, image, 1, false);
    const usvAppearance = usvSpriteAppearance({ sensor_mode: "passive" } as USVView, image, 1);

    expect(uuvAppearance.markerRing.color).toBe("#21d0c3");
    expect(uuvAppearance.markerRing.highlightColor).toBeNull();
    expect(usvAppearance.markerRing.color).toBe("#66e0ad");
  });

  it("aligns the submarine asset and exposes range-driven platform visibility", () => {
    expect(submarineAssetRotation(0)).toBeCloseTo(Math.PI);
    const target = {
      target_id: "T1",
      mean: { x: 0, y: 0 },
      covariance_ellipse: { semimajor_m: 20, semiminor_m: 10, rotation_rad: 0 },
      intent: { label: "unknown", confidence: 0, alternatives: {} },
      prediction: null,
      quality: { quality_score: 0.8, estimated_rmse_m: 20, fim_min_eigenvalue: 1, fim_condition: 1 },
      classification: "submarine",
      last_ping_s: null,
      detection_range_m: 100,
    } as TargetEstimateView;
    const frame = {
      map_bounds: { min_x: -1000, min_y: -1000, max_x: 1000, max_y: 1000 },
      uuvs: [{ ...uuv, uuv_id: "UUV-NEAR", position: { x: 80, y: 0 } }, { ...uuv, uuv_id: "UUV-FAR", position: { x: 160, y: 0 } }],
      usvs: [],
      communication_links: [{ source_id: "USV-01", target_id: "CARRIER-01", medium: "surface", distance_m: 400, limit_m: 900, status: "connected", relay: true }],
    } as unknown as OperationalFrame;

    expect(communicationRangeForUsv(frame, "USV-01")).toBe(900);
    expect(targetDetectionRange(target)).toBe(100);
    expect(detectedPlatformIds(frame, target)).toEqual(["UUV-NEAR"]);
    expect(DEFAULT_SUBMARINE_DETECTION_RANGE_M).toBeGreaterThan(0);
  });

  it("keeps detection range opt-in while using a fine base grid", () => {
    expect(shouldDrawDetectionRange(false)).toBe(false);
    expect(shouldDrawDetectionRange(true)).toBe(true);
    expect(GRID_DIVISIONS).toBe(16);
  });
});
