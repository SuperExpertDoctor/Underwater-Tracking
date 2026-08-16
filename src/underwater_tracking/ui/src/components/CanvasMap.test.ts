import { describe, expect, it } from "vitest";
import type { UUVView } from "../types/frames";
import {
  CARRIER_ASSET_HEADING_OFFSET,
  carrierAssetRotation,
  uuvSpriteAppearance,
} from "./CanvasMap";

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
      .toContain("#f6b94a");
    expect(uuvSpriteAppearance({ ...uuv, status: "failed" }, image, 1, false).cueColors)
      .toContain("#ff6f7f");
    expect(uuvSpriteAppearance({ ...uuv, reserved: true }, image, 1, false).cueColors)
      .toContain("#b29cff");
    expect(uuvSpriteAppearance(uuv, image, 1, true).cueColors)
      .toContain("#dbeafe");
  });
});
