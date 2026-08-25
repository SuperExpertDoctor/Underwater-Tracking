import { afterEach, describe, expect, it, vi } from "vitest";

import {
  SCENE_ASSET_URLS,
  coverImageRect,
  loadSceneAssets,
  type ImageLoader,
} from "./sceneAssets";

const validImage = {} as HTMLImageElement;

describe("scene assets", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("uses stable browser URLs for every scene asset", () => {
    expect(SCENE_ASSET_URLS).toEqual({
      background: "/assets/background.png",
      aircraftCarrier: "/assets/aircraft-carrier.png",
      warship: "/assets/warship.png",
      uuv: "/assets/UUV.png",
      submarine: "/assets/submarine.png",
    });
  });

  it("keeps a failed scene image nullable without rejecting other assets", async () => {
    const loader: ImageLoader = async (url) => (url.endsWith("warship.png") ? null : validImage);

    const assets = await loadSceneAssets(loader);

    expect(assets.background).not.toBeNull();
    expect(assets.warship).toBeNull();
  });

  it("resolves failed browser image loading as null", async () => {
    class ControlledImage {
      onerror: (() => void) | null = null;
      onload: (() => void) | null = null;

      set src(url: string) {
        if (url.endsWith("warship.png")) this.onerror?.();
        else this.onload?.();
      }
    }

    vi.stubGlobal("Image", ControlledImage);

    const assets = await loadSceneAssets();

    expect(assets.background).toBeInstanceOf(ControlledImage);
    expect(assets.warship).toBeNull();
  });

  it("computes a centered cover rectangle", () => {
    const rect = coverImageRect(1672, 941, 1200, 700);

    expect(rect.x).toBeCloseTo(-21.89, 1);
    expect(rect.y).toBe(0);
    expect(rect.width).toBeCloseTo(1243.78, 1);
    expect(rect.height).toBe(700);
  });
});
