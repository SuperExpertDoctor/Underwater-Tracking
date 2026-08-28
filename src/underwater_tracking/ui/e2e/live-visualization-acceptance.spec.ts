import { mkdirSync, appendFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test, type Page, type TestInfo } from "@playwright/test";

const CHECKPOINTS_S = [600, 1800, 3600, 7200, 14400, 21600, 28800] as const;
const ACCEPTANCE_DIR = process.env.UNDERWATER_TRACKING_ACCEPTANCE_DIR;

type JsonObject = Record<string, unknown>;

interface CanvasPixels {
  variance: number;
  nonBackground: number;
  red: number;
  amber: number;
  cyan: number;
}

interface ConsoleRecord {
  timestamp_utc: string;
  type: "error" | "pageerror";
  message: string;
  checkpoint_s: number | null;
}

function configuredAcceptanceDir(testInfo: TestInfo): string {
  return ACCEPTANCE_DIR ?? join(testInfo.outputDir, "acceptance");
}

function appendConsoleRecords(records: ConsoleRecord[], testInfo: TestInfo): void {
  const acceptanceDir = configuredAcceptanceDir(testInfo);
  mkdirSync(acceptanceDir, { recursive: true });
  const outputPath = join(acceptanceDir, "browser-console.jsonl");
  if (records.length > 0) {
    appendFileSync(
      outputPath,
      `${records.map((record) => JSON.stringify(record)).join("\n")}\n`,
      "utf8",
    );
  }
}

async function readSnapshot(page: Page): Promise<JsonObject> {
  const response = await page.request.get("/api/operational/snapshot", {
    timeout: 2_000,
  });
  expect(response.ok()).toBeTruthy();
  const payload: unknown = await response.json();
  expect(payload).toBeTruthy();
  expect(typeof payload).toBe("object");
  return payload as JsonObject;
}

async function waitForCheckpoint(page: Page, checkpointS: number): Promise<JsonObject> {
  const deadline = Date.now() + 20 * 60 * 1_000;
  let lastSimTime: unknown = null;
  while (Date.now() < deadline) {
    const frame = await readSnapshot(page);
    lastSimTime = frame.sim_time_s;
    if (
      typeof frame.sim_time_s === "number" &&
      Number.isFinite(frame.sim_time_s) &&
      frame.sim_time_s >= checkpointS
    ) {
      return frame;
    }
    await page.waitForTimeout(250);
  }
  throw new Error(`checkpoint ${checkpointS}s was not published; last sim_time_s=${String(lastSimTime)}`);
}

async function waitForRenderedFrame(page: Page, frame: JsonObject): Promise<void> {
  const expectedSimTime = typeof frame.sim_time_s === "number" ? frame.sim_time_s : 0;
  const execution = frame.execution;
  const regions =
    execution && typeof execution === "object"
      ? (execution as JsonObject).regions
      : undefined;
  const expectedRegionCount = Array.isArray(regions) ? regions.length : 0;
  await expect.poll(
    async () => {
      const readout = await page.locator("[data-sim-time]").textContent();
      const canvas = page.locator("canvas").first();
      const planVersion = await canvas.getAttribute("data-plan-version");
      const regionCount = await canvas.getAttribute("data-execution-region-count");
      const renderedSimTime = Number(readout?.replace(/[^0-9.]+/g, ""));
      return {
        planVersion,
        regionCount,
        renderedSimTime: Number.isFinite(renderedSimTime) ? renderedSimTime : -1,
      };
    },
    { timeout: 15_000, intervals: [100, 250, 500] },
  ).toMatchObject({
    planVersion: String(frame.plan_version ?? 0),
    regionCount: String(expectedRegionCount),
  });
  await expect.poll(
    async (): Promise<number> => {
      const readout = await page.locator("[data-sim-time]").textContent();
      return Number(readout?.replace(/[^0-9.]+/g, ""));
    },
    { timeout: 15_000, intervals: [100, 250, 500] },
  ).toBeGreaterThanOrEqual(expectedSimTime);
}

async function canvasPixels(page: Page): Promise<CanvasPixels> {
  return page.locator("canvas").first().evaluate((element) => {
    const canvas = element as HTMLCanvasElement;
    const context = canvas.getContext("2d");
    if (!context || canvas.width === 0 || canvas.height === 0) {
      return { variance: 0, nonBackground: 0, red: 0, amber: 0, cyan: 0 };
    }
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let sum = 0;
    let sumSquared = 0;
    let nonBackground = 0;
    let red = 0;
    let amber = 0;
    let cyan = 0;
    const stride = Math.max(1, Math.floor(pixels.length / 4 / 120_000));
    for (let index = 0; index < pixels.length; index += 4 * stride) {
      const r = pixels[index] ?? 0;
      const g = pixels[index + 1] ?? 0;
      const b = pixels[index + 2] ?? 0;
      const luminance = (r + g + b) / 3;
      sum += luminance;
      sumSquared += luminance * luminance;
      if (Math.max(r, g, b) - Math.min(r, g, b) > 12) nonBackground += 1;
      if (r > 170 && g < 155 && b < 175 && r - g > 35) red += 1;
      if (r > 145 && g > 100 && b < 135 && r > b * 1.35) amber += 1;
      if (r < 145 && g > 125 && b > 125 && g + b > r * 2.2) cyan += 1;
    }
    const samples = Math.max(1, Math.ceil(pixels.length / (4 * stride)));
    const mean = sum / samples;
    return {
      variance: Math.max(0, sumSquared / samples - mean * mean),
      nonBackground,
      red,
      amber,
      cyan,
    };
  });
}

function targetPredictions(frame: JsonObject): JsonObject[] {
  const estimates = frame.target_estimates;
  if (!Array.isArray(estimates)) return [];
  return estimates.flatMap((estimate) => {
    if (!estimate || typeof estimate !== "object") return [];
    const prediction = (estimate as JsonObject).prediction;
    return prediction && typeof prediction === "object"
      ? [prediction as JsonObject]
      : [];
  });
}

function executionObject(frame: JsonObject): JsonObject {
  return frame.execution && typeof frame.execution === "object"
    ? frame.execution as JsonObject
    : {};
}

async function assertRegionGeometryAndStatus(page: Page, frame: JsonObject): Promise<void> {
  const polygons = page.locator(".region-map-overlay g[data-execution-region-id] polygon");
  await expect(polygons).toHaveCount(4);
  const geometry = await polygons.evaluateAll((items) => items.map((item) => {
    const element = item as SVGGraphicsElement;
    const bounds = element.getBoundingClientRect();
    const style = getComputedStyle(element);
    return {
      area: bounds.width * bounds.height,
      stroke: element.getAttribute("stroke") ?? style.stroke,
      fill: element.getAttribute("fill") ?? style.fill,
    };
  }));
  expect(geometry.every((item) => item.area > 20)).toBeTruthy();
  expect(geometry.every((item) => item.stroke && item.fill)).toBeTruthy();
  const regions = Array.isArray(executionObject(frame).regions)
    ? executionObject(frame).regions as JsonObject[]
    : [];
  const statuses = new Set(regions.map((region) => String(region.status ?? "")));
  if (statuses.size > 1) {
    expect(new Set(geometry.map((item) => item.stroke)).size).toBeGreaterThan(1);
  }
}

async function assertPredictionRendering(page: Page, frame: JsonObject): Promise<void> {
  const predictions = targetPredictions(frame);
  const usable = predictions.some((prediction) => {
    const health = prediction.health;
    return health && typeof health === "object" &&
      ((health as JsonObject).status === "valid" || (health as JsonObject).status === "degraded") &&
      Array.isArray(prediction.centerline_xy) && prediction.centerline_xy.length >= 2;
  });
  if (usable) {
    const bands = page.locator(".imm-confidence-band");
    expect(await bands.count()).toBeGreaterThan(0);
    const centerlineLength = await page.locator(".imm-prediction-centerline").evaluateAll((items) =>
      items.reduce((total, item) => {
        const path = item as SVGGeometryElement;
        return total + (typeof path.getTotalLength === "function" ? path.getTotalLength() : 0);
      }, 0),
    );
    expect(centerlineLength).toBeGreaterThan(5);
    const bandArea = await bands.evaluateAll((items) => items.reduce((total, item) => {
      const bounds = (item as SVGGraphicsElement).getBoundingClientRect();
      return total + bounds.width * bounds.height;
    }, 0));
    expect(bandArea).toBeGreaterThan(20);
  } else {
    await expect(page.locator(".imm-confidence-band")).toHaveCount(0);
    await expect(page.locator(".imm-prediction-centerline")).toHaveCount(0);
  }
}

async function assertViewportContainment(page: Page): Promise<void> {
  const boxes = await page.locator(
    ".canvas-area text, .map-tools, .map-scale, .map-region-selection",
  ).evaluateAll((elements) => elements.map((element) => {
    const box = element.getBoundingClientRect();
    return {
      left: box.left,
      top: box.top,
      right: box.right,
      bottom: box.bottom,
      visible: box.width > 0 && box.height > 0,
    };
  }));
  const viewport = page.viewportSize();
  expect(viewport).not.toBeNull();
  const viewportWidth = viewport?.width ?? 0;
  const viewportHeight = viewport?.height ?? 0;
  for (const box of boxes.filter((item) => item.visible)) {
    expect(box.left).toBeGreaterThanOrEqual(-1);
    expect(box.top).toBeGreaterThanOrEqual(-1);
    expect(box.right).toBeLessThanOrEqual(viewportWidth + 1);
    expect(box.bottom).toBeLessThanOrEqual(viewportHeight + 1);
  }
}

test.describe("real owned live visualization", () => {
  test("renders every operational checkpoint without synthetic data", async ({ page }, testInfo) => {
    const consoleErrors: ConsoleRecord[] = [];
    let activeCheckpoint: number | null = null;
    page.on("console", (message) => {
      if (message.type() === "error") {
        consoleErrors.push({
          timestamp_utc: new Date().toISOString(),
          type: "error",
          message: message.text(),
          checkpoint_s: activeCheckpoint,
        });
      }
    });
    page.on("pageerror", (error) => {
      consoleErrors.push({
        timestamp_utc: new Date().toISOString(),
        type: "pageerror",
        message: error.message,
        checkpoint_s: activeCheckpoint,
      });
    });

    try {
      await page.goto("/", { waitUntil: "domcontentloaded" });
      await expect(page.locator("canvas").first()).toBeVisible();
      for (const checkpointS of CHECKPOINTS_S) {
        activeCheckpoint = checkpointS;
        const frame = await waitForCheckpoint(page, checkpointS);
        await waitForRenderedFrame(page, frame);
        const pixels = await canvasPixels(page);
        expect(pixels.variance).toBeGreaterThan(1);
        expect(pixels.nonBackground).toBeGreaterThan(100);

        const estimates = Array.isArray(frame.target_estimates) ? frame.target_estimates : [];
        const hasDetection = estimates.some((estimate) =>
          estimate && typeof estimate === "object" &&
          typeof (estimate as JsonObject).detection_range_m === "number" &&
          Number.isFinite((estimate as JsonObject).detection_range_m as number),
        );
        if (hasDetection) expect(pixels.red).toBeGreaterThan(4);

        const uuvs = Array.isArray(frame.uuvs) ? frame.uuvs : [];
        const activeDeclared = uuvs.some((uuv) =>
          uuv && typeof uuv === "object" && (uuv as JsonObject).sensor_mode === "active",
        );
        const passiveDeclared = uuvs.some((uuv) =>
          uuv && typeof uuv === "object" && (uuv as JsonObject).sensor_mode === "passive",
        );
        if (activeDeclared) expect(pixels.amber).toBeGreaterThan(10);
        if (passiveDeclared) expect(pixels.cyan).toBeGreaterThan(10);

        await assertRegionGeometryAndStatus(page, frame);
        await assertPredictionRendering(page, frame);
        await assertViewportContainment(page);

        const screenshotDir = join(configuredAcceptanceDir(testInfo), "screenshots");
        mkdirSync(screenshotDir, { recursive: true });
        const viewportName = (testInfo.project.name === "mobile" || page.viewportSize()?.width === 390)
          ? "mobile"
          : "desktop";
        await page.screenshot({
          path: join(screenshotDir, `${viewportName}-${checkpointS}.png`),
          animations: "disabled",
        });
      }
    } finally {
      appendConsoleRecords(consoleErrors, testInfo);
    }
    expect(consoleErrors).toEqual([]);
  });
});
