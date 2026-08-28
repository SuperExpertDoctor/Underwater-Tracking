import { mkdirSync, appendFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test, type Page, type TestInfo } from "@playwright/test";

const CHECKPOINTS_S = [600, 1800, 3600, 7200, 14400, 21600, 28800] as const;
const ACCEPTANCE_DIR = process.env.UNDERWATER_TRACKING_ACCEPTANCE_DIR;

type JsonObject = Record<string, unknown>;

interface CanvasPixels {
  variance: number;
  nonBackground: number;
}

interface WorldPoint {
  x: number;
  y: number;
}

interface WorldBounds {
  min_x: number;
  min_y: number;
  max_x: number;
  max_y: number;
}

interface CanvasProjection extends WorldBounds {
  width: number;
  height: number;
  scale: number;
  offsetX: number;
  offsetY: number;
}

interface ScreenRect {
  left: number;
  top: number;
  right: number;
  bottom: number;
  width: number;
  height: number;
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
      return { variance: 0, nonBackground: 0 };
    }
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let sum = 0;
    let sumSquared = 0;
    let nonBackground = 0;
    const stride = Math.max(1, Math.floor(pixels.length / 4 / 120_000));
    for (let index = 0; index < pixels.length; index += 4 * stride) {
      const r = pixels[index] ?? 0;
      const g = pixels[index + 1] ?? 0;
      const b = pixels[index + 2] ?? 0;
      const luminance = (r + g + b) / 3;
      sum += luminance;
      sumSquared += luminance * luminance;
      if (Math.max(r, g, b) - Math.min(r, g, b) > 12) nonBackground += 1;
    }
    const samples = Math.max(1, Math.ceil(pixels.length / (4 * stride)));
    const mean = sum / samples;
    return {
      variance: Math.max(0, sumSquared / samples - mean * mean),
      nonBackground,
    };
  });
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function worldPoint(value: unknown): WorldPoint | null {
  if (!value || typeof value !== "object") return null;
  const point = value as JsonObject;
  const x = finiteNumber(point.x);
  const y = finiteNumber(point.y);
  return x === null || y === null ? null : { x, y };
}

function worldBounds(frame: JsonObject): WorldBounds | null {
  const raw = frame.map_bounds;
  if (!raw || typeof raw !== "object") return null;
  const bounds = raw as JsonObject;
  const minX = finiteNumber(bounds.min_x);
  const minY = finiteNumber(bounds.min_y);
  const maxX = finiteNumber(bounds.max_x);
  const maxY = finiteNumber(bounds.max_y);
  if (minX === null || minY === null || maxX === null || maxY === null) return null;
  if (maxX <= minX || maxY <= minY) return null;
  return { min_x: minX, min_y: minY, max_x: maxX, max_y: maxY };
}

function detectionRange(frame: JsonObject, target: JsonObject): number {
  const adversary = frame.adversary && typeof frame.adversary === "object"
    ? frame.adversary as JsonObject
    : null;
  const configured = finiteNumber(adversary?.detection_range_m)
    ?? finiteNumber(target.detection_range_m);
  return configured !== null && configured > 1 ? configured : 5_000;
}

function projectWorld(point: WorldPoint, projection: CanvasProjection): WorldPoint {
  return {
    x: projection.offsetX + (point.x - projection.min_x) * projection.scale,
    y: projection.height - projection.offsetY - (point.y - projection.min_y) * projection.scale,
  };
}

async function canvasProjection(page: Page, frame: JsonObject): Promise<CanvasProjection> {
  const fallback = worldBounds(frame);
  if (!fallback) throw new Error("live frame has no usable map bounds");
  const projection = await page.locator(".canvas-area").evaluate((element, fallbackBounds) => {
    const canvas = element.querySelector("canvas");
    if (!canvas) throw new Error("canvas is missing from map area");
    const raw = element.getAttribute("data-visible-bounds");
    const parsed = raw ? JSON.parse(raw) as Partial<WorldBounds> : fallbackBounds;
    const min_x = Number(parsed.min_x);
    const min_y = Number(parsed.min_y);
    const max_x = Number(parsed.max_x);
    const max_y = Number(parsed.max_y);
    const rect = canvas.getBoundingClientRect();
    const width = canvas.clientWidth || rect.width;
    const height = canvas.clientHeight || rect.height;
    const scale = Math.min(width / (max_x - min_x), height / (max_y - min_y));
    return {
      min_x,
      min_y,
      max_x,
      max_y,
      width,
      height,
      scale,
      offsetX: (width - (max_x - min_x) * scale) / 2,
      offsetY: (height - (max_y - min_y) * scale) / 2,
    };
  }, fallback);
  expect(projection.width).toBeGreaterThan(0);
  expect(projection.height).toBeGreaterThan(0);
  expect(projection.scale).toBeGreaterThan(0);
  return projection as CanvasProjection;
}

type CanvasColor = "red" | "amber" | "cyan";

async function sampleCanvasColorNear(
  page: Page,
  points: WorldPoint[],
  color: CanvasColor,
): Promise<{ matched: number; total: number }> {
  return page.locator("canvas").first().evaluate((element, input) => {
    const canvas = element as HTMLCanvasElement;
    const context = canvas.getContext("2d");
    const rect = canvas.getBoundingClientRect();
    if (!context || !rect.width || !rect.height) return { matched: 0, total: input.points.length };
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
    const matches = (r: number, g: number, b: number): boolean => {
      if (input.color === "red") return r > 100 && r > g + 25 && r > b + 15;
      if (input.color === "amber") return r > 105 && r > g + 8 && r > b + 20;
      return g > 105 && b > 110 && g > r + 35 && b > r + 30 && g + b > 230;
    };
    let matched = 0;
    for (const point of input.points) {
      const pixelX = Math.round(point.x * canvas.width / rect.width);
      const pixelY = Math.round(point.y * canvas.height / rect.height);
      let pointMatched = false;
      for (let dy = -2; dy <= 2 && !pointMatched; dy += 1) {
        for (let dx = -2; dx <= 2 && !pointMatched; dx += 1) {
          const x = pixelX + dx;
          const y = pixelY + dy;
          if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) continue;
          const offset = (y * canvas.width + x) * 4;
          if (matches(pixels.data[offset] ?? 0, pixels.data[offset + 1] ?? 0, pixels.data[offset + 2] ?? 0)) {
            pointMatched = true;
          }
        }
      }
      if (pointMatched) matched += 1;
    }
    return { matched, total: input.points.length };
  }, { points, color });
}

function visiblePoints(points: WorldPoint[], projection: CanvasProjection): WorldPoint[] {
  return points.filter((point) =>
    point.x >= 0 && point.x <= projection.width && point.y >= 0 && point.y <= projection.height,
  );
}

function targetEstimate(frame: JsonObject): JsonObject {
  const estimates = Array.isArray(frame.target_estimates) ? frame.target_estimates : [];
  const targetId = executionObject(frame).target_id;
  const target = estimates.find((estimate) =>
    estimate && typeof estimate === "object" &&
    (targetId === undefined || (estimate as JsonObject).target_id === targetId),
  );
  if (!target || typeof target !== "object") throw new Error("live frame has no execution target");
  return target as JsonObject;
}

function currentTaskUuvs(frame: JsonObject): JsonObject[] {
  const execution = executionObject(frame);
  const groups = Array.isArray(execution.task_groups) ? execution.task_groups : [];
  const currentRegionId = execution.current_region_id;
  const group = groups.find((item) =>
    item && typeof item === "object" && (item as JsonObject).region_id === currentRegionId,
  );
  const memberIds = new Set(
    group && typeof group === "object" && Array.isArray((group as JsonObject).member_uuv_ids)
      ? ((group as JsonObject).member_uuv_ids as unknown[]).filter((id): id is string => typeof id === "string")
      : [],
  );
  const uuvs = Array.isArray(frame.uuvs) ? frame.uuvs : [];
  return uuvs.filter((uuv) =>
    uuv && typeof uuv === "object" &&
    (uuv as JsonObject).physically_exposed === true &&
    memberIds.has(String((uuv as JsonObject).uuv_id)),
  ) as JsonObject[];
}

function sensorRange(uuv: JsonObject): number {
  const mode = uuv.sensor_mode;
  const preferred = mode === "active" ? uuv.active_range_m : uuv.passive_range_m;
  const fallback = mode === "active" ? uuv.passive_range_m : uuv.active_range_m;
  return finiteNumber(preferred) ?? finiteNumber(fallback) ?? 2_000;
}

function sensorArcPoints(uuv: JsonObject, projection: CanvasProjection): WorldPoint[] {
  const center = worldPoint(uuv.position);
  if (!center) return [];
  const screenCenter = projectWorld(center, projection);
  const radius = sensorRange(uuv) * projection.scale;
  const heading = finiteNumber(uuv.sensor_heading_rad) ?? finiteNumber(uuv.heading_rad) ?? 0;
  const centerAngle = heading === 0 ? 0 : -heading;
  const points: WorldPoint[] = [];
  for (const fraction of [0.72, 0.96]) {
    for (let index = 0; index <= 24; index += 1) {
      const angle = centerAngle - Math.PI / 4 + (Math.PI / 2) * index / 24;
      points.push({
        x: screenCenter.x + Math.cos(angle) * radius * fraction,
        y: screenCenter.y + Math.sin(angle) * radius * fraction,
      });
    }
  }
  return visiblePoints(points, projection);
}

async function assertDetectionGeometry(page: Page, frame: JsonObject, projection: CanvasProjection): Promise<void> {
  const target = targetEstimate(frame);
  const center = worldPoint(target.mean);
  if (!center) throw new Error("execution target has no finite mean");
  const screenCenter = projectWorld(center, projection);
  const radiusPx = detectionRange(frame, target) * projection.scale;
  expect(screenCenter.x).toBeGreaterThanOrEqual(0);
  expect(screenCenter.x).toBeLessThanOrEqual(projection.width);
  expect(screenCenter.y).toBeGreaterThanOrEqual(0);
  expect(screenCenter.y).toBeLessThanOrEqual(projection.height);
  expect(radiusPx * 2).toBeGreaterThanOrEqual(24);
  const perimeter = visiblePoints(
    Array.from({ length: 96 }, (_, index) => {
      const angle = 2 * Math.PI * index / 96;
      return {
        x: screenCenter.x + Math.cos(angle) * radiusPx,
        y: screenCenter.y + Math.sin(angle) * radiusPx,
      };
    }),
    projection,
  );
  expect(perimeter.length).toBeGreaterThan(16);
  const samples = await sampleCanvasColorNear(page, perimeter, "red");
  expect(samples.matched).toBeGreaterThan(6);
  expect(samples.matched / samples.total).toBeGreaterThan(0.08);
  expect(samples.matched).toBeLessThan(samples.total * 0.92);
}

async function assertSonarAttribution(page: Page, frame: JsonObject, projection: CanvasProjection): Promise<void> {
  const taskUuvs = currentTaskUuvs(frame);
  const renderedIds = (await page.locator("canvas").first().getAttribute("data-current-task-uuv-ids")) ?? "";
  for (const mode of ["active", "passive"] as const) {
    const candidates = taskUuvs.filter((uuv) => uuv.sensor_mode === mode);
    if (candidates.length === 0) continue;
    expect(candidates.every((uuv) => renderedIds.split(",").includes(String(uuv.uuv_id)))).toBeTruthy();
    let best = { matched: 0, total: 0 };
    for (const uuv of candidates) {
      const points = sensorArcPoints(uuv, projection);
      const sample = await sampleCanvasColorNear(page, points, mode === "active" ? "amber" : "cyan");
      if (sample.matched > best.matched) best = sample;
    }
    expect(best.total).toBeGreaterThan(6);
    expect(best.matched).toBeGreaterThan(4);
    expect(best.matched / best.total).toBeGreaterThan(0.03);
  }
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
  const regionGroups = page.locator(".region-map-overlay g[data-execution-region-id]");
  await expect(regionGroups).toHaveCount(4);
  const rendered = await regionGroups.evaluateAll((items) => items.map((item) => {
    const element = item as SVGGElement;
    const polygon = element.querySelector("polygon");
    const bounds = polygon?.getBoundingClientRect();
    const style = polygon ? getComputedStyle(polygon) : null;
    return {
      regionId: element.getAttribute("data-execution-region-id"),
      taskGroupId: element.getAttribute("data-task-group-id"),
      label: element.getAttribute("aria-label") ?? element.textContent ?? "",
      area: (bounds?.width ?? 0) * (bounds?.height ?? 0),
      left: bounds?.left ?? 0,
      top: bounds?.top ?? 0,
      right: bounds?.right ?? 0,
      bottom: bounds?.bottom ?? 0,
      stroke: polygon?.getAttribute("stroke") ?? style?.stroke ?? "",
      fill: polygon?.getAttribute("fill") ?? style?.fill ?? "",
      current: element.getAttribute("data-current-region") === "true",
      next: element.getAttribute("data-next-region") === "true",
    };
  }));
  const regions = Array.isArray(executionObject(frame).regions)
    ? executionObject(frame).regions as JsonObject[]
    : [];
  const canvasBounds = await page.locator("canvas").first().evaluate((item) => {
    const box = item.getBoundingClientRect();
    return { left: box.left, top: box.top, right: box.right, bottom: box.bottom };
  });
  expect(regions).toHaveLength(4);
  const expectedById = new Map(
    regions.map((region) => [String(region.region_id), region]),
  );
  expect(rendered.every((item) => expectedById.has(String(item.regionId)))).toBeTruthy();
  for (const item of rendered) {
    const expected = expectedById.get(String(item.regionId));
    expect(expected).toBeDefined();
    expect(item.area).toBeGreaterThan(20);
    expect(item.left).toBeGreaterThanOrEqual(canvasBounds.left - 1);
    expect(item.top).toBeGreaterThanOrEqual(canvasBounds.top - 1);
    expect(item.right).toBeLessThanOrEqual(canvasBounds.right + 1);
    expect(item.bottom).toBeLessThanOrEqual(canvasBounds.bottom + 1);
    expect(item.stroke).toBeTruthy();
    expect(item.fill).toBeTruthy();
    expect(item.label).toContain("R");
    expect(item.taskGroupId).toBe(String(expected?.task_group_id));
    if (!item.current && !item.next) {
      const status = String(expected?.status ?? "").toLowerCase();
      const colorToken = status === "degraded" || status === "uncovered"
        ? status === "degraded" ? "255, 120, 130" : "173, 190, 205"
        : status === "handoff_pending" ? "247, 189, 69"
          : status === "active" || status === "passive" ? "33, 208, 195"
            : "196, 180, 255";
      expect(item.stroke).toContain(colorToken);
    }
  }
  const statusColors = new Set(
    rendered.filter((item) => !item.current && !item.next).map((item) => item.stroke),
  );
  const statuses = new Set(regions.map((region) => String(region.status ?? "")));
  if (statuses.size > 1 && rendered.some((item) => !item.current && !item.next)) {
    expect(statusColors.size).toBeGreaterThan(1);
  }
}

async function assertPredictionRendering(page: Page, frame: JsonObject): Promise<void> {
  const predictions = targetPredictions(frame);
  const prediction = predictions[0];
  const health = prediction?.health;
  const usable = Boolean(health && typeof health === "object" &&
    ((health as JsonObject).status === "valid" || (health as JsonObject).status === "degraded") &&
    Array.isArray(prediction?.centerline_xy) && prediction.centerline_xy.length >= 2);
  if (usable) {
    const bands = page.locator(".imm-confidence-band");
    expect(await bands.count()).toBeGreaterThan(0);
    const targetId = String(executionObject(frame).target_id ?? "");
    const overlay = page.locator(`.imm-prediction-overlay g[data-target-id="${targetId}"]`);
    await expect(overlay).toHaveCount(1);
    await expect(overlay).toHaveAttribute(
      "data-prediction-id",
      String(prediction.prediction_id),
    );
    const centerlineLength = await page.locator(".imm-prediction-centerline").evaluateAll((items) =>
      items.reduce((total, item) => {
        const path = item as SVGGeometryElement;
        return total + (typeof path.getTotalLength === "function" ? path.getTotalLength() : 0);
      }, 0),
    );
    expect(centerlineLength).toBeGreaterThanOrEqual(16);
    const bandArea = await bands.evaluateAll((items) => items.reduce((total, item) => {
      const bounds = (item as SVGGraphicsElement).getBoundingClientRect();
      return total + bounds.width * bounds.height;
    }, 0));
    expect(bandArea).toBeGreaterThanOrEqual(40);
    const bandBounds = await bands.first().evaluate((item) => {
      const box = (item as SVGGraphicsElement).getBoundingClientRect();
      return { left: box.left, top: box.top, right: box.right, bottom: box.bottom };
    });
    const canvasBounds = await page.locator("canvas").first().evaluate((item) => {
      const box = item.getBoundingClientRect();
      return { left: box.left, top: box.top, right: box.right, bottom: box.bottom };
    });
    expect(bandBounds.left).toBeGreaterThanOrEqual(canvasBounds.left - 1);
    expect(bandBounds.top).toBeGreaterThanOrEqual(canvasBounds.top - 1);
    expect(bandBounds.right).toBeLessThanOrEqual(canvasBounds.right + 1);
    expect(bandBounds.bottom).toBeLessThanOrEqual(canvasBounds.bottom + 1);
  } else {
    await expect(page.locator(".imm-confidence-band")).toHaveCount(0);
    await expect(page.locator(".imm-prediction-centerline")).toHaveCount(0);
  }
}

function rectanglesOverlap(left: ScreenRect, right: ScreenRect): boolean {
  return left.left < right.right && left.right > right.left
    && left.top < right.bottom && left.bottom > right.top;
}

async function assertViewportContainment(page: Page): Promise<void> {
  const canvasBox = await page.locator("canvas").first().evaluate((element) => {
    const box = element.getBoundingClientRect();
    return { left: box.left, top: box.top, right: box.right, bottom: box.bottom };
  });
  const boxes = await page.locator(
    ".region-map-overlay text, .imm-prediction-overlay text, .map-tools, .map-scale, .map-region-selection",
  ).evaluateAll((elements) => elements.map((element) => {
    const box = element.getBoundingClientRect();
    return {
      left: box.left,
      top: box.top,
      right: box.right,
      bottom: box.bottom,
      width: box.width,
      height: box.height,
      kind: element.className instanceof SVGAnimatedString
        ? element.className.baseVal
        : typeof element.className === "string" ? element.className : element.tagName,
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
    expect(box.left).toBeGreaterThanOrEqual(canvasBox.left - 1);
    expect(box.top).toBeGreaterThanOrEqual(canvasBox.top - 1);
    expect(box.right).toBeLessThanOrEqual(canvasBox.right + 1);
    expect(box.bottom).toBeLessThanOrEqual(canvasBox.bottom + 1);
  }
  const controls = boxes.filter((item) =>
    item.visible && ["map-tools", "map-scale", "map-region-selection"].some((name) => item.kind.includes(name)),
  );
  for (let leftIndex = 0; leftIndex < controls.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < controls.length; rightIndex += 1) {
      expect(rectanglesOverlap(controls[leftIndex], controls[rightIndex])).toBeFalsy();
    }
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
        const projection = await canvasProjection(page, frame);
        await assertDetectionGeometry(page, frame, projection);
        await assertSonarAttribution(page, frame, projection);
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
