import { mkdirSync, appendFileSync, renameSync, unlinkSync } from "node:fs";
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
  panX: number;
  panY: number;
}

interface RenderedFrameIdentity {
  frameId: number | null;
  simTimeS: number | null;
  executionRevision: number | null;
  predictionId: string | null;
  predictionRevision: number | null;
  targetId: string | null;
  paintSequence: number | null;
}

interface PaintedDetectionLayer {
  target_id: string;
  center: WorldPoint;
  radius_px: number;
  stroke_style: string;
  line_dash: number[];
}

interface PaintedSonarLayer {
  uuv_id: string;
  target_id: string | null;
  task_group_id: string | null;
  role: "active_verifier" | "passive_tracker" | null;
  sensor_mode: "active" | "passive";
  center: WorldPoint;
  radius_px: number;
  start_angle_rad: number;
  end_angle_rad: number;
  stroke_style: string;
  fill_style: string;
}

interface PaintedVisualLayers {
  detection: PaintedDetectionLayer[];
  sonar: PaintedSonarLayer[];
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

async function readRenderedFrameIdentity(page: Page): Promise<RenderedFrameIdentity> {
  return page.locator("canvas").first().evaluate((element) => {
    const readNumber = (value: string | undefined): number | null => {
      if (value === undefined || value === "") return null;
      const number = Number(value);
      return Number.isFinite(number) ? number : null;
    };
    const readString = (value: string | undefined): string | null => value || null;
    return {
      frameId: readNumber(element.getAttribute("data-last-painted-frame-id") ?? undefined),
      simTimeS: readNumber(element.getAttribute("data-last-painted-sim-time-s") ?? undefined),
      executionRevision: readNumber(element.getAttribute("data-last-painted-execution-revision") ?? undefined),
      predictionId: readString(element.getAttribute("data-last-painted-prediction-id") ?? undefined),
      predictionRevision: readNumber(element.getAttribute("data-last-painted-prediction-revision") ?? undefined),
      targetId: readString(element.getAttribute("data-last-painted-target-id") ?? undefined),
      paintSequence: readNumber(element.getAttribute("data-last-painted-paint-sequence") ?? undefined),
    };
  });
}

async function readPaintedVisualLayers(page: Page): Promise<PaintedVisualLayers> {
  const raw = await page.locator("canvas").first().getAttribute("data-last-painted-visual-layers");
  if (!raw) throw new Error("canvas has no last-painted visual layer contract");
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    throw new Error(`last-painted visual layer contract is not JSON: ${String(error)}`);
  }
  if (!parsed || typeof parsed !== "object") {
    throw new Error("last-painted visual layer contract is not an object");
  }
  const value = parsed as JsonObject;
  const readObjects = (key: "detection" | "sonar"): JsonObject[] => {
    const entries = value[key];
    if (!Array.isArray(entries) || entries.some((entry) => !entry || typeof entry !== "object")) {
      throw new Error(`last-painted ${key} layer contract is not an object array`);
    }
    return entries as JsonObject[];
  };
  return {
    detection: readObjects("detection") as unknown as PaintedDetectionLayer[],
    sonar: readObjects("sonar") as unknown as PaintedSonarLayer[],
  };
}

function assertRenderedFrameBinding(
  frame: JsonObject,
  identity: RenderedFrameIdentity,
): void {
  const execution = executionObject(frame);
  const prediction = targetPredictions(frame)[0];
  const targetId = typeof execution.target_id === "string" ? execution.target_id : null;
  expect(identity.frameId).toBe(finiteNumber(frame.frame_id));
  expect(identity.simTimeS).toBe(finiteNumber(frame.sim_time_s));
  expect(identity.executionRevision).toBe(finiteNumber(execution.execution_revision));
  expect(identity.targetId).toBe(targetId);
  expect(identity.predictionId).toBe(
    prediction ? String(prediction.prediction_id) : null,
  );
  expect(identity.predictionRevision).toBe(
    prediction ? finiteNumber(prediction.prediction_revision) : null,
  );
  expect(identity.paintSequence).toBeGreaterThan(0);
}

async function waitForRenderedFrame(page: Page, frame: JsonObject): Promise<JsonObject> {
  const expectedSimTime = finiteNumber(frame.sim_time_s) ?? 0;
  const executionRegions: unknown = executionObject(frame).regions;
  const expectedRegionCount = Array.isArray(executionRegions)
    ? executionRegions.length
    : 0;
  let renderedFrame: JsonObject | null = null;
  await expect.poll(
    async () => {
      const identity = await readRenderedFrameIdentity(page);
      if (identity.frameId === null || identity.simTimeS === null || identity.simTimeS < expectedSimTime) {
        return false;
      }
      const candidate = await readSnapshot(page);
      const candidatePrediction = targetPredictions(candidate)[0];
      const candidateExecution = executionObject(candidate);
      const candidateIdentity: Omit<RenderedFrameIdentity, "paintSequence"> = {
        frameId: finiteNumber(candidate.frame_id),
        simTimeS: finiteNumber(candidate.sim_time_s),
        executionRevision: finiteNumber(candidateExecution.execution_revision),
        predictionId: candidatePrediction ? String(candidatePrediction.prediction_id) : null,
        predictionRevision: candidatePrediction
          ? finiteNumber(candidatePrediction.prediction_revision)
          : null,
        targetId: typeof candidateExecution.target_id === "string"
          ? candidateExecution.target_id
          : null,
      };
      const planVersion = await page.locator("canvas").first().getAttribute("data-last-painted-plan-version");
      const regionCount = await page.locator("canvas").first().getAttribute("data-last-painted-execution-region-count");
      const identityMatchesCandidate =
        identity.frameId === candidateIdentity.frameId
        && identity.simTimeS === candidateIdentity.simTimeS
        && identity.executionRevision === candidateIdentity.executionRevision
        && identity.predictionId === candidateIdentity.predictionId
        && identity.predictionRevision === candidateIdentity.predictionRevision
        && identity.targetId === candidateIdentity.targetId
        && identity.paintSequence !== null
        && identity.paintSequence > 0;
      if (
        !identityMatchesCandidate
        || planVersion !== String(candidate.plan_version ?? 0)
        || regionCount !== String(expectedRegionCount)
      ) {
        return false;
      }
      renderedFrame = candidate;
      return true;
    },
    { timeout: 15_000, intervals: [100, 250, 500] },
  ).toBe(true);
  if (!renderedFrame) throw new Error("the page did not expose a bound rendered frame");
  assertRenderedFrameBinding(renderedFrame, await readRenderedFrameIdentity(page));
  return renderedFrame;
}

async function assertPaintStillBound(
  page: Page,
  frame: JsonObject,
  binding: RenderedFrameIdentity,
): Promise<void> {
  const current = await readRenderedFrameIdentity(page);
  assertRenderedFrameBinding(frame, current);
  expect(current.paintSequence).toBe(binding.paintSequence);
}

async function waitForPaintSequenceIncrease(
  page: Page,
  previousPaintSequence: number,
): Promise<RenderedFrameIdentity> {
  await expect.poll(
    async () => {
      const identity = await readRenderedFrameIdentity(page);
      return identity.paintSequence !== null && identity.paintSequence > previousPaintSequence;
    },
    { timeout: 5_000, intervals: [50, 100, 250] },
  ).toBe(true);
  return readRenderedFrameIdentity(page);
}

async function readCameraState(page: Page): Promise<{ zoom: number; pan: WorldPoint; width: number; height: number }> {
  return page.locator(".canvas-area").evaluate((element) => {
    const rawPan = element.getAttribute("data-camera-pan");
    const parsedPan = rawPan ? JSON.parse(rawPan) as Partial<WorldPoint> : {};
    const canvas = element.querySelector("canvas");
    const box = canvas?.getBoundingClientRect();
    return {
      zoom: Number(element.getAttribute("data-camera-zoom") ?? "0"),
      pan: { x: Number(parsedPan.x ?? 0), y: Number(parsedPan.y ?? 0) },
      width: box?.width ?? 0,
      height: box?.height ?? 0,
    };
  });
}

async function assertCameraInteractionContract(page: Page, frame: JsonObject): Promise<void> {
  await waitForRenderedFrame(page, frame);
  const initialIdentity = await readRenderedFrameIdentity(page);
  const initialCamera = await readCameraState(page);
  expect(initialCamera.zoom).toBeGreaterThan(0);
  const canvas = page.locator("canvas").first();
  const initialBox = await canvas.boundingBox();
  expect(initialBox).not.toBeNull();
  if (!initialBox) throw new Error("live canvas has no measurable bounds");
  const initialGeometry = await renderedRegionGeometry(page);
  const initialArea = polygonArea(initialGeometry.points);
  expect(initialArea).toBeGreaterThan(1);

  await canvas.hover();
  await page.mouse.wheel(0, -360);
  await expect.poll(
    async () => (await readCameraState(page)).zoom,
    { timeout: 5_000, intervals: [50, 100, 250] },
  ).toBeGreaterThan(initialCamera.zoom);
  const zoomIdentity = await waitForPaintSequenceIncrease(page, initialIdentity.paintSequence ?? 0);
  const zoomCamera = await readCameraState(page);
  expect(zoomCamera.zoom).toBeGreaterThan(initialCamera.zoom);
  expect(zoomIdentity.paintSequence).toBeGreaterThan(initialIdentity.paintSequence ?? 0);
  const zoomGeometry = await renderedRegionGeometry(page);
  expect(geometryMoved(initialGeometry.points, zoomGeometry.points)).toBeTruthy();
  expect(polygonArea(zoomGeometry.points)).toBeGreaterThan(initialArea * 1.05);

  const dragStart = { x: initialBox.x + initialBox.width / 2, y: initialBox.y + initialBox.height / 2 };
  await page.mouse.move(dragStart.x, dragStart.y);
  await page.mouse.down();
  await page.mouse.move(dragStart.x + 44, dragStart.y + 27, { steps: 3 });
  await page.mouse.up();
  await expect.poll(
    async () => {
      const camera = await readCameraState(page);
      return Math.hypot(camera.pan.x - zoomCamera.pan.x, camera.pan.y - zoomCamera.pan.y);
    },
    { timeout: 5_000, intervals: [50, 100, 250] },
  ).toBeGreaterThan(10);
  const panIdentity = await waitForPaintSequenceIncrease(page, zoomIdentity.paintSequence ?? 0);
  expect(panIdentity.paintSequence).toBeGreaterThan(zoomIdentity.paintSequence ?? 0);
  const panGeometry = await renderedRegionGeometry(page);
  const zoomCenter = geometryCentroid(zoomGeometry.points);
  const panCenter = geometryCentroid(panGeometry.points);
  expect(Math.hypot(panCenter.x - zoomCenter.x, panCenter.y - zoomCenter.y)).toBeGreaterThan(10);

  const viewport = page.viewportSize();
  expect(viewport).not.toBeNull();
  if (!viewport) throw new Error("live page has no configured viewport");
  const resizedViewport = {
    width: Math.max(320, viewport.width - 32),
    height: Math.max(400, viewport.height - 32),
  };
  const beforeResize = await readCameraState(page);
  await page.setViewportSize(resizedViewport);
  await expect.poll(
    async () => {
      const camera = await readCameraState(page);
      return Math.abs(camera.width - beforeResize.width) + Math.abs(camera.height - beforeResize.height);
    },
    { timeout: 5_000, intervals: [50, 100, 250] },
  ).toBeGreaterThan(10);
  const resizeIdentity = await waitForPaintSequenceIncrease(page, panIdentity.paintSequence ?? 0);
  expect(resizeIdentity.paintSequence).toBeGreaterThan(panIdentity.paintSequence ?? 0);
  const resizeGeometry = await renderedRegionGeometry(page);
  expect(geometryMoved(panGeometry.points, resizeGeometry.points)).toBeTruthy();
  expect(resizeGeometry.bounds.width).not.toBe(panGeometry.bounds.width);
  expect(resizeGeometry.bounds.height).not.toBe(panGeometry.bounds.height);

  await page.setViewportSize(viewport);
  await expect(page.locator(".map-tools button")).toBeVisible();
  const beforeFit = await readRenderedFrameIdentity(page);
  await page.locator(".map-tools button").click();
  const fitIdentity = await waitForPaintSequenceIncrease(page, beforeFit.paintSequence ?? 0);
  expect(fitIdentity.paintSequence).toBeGreaterThan(beforeFit.paintSequence ?? 0);
  await expect.poll(
    async () => (await readCameraState(page)).zoom,
    { timeout: 5_000, intervals: [50, 100, 250] },
  ).toBeCloseTo(1, 5);
  const fittedCamera = await readCameraState(page);
  expect(Math.hypot(fittedCamera.pan.x, fittedCamera.pan.y)).toBeLessThanOrEqual(1);
  const fittedGeometry = await renderedRegionGeometry(page);
  assertPointSetsMatch(fittedGeometry.points, initialGeometry.points, 2.5);
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
    x: projection.offsetX + (point.x - projection.min_x) * projection.scale + projection.panX,
    y: projection.height - projection.offsetY - (point.y - projection.min_y) * projection.scale + projection.panY,
  };
}

async function canvasProjection(page: Page, frame: JsonObject): Promise<CanvasProjection> {
  const fallback = worldBounds(frame);
  if (!fallback) throw new Error("live frame has no usable map bounds");
  const projection = await page.locator(".canvas-area").evaluate((element) => {
    const canvas = element.querySelector("canvas");
    if (!canvas) throw new Error("canvas is missing from map area");
    const currentBoundsAttribute = element.getAttribute("data-visible-bounds");
    if (!currentBoundsAttribute) throw new Error("map has no visible-bounds contract");
    const raw = canvas.getAttribute("data-last-painted-visible-bounds");
    if (!raw) throw new Error("canvas has no last-painted projection acknowledgement");
    const parsed = JSON.parse(raw) as Partial<WorldBounds>;
    const renderedFrameId = element.getAttribute("data-rendered-frame-id");
    const renderedSimTime = element.getAttribute("data-rendered-sim-time-s");
    const renderedExecutionRevision = element.getAttribute("data-rendered-execution-revision");
    const renderedPredictionId = element.getAttribute("data-rendered-prediction-id");
    const renderedPredictionRevision = element.getAttribute("data-rendered-prediction-revision");
    const renderedTargetId = element.getAttribute("data-rendered-target-id");
    if (
      renderedFrameId !== canvas.getAttribute("data-last-painted-frame-id")
      || renderedSimTime !== canvas.getAttribute("data-last-painted-sim-time-s")
      || renderedExecutionRevision !== canvas.getAttribute("data-last-painted-execution-revision")
      || renderedPredictionId !== canvas.getAttribute("data-last-painted-prediction-id")
      || renderedPredictionRevision !== canvas.getAttribute("data-last-painted-prediction-revision")
      || renderedTargetId !== canvas.getAttribute("data-last-painted-target-id")
    ) {
      throw new Error("React rendered identity is not acknowledged by the last-painted canvas");
    }
    const rawPan = canvas.getAttribute("data-last-painted-camera-pan");
    const parsedPan = rawPan ? JSON.parse(rawPan) as Partial<WorldPoint> : {};
    const zoom = Number(canvas.getAttribute("data-last-painted-camera-zoom") ?? "NaN");
    const panX = Number(parsedPan.x ?? 0);
    const panY = Number(parsedPan.y ?? 0);
    const min_x = Number(parsed.min_x);
    const min_y = Number(parsed.min_y);
    const max_x = Number(parsed.max_x);
    const max_y = Number(parsed.max_y);
    const rect = canvas.getBoundingClientRect();
    const width = canvas.clientWidth || rect.width;
    const height = canvas.clientHeight || rect.height;
    const scale = Math.min(width / (max_x - min_x), height / (max_y - min_y)) * zoom;
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
      panX: Number.isFinite(panX) ? panX : 0,
      panY: Number.isFinite(panY) ? panY : 0,
    };
  });
  expect(projection.width).toBeGreaterThan(0);
  expect(projection.height).toBeGreaterThan(0);
  expect(projection.scale).toBeGreaterThan(0);
  return projection as CanvasProjection;
}

function parseSvgPoints(value: string | null): WorldPoint[] {
  if (!value?.trim()) return [];
  return value.trim().split(/\s+/).flatMap((token) => {
    const [rawX, rawY] = token.split(",");
    const x = Number(rawX);
    const y = Number(rawY);
    return Number.isFinite(x) && Number.isFinite(y) ? [{ x, y }] : [];
  });
}

function assertPointSetsMatch(
  actual: WorldPoint[],
  expected: WorldPoint[],
  tolerancePx = 1.5,
): void {
  expect(actual).toHaveLength(expected.length);
  actual.forEach((point, index) => {
    const expectedPoint = expected[index];
    expect(Math.hypot(point.x - expectedPoint.x, point.y - expectedPoint.y)).toBeLessThanOrEqual(tolerancePx);
  });
}

interface RenderedGeometrySample {
  points: WorldPoint[];
  bounds: ScreenRect;
}

async function renderedRegionGeometry(page: Page): Promise<RenderedGeometrySample> {
  const polygon = page.locator(".region-map-overlay polygon").first();
  await expect(polygon).toHaveCount(1);
  const points = parseSvgPoints(await polygon.getAttribute("points"));
  const bounds = await polygon.evaluate((element) => {
    const box = element.getBoundingClientRect();
    return {
      left: box.left,
      top: box.top,
      right: box.right,
      bottom: box.bottom,
      width: box.width,
      height: box.height,
    };
  });
  expect(points.length).toBeGreaterThanOrEqual(3);
  expect(bounds.width).toBeGreaterThan(0);
  expect(bounds.height).toBeGreaterThan(0);
  return { points, bounds };
}

function polygonArea(points: WorldPoint[]): number {
  return Math.abs(points.reduce((area, point, index) => {
    const next = points[(index + 1) % points.length];
    return area + point.x * next.y - next.x * point.y;
  }, 0) / 2);
}

function geometryCentroid(points: WorldPoint[]): WorldPoint {
  const total = points.reduce(
    (sum, point) => ({ x: sum.x + point.x, y: sum.y + point.y }),
    { x: 0, y: 0 },
  );
  return { x: total.x / points.length, y: total.y / points.length };
}

function geometryMoved(left: WorldPoint[], right: WorldPoint[], minimumDistance = 2): boolean {
  if (left.length !== right.length) return true;
  return left.some((point, index) => {
    const other = right[index];
    return Math.hypot(point.x - other.x, point.y - other.y) >= minimumDistance;
  });
}

function corridorPolygonPoints(centerline: WorldPoint[], radii: number[]): WorldPoint[] {
  if (centerline.length === 0) return [];
  const right: WorldPoint[] = [];
  const left: WorldPoint[] = [];
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

type CanvasColor = "red" | "amber" | "cyan";

async function sampleCanvasColorNear(
  page: Page,
  points: WorldPoint[],
  color: CanvasColor,
  searchRadius = 1,
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
      for (let dy = -input.searchRadius; dy <= input.searchRadius && !pointMatched; dy += 1) {
        for (let dx = -input.searchRadius; dx <= input.searchRadius && !pointMatched; dx += 1) {
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
  }, { points, color, searchRadius });
}

async function sampleCanvasRingProfile(
  page: Page,
  center: WorldPoint,
  radius: number,
  color: CanvasColor,
  sampleCount = 144,
): Promise<{ matched: number; total: number; transitions: number; longestGap: number }> {
  return page.locator("canvas").first().evaluate((element, input) => {
    const canvas = element as HTMLCanvasElement;
    const context = canvas.getContext("2d");
    const rect = canvas.getBoundingClientRect();
    if (!context || !rect.width || !rect.height || input.radius <= 0 || input.sampleCount < 8) {
      return { matched: 0, total: input.sampleCount, transitions: 0, longestGap: input.sampleCount };
    }
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    const matches = (r: number, g: number, b: number): boolean => {
      if (input.color === "red") return r > 100 && r > g + 25 && r > b + 15;
      if (input.color === "amber") return r > 105 && r > g + 8 && r > b + 20;
      return g > 105 && b > 110 && g > r + 35 && b > r + 30 && g + b > 230;
    };
    const matchedSamples: boolean[] = [];
    for (let index = 0; index < input.sampleCount; index += 1) {
      const angle = 2 * Math.PI * index / input.sampleCount;
      const pointX = input.center.x + Math.cos(angle) * input.radius;
      const pointY = input.center.y + Math.sin(angle) * input.radius;
      const pixelX = Math.round(pointX * canvas.width / rect.width);
      const pixelY = Math.round(pointY * canvas.height / rect.height);
      let matched = false;
      for (let dy = -1; dy <= 1 && !matched; dy += 1) {
        for (let dx = -1; dx <= 1 && !matched; dx += 1) {
          const x = pixelX + dx;
          const y = pixelY + dy;
          if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) continue;
          const offset = (y * canvas.width + x) * 4;
          matched = matches(pixels[offset] ?? 0, pixels[offset + 1] ?? 0, pixels[offset + 2] ?? 0);
        }
      }
      matchedSamples.push(matched);
    }
    const matched = matchedSamples.filter(Boolean).length;
    let transitions = 0;
    let longestGap = 0;
    let currentGap = 0;
    for (let index = 0; index < matchedSamples.length * 2; index += 1) {
      const value = matchedSamples[index % matchedSamples.length];
      if (value) {
        currentGap = 0;
      } else {
        currentGap += 1;
        longestGap = Math.max(longestGap, currentGap);
      }
      if (index < matchedSamples.length) {
        const next = matchedSamples[(index + 1) % matchedSamples.length];
        if (value !== next) transitions += 1;
      }
    }
    return { matched, total: matchedSamples.length, transitions, longestGap };
  }, { center, radius, color, sampleCount });
}

async function measureLocalColorRing(
  page: Page,
  center: WorldPoint,
  expectedRadius: number,
  color: CanvasColor,
): Promise<{ pixelCount: number; minRadius: number; maxRadius: number }> {
  return page.locator("canvas").first().evaluate((element, input) => {
    const canvas = element as HTMLCanvasElement;
    const context = canvas.getContext("2d");
    const rect = canvas.getBoundingClientRect();
    if (!context || !rect.width || !rect.height || input.radius <= 0) {
      return { pixelCount: 0, minRadius: 0, maxRadius: 0 };
    }
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    const matches = (r: number, g: number, b: number): boolean => {
      if (input.color === "red") return r > 100 && r > g + 25 && r > b + 15;
      if (input.color === "amber") return r > 105 && r > g + 8 && r > b + 20;
      return g > 105 && b > 110 && g > r + 35 && b > r + 30 && g + b > 230;
    };
    const minX = Math.max(0, Math.floor((input.center.x - input.radius * 1.35) * canvas.width / rect.width));
    const maxX = Math.min(canvas.width - 1, Math.ceil((input.center.x + input.radius * 1.35) * canvas.width / rect.width));
    const minY = Math.max(0, Math.floor((input.center.y - input.radius * 1.35) * canvas.height / rect.height));
    const maxY = Math.min(canvas.height - 1, Math.ceil((input.center.y + input.radius * 1.35) * canvas.height / rect.height));
    let pixelCount = 0;
    let minRadius = Number.POSITIVE_INFINITY;
    let maxRadius = 0;
    for (let y = minY; y <= maxY; y += 1) {
      for (let x = minX; x <= maxX; x += 1) {
        const offset = (y * canvas.width + x) * 4;
        if (!matches(pixels[offset] ?? 0, pixels[offset + 1] ?? 0, pixels[offset + 2] ?? 0)) continue;
        const localX = (x + 0.5) * rect.width / canvas.width;
        const localY = (y + 0.5) * rect.height / canvas.height;
        const radius = Math.hypot(localX - input.center.x, localY - input.center.y);
        if (radius < input.radius * 0.82 || radius > input.radius * 1.18) continue;
        pixelCount += 1;
        minRadius = Math.min(minRadius, radius);
        maxRadius = Math.max(maxRadius, radius);
      }
    }
    return {
      pixelCount,
      minRadius: Number.isFinite(minRadius) ? minRadius : 0,
      maxRadius,
    };
  }, { center, radius: expectedRadius, color });
}

function visiblePoints(points: WorldPoint[], projection: CanvasProjection): WorldPoint[] {
  return points.filter((point) =>
    point.x >= 0 && point.x <= projection.width && point.y >= 0 && point.y <= projection.height,
  );
}

function targetEstimate(frame: JsonObject): JsonObject {
  const estimates = Array.isArray(frame.target_estimates) ? frame.target_estimates : [];
  const targetId = executionObject(frame).target_id;
  if (typeof targetId !== "string" || !targetId) {
    throw new Error("live frame has no execution target id");
  }
  const matchingTargets = estimates.filter((estimate) =>
    estimate && typeof estimate === "object" &&
    (estimate as JsonObject).target_id === targetId,
  ) as JsonObject[];
  if (matchingTargets.length !== 1) {
    throw new Error(`live frame must contain one execution target estimate for ${targetId}`);
  }
  return matchingTargets[0];
}

function currentTaskUuvs(frame: JsonObject): JsonObject[] {
  const group = currentTaskGroup(frame);
  const memberList = group.member_uuv_ids;
  if (!Array.isArray(memberList) || memberList.length !== 2 || memberList.some((id) => typeof id !== "string")) {
    throw new Error("current execution group must contain exactly two UUV members");
  }
  const memberIds = new Set(memberList as string[]);
  if (memberIds.size !== 2) throw new Error("current execution group contains duplicate UUV members");
  const activeId = group.active_verifier_uuv_id;
  const passiveId = group.passive_tracker_uuv_id;
  if (
    typeof activeId !== "string"
    || typeof passiveId !== "string"
    || activeId === passiveId
    || new Set([activeId, passiveId]).size !== 2
    || !memberIds.has(activeId)
    || !memberIds.has(passiveId)
  ) {
    throw new Error("current execution group roles must bind its two UUV members");
  }
  const uuvs = Array.isArray(frame.uuvs) ? frame.uuvs : [];
  const selected = uuvs.filter((uuv) =>
    uuv && typeof uuv === "object" &&
    memberIds.has(String((uuv as JsonObject).uuv_id)),
  ) as JsonObject[];
  const selectedIds = new Set(selected.map((uuv) => uuv.uuv_id));
  if (selected.length !== memberIds.size || selectedIds.size !== memberIds.size) {
    throw new Error("current execution group must bind exactly two distinct frame UUVs");
  }
  for (const memberId of memberIds) {
    if (!selectedIds.has(memberId)) {
      throw new Error(`current execution group member ${memberId} is missing from the frame`);
    }
  }
  const targetId = executionObject(frame).target_id;
  const taskGroupId = group.task_group_id;
  if (typeof taskGroupId !== "string" || !taskGroupId) {
    throw new Error("current execution group has no task_group_id");
  }
  for (const uuv of selected) {
    const expectedMode = uuv.uuv_id === activeId ? "active" : "passive";
    expect(uuv.physically_exposed).toBe(true);
    expect(uuv.sensor_mode).toBe(expectedMode);
    expect(uuv.group_id).toBe(taskGroupId);
    const trackedTarget = typeof uuv.tracked_target_id === "string"
      ? uuv.tracked_target_id
      : uuv.tracked_target;
    expect(trackedTarget).toBe(targetId);
  }
  return selected;
}

function currentTaskGroup(frame: JsonObject): JsonObject {
  const execution = executionObject(frame);
  const groups = Array.isArray(execution.task_groups) ? execution.task_groups : [];
  const group = groups.find((item) =>
    item && typeof item === "object" && (item as JsonObject).region_id === execution.current_region_id,
  );
  if (!group || typeof group !== "object") {
    throw new Error("live frame has no current execution task group");
  }
  const current = group as JsonObject;
  if (current.target_id !== execution.target_id) {
    throw new Error("current execution task group targets a different target");
  }
  if (typeof current.task_group_id !== "string" || !current.task_group_id) {
    throw new Error("current execution task group has no task_group_id");
  }
  return current;
}

function sensorRange(uuv: JsonObject): number {
  const mode = uuv.sensor_mode;
  if (mode !== "active" && mode !== "passive") {
    throw new Error(`UUV ${String(uuv.uuv_id)} has an invalid sensor mode`);
  }
  const field = mode === "active" ? "active_range_m" : "passive_range_m";
  const range = finiteNumber(uuv[field]);
  if (range === null || range <= 0) {
    throw new Error(`UUV ${String(uuv.uuv_id)} has no positive ${field}`);
  }
  return range;
}

async function measureLocalColorArc(
  page: Page,
  center: WorldPoint,
  expectedRadius: number,
  centerAngle: number,
  spanAngle: number,
  color: CanvasColor,
): Promise<{ pixelCount: number; minRadius: number; maxRadius: number }> {
  return page.locator("canvas").first().evaluate((element, input) => {
    const canvas = element as HTMLCanvasElement;
    const context = canvas.getContext("2d");
    const rect = canvas.getBoundingClientRect();
    if (!context || !rect.width || !rect.height || input.radius <= 0) {
      return { pixelCount: 0, minRadius: 0, maxRadius: 0 };
    }
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    const matches = (r: number, g: number, b: number): boolean => {
      if (input.color === "red") return r > 100 && r > g + 25 && r > b + 15;
      if (input.color === "amber") return r > 105 && r > g + 8 && r > b + 20;
      return g > 105 && b > 110 && g > r + 35 && b > r + 30 && g + b > 230;
    };
    const minX = Math.max(0, Math.floor((input.center.x - input.radius * 1.22) * canvas.width / rect.width));
    const maxX = Math.min(canvas.width - 1, Math.ceil((input.center.x + input.radius * 1.22) * canvas.width / rect.width));
    const minY = Math.max(0, Math.floor((input.center.y - input.radius * 1.22) * canvas.height / rect.height));
    const maxY = Math.min(canvas.height - 1, Math.ceil((input.center.y + input.radius * 1.22) * canvas.height / rect.height));
    let pixelCount = 0;
    let minRadius = Number.POSITIVE_INFINITY;
    let maxRadius = 0;
    for (let y = minY; y <= maxY; y += 1) {
      for (let x = minX; x <= maxX; x += 1) {
        const offset = (y * canvas.width + x) * 4;
        if (!matches(pixels[offset] ?? 0, pixels[offset + 1] ?? 0, pixels[offset + 2] ?? 0)) continue;
        const localX = (x + 0.5) * rect.width / canvas.width;
        const localY = (y + 0.5) * rect.height / canvas.height;
        const radius = Math.hypot(localX - input.center.x, localY - input.center.y);
        if (radius < input.radius * 0.84 || radius > input.radius * 1.16) continue;
        const angle = Math.atan2(localY - input.center.y, localX - input.center.x);
        const delta = Math.atan2(
          Math.sin(angle - input.centerAngle),
          Math.cos(angle - input.centerAngle),
        );
        if (Math.abs(delta) > input.spanAngle / 2 + 0.08) continue;
        pixelCount += 1;
        minRadius = Math.min(minRadius, radius);
        maxRadius = Math.max(maxRadius, radius);
      }
    }
    return {
      pixelCount,
      minRadius: Number.isFinite(minRadius) ? minRadius : 0,
      maxRadius,
    };
  }, { center, radius: expectedRadius, centerAngle, spanAngle, color });
}

function paintedPoint(value: unknown, label: string): WorldPoint {
  const point = worldPoint(value);
  if (!point) throw new Error(`${label} is not a finite screen point`);
  return point;
}

function paintedNumber(value: unknown, label: string): number {
  const number = finiteNumber(value);
  if (number === null) throw new Error(`${label} is not a finite number`);
  return number;
}

function assertPaintedPointMatches(
  actual: WorldPoint,
  expected: WorldPoint,
  label: string,
  tolerance = 0.75,
): void {
  const distance = Math.hypot(actual.x - expected.x, actual.y - expected.y);
  expect(distance).toBeLessThanOrEqual(tolerance);
  if (distance > tolerance) throw new Error(`${label} does not match backend geometry`);
}

function paintedArcPoints(
  layer: PaintedSonarLayer,
  projection: CanvasProjection,
  radiusFactor = 1,
): WorldPoint[] {
  const center = paintedPoint(layer.center, `${layer.uuv_id} painted sonar center`);
  const radius = paintedNumber(layer.radius_px, `${layer.uuv_id} painted sonar radius`) * radiusFactor;
  const start = paintedNumber(layer.start_angle_rad, `${layer.uuv_id} painted sonar start angle`);
  const end = paintedNumber(layer.end_angle_rad, `${layer.uuv_id} painted sonar end angle`);
  const points: WorldPoint[] = [];
  for (let index = 0; index <= 64; index += 1) {
    const angle = start + (end - start) * index / 64;
    points.push({
      x: center.x + Math.cos(angle) * radius,
      y: center.y + Math.sin(angle) * radius,
    });
  }
  return visiblePoints(points, projection);
}

async function assertPaintedVisualLayerContract(
  page: Page,
  frame: JsonObject,
  projection: CanvasProjection,
): Promise<PaintedVisualLayers> {
  const layers = await readPaintedVisualLayers(page);
  const execution = executionObject(frame);
  const target = targetEstimate(frame);
  const targetId = String(execution.target_id);
  expect(layers.detection).toHaveLength(1);
  const detection = layers.detection[0];
  if (!detection) throw new Error("painted detection layer is missing");
  expect(detection.target_id).toBe(targetId);
  expect(detection.stroke_style).toBe("#ff7882");
  expect(detection.line_dash).toEqual([4, 7]);
  const targetCenter = worldPoint(target.mean);
  if (!targetCenter) throw new Error("execution target has no finite mean");
  assertPaintedPointMatches(
    paintedPoint(detection.center, "painted detection center"),
    projectWorld(targetCenter, projection),
    "painted detection center",
  );
  const detectionRadius = paintedNumber(detection.radius_px, "painted detection radius");
  expect(detectionRadius).toBeGreaterThan(0);
  expect(Math.abs(detectionRadius - detectionRange(frame, target) * projection.scale)).toBeLessThanOrEqual(0.75);

  const group = currentTaskGroup(frame);
  const taskUuvs = currentTaskUuvs(frame);
  const memberIds = group.member_uuv_ids as string[];
  expect(layers.sonar).toHaveLength(memberIds.length);
  const renderedIds = layers.sonar.map((layer) => layer.uuv_id);
  expect(new Set(renderedIds).size).toBe(renderedIds.length);
  expect(new Set(renderedIds)).toEqual(new Set(memberIds));
  const taskGroupId = String(group.task_group_id);
  const targetUuvs = new Map(taskUuvs.map((uuv) => [String(uuv.uuv_id), uuv]));
  for (const mode of ["active", "passive"] as const) {
    const role = mode === "active" ? "active_verifier" : "passive_tracker";
    const roleKey = mode === "active" ? "active_verifier_uuv_id" : "passive_tracker_uuv_id";
    const requiredId = String(group[roleKey]);
    const uuv = targetUuvs.get(requiredId);
    if (!uuv) throw new Error(`${mode} UUV ${requiredId} is missing from the backend frame`);
    const layer = layers.sonar.find((candidate) => candidate.uuv_id === requiredId);
    if (!layer) throw new Error(`${mode} UUV ${requiredId} has no painted sonar layer`);
    expect(layer.target_id).toBe(targetId);
    expect(layer.task_group_id).toBe(taskGroupId);
    expect(layer.role).toBe(role);
    expect(layer.sensor_mode).toBe(mode);
    expect(layer.stroke_style).toBe(
      mode === "active" ? "rgba(247, 189, 69, 0.88)" : "rgba(33, 208, 195, 0.82)",
    );
    expect(layer.fill_style).toBe(
      mode === "active" ? "rgba(247, 189, 69, 0.14)" : "rgba(33, 208, 195, 0.11)",
    );
    const position = worldPoint(uuv.position);
    if (!position) throw new Error(`${mode} UUV ${requiredId} has no finite position`);
    assertPaintedPointMatches(
      paintedPoint(layer.center, `${mode} painted sonar center`),
      projectWorld(position, projection),
      `${mode} painted sonar center`,
    );
    const radius = paintedNumber(layer.radius_px, `${mode} painted sonar radius`);
    expect(radius).toBeGreaterThan(0);
    expect(Math.abs(radius - sensorRange(uuv) * projection.scale)).toBeLessThanOrEqual(0.75);
    const heading = finiteNumber(uuv.sensor_heading_rad) ?? finiteNumber(uuv.heading_rad) ?? 0;
    const centerAngle = heading === 0 ? 0 : -heading;
    expect(Math.abs(
      paintedNumber(layer.start_angle_rad, `${mode} painted sonar start angle`)
      - (centerAngle - Math.PI / 4),
    )).toBeLessThanOrEqual(0.001);
    expect(Math.abs(
      paintedNumber(layer.end_angle_rad, `${mode} painted sonar end angle`)
      - (centerAngle + Math.PI / 4),
    )).toBeLessThanOrEqual(0.001);
  }
  return layers;
}

async function assertDetectionGeometry(
  page: Page,
  frame: JsonObject,
  projection: CanvasProjection,
  layers: PaintedVisualLayers,
): Promise<void> {
  const target = targetEstimate(frame);
  const detection = layers.detection.find((layer) => layer.target_id === String(executionObject(frame).target_id));
  if (!detection) throw new Error("painted detection layer is not bound to the execution target");
  const screenCenter = paintedPoint(detection.center, "painted detection center");
  const radiusPx = paintedNumber(detection.radius_px, "painted detection radius");
  const center = worldPoint(target.mean);
  if (!center) throw new Error("execution target has no finite mean");
  assertPaintedPointMatches(screenCenter, projectWorld(center, projection), "painted detection center");
  expect(Math.abs(radiusPx - detectionRange(frame, target) * projection.scale)).toBeLessThanOrEqual(0.75);
  expect(screenCenter.x).toBeGreaterThanOrEqual(0);
  expect(screenCenter.x).toBeLessThanOrEqual(projection.width);
  expect(screenCenter.y).toBeGreaterThanOrEqual(0);
  expect(screenCenter.y).toBeLessThanOrEqual(projection.height);
  expect(radiusPx * 2).toBeGreaterThanOrEqual(24);
  const perimeter = visiblePoints(
    Array.from({ length: 144 }, (_, index) => {
      const angle = 2 * Math.PI * index / 144;
      return {
        x: screenCenter.x + Math.cos(angle) * radiusPx,
        y: screenCenter.y + Math.sin(angle) * radiusPx,
      };
    }),
    projection,
  );
  expect(perimeter.length).toBeGreaterThan(32);
  const samples = await sampleCanvasColorNear(page, perimeter, "red");
  const ringProfile = await sampleCanvasRingProfile(page, screenCenter, radiusPx, "red");
  const ringPoints = (factor: number): WorldPoint[] => visiblePoints(
    Array.from({ length: 72 }, (_, index) => {
      const angle = 2 * Math.PI * index / 72;
      return {
        x: screenCenter.x + Math.cos(angle) * radiusPx * factor,
        y: screenCenter.y + Math.sin(angle) * radiusPx * factor,
      };
    }),
    projection,
  );
  const innerRing = await sampleCanvasColorNear(page, ringPoints(0.72), "red");
  const outerRing = await sampleCanvasColorNear(page, ringPoints(1.28), "red");
  const ringGeometry = await measureLocalColorRing(page, screenCenter, radiusPx, "red");
  expect(samples.matched).toBeGreaterThan(20);
  expect(samples.matched / samples.total).toBeGreaterThan(0.2);
  expect(samples.matched).toBeGreaterThan(innerRing.matched + outerRing.matched);
  expect(ringProfile.matched).toBeGreaterThan(20);
  expect(ringProfile.matched / ringProfile.total).toBeGreaterThan(0.2);
  expect(ringProfile.matched / ringProfile.total).toBeLessThan(0.9);
  expect(ringProfile.transitions).toBeGreaterThanOrEqual(4);
  expect(ringProfile.longestGap).toBeGreaterThanOrEqual(1);
  expect(ringGeometry.pixelCount).toBeGreaterThan(20);
  expect(ringGeometry.minRadius).toBeGreaterThan(radiusPx * 0.82);
  expect(ringGeometry.maxRadius).toBeLessThan(radiusPx * 1.18);
}

async function assertSonarAttribution(
  page: Page,
  frame: JsonObject,
  projection: CanvasProjection,
  layers: PaintedVisualLayers,
): Promise<void> {
  const group = currentTaskGroup(frame);
  const taskUuvs = currentTaskUuvs(frame);
  const memberIds = Array.isArray(group.member_uuv_ids)
    ? group.member_uuv_ids.filter((id): id is string => typeof id === "string")
    : [];
  const renderedIds = ((await page.locator("canvas").first().getAttribute("data-current-task-uuv-ids")) ?? "")
    .split(",")
    .filter(Boolean);
  expect(new Set(renderedIds)).toEqual(new Set(memberIds));
  const telemetryText = await page.locator("canvas").first().getAttribute("data-current-task-uuv-telemetry");
  let renderedTelemetry: JsonObject[];
  try {
    const parsed: unknown = JSON.parse(telemetryText ?? "null");
    if (!Array.isArray(parsed) || parsed.some((item) => !item || typeof item !== "object")) {
      throw new Error("telemetry is not an object array");
    }
    renderedTelemetry = parsed as JsonObject[];
  } catch (error) {
    throw new Error(`rendered task UUV telemetry is invalid: ${String(error)}`);
  }
  expect(new Set(renderedTelemetry.map((item) => String(item.uuv_id)))).toEqual(new Set(memberIds));
  for (const mode of ["active", "passive"] as const) {
    const roleKey = mode === "active" ? "active_verifier_uuv_id" : "passive_tracker_uuv_id";
    const requiredId = group[roleKey];
    expect(typeof requiredId).toBe("string");
    if (typeof requiredId !== "string") throw new Error(`current group has no ${mode} UUV role`);
    const uuv = taskUuvs.find((item) => item.uuv_id === requiredId);
    expect(uuv).toBeDefined();
    if (!uuv) throw new Error(`${mode} UUV ${requiredId} is not rendered in the current task group`);
    expect(uuv.sensor_mode).toBe(mode);
    const telemetry = renderedTelemetry.find((item) => item.uuv_id === requiredId);
    expect(telemetry).toBeDefined();
    if (!telemetry) throw new Error(`${mode} UUV ${requiredId} has no rendered telemetry`);
    expect(telemetry.sensor_mode).toBe(mode);
    expect(telemetry.physically_exposed).toBe(true);
    expect(telemetry.task_group_id).toBe(group.task_group_id);
    expect(telemetry.role).toBe(mode === "active" ? "active_verifier" : "passive_tracker");
    expect(telemetry.tracked_target_id).toBe(executionObject(frame).target_id);
    const expectedPosition = worldPoint(uuv.position);
    const renderedPosition = worldPoint(telemetry.position);
    expect(expectedPosition).not.toBeNull();
    expect(renderedPosition).not.toBeNull();
    if (!expectedPosition || !renderedPosition) throw new Error(`${mode} UUV has no finite position`);
    expect(Math.hypot(
      expectedPosition.x - renderedPosition.x,
      expectedPosition.y - renderedPosition.y,
    )).toBeLessThanOrEqual(0.01);
    const color = mode === "active" ? "amber" : "cyan";
    const layer = layers.sonar.find((candidate) => candidate.uuv_id === requiredId);
    if (!layer) throw new Error(`${mode} UUV ${requiredId} has no painted sonar layer`);
    const boundaryPoints = paintedArcPoints(layer, projection, 1);
    const innerPoints = paintedArcPoints(layer, projection, 0.72);
    const outerPoints = paintedArcPoints(layer, projection, 1.3);
    expect(boundaryPoints.length).toBeGreaterThan(20);
    const boundary = await sampleCanvasColorNear(page, boundaryPoints, color);
    const inner = await sampleCanvasColorNear(page, innerPoints, color);
    const outer = await sampleCanvasColorNear(page, outerPoints, color);
    const screenCenter = paintedPoint(layer.center, `${mode} painted sonar center`);
    const radiusPx = paintedNumber(layer.radius_px, `${mode} painted sonar radius`);
    const startAngle = paintedNumber(layer.start_angle_rad, `${mode} painted sonar start angle`);
    const endAngle = paintedNumber(layer.end_angle_rad, `${mode} painted sonar end angle`);
    const arcGeometry = await measureLocalColorArc(
      page,
      screenCenter,
      radiusPx,
      (startAngle + endAngle) / 2,
      endAngle - startAngle,
      color,
    );
    expect(boundary.total).toBeGreaterThan(20);
    expect(boundary.matched).toBeGreaterThan(8);
    expect(boundary.matched / boundary.total).toBeGreaterThan(0.2);
    expect(boundary.matched).toBeGreaterThan(outer.matched + 2);
    expect(arcGeometry.pixelCount).toBeGreaterThan(10);
    expect(arcGeometry.minRadius).toBeGreaterThan(radiusPx * 0.84);
    expect(arcGeometry.maxRadius).toBeLessThan(radiusPx * 1.16);
    expect(inner.matched).toBeGreaterThan(0);
  }
}

function targetPredictions(frame: JsonObject): JsonObject[] {
  const execution = executionObject(frame);
  const targetId = execution.target_id;
  if (typeof targetId !== "string" || !targetId) {
    throw new Error("live frame has no execution target id");
  }
  const estimates = frame.target_estimates;
  if (!Array.isArray(estimates)) return [];
  const targetEstimates = estimates.filter((estimate) =>
    estimate && typeof estimate === "object" && (estimate as JsonObject).target_id === targetId,
  ) as JsonObject[];
  if (targetEstimates.length !== 1) {
    throw new Error(`live frame must contain one estimate for execution target ${targetId}`);
  }
  const prediction = targetEstimates[0].prediction;
  if (!prediction || typeof prediction !== "object") {
    throw new Error(`execution target ${targetId} has no prediction payload`);
  }
  const predictionObject = prediction as JsonObject;
  const executionPredictionId = execution.prediction_id;
  const executionPredictionRevision = finiteNumber(execution.prediction_revision);
  if (typeof executionPredictionId !== "string" || !executionPredictionId) {
    throw new Error("execution prediction_id is missing");
  }
  if (executionPredictionId !== predictionObject.prediction_id) {
    throw new Error("execution and target prediction IDs differ");
  }
  if (
    executionPredictionRevision === null
    || executionPredictionRevision !== finiteNumber(predictionObject.prediction_revision)
  ) {
    throw new Error("execution and target prediction revisions differ");
  }
  return [predictionObject];
}

function executionObject(frame: JsonObject): JsonObject {
  return frame.execution && typeof frame.execution === "object"
    ? frame.execution as JsonObject
    : {};
}

function expectedRegionOverlayState(status: string): string {
  if (status === "active" || status === "passive") return "active";
  if (status === "handoff_pending" || status === "handoff_completed") return "handoff";
  if (status === "degraded") return "degraded";
  if (status === "uncovered") return "uncovered";
  return "planned";
}

async function assertRegionGeometryAndStatus(
  page: Page,
  frame: JsonObject,
  projection: CanvasProjection,
): Promise<void> {
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
      predictionId: element.getAttribute("data-prediction-id"),
      executionRevision: Number(element.getAttribute("data-execution-revision")),
      state: element.getAttribute("data-region-state"),
      points: polygon?.getAttribute("points") ?? null,
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
  const rawRegions: unknown = executionObject(frame).regions;
  const regions = Array.isArray(rawRegions)
    ? (rawRegions as unknown[]).filter((region): region is JsonObject => Boolean(region && typeof region === "object"))
    : [];
  const canvasBounds = await page.locator("canvas").first().evaluate((item) => {
    const box = item.getBoundingClientRect();
    return { left: box.left, top: box.top, right: box.right, bottom: box.bottom };
  });
  expect(regions).toHaveLength(4);
  const expectedIds = regions.map((region) => String(region.region_id));
  const renderedIds = rendered.map((item) => String(item.regionId));
  expect(new Set(expectedIds).size).toBe(4);
  expect(new Set(renderedIds).size).toBe(4);
  expect(new Set(renderedIds)).toEqual(new Set(expectedIds));
  const expectedById = new Map(
    regions.map((region) => [String(region.region_id), region]),
  );
  for (const item of rendered) {
    const expected = expectedById.get(String(item.regionId));
    expect(expected).toBeDefined();
    if (!expected) throw new Error(`rendered unknown region ${String(item.regionId)}`);
    expect(item.area).toBeGreaterThan(20);
    expect(item.left).toBeGreaterThanOrEqual(canvasBounds.left - 1);
    expect(item.top).toBeGreaterThanOrEqual(canvasBounds.top - 1);
    expect(item.right).toBeLessThanOrEqual(canvasBounds.right + 1);
    expect(item.bottom).toBeLessThanOrEqual(canvasBounds.bottom + 1);
    expect(item.stroke).toBeTruthy();
    expect(item.fill).toBeTruthy();
    expect(item.label).toContain("R");
    expect(item.taskGroupId).toBe(String(expected.task_group_id));
    expect(item.predictionId).toBe(String(expected.prediction_id));
    expect(item.executionRevision).toBe(finiteNumber(executionObject(frame).execution_revision));
    expect(item.state).toBe(expectedRegionOverlayState(String(expected.status ?? "")));
    const geometry = Array.isArray(expected.geometry)
      ? expected.geometry.flatMap((point) => {
        const parsed = worldPoint(point);
        return parsed ? [parsed] : [];
      })
      : [];
    expect(geometry.length).toBeGreaterThanOrEqual(3);
    assertPointSetsMatch(
      parseSvgPoints(item.points),
      geometry.map((point) => projectWorld(point, projection)),
    );
    if (!item.current && !item.next) {
      const status = String(expected.status ?? "").toLowerCase();
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

async function assertPredictionRendering(
  page: Page,
  frame: JsonObject,
  projection: CanvasProjection,
): Promise<void> {
  const predictions = targetPredictions(frame);
  const prediction = predictions[0];
  const health = prediction?.health;
  expect(prediction).toBeDefined();
  if (!prediction || !health || typeof health !== "object") {
    throw new Error("live frame has no prediction health payload");
  }
  const status = String((health as JsonObject).status ?? "");
  const targetId = String(executionObject(frame).target_id ?? "");
  const overlay = page.locator(`.imm-prediction-overlay g[data-target-id="${targetId}"]`);
  await expect(overlay).toHaveCount(1);
  await expect(overlay).toHaveAttribute("data-prediction-id", String(prediction.prediction_id));
  await expect(overlay).toHaveAttribute(
    "data-prediction-revision",
    String(prediction.prediction_revision),
  );
  await expect(overlay).toHaveAttribute("data-health-status", status);
  if (status === "unavailable") {
    await expect(overlay).toHaveClass(/prediction-unavailable/);
    await expect(overlay.locator(".imm-confidence-band")).toHaveCount(0);
    await expect(overlay.locator(".imm-prediction-centerline")).toHaveCount(0);
    return;
  }
  expect(["valid", "degraded"]).toContain(status);
  expect(Array.isArray(prediction.centerline_xy)).toBeTruthy();
  if (!Array.isArray(prediction.centerline_xy) || prediction.centerline_xy.length < 2) {
    throw new Error("usable live prediction has no sampled centerline");
  }
  expect(prediction.centerline_xy.length).toBeGreaterThanOrEqual(2);
  const centerline = prediction.centerline_xy.flatMap((point) => {
    const parsed = worldPoint(point);
    return parsed ? [parsed] : [];
  });
  expect(centerline).toHaveLength(prediction.centerline_xy.length);
  const centerlineOverlay = overlay.locator(".imm-prediction-centerline");
  const corridorOverlay = overlay.locator(".imm-confidence-band");
  await expect(centerlineOverlay).toHaveCount(1);
  await expect(corridorOverlay).toHaveCount(1);
  const renderedCenterline = parseSvgPoints(await centerlineOverlay.getAttribute("points"));
  assertPointSetsMatch(renderedCenterline, centerline.map((point) => projectWorld(point, projection)));
  const radii = Array.isArray(prediction.radius_m)
    ? prediction.radius_m.flatMap((radius) => finiteNumber(radius) ?? [])
    : [];
  expect(radii).toHaveLength(centerline.length);
  const expectedCorridor = corridorPolygonPoints(centerline, radii)
    .map((point) => projectWorld(point, projection));
  const renderedCorridor = parseSvgPoints(await corridorOverlay.getAttribute("points"));
  assertPointSetsMatch(renderedCorridor, expectedCorridor);
  const centerlineLength = await centerlineOverlay.evaluate((item) => {
    const path = item as SVGGeometryElement;
    return typeof path.getTotalLength === "function" ? path.getTotalLength() : 0;
  });
  expect(centerlineLength).toBeGreaterThanOrEqual(16);
  const bandBounds = await corridorOverlay.evaluate((item) => {
    const box = (item as SVGGraphicsElement).getBoundingClientRect();
    return { left: box.left, top: box.top, right: box.right, bottom: box.bottom, width: box.width, height: box.height };
  });
  expect(bandBounds.width).toBeGreaterThanOrEqual(12);
  expect(bandBounds.height).toBeGreaterThanOrEqual(12);
  const canvasBounds = await page.locator("canvas").first().evaluate((item) => {
    const box = item.getBoundingClientRect();
    return { left: box.left, top: box.top, right: box.right, bottom: box.bottom };
  });
  expect(bandBounds.left).toBeGreaterThanOrEqual(canvasBounds.left - 1);
  expect(bandBounds.top).toBeGreaterThanOrEqual(canvasBounds.top - 1);
  expect(bandBounds.right).toBeLessThanOrEqual(canvasBounds.right + 1);
  expect(bandBounds.bottom).toBeLessThanOrEqual(canvasBounds.bottom + 1);
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

async function screenshotForBoundPaint(
  page: Page,
  frame: JsonObject,
  binding: RenderedFrameIdentity,
  screenshotPath: string,
): Promise<void> {
  const pendingPath = `${screenshotPath}.pending`;
  try {
    await assertPaintStillBound(page, frame, binding);
    await page.screenshot({ path: pendingPath, animations: "disabled" });
    const after = await readRenderedFrameIdentity(page);
    assertRenderedFrameBinding(frame, after);
    expect(after.paintSequence).toBe(binding.paintSequence);
    renameSync(pendingPath, screenshotPath);
  } catch (error) {
    try {
      unlinkSync(pendingPath);
    } catch {
      // There may be no temporary file when the precondition itself failed.
    }
    throw error;
  }
}

async function verifyAndCaptureCheckpoint(
  page: Page,
  checkpointFrame: JsonObject,
  screenshotPath: string,
): Promise<void> {
  let renderedFrame = await waitForRenderedFrame(page, checkpointFrame);
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      const binding = await readRenderedFrameIdentity(page);
      await assertPaintStillBound(page, renderedFrame, binding);
      const pixels = await canvasPixels(page);
      expect(pixels.variance).toBeGreaterThan(1);
      expect(pixels.nonBackground).toBeGreaterThan(100);
      await assertPaintStillBound(page, renderedFrame, binding);
      const projection = await canvasProjection(page, renderedFrame);
      const paintedLayers = await assertPaintedVisualLayerContract(page, renderedFrame, projection);
      await assertPaintStillBound(page, renderedFrame, binding);
      await assertDetectionGeometry(page, renderedFrame, projection, paintedLayers);
      await assertPaintStillBound(page, renderedFrame, binding);
      await assertSonarAttribution(page, renderedFrame, projection, paintedLayers);
      await assertPaintStillBound(page, renderedFrame, binding);
      await assertRegionGeometryAndStatus(page, renderedFrame, projection);
      await assertPaintStillBound(page, renderedFrame, binding);
      await assertPredictionRendering(page, renderedFrame, projection);
      await assertPaintStillBound(page, renderedFrame, binding);
      await assertViewportContainment(page);
      await screenshotForBoundPaint(page, renderedFrame, binding, screenshotPath);
      return;
    } catch (error) {
      if (attempt === 2) throw error;
      renderedFrame = await waitForRenderedFrame(page, checkpointFrame);
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
      let cameraContractChecked = false;
      for (const checkpointS of CHECKPOINTS_S) {
        activeCheckpoint = checkpointS;
        const checkpointFrame = await waitForCheckpoint(page, checkpointS);
        if (!cameraContractChecked) {
          await assertCameraInteractionContract(page, checkpointFrame);
          cameraContractChecked = true;
        }

        const screenshotDir = join(configuredAcceptanceDir(testInfo), "screenshots");
        mkdirSync(screenshotDir, { recursive: true });
        const viewportName = (testInfo.project.name === "mobile" || page.viewportSize()?.width === 390)
          ? "mobile"
          : "desktop";
        await verifyAndCaptureCheckpoint(
          page,
          checkpointFrame,
          join(screenshotDir, `${viewportName}-${checkpointS}.png`),
        );
      }
    } finally {
      appendConsoleRecords(consoleErrors, testInfo);
    }
    expect(consoleErrors).toEqual([]);
  });
});
