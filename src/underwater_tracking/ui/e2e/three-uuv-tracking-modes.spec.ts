import { expect, test, type Page, type TestInfo } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { join } from "node:path";

type JsonObject = Record<string, unknown>;

type Stage =
  | "reset"
  | "active_scan"
  | "passive_track"
  | "regional_handoff"
  | "dedicated_track"
  | "dedicated_steady"
  | "dedicated_restore_pending"
  | "regional_restore"
  | "regional_final"
  | "parallel_replacement";

interface StageExpectation {
  stage: Exclude<Stage, "reset">;
  mode: "regional" | "dedicated";
  groupCount: number;
  visibleUuvCount: number;
  activeScanGroups?: number;
  passiveTrackGroups?: number;
  enteringGroups?: number;
  exitingGroups?: number;
}

interface Rect {
  left: number;
  top: number;
  right: number;
  bottom: number;
  width: number;
  height: number;
}

interface CanvasPixels {
  width: number;
  height: number;
  variance: number;
  nonBackground: number;
}

const STAGES: StageExpectation[] = [
  {
    stage: "active_scan",
    mode: "regional",
    groupCount: 4,
    visibleUuvCount: 12,
    activeScanGroups: 4,
    passiveTrackGroups: 0,
    enteringGroups: 0,
    exitingGroups: 0,
  },
  {
    stage: "passive_track",
    mode: "regional",
    groupCount: 4,
    visibleUuvCount: 12,
    activeScanGroups: 3,
    passiveTrackGroups: 1,
    enteringGroups: 0,
    exitingGroups: 0,
  },
  {
    stage: "regional_handoff",
    mode: "regional",
    groupCount: 4,
    visibleUuvCount: 12,
    activeScanGroups: 2,
    passiveTrackGroups: 1,
    enteringGroups: 0,
    exitingGroups: 1,
  },
  {
    stage: "dedicated_track",
    mode: "dedicated",
    groupCount: 4,
    visibleUuvCount: 12,
    activeScanGroups: 0,
    passiveTrackGroups: 0,
    enteringGroups: 0,
    exitingGroups: 3,
  },
  {
    stage: "dedicated_steady",
    mode: "dedicated",
    groupCount: 1,
    visibleUuvCount: 3,
    activeScanGroups: 0,
    passiveTrackGroups: 0,
    enteringGroups: 0,
    exitingGroups: 0,
  },
  {
    stage: "dedicated_restore_pending",
    mode: "dedicated",
    groupCount: 5,
    visibleUuvCount: 15,
    activeScanGroups: 0,
    passiveTrackGroups: 1,
    enteringGroups: 3,
    exitingGroups: 0,
  },
  {
    stage: "regional_restore",
    mode: "regional",
    groupCount: 5,
    visibleUuvCount: 15,
    activeScanGroups: 0,
    passiveTrackGroups: 1,
    enteringGroups: 3,
    exitingGroups: 1,
  },
  {
    stage: "regional_final",
    mode: "regional",
    groupCount: 4,
    visibleUuvCount: 12,
    activeScanGroups: 0,
    passiveTrackGroups: 1,
    enteringGroups: 3,
    exitingGroups: 0,
  },
  {
    stage: "parallel_replacement",
    mode: "regional",
    groupCount: 8,
    visibleUuvCount: 24,
    activeScanGroups: 0,
    passiveTrackGroups: 1,
    enteringGroups: 3,
    exitingGroups: 4,
  },
];

const EXPECTED_EVENTS = [
  "task_group_entering",
  "active_scan_started",
  "passive_track_started",
  "tracking_ownership_transferred",
  "dedicated_tracking_started",
  "task_group_exiting",
  "task_group_disappeared",
  "dedicated_release_threshold_reached",
  "regional_mode_restored",
];

function asObject(value: unknown, label: string): JsonObject {
  expect(value, label).toBeTruthy();
  expect(typeof value, label).toBe("object");
  expect(Array.isArray(value), label).toBeFalsy();
  return value as JsonObject;
}

function asArray(value: unknown, label: string): unknown[] {
  expect(Array.isArray(value), label).toBeTruthy();
  return value as unknown[];
}

function strictOperationalFrame(value: unknown, requireExecution = true): JsonObject {
  const frame = asObject(value, "operational frame");
  expect(frame).toHaveProperty("schema_version");
  expect(frame.uuv_only).toBe(true);
  expect(frame).not.toHaveProperty("usvs");
  expect(frame.carrier ?? null).toBeNull();
  expect(frame.carriers ?? []).toEqual([]);
  expect(frame.carrier_missions ?? []).toEqual([]);
  expect(frame.planned_assignments ?? []).toEqual([]);

  if (frame.execution === null && !requireExecution) return frame;
  const execution = asObject(frame.execution, "execution snapshot");
  const regions = asArray(execution.regions, "execution regions");
  expect(regions).toHaveLength(4);
  const groups = asArray(execution.task_groups, "runtime task groups");
  expect(groups.length).toBeGreaterThanOrEqual(1);
  const groupIds = new Set<string>();
  const memberIds = new Set<string>();
  for (const rawGroup of groups) {
    const group = asObject(rawGroup, "runtime task group");
    expect(group).not.toHaveProperty("active_verifier_uuv_id");
    expect(group).not.toHaveProperty("passive_tracker_uuv_id");
    expect(typeof group.group_instance_id).toBe("string");
    expect(typeof group.lifecycle).toBe("string");
    expect(typeof group.sensor_mode).toBe("string");
    expect(groupIds.has(String(group.group_instance_id))).toBeFalsy();
    groupIds.add(String(group.group_instance_id));
    const members = asArray(group.member_uuv_ids, "task group members");
    expect(members).toHaveLength(3);
    for (const member of members) {
      expect(typeof member).toBe("string");
      expect(memberIds.has(String(member))).toBeFalsy();
      memberIds.add(String(member));
    }
  }

  const uuvs = asArray(frame.uuvs, "UUV entities");
  for (const rawUuv of uuvs) {
    const uuv = asObject(rawUuv, "UUV entity");
    expect(uuv).not.toHaveProperty("active_verifier_uuv_id");
    expect(uuv).not.toHaveProperty("passive_tracker_uuv_id");
    expect(typeof uuv.uuv_id).toBe("string");
    expect(typeof uuv.sensor_mode).toBe("string");
    if (uuv.physically_exposed === true) {
      expect(typeof uuv.group_instance_id).toBe("string");
      expect(typeof uuv.group_lifecycle).toBe("string");
    }
  }
  expect(frame).not.toHaveProperty("legacy_frame");
  return frame;
}

function latestFrameId(frames: JsonObject[]): number {
  const ids = frames.flatMap((frame) =>
    typeof frame.frame_id === "number" ? [frame.frame_id] : [],
  );
  return ids.length ? Math.max(...ids) : -1;
}

function eventSubsequence(eventTypes: string[]): boolean {
  let index = 0;
  for (const eventType of eventTypes) {
    if (eventType === EXPECTED_EVENTS[index]) index += 1;
    if (index === EXPECTED_EVENTS.length) return true;
  }
  return false;
}

async function canvasPixels(page: Page): Promise<CanvasPixels> {
  return page.locator("canvas").first().evaluate((element) => {
    const canvas = element as HTMLCanvasElement;
    const context = canvas.getContext("2d");
    if (!context || canvas.width === 0 || canvas.height === 0) {
      return { width: canvas.width, height: canvas.height, variance: 0, nonBackground: 0 };
    }
    const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
    const stride = Math.max(1, Math.floor(pixels.length / 4 / 120_000));
    let sum = 0;
    let sumSquared = 0;
    let nonBackground = 0;
    let samples = 0;
    for (let index = 0; index < pixels.length; index += 4 * stride) {
      const red = pixels[index] ?? 0;
      const green = pixels[index + 1] ?? 0;
      const blue = pixels[index + 2] ?? 0;
      const luminance = (red + green + blue) / 3;
      sum += luminance;
      sumSquared += luminance * luminance;
      if (Math.max(red, green, blue) - Math.min(red, green, blue) > 12) {
        nonBackground += 1;
      }
      samples += 1;
    }
    const mean = sum / Math.max(1, samples);
    return {
      width: canvas.width,
      height: canvas.height,
      variance: Math.max(0, sumSquared / Math.max(1, samples) - mean * mean),
      nonBackground,
    };
  });
}

async function rectFor(page: Page, selector: string): Promise<Rect | null> {
  const locator = page.locator(selector).first();
  if (!(await locator.isVisible().catch(() => false))) return null;
  return locator.evaluate((element) => {
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
}

function contains(outer: Rect, inner: Rect, tolerance = 2): boolean {
  return inner.left >= outer.left - tolerance
    && inner.top >= outer.top - tolerance
    && inner.right <= outer.right + tolerance
    && inner.bottom <= outer.bottom + tolerance;
}

function overlaps(left: Rect, right: Rect): boolean {
  return left.left < right.right
    && left.right > right.left
    && left.top < right.bottom
    && left.bottom > right.top;
}

async function assertVisualLayout(page: Page): Promise<void> {
  const viewport = page.viewportSize();
  expect(viewport).toBeTruthy();
  const canvasArea = await rectFor(page, ".canvas-area");
  const canvas = await rectFor(page, "canvas");
  expect(canvasArea).toBeTruthy();
  expect(canvas).toBeTruthy();
  if (!viewport || !canvasArea || !canvas) return;
  expect(canvas.width).toBeGreaterThan(200);
  expect(canvas.height).toBeGreaterThan(200);

  const boundedElements = await page.locator(
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
    };
  }));
  for (const element of boundedElements) {
    if (element.width <= 0 || element.height <= 0) continue;
    expect(element.left).toBeGreaterThanOrEqual(-1);
    expect(element.top).toBeGreaterThanOrEqual(-1);
    expect(element.right).toBeLessThanOrEqual(viewport.width + 1);
    expect(element.bottom).toBeLessThanOrEqual(viewport.height + 1);
    expect(contains(canvasArea, element, 3)).toBeTruthy();
  }

  const tools = await rectFor(page, ".map-tools");
  expect(tools).toBeTruthy();
  if (tools) expect(contains(canvasArea, tools, 3)).toBeTruthy();
  const drawer = await rectFor(page, ".bottom-drawer");
  if (drawer) {
    expect(drawer.left).toBeGreaterThanOrEqual(-1);
    expect(drawer.right).toBeLessThanOrEqual(viewport.width + 1);
    expect(drawer.bottom).toBeLessThanOrEqual(viewport.height + 1);
    if (tools) expect(overlaps(drawer, tools)).toBeFalsy();
  }
  const sidebar = await rectFor(page, ".sidebar.open");
  if (sidebar) {
    expect(sidebar.left).toBeGreaterThanOrEqual(-1);
    expect(sidebar.right).toBeLessThanOrEqual(viewport.width + 1);
    expect(sidebar.bottom).toBeLessThanOrEqual(viewport.height + 1);
  }
}

async function waitForPaintedFrame(page: Page, frame: JsonObject): Promise<void> {
  const frameId = frame.frame_id;
  expect(typeof frameId).toBe("number");
  await expect.poll(
    async () => page.locator("canvas").first().getAttribute("data-last-painted-frame-id"),
    { timeout: 10_000, intervals: [100, 250, 500] },
  ).toBe(String(frameId));
  await expect.poll(async () => {
    const pixels = await canvasPixels(page);
    return pixels.variance > 1 && pixels.nonBackground > 20;
  }, { timeout: 10_000, intervals: [100, 250, 500] }).toBe(true);
}

async function assertPolicyAttributes(page: Page): Promise<void> {
  const canvas = page.locator("canvas").first();
  await expect(canvas).toHaveAttribute("data-region-count", "4");
  await expect(canvas).toHaveAttribute("data-task-group-size", "3");
  await expect(canvas).toHaveAttribute("data-region-side-m", "2000");
  await expect(canvas).toHaveAttribute("data-target-radius-m", "1000");
  await expect(canvas).toHaveAttribute("data-uuv-radius-m", "600");
}

async function postStage(page: Page, stage: Stage): Promise<JsonObject> {
  const response = await page.request.post("/api/verification/three-uuv-tracking-modes", {
    data: { stage },
    timeout: 30_000,
  });
  expect(response.ok(), `${stage} request failed`).toBeTruthy();
  const body = asObject(await response.json(), `${stage} response`);
  expect(body.stage).toBe(stage);
  return strictOperationalFrame(body.frame);
}

async function saveStageScreenshot(page: Page, testInfo: TestInfo, stage: string): Promise<void> {
  const directory = testInfo.outputPath("three-uuv-tracking-modes");
  mkdirSync(directory, { recursive: true });
  await page.screenshot({
    path: join(directory, `${testInfo.project.name}-${stage}.png`),
    animations: "disabled",
    fullPage: true,
  });
}

test.describe("three-UUV tracking modes live acceptance", () => {
  test("renders the strict runtime sequence over WebSocket and replay", async ({ page }, testInfo) => {
    const receivedFrames: JsonObject[] = [];
    const browserErrors: string[] = [];
    const eventTypes: string[] = [];
    const missionEventTypes: string[] = [];
    const seenMissionEventIds = new Set<string>();
    let websocketFrameCount = 0;
    page.on("pageerror", (error) => browserErrors.push(error.message));
    page.on("console", (message) => {
      if (message.type() === "error") browserErrors.push(message.text());
    });
    page.on("websocket", (socket) => {
      if (!socket.url().includes("/ws/operational")) return;
      socket.on("framereceived", ({ payload }) => {
        const raw = typeof payload === "string" ? payload : payload.toString();
        websocketFrameCount += 1;
        let value: unknown;
        try {
          value = JSON.parse(raw) as unknown;
        } catch {
          return;
        }
        if (!value || typeof value !== "object" || Array.isArray(value)) return;
        const frame = value as JsonObject;
        if (typeof frame.frame_id !== "number") return;
        try {
          receivedFrames.push(strictOperationalFrame(frame));
        } catch (error) {
          browserErrors.push(`WebSocket frame contract: ${String(error)}`);
          return;
        }
        for (const key of ["events", "mission_events"]) {
          for (const rawEvent of Array.isArray(frame[key]) ? frame[key] : []) {
            if (!rawEvent || typeof rawEvent !== "object") continue;
            const eventType = (rawEvent as JsonObject).event_type;
            if (typeof eventType === "string" && !eventTypes.includes(eventType)) {
              eventTypes.push(eventType);
            }
            if (
              key === "mission_events"
              && typeof eventType === "string"
            ) {
              const eventId = (rawEvent as JsonObject).event_id;
              if (typeof eventId === "string" && !seenMissionEventIds.has(eventId)) {
                seenMissionEventIds.add(eventId);
                missionEventTypes.push(eventType);
              }
            }
          }
        }
      });
    });

    strictOperationalFrame(await postStage(page, "reset"));
    await page.goto("/");
    const canvas = page.locator("canvas").first();
    await expect(canvas).toBeVisible();
    await expect(canvas).toHaveAttribute("data-region-count", "4", { timeout: 30_000 });
    await assertPolicyAttributes(page);

    let regionalOwner: string | null = null;
    let dedicatedOwner: string | null = null;
    let previousExecutionRevision: number | null = null;
    for (const expected of STAGES) {
      const frame = await postStage(page, expected.stage);
      const execution = asObject(frame.execution, `${expected.stage} execution`);
      const groups = asArray(execution.task_groups, `${expected.stage} groups`);
      expect(groups).toHaveLength(expected.groupCount);
      expect(execution.tracking_control).toMatchObject({ mode: expected.mode });

      await waitForPaintedFrame(page, frame);
      await assertPolicyAttributes(page);
      await expect(canvas).toHaveAttribute("data-tracking-mode", expected.mode);
      await expect(canvas).toHaveAttribute("data-visible-uuv-count", String(expected.visibleUuvCount));
      if (expected.activeScanGroups !== undefined) {
        await expect(canvas).toHaveAttribute("data-active-scan-group-count", String(expected.activeScanGroups));
      }
      if (expected.passiveTrackGroups !== undefined) {
        await expect(canvas).toHaveAttribute("data-passive-track-group-count", String(expected.passiveTrackGroups));
      }
      if (expected.enteringGroups !== undefined) {
        await expect(canvas).toHaveAttribute("data-entering-group-count", String(expected.enteringGroups));
      }
      if (expected.exitingGroups !== undefined) {
        await expect(canvas).toHaveAttribute("data-exiting-group-count", String(expected.exitingGroups));
      }
      await assertVisualLayout(page);
      await saveStageScreenshot(page, testInfo, expected.stage);

      const control = asObject(execution.tracking_control, `${expected.stage} tracking control`);
      const owner = typeof control.tracking_owner_group_id === "string"
        ? control.tracking_owner_group_id
        : null;
      if (expected.stage === "passive_track") regionalOwner = owner;
      if (expected.stage === "regional_handoff") {
        expect(owner).toBeTruthy();
        expect(owner).not.toBe(regionalOwner);
        regionalOwner = owner;
      }
      if (expected.stage === "dedicated_track") {
        expect(owner).toBeTruthy();
        dedicatedOwner = owner;
      }
      if (
        (expected.stage === "dedicated_steady"
          || expected.stage === "dedicated_restore_pending")
        && dedicatedOwner !== null
      ) {
        expect(owner).toBe(dedicatedOwner);
      }
      if (expected.stage === "dedicated_restore_pending") {
        expect(control.dedicated_release_triggered_at_m).toBe(5_000);
      }
      if (previousExecutionRevision !== null && expected.stage !== "parallel_replacement") {
        expect(Number(execution.execution_revision)).toBeGreaterThanOrEqual(previousExecutionRevision);
      }
      previousExecutionRevision = Number(execution.execution_revision);
    }

    await page.waitForTimeout(1_000);
    if (receivedFrames.length === 0) {
      throw new Error(
        `WebSocket frames=${websocketFrameCount}; contract errors=${browserErrors.join(" | ")}`,
      );
    }
    expect(latestFrameId(receivedFrames)).toBeGreaterThan(0);
    expect(
      eventSubsequence(missionEventTypes),
      missionEventTypes.join(" -> "),
    ).toBeTruthy();
    expect(eventTypes).toContain("region_replacement_started");
    expect(browserErrors).toEqual([]);

    const replayResponse = await page.request.get("/api/replay?start_s=0&limit=10000", {
      timeout: 30_000,
    });
    expect(replayResponse.ok()).toBeTruthy();
    const replayBody = asObject(await replayResponse.json(), "replay response");
    const replayFrames = asArray(replayBody.frames, "replay frames").map((frame) =>
      strictOperationalFrame(frame, false),
    );
    const replayExecutionFrames = replayFrames.filter((frame) => frame.execution !== null);
    expect(replayExecutionFrames.length).toBeGreaterThanOrEqual(STAGES.length);
    expect(replayExecutionFrames.some((frame) => frame.frame_id === latestFrameId(receivedFrames))).toBeTruthy();
  });
});
