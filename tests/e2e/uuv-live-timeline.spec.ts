import {
  expect,
  test,
  type APIRequestContext,
  type Page,
} from "../../src/underwater_tracking/ui/node_modules/@playwright/test";

type RuntimeEvent = {
  event_id: string;
  event_type: string;
  entity_id?: string | null;
  sim_time_s: number;
};

type OperationalFrame = {
  sim_time_s: number;
  uuvs: Array<{ uuv_id: string; deployment_state: string }>;
  events: RuntimeEvent[];
  [key: string]: unknown;
};

type ReplayPayload = { frames: OperationalFrame[] };

const realBaseURL = process.env.PLAYWRIGHT_BASE_URL;
const timelineTimeoutMs = 10 * 60 * 1000;

test.describe("live UUV initialization timeline", () => {
  test.skip(
    !realBaseURL,
    "set PLAYWRIGHT_BASE_URL to the real main.py command center",
  );
  test.setTimeout(timelineTimeoutMs);

  async function readSnapshot(request: APIRequestContext): Promise<OperationalFrame> {
    const response = await request.get("/api/operational/snapshot");
    expect(response.ok()).toBeTruthy();
    return (await response.json()) as OperationalFrame;
  }

  async function waitForEvent(
    request: APIRequestContext,
    eventType: string,
    entityId?: string,
  ): Promise<{ event: RuntimeEvent; replay: ReplayPayload }> {
    let latest: ReplayPayload = { frames: [] };
    let match: RuntimeEvent | undefined;
    await expect
      .poll(
        async () => {
          const response = await request.get("/api/replay?start_s=0&limit=10000");
          expect(response.ok()).toBeTruthy();
          latest = (await response.json()) as ReplayPayload;
          match = latest.frames
            .flatMap((frame) => frame.events ?? [])
            .find(
              (event) =>
                event.event_type === eventType &&
                (entityId === undefined || event.entity_id === entityId),
            );
          return Boolean(match);
        },
        { timeout: timelineTimeoutMs, intervals: [500, 1000, 2000, 5000] },
      )
      .toBe(true);
    if (!match) throw new Error(`event ${eventType} was not found`);
    return { event: match, replay: latest };
  }

  async function assertCanvasHasPixels(page: Page) {
    const canvas = page.locator("canvas").first();
    await expect(canvas).toBeVisible();
    await expect
      .poll(
        async () =>
          canvas.evaluate((element) => {
            const canvasElement = element as HTMLCanvasElement;
            const context = canvasElement.getContext("2d");
            if (!context || canvasElement.width === 0 || canvasElement.height === 0) return 0;
            const pixels = context.getImageData(0, 0, canvasElement.width, canvasElement.height).data;
            let nonTransparent = 0;
            for (let index = 3; index < pixels.length; index += 32) {
              if (pixels[index] > 0) nonTransparent += 1;
            }
            return nonTransparent;
          }),
        { timeout: 30_000, intervals: [250, 500, 1000] },
      )
      .toBeGreaterThan(10);
  }

  test("renders the real default initial state without surface-node fields", async ({
    page,
    request,
  }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto("/");
    await page.waitForLoadState("networkidle");
    await expect(page.locator('canvas[aria-label="水下跟踪态势地图，支持拖动、滚轮缩放、UUV 与区域选择"]')).toBeVisible();

    const initial = await readSnapshot(request);
    expect(initial.uuvs).toHaveLength(12);
    expect(initial.uuvs.every((uuv) => uuv.deployment_state === "onboard")).toBeTruthy();
    expect(initial).not.toHaveProperty("usvs");
    expect(JSON.stringify(initial).toLowerCase()).not.toContain("usv");
    await assertCanvasHasPixels(page);
    await expect(page.locator("body")).not.toContainText(/USV/i);
    await page.screenshot({ path: "test-results/uuv-live-default-initial-1440.png", fullPage: true });
  });

  test("polls the real deployment, handoff, recovery, and returned-to-fleet states", async ({
    page,
    request,
  }) => {
    const consoleErrors: string[] = [];
    page.on("console", (message) => {
      if (message.type() === "error") consoleErrors.push(message.text());
    });
    page.on("pageerror", (error) => consoleErrors.push(error.message));

    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto("/");
    await page.waitForLoadState("networkidle");
    await expect(page.locator('canvas[aria-label="水下跟踪态势地图，支持拖动、滚轮缩放、UUV 与区域选择"]')).toBeVisible();

    const initial = await readSnapshot(request);
    expect(initial.uuvs).toHaveLength(12);
    expect(initial.uuvs.every((uuv) => uuv.deployment_state === "onboard")).toBeTruthy();
    expect(initial).not.toHaveProperty("usvs");
    expect(JSON.stringify(initial).toLowerCase()).not.toContain("usv");
    await assertCanvasHasPixels(page);
    await page.screenshot({ path: "test-results/uuv-live-initial-1440.png", fullPage: true });

    const deployed = await waitForEvent(request, "uuv_deployed");
    expect(deployed.event.sim_time_s).toBeGreaterThan(0);
    const deployedFrame = deployed.replay.frames.find(
      (frame) => frame.sim_time_s >= deployed.event.sim_time_s,
    );
    expect(deployedFrame?.uuvs.some((uuv) => uuv.deployment_state === "deployed")).toBeTruthy();
    await assertCanvasHasPixels(page);
    await expect(page.locator("body")).not.toContainText(/USV/i);
    await page.screenshot({ path: "test-results/uuv-live-post-deployment-1440.png", fullPage: true });

    const handoff = await waitForEvent(request, "handoff_completed");
    expect(handoff.event.sim_time_s).toBeGreaterThan(deployed.event.sim_time_s);
    await page.screenshot({ path: "test-results/uuv-live-handoff-1440.png", fullPage: true });

    const returned = await waitForEvent(request, "carrier_returned_to_fleet", "carrier_02");
    expect(returned.event.sim_time_s).toBeGreaterThan(handoff.event.sim_time_s);
    const returnedCount = returned.replay.frames
      .flatMap((frame) => frame.events ?? [])
      .filter((event) => event.event_type === "carrier_returned_to_fleet" && event.entity_id === "carrier_02");
    expect(returnedCount).toHaveLength(1);
    await page.screenshot({ path: "test-results/uuv-live-returned-1440.png", fullPage: true });

    const snapshot = await readSnapshot(request);
    const scenarioId = String(snapshot.scenario_id ?? "uuv-only-single-target");
    const memoryStream = await request.get(
      `/api/assistant/memory/stream?user_id=operator&conversation_id=uuv-live-timeline&scenario_id=${encodeURIComponent(scenarioId)}&after_cursor=0&limit=100`,
    );
    expect(memoryStream.ok()).toBeTruthy();
    expect(Array.isArray(((await memoryStream.json()) as { events?: unknown[] }).events)).toBeTruthy();
    expect(JSON.stringify(returned.replay).toLowerCase()).not.toContain("usv");
    expect(consoleErrors).toEqual([]);
  });

  test("keeps the real deployment view usable on a narrow screen", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto("/");
    await page.waitForLoadState("networkidle");
    await assertCanvasHasPixels(page);
    await expect(page.locator("body")).not.toContainText(/USV/i);
    await page.screenshot({ path: "test-results/uuv-live-returned-390.png", fullPage: true });
  });
});
