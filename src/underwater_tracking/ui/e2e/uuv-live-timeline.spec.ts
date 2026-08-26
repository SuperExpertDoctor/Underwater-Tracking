import {
  expect,
  test,
  type APIRequestContext,
  type Page,
} from "@playwright/test";

type RuntimeEvent = {
  event_id: string;
  event_type: string;
  entity_id?: string | null;
  sim_time_s: number;
};

type UuvSnapshot = {
  uuv_id: string;
  deployment_state: string;
  physically_exposed?: boolean;
};

type OperationalFrame = {
  frame_id: number;
  sim_time_s: number;
  plan_version: number;
  scenario_id?: string | null;
  uuvs: UuvSnapshot[];
  target_estimates: Array<{ target_id: string }>;
  events: RuntimeEvent[];
  carriers?: unknown[];
  carrier?: unknown | null;
  groups?: Array<{ mode?: string }>;
  [key: string]: unknown;
};

type ReplayPayload = {
  frames: OperationalFrame[];
  count?: number;
  total_count?: number;
  offset?: number;
};

const realBaseURL = process.env.PLAYWRIGHT_BASE_URL;
const timelineTimeoutMs = 10 * 60 * 1000;
const canvasLabel = "canvas[aria-label=\"水下跟踪态势地图，支持拖动、滚轮缩放、UUV 与区域选择\"]";

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

  async function waitForSnapshot(
    request: APIRequestContext,
    predicate: (frame: OperationalFrame) => boolean,
  ): Promise<OperationalFrame> {
    let latest: OperationalFrame | null = null;
    await expect
      .poll(
        async () => {
          const snapshot = await readSnapshot(request);
          latest = snapshot;
          return predicate(snapshot);
        },
        { timeout: timelineTimeoutMs, intervals: [500, 1000, 2000, 5000] },
      )
      .toBe(true);
    if (!latest) throw new Error("operational snapshot was not published");
    return latest;
  }

  async function waitForEvent(
    request: APIRequestContext,
    eventType: string,
    afterSimTime = -Infinity,
    entityId?: string,
  ): Promise<{ event: RuntimeEvent; replay: ReplayPayload }> {
    let latest: ReplayPayload = { frames: [] };
    let match: RuntimeEvent | undefined;
    await expect
      .poll(
        async () => {
          let offset = 0;
          latest = { frames: [] };
          match = undefined;
          for (let pageIndex = 0; pageIndex < 64; pageIndex += 1) {
            const response = await request.get(
              `/api/replay?start_s=0&offset=${offset}&limit=250`,
            );
            expect(response.ok()).toBeTruthy();
            const page = (await response.json()) as ReplayPayload;
            latest = {
              ...page,
              frames: [...latest.frames, ...(page.frames ?? [])],
            };
            match = latest.frames
              .flatMap((frame) => frame.events ?? [])
              .find(
                (event) =>
                  event.event_type === eventType &&
                  event.sim_time_s >= afterSimTime &&
                  (entityId === undefined || event.entity_id === entityId),
              );
            const pageCount = page.count ?? page.frames?.length ?? 0;
            if (match || pageCount === 0 || pageCount < 250) break;
            offset += pageCount;
          }
          return Boolean(match);
        },
        { timeout: timelineTimeoutMs, intervals: [500, 1000, 2000, 5000] },
      )
      .toBe(true);
    if (!match) throw new Error(`event ${eventType} was not found`);
    return { event: match, replay: latest };
  }

  async function assertCanvasHasPixels(page: Page) {
    const canvas = page.locator(canvasLabel);
    await expect(canvas).toBeVisible();
    await expect
      .poll(
        async () =>
          canvas.evaluate((element) => {
            const canvasElement = element as HTMLCanvasElement;
            const context = canvasElement.getContext("2d");
            if (!context || canvasElement.width === 0 || canvasElement.height === 0) return 0;
            const pixels = context.getImageData(0, 0, canvasElement.width, canvasElement.height).data;
            const luminances: number[] = [];
            for (let index = 0; index < pixels.length; index += 32) {
              const alpha = pixels[index + 3];
              if (!alpha) continue;
              luminances.push(
                pixels[index] * 0.2126 +
                  pixels[index + 1] * 0.7152 +
                  pixels[index + 2] * 0.0722,
              );
            }
            if (luminances.length < 10) return 0;
            const mean = luminances.reduce((sum, value) => sum + value, 0) / luminances.length;
            return luminances.reduce((sum, value) => sum + (value - mean) ** 2, 0) / luminances.length;
          }),
        { timeout: 30_000, intervals: [250, 500, 1000] },
      )
      .toBeGreaterThan(2);
  }

  async function assertCanvasSemantics(page: Page, frame: OperationalFrame) {
    const canvas = page.locator(canvasLabel);
    const expectedCarrierCount = Array.isArray(frame.carriers)
      ? frame.carriers.length
      : frame.carrier
        ? 1
        : 0;
    const expectedWaterborneCount = frame.uuvs.filter(
      (uuv) => uuv.physically_exposed !== false,
    ).length;
    await expect
      .poll(async () =>
        canvas.evaluate((element) => [
          element.getAttribute("data-carrier-count"),
          element.getAttribute("data-waterborne-uuv-count"),
          element.getAttribute("data-target-estimate-count"),
          element.getAttribute("data-plan-version"),
        ]),
      )
      .toEqual([
        String(expectedCarrierCount),
        String(expectedWaterborneCount),
        String(frame.target_estimates.length),
        String(frame.plan_version),
      ]);
  }

  async function assertNoOverflowOrClipping(page: Page) {
    const layout = await page.evaluate(() => {
      const panels = Array.from(
        document.querySelectorAll<HTMLElement>(
          '[role="complementary"], [role="dialog"], .bottom-drawer, .smart-assistant, .memory-window',
        ),
      );
      return {
        viewportWidth: window.innerWidth,
        documentOverflow: document.documentElement.scrollWidth - window.innerWidth,
        panels: panels
          .filter((panel) => panel.getClientRects().length > 0)
          .map((panel) => ({
            right: panel.getBoundingClientRect().right,
            scrollWidth: panel.scrollWidth,
            clientWidth: panel.clientWidth,
          })),
      };
    });
    expect(layout.documentOverflow).toBeLessThanOrEqual(0);
    expect(layout.panels.every((panel) => panel.right <= layout.viewportWidth + 1)).toBeTruthy();
    expect(
      layout.panels.every((panel) => panel.scrollWidth <= panel.clientWidth + 1),
    ).toBeTruthy();
  }

  async function exerciseOperatorSurface(
    page: Page,
    request: APIRequestContext,
    snapshot: OperationalFrame,
  ) {
    const targetId = snapshot.target_estimates[0]?.target_id;
    const uuvId = snapshot.uuvs.find((uuv) => uuv.physically_exposed !== false)?.uuv_id;
    expect(targetId).toBeTruthy();
    expect(uuvId).toBeTruthy();

    await page.getByText("当前态势", { exact: true }).click();
    await expect(page.getByText("UUV 资源", { exact: true })).toBeVisible();

    const assistantSummary = page.locator("details.assistant-panel > summary");
    await assistantSummary.getByText("智能助理", { exact: true }).click();
    const conversation = page.getByRole("textbox", { name: "智能助理输入" });
    await expect(conversation).toBeEditable();
    await conversation.fill("请基于当前态势复核下一交接窗口");
    const conversationRequest = page.waitForRequest((requestEvent) =>
      new URL(requestEvent.url()).pathname === "/api/conversation/messages",
    );
    const conversationResponse = page.waitForResponse((response) =>
      new URL(response.url()).pathname === "/api/conversation/messages",
    );
    await page.getByRole("button", { name: "发送", exact: true }).click();
    expect((await conversationResponse).status()).toBe(200);
    const conversationPayload = (await conversationRequest).postDataJSON() as {
      conversation_id: string;
    };

    const scopeResponse = await request.get("/api/operational/snapshot");
    expect(scopeResponse.ok()).toBeTruthy();
    const current = (await scopeResponse.json()) as OperationalFrame;
    const scenarioId = String(current.scenario_id ?? "uuv-only-single-target");
    const memorySnapshotResponse = await request.get(
      `/api/assistant/memory?user_id=operator&conversation_id=${encodeURIComponent(conversationPayload.conversation_id)}&scenario_id=${encodeURIComponent(scenarioId)}`,
    );
    expect(memorySnapshotResponse.ok()).toBeTruthy();
    const memorySnapshot = await memorySnapshotResponse.json();
    expect(memorySnapshot.user_id).toBe("operator");
    expect(["pending", "completed", "degraded", "failed"]).toContain(memorySnapshot.memory_status);
    const memoryStreamResponse = await request.get(
      `/api/assistant/memory/stream?user_id=operator&conversation_id=${encodeURIComponent(conversationPayload.conversation_id)}&scenario_id=${encodeURIComponent(scenarioId)}&after_cursor=0&limit=100`,
    );
    expect(memoryStreamResponse.ok()).toBeTruthy();
    const memoryStream = await memoryStreamResponse.json();
    expect(Array.isArray(memoryStream.events)).toBeTruthy();

    const uuvButton = page.getByRole("button", { name: /UUV|uuv/ }).first();
    await expect(uuvButton).toBeVisible();
    await uuvButton.click();
    const sensorMode = page.locator('select[aria-label$="人工声纳模式"]').first();
    if (await sensorMode.isEnabled()) {
      const sensorResponse = page.waitForResponse((response) =>
        new URL(response.url()).pathname === "/api/sensor-modes",
      );
      await sensorMode.selectOption("active");
      expect((await sensorResponse).status()).toBe(202);
    }

    await page.getByRole("button", { name: "切换任务详情" }).click();
    for (const tab of [
      "时间线",
      "方案",
      "事件",
      "决策台账",
      "指标",
      "分段跟踪",
      "LLM 思考过程",
      "Memory Steam",
    ]) {
      await expect(page.getByRole("tab", { name: tab, exact: true })).toBeVisible();
    }
    await page.getByRole("tab", { name: "LLM 思考过程", exact: true }).click();
    await expect(page.getByLabel("LLM 思考过程演进")).toBeVisible();
    await page.getByRole("tab", { name: "Memory Steam", exact: true }).click();
    await expect(page.getByLabel("Memory Steam")).toBeVisible();

    const replayResponse = await request.get("/api/replay?start_s=0&limit=250");
    expect(replayResponse.ok()).toBeTruthy();
    const replay = (await replayResponse.json()) as ReplayPayload;
    expect(replay.frames.length).toBeGreaterThan(0);
    expect(typeof current.frame_id).toBe("number");
    expect(typeof current.llm_thinking).toBe("string");

    const expectedPlanVersion = current.plan_version;
    const directiveResponse = await request.post("/api/directives", {
      data: {
        text: "继续保持当前跟踪，优先检查下一交接窗口",
        author: "playwright-operator",
        expected_plan_version: expectedPlanVersion,
        target_ids: [targetId],
      },
    });
    expect([202, 409]).toContain(directiveResponse.status());
    if (directiveResponse.status() === 202) {
      const directive = await directiveResponse.json();
      expect(typeof directive.request_id).toBe("string");
    }

    const assignmentResponse = await request.post("/api/assignments", {
      data: {
        target_id: targetId,
        uuv_ids: [uuvId],
        expected_plan_version: expectedPlanVersion,
      },
    });
    expect([202, 409]).toContain(assignmentResponse.status());
    const questionResponse = await request.post("/api/questions", {
      data: { text: "请说明当前编组的主要证据" },
    });
    expect([200, 422]).toContain(questionResponse.status());
  }

  test("runs the owned live sequence and validates the operator surface", async ({
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
    await expect(page.locator(canvasLabel)).toBeVisible();

    const initial = await readSnapshot(request);
    expect(initial.uuvs).toHaveLength(12);
    expect(initial.uuvs.some((uuv) => uuv.physically_exposed !== false)).toBeTruthy();
    expect(initial).not.toHaveProperty("usvs");
    expect(JSON.stringify(initial).toLowerCase()).not.toContain("usv");
    await assertCanvasSemantics(page, initial);
    await assertCanvasHasPixels(page);
    await assertNoOverflowOrClipping(page);
    await expect(page.locator("body")).not.toContainText(/USV/i);
    await page.screenshot({ path: "test-results/uuv-live-default-initial-1440.png", fullPage: true });

    const planned = await waitForSnapshot(request, (frame) => frame.plan_version > 0);
    await assertCanvasSemantics(page, planned);
    await exerciseOperatorSurface(page, request, planned);

    const boundaryEntry = await waitForEvent(request, "uuv_boundary_entry_started", 0);
    const deployedFrame = boundaryEntry.replay.frames.find(
      (frame) => frame.sim_time_s >= boundaryEntry.event.sim_time_s,
    );
    expect(deployedFrame?.uuvs.some((uuv) => uuv.deployment_state === "deployed")).toBeTruthy();
    await assertCanvasSemantics(page, deployedFrame ?? (await readSnapshot(request)));
    await assertCanvasHasPixels(page);
    await page.screenshot({ path: "test-results/uuv-live-post-deployment-1440.png", fullPage: true });

    const activeScan = await waitForEvent(request, "active_ping", boundaryEntry.event.sim_time_s);
    const detection = await waitForEvent(
      request,
      "target_detection_acquired",
      activeScan.event.sim_time_s,
    );
    const adversary = await waitForEvent(
      request,
      "target_mission_decision",
      detection.event.sim_time_s,
    );
    const passiveTrack = await waitForSnapshot(
      request,
      (frame) =>
        frame.sim_time_s > adversary.event.sim_time_s &&
        (frame.groups ?? []).some((group) => group.mode === "passive_track"),
    );
    expect(passiveTrack.sim_time_s).toBeGreaterThan(adversary.event.sim_time_s);

    const handoff = await waitForEvent(request, "handoff_completed", passiveTrack.sim_time_s);
    const legacyLifecycleEvents = handoff.replay.frames
      .flatMap((frame) => frame.events ?? [])
      .filter(
        (event) =>
          [
            "carrier_dispatch_completed",
            "uuv_deployed",
            "uuv_recovery_requested",
            "uuv_recovered",
            "carrier_returned_to_fleet",
          ].includes(event.event_type),
      );
    expect(legacyLifecycleEvents).toHaveLength(0);

    const finalFrame = await readSnapshot(request);
    await assertCanvasSemantics(page, finalFrame);
    await assertCanvasHasPixels(page);
    await assertNoOverflowOrClipping(page);
    await page.screenshot({ path: "test-results/uuv-live-returned-1440.png", fullPage: true });
    expect(JSON.stringify(handoff.replay).toLowerCase()).not.toContain("usv");

    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto("/");
    await page.waitForLoadState("networkidle");
    await expect(page.locator(canvasLabel)).toBeVisible();
    await assertCanvasSemantics(page, await readSnapshot(request));
    await assertCanvasHasPixels(page);
    await assertNoOverflowOrClipping(page);
    await expect(page.locator("body")).not.toContainText(/USV/i);
    await page.screenshot({ path: "test-results/uuv-live-returned-390.png", fullPage: true });
    expect(consoleErrors).toEqual([]);
  });
});
