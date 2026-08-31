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
  phase?: string;
};

type UuvSnapshot = {
  uuv_id: string;
  deployment_state: string;
  physically_exposed?: boolean;
  tracked_target_id?: string | null;
  tracked_target?: string | null;
};

type ExecutionSnapshot = {
  target_id: string;
  execution_revision: number;
  prediction_revision: number;
  intent_revision: number;
  current_region_id: string;
  next_region_id: string;
  evidence_ids: string[];
  reserve_uuv_ids: string[];
  regions: Array<{
    region_id: string;
    target_id: string;
    execution_revision: number;
    prediction_id: string;
    task_group_id: string;
    evidence_ids: string[];
  }>;
  task_groups: Array<{
    task_group_id: string;
    target_id: string;
    region_id: string;
    execution_revision: number;
    member_uuv_ids: string[];
    active_verifier_uuv_id: string;
    passive_tracker_uuv_id: string;
    evidence_ids: string[];
  }>;
};

type OperationalFrame = {
  frame_id: number;
  sim_time_s: number;
  plan_version: number;
  scenario_id?: string | null;
  uuvs: UuvSnapshot[];
  target_estimates: Array<{ target_id: string }>;
  execution?: ExecutionSnapshot | null;
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

function assertExecutionContext(payload: Record<string, unknown>, frame: OperationalFrame) {
  expect(frame.execution).toBeTruthy();
  expect(payload.execution_revision).toBe(frame.execution?.execution_revision);
  expect(payload.frame_id).toBe(frame.frame_id);
}

function assertContextValues(
  payload: Record<string, unknown>,
  executionRevision: number,
  frameId: number,
) {
  expect(payload.execution_revision).toBe(executionRevision);
  expect(payload.frame_id).toBe(frameId);
}

const legacyCarrierLifecycleEvents = new Set([
  "carrier_dispatch_completed",
  "uuv_deployed",
  "uuv_recovery_requested",
  "uuv_recovered",
  "carrier_returned_to_fleet",
]);

function assertExecutionContract(frame: OperationalFrame) {
  expect(typeof frame.frame_id).toBe("number");
  expect(frame.uuv_only).toBe(true);
  expect(frame).not.toHaveProperty("usvs");
  expect(frame.carrier ?? null).toBeNull();
  expect(frame.carriers ?? []).toHaveLength(0);
  expect(frame.carrier_missions ?? []).toHaveLength(0);
  expect(frame.planned_assignments ?? []).toHaveLength(0);
  expect(
    frame.events.every((event) => !legacyCarrierLifecycleEvents.has(event.event_type)),
  ).toBeTruthy();
  if (!frame.execution) {
    expect(frame.plan_version).toBe(0);
    return;
  }

  const execution = frame.execution;
  expect(execution.execution_revision).toBeGreaterThanOrEqual(1);
  expect(execution.prediction_revision).toBeGreaterThanOrEqual(1);
  expect(execution.intent_revision).toBeGreaterThanOrEqual(1);
  expect(execution.evidence_ids.length).toBeGreaterThan(0);
  const regionIds = Array.from({ length: 4 }, (_, index) =>
    `${execution.target_id}:task:${String(index + 1).padStart(2, "0")}`,
  );
  expect(execution.regions.map((region) => region.region_id)).toEqual(regionIds);
  expect(new Set(execution.regions.map((region) => region.region_id)).size).toBe(4);
  expect(new Set(execution.regions.map((region) => region.execution_revision))).toEqual(
    new Set([execution.execution_revision]),
  );
  expect(new Set(execution.regions.map((region) => region.target_id))).toEqual(
    new Set([execution.target_id]),
  );
  expect(new Set(execution.regions.map((region) => region.prediction_id)).size).toBe(1);
  expect(execution.regions.every((region) => region.evidence_ids.length > 0)).toBeTruthy();

  expect(execution.task_groups).toHaveLength(4);
  expect(new Set(execution.task_groups.map((group) => group.task_group_id)).size).toBe(4);
  expect(new Set(execution.task_groups.map((group) => group.region_id))).toEqual(
    new Set(regionIds),
  );
  expect(
    execution.task_groups.every(
      (group) =>
        group.target_id === execution.target_id &&
        group.execution_revision === execution.execution_revision &&
        group.member_uuv_ids.length === 2 &&
        new Set(group.member_uuv_ids).size === 2 &&
        new Set([
          group.active_verifier_uuv_id,
          group.passive_tracker_uuv_id,
        ]).size === 2 &&
        group.member_uuv_ids.includes(group.active_verifier_uuv_id) &&
        group.member_uuv_ids.includes(group.passive_tracker_uuv_id) &&
        group.evidence_ids.length > 0,
    ),
  ).toBeTruthy();
  const executionMembers = execution.task_groups.flatMap((group) => group.member_uuv_ids);
  expect(executionMembers).toHaveLength(8);
  expect(new Set(executionMembers).size).toBe(8);
  expect(execution.reserve_uuv_ids).toHaveLength(4);
  expect(new Set(execution.reserve_uuv_ids).size).toBe(4);
  expect(
    execution.reserve_uuv_ids.every((uuvId) => !executionMembers.includes(uuvId)),
  ).toBeTruthy();
  expect(execution.current_region_id).toBeTruthy();
  expect(regionIds).toContain(execution.current_region_id);
  expect(regionIds).toContain(execution.next_region_id);

  expect(frame.uuvs).toHaveLength(12);
  const uuvById = new Map(frame.uuvs.map((uuv) => [uuv.uuv_id, uuv]));
  expect(executionMembers.every((uuvId) => uuvById.has(uuvId))).toBeTruthy();
  expect(execution.reserve_uuv_ids.every((uuvId) => uuvById.has(uuvId))).toBeTruthy();
  const currentGroup = execution.task_groups.find(
    (group) => group.region_id === execution.current_region_id,
  );
  expect(currentGroup).toBeTruthy();
  if (currentGroup) {
    const trackedValues = currentGroup.member_uuv_ids.flatMap((uuvId) => {
      const uuv = uuvById.get(uuvId);
      return uuv && ("tracked_target_id" in uuv || "tracked_target" in uuv)
        ? [uuv.tracked_target_id ?? uuv.tracked_target ?? null]
        : [];
    });
    if (trackedValues.length > 0) expect(trackedValues).toContain(execution.target_id);
  }
}

function assertTransportContextEqual(left: OperationalFrame, right: OperationalFrame) {
  expect(right.frame_id).toBe(left.frame_id);
  expect(right.plan_version).toBe(left.plan_version);
  expect(Boolean(right.execution)).toBe(Boolean(left.execution));
  if (left.execution && right.execution) {
    expect(right.execution.execution_revision).toBe(left.execution.execution_revision);
    expect(right.execution.target_id).toBe(left.execution.target_id);
    expect(right.execution.regions.map((region) => region.region_id)).toEqual(
      left.execution.regions.map((region) => region.region_id),
    );
    expect(right.execution.task_groups.map((group) => group.task_group_id)).toEqual(
      left.execution.task_groups.map((group) => group.task_group_id),
    );
  }
}

function assertReplayContract(replay: ReplayPayload) {
  let previousFrameId = -1;
  for (const frame of replay.frames ?? []) {
    expect(frame.frame_id).toBeGreaterThan(previousFrameId);
    previousFrameId = frame.frame_id;
    assertExecutionContract(frame);
  }
}

const realBaseURL = process.env.PLAYWRIGHT_BASE_URL;
const timelineTimeoutMs = 10 * 60 * 1000;
const canvasLabel = "canvas[aria-label=\"水下跟踪态势地图，支持拖动、滚轮缩放、区域双击聚焦与 UUV、区域选择\"]";

test.describe("live UUV initialization timeline", () => {
  test.skip(
    !realBaseURL,
    "set PLAYWRIGHT_BASE_URL to the real main.py command center",
  );
  test.setTimeout(timelineTimeoutMs);

  async function readSnapshot(request: APIRequestContext): Promise<OperationalFrame> {
    const response = await request.get("/api/operational/snapshot");
    expect(response.ok()).toBeTruthy();
    const frame = (await response.json()) as OperationalFrame;
    assertExecutionContract(frame);
    return frame;
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
    predicate?: (event: RuntimeEvent) => boolean,
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
            assertReplayContract(latest);
            match = latest.frames
              .flatMap((frame) => frame.events ?? [])
              .find(
                (event) =>
                  event.event_type === eventType &&
                  event.sim_time_s >= afterSimTime &&
                  (entityId === undefined || event.entity_id === entityId) &&
                  (predicate === undefined || predicate(event)),
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
    const expectedCarrierCount = frame.execution
      ? 0
      : Array.isArray(frame.carriers)
      ? frame.carriers.length
      : frame.carrier
        ? 1
        : 0;
    const executionIds = new Set(
      frame.execution?.task_groups.flatMap((group) => group.member_uuv_ids),
    );
    const expectedWaterborneCount = frame.uuvs.filter(
      (uuv) =>
        uuv.physically_exposed !== false &&
        (!frame.execution || executionIds.has(uuv.uuv_id)),
    ).length;
    const expectedTargetCount = frame.execution
      ? frame.target_estimates.filter(
          (target) => target.target_id === frame.execution?.target_id,
        ).length
      : frame.target_estimates.length;
    if (frame.execution) {
      expect(frame.execution.regions).toHaveLength(4);
      expect(frame.execution.task_groups).toHaveLength(4);
      expect(executionIds.size).toBe(8);
    }
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
        String(expectedTargetCount),
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
    const executionIds = new Set(
      snapshot.execution?.task_groups.flatMap((group) => group.member_uuv_ids),
    );
    const uuvId = snapshot.uuvs.find(
      (uuv) =>
        uuv.physically_exposed !== false &&
        (!snapshot.execution || executionIds.has(uuv.uuv_id)),
    )?.uuv_id;
    expect(targetId).toBeTruthy();
    expect(uuvId).toBeTruthy();
    expect(snapshot.execution).toBeTruthy();
    if (!snapshot.execution || !targetId || !uuvId) {
      throw new Error("execution snapshot did not expose a target and executing UUV");
    }

    async function getContextBoundJson(
      pathFor: (frame: OperationalFrame) => string,
    ): Promise<{ frame: OperationalFrame; payload: Record<string, unknown> }> {
      for (let attempt = 0; attempt < 8; attempt += 1) {
        const frame = await readSnapshot(request);
        if (!frame.execution) continue;
        const response = await request.get(pathFor(frame));
        if (response.status() === 409) continue;
        expect(response.ok()).toBeTruthy();
        return {
          frame,
          payload: (await response.json()) as Record<string, unknown>,
        };
      }
      throw new Error("could not bind an operator request to a current execution frame");
    }

    async function postContextBound(
      path: string,
      payloadFor: (frame: OperationalFrame) => Record<string, unknown>,
    ): Promise<{
      frame: OperationalFrame;
      status: number;
      payload: Record<string, unknown>;
    }> {
      for (let attempt = 0; attempt < 8; attempt += 1) {
        const frame = await readSnapshot(request);
        if (!frame.execution) continue;
        const response = await request.post(path, { data: payloadFor(frame) });
        if (response.status() === 409) continue;
        expect(response.ok()).toBeTruthy();
        return {
          frame,
          status: response.status(),
          payload: (await response.json()) as Record<string, unknown>,
        };
      }
      throw new Error(`could not bind ${path} to a current execution frame`);
    }

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
    const conversationResponseValue = await conversationResponse;
    expect(conversationResponseValue.status()).toBe(200);
    const conversationPayload = (await conversationRequest).postDataJSON() as {
      conversation_id: string;
      expected_plan_version: number;
      execution_revision?: number;
      frame_id?: number;
    };
    const conversationResult =
      (await conversationResponseValue.json()) as Record<string, unknown>;
    expect(conversationPayload.expected_plan_version).toBeGreaterThan(0);
    expect(conversationPayload.execution_revision).toBeGreaterThanOrEqual(1);
    expect(typeof conversationPayload.frame_id).toBe("number");
    assertContextValues(
      conversationResult,
      conversationPayload.execution_revision as number,
      conversationPayload.frame_id as number,
    );

    const current = await readSnapshot(request);
    const scenarioId = String(current.scenario_id ?? "uuv-only-single-target");
    const memorySnapshotResult = await getContextBoundJson(
      (frame) =>
        `/api/assistant/memory?user_id=operator&conversation_id=${encodeURIComponent(conversationPayload.conversation_id)}&scenario_id=${encodeURIComponent(scenarioId)}&execution_revision=${frame.execution?.execution_revision}&frame_id=${frame.frame_id}`,
    );
    const memorySnapshot = memorySnapshotResult.payload;
    expect(memorySnapshot.user_id).toBe("operator");
    expect(["pending", "completed", "degraded", "failed"]).toContain(memorySnapshot.memory_status);
    assertExecutionContext(memorySnapshot, memorySnapshotResult.frame);
    const memoryStreamResult = await getContextBoundJson(
      (frame) =>
        `/api/assistant/memory/stream?user_id=operator&conversation_id=${encodeURIComponent(conversationPayload.conversation_id)}&scenario_id=${encodeURIComponent(scenarioId)}&after_cursor=0&limit=100&execution_revision=${frame.execution?.execution_revision}&frame_id=${frame.frame_id}`,
    );
    const memoryStream = memoryStreamResult.payload;
    expect(Array.isArray(memoryStream.events)).toBeTruthy();
    assertExecutionContext(memoryStream, memoryStreamResult.frame);

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
    assertReplayContract(replay);
    expect(typeof current.frame_id).toBe("number");
    expect(typeof current.llm_thinking).toBe("string");

    const directiveResult = await postContextBound("/api/directives", (frame) => ({
      text: "继续保持当前跟踪，优先检查下一交接窗口",
      author: "playwright-operator",
      expected_plan_version: frame.plan_version,
      target_ids: [targetId],
      execution_revision: frame.execution?.execution_revision,
      frame_id: frame.frame_id,
    }));
    expect(directiveResult.status).toBe(202);
    expect(typeof directiveResult.payload.request_id).toBe("string");
    assertExecutionContext(directiveResult.payload, directiveResult.frame);

    const assignmentResult = await postContextBound("/api/assignments", (frame) => ({
      target_id: targetId,
      uuv_ids: [uuvId],
      expected_plan_version: frame.plan_version,
      execution_revision: frame.execution?.execution_revision,
      frame_id: frame.frame_id,
    }));
    expect(assignmentResult.status).toBe(202);
    expect(typeof assignmentResult.payload.request_id).toBe("string");
    assertExecutionContext(assignmentResult.payload, assignmentResult.frame);

    const questionResult = await postContextBound("/api/questions", (frame) => ({
      text: "请说明当前编组的主要证据",
      evidence_ids: frame.execution?.evidence_ids ?? [],
      execution_revision: frame.execution?.execution_revision,
      frame_id: frame.frame_id,
    }));
    expect(questionResult.status).toBe(200);
    assertExecutionContext(questionResult.payload, questionResult.frame);
    const questionEvidenceIds = questionResult.payload.evidence_ids;
    expect(Array.isArray(questionEvidenceIds)).toBeTruthy();
    expect((questionEvidenceIds as unknown[]).length).toBeGreaterThan(0);
    const evidenceResult = await getContextBoundJson(
      (frame) =>
        `/api/evidence?${(questionEvidenceIds as string[])
          .map((evidenceId) => `evidence_ids=${encodeURIComponent(evidenceId)}`)
          .join("&")}&execution_revision=${frame.execution?.execution_revision}&frame_id=${frame.frame_id}`,
    );
    assertExecutionContext(evidenceResult.payload, evidenceResult.frame);
    const resolvedEvidence = evidenceResult.payload.resolved;
    expect(Array.isArray(resolvedEvidence)).toBeTruthy();
    expect((resolvedEvidence as unknown[]).length).toBeGreaterThan(0);
  }

  test("runs the owned live sequence and validates the operator surface", async ({
    page,
    request,
  }) => {
    const consoleErrors: string[] = [];
    const websocketFrames: OperationalFrame[] = [];
    page.on("console", (message) => {
      if (message.type() === "error") consoleErrors.push(message.text());
    });
    page.on("pageerror", (error) => consoleErrors.push(error.message));
    page.on("websocket", (websocket) => {
      websocket.on("framereceived", (payload) => {
        try {
          const raw = typeof payload === "string" ? payload : payload.toString();
          const candidate = JSON.parse(raw) as unknown;
          if (
            candidate &&
            typeof candidate === "object" &&
            typeof (candidate as { frame_id?: unknown }).frame_id === "number"
          ) {
            websocketFrames.push(candidate as OperationalFrame);
          }
        } catch {
          // The server can use text frames for heartbeats; those are not snapshots.
        }
      });
    });

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

    const planned = await waitForSnapshot(
      request,
      (frame) => frame.plan_version > 0 && Boolean(frame.execution),
    );
    await assertCanvasSemantics(page, planned);
    await exerciseOperatorSurface(page, request, planned);

    const boundaryEntry = await waitForEvent(request, "uuv_boundary_entry_started", 0);
    const boundaryEntryUuvIds = new Set(
      boundaryEntry.replay.frames
        .flatMap((frame) => frame.events ?? [])
        .filter((event) => event.event_type === "uuv_boundary_entry_started")
        .map((event) => event.entity_id)
        .filter((entityId): entityId is string => Boolean(entityId)),
    );
    expect(boundaryEntryUuvIds).toHaveLength(8);
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
    const targetManeuver = await waitForEvent(
      request,
      "target_maneuver_observed",
      detection.event.sim_time_s,
    );
    const adversary = await waitForEvent(
      request,
      "target_mission_decision",
      detection.event.sim_time_s,
    );
    const blueResponse = await waitForEvent(
      request,
      "state_changed",
      Math.max(targetManeuver.event.sim_time_s, adversary.event.sim_time_s),
      undefined,
      (event) => event.phase === "blue_response",
    );
    expect(blueResponse.event.phase).toBe("blue_response");
    const passiveTrack = await waitForSnapshot(
      request,
      (frame) =>
        frame.sim_time_s > adversary.event.sim_time_s &&
        (frame.groups ?? []).some((group) => group.mode === "passive_track"),
    );
    expect(passiveTrack.sim_time_s).toBeGreaterThan(adversary.event.sim_time_s);

    const handoff = await waitForEvent(request, "handoff_completed", passiveTrack.sim_time_s);
    const replacement = await waitForEvent(
      request,
      "uuv_boundary_replacement",
      handoff.event.sim_time_s,
    );
    expect(replacement.event.event_type).toBe("uuv_boundary_replacement");
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

    const finalFrame = await waitForSnapshot(request, (frame) =>
      websocketFrames.some((candidate) => candidate.frame_id === frame.frame_id),
    );
    await assertCanvasSemantics(page, finalFrame);
    await assertCanvasHasPixels(page);
    await assertNoOverflowOrClipping(page);
    await page.screenshot({ path: "test-results/uuv-live-returned-1440.png", fullPage: true });
    expect(JSON.stringify(handoff.replay).toLowerCase()).not.toContain("usv");

    expect(websocketFrames.length).toBeGreaterThan(0);
    websocketFrames.forEach((frame) => assertExecutionContract(frame));
    const matchingWebsocket = websocketFrames.find(
      (frame) => frame.frame_id === finalFrame.frame_id,
    );
    expect(matchingWebsocket).toBeTruthy();
    if (matchingWebsocket) assertTransportContextEqual(finalFrame, matchingWebsocket);

    const finalReplayResponse = await request.get("/api/replay?start_s=0&limit=10000");
    expect(finalReplayResponse.ok()).toBeTruthy();
    const finalReplay = (await finalReplayResponse.json()) as ReplayPayload;
    assertReplayContract(finalReplay);
    const matchingReplay = finalReplay.frames.find(
      (frame) => frame.frame_id === finalFrame.frame_id,
    );
    expect(matchingReplay).toBeTruthy();
    if (matchingReplay) assertTransportContextEqual(finalFrame, matchingReplay);

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
