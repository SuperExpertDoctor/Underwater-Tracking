import { expect, test } from "@playwright/test";

const realBackendURL = process.env.PLAYWRIGHT_BASE_URL;

test.describe("real backend command center flow", () => {
  test.skip(
    !realBackendURL,
    "set PLAYWRIGHT_BASE_URL to a running UI reverse-proxied to the real backend",
  );

  test("renders live frames, uses operator APIs, and replays the same contract", async ({
    page,
    request,
  }) => {
    const consoleErrors: string[] = [];
    const apiRequests: string[] = [];
    page.on("request", (request) => {
      if (new URL(request.url()).pathname.startsWith("/api/")) {
        apiRequests.push(new URL(request.url()).pathname);
      }
    });
    page.on("console", (message) => {
      if (message.type() === "error") consoleErrors.push(message.text());
    });
    page.on("pageerror", (error) => consoleErrors.push(error.message));
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto("/");
    await page.waitForLoadState("networkidle");

    await expect(
      page.locator(
        'canvas[aria-label="水下跟踪态势地图，支持拖动、滚轮缩放、UUV 与区域选择"]',
      ),
    ).toBeVisible();
    await expect(page.getByText("实时连接", { exact: true })).toBeVisible();
    await expect(page.getByText("任务执行", { exact: true })).toBeVisible();
    await expect(page.getByText("事件触发", { exact: true })).toBeVisible();
    await expect(page.locator("details.sidebar-collapsible")).toHaveCount(3);

    const currentPanel = page.getByText("当前态势", { exact: true });
    await currentPanel.click();
    await expect(page.getByText("UUV 资源", { exact: true })).toBeVisible();

    const clientPanel = page.getByText("LLM Client", { exact: true });
    await clientPanel.click();
    const conversation = page.getByRole("textbox", { name: "LLM 输入" });
    await expect(conversation).toBeEditable();
    await conversation.fill("请基于当前态势复核下一交接窗口");
    const conversationResponse = page.waitForResponse((response) =>
      new URL(response.url()).pathname === "/api/conversation/messages",
    );
    await page.getByRole("button", { name: "发送", exact: true }).click();
    expect((await conversationResponse).status()).toBe(200);
    await expect(conversation).toHaveValue("");

    const uuvButton = page.getByRole("button", { name: /UUV|uuv/ }).first();
    await expect(uuvButton).toBeVisible();
    await uuvButton.click();
    const sensorMode = page.locator('select[aria-label$="人工声纳模式"]').first();
    await expect(sensorMode).toBeEnabled();
    const sensorResponse = page.waitForResponse((response) =>
      new URL(response.url()).pathname === "/api/sensor-modes",
    );
    await sensorMode.selectOption("active");
    expect((await sensorResponse).status()).toBe(202);

    await page.getByRole("button", { name: "切换任务详情" }).click();
    for (const tab of [
      "时间线",
      "方案",
      "事件",
      "决策台账",
      "指标",
      "分段跟踪",
      "LLM 思考过程",
    ]) {
      await expect(page.getByRole("tab", { name: tab, exact: true })).toBeVisible();
    }
    await page.getByRole("tab", { name: "LLM 思考过程", exact: true }).click();
    await expect(page.getByLabel("LLM 思考过程演进")).toBeVisible();

    const snapshotResponse = await request.get("/api/operational/snapshot");
    const replayResponse = await request.get("/api/replay?start_s=0");
    expect(snapshotResponse.ok()).toBeTruthy();
    expect(replayResponse.ok()).toBeTruthy();
    const snapshot = await snapshotResponse.json();
    const replay = await replayResponse.json();
    expect(replay.frames.length).toBeGreaterThan(0);
    expect(typeof snapshot.frame_id).toBe("number");
    expect(typeof snapshot.llm_thinking).toBe("string");
    expect(Array.isArray(snapshot.operational_stage_flags)).toBeTruthy();
    expect(typeof replay.frames.at(-1).llm_thinking).toBe("string");

    const targetId = snapshot.target_estimates[0]?.target_id;
    const uuvId = snapshot.uuvs[0]?.uuv_id;
    expect(typeof targetId).toBe("string");
    expect(typeof uuvId).toBe("string");
    const expectedPlanVersion = snapshot.plan_version;

    const directiveResponse = await request.post("/api/directives", {
      data: {
        text: "继续保持当前跟踪，优先检查下一交接窗口",
        author: "playwright-operator",
        expected_plan_version: expectedPlanVersion,
        target_ids: [targetId],
      },
    });
    expect(directiveResponse.status()).toBe(202);
    const directive = await directiveResponse.json();
    expect(typeof directive.request_id).toBe("string");
    const directiveStatusResponse = await request.get(
      `/api/directives/${encodeURIComponent(directive.request_id)}`,
    );
    expect(directiveStatusResponse.ok()).toBeTruthy();
    expect([
      "queued",
      "processing",
      "preview",
      "applying",
      "applied",
      "needs_clarification",
      "rejected",
      "error",
    ]).toContain((await directiveStatusResponse.json()).status);

    const assignmentResponse = await request.post("/api/assignments", {
      data: {
        target_id: targetId,
        uuv_ids: [uuvId],
        expected_plan_version: expectedPlanVersion,
      },
    });
    expect(assignmentResponse.status()).toBe(202);
    expect(typeof (await assignmentResponse.json()).request_id).toBe("string");

    const questionResponse = await request.post("/api/questions", {
      data: { text: "请说明当前编组的主要证据" },
    });
    expect([200, 422]).toContain(questionResponse.status());
    const question = await questionResponse.json();
    expect(typeof question.status === "string" || typeof question.answer === "string").toBeTruthy();
    expect(apiRequests).toEqual(
      expect.arrayContaining(["/api/conversation/messages", "/api/sensor-modes"]),
    );

    const wsFrame = await page.evaluate(
      () =>
        new Promise<{ frame_id: number; sim_time_s: number }>((resolve, reject) => {
          const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
          const socket = new WebSocket(`${protocol}//${window.location.host}/ws/operational`);
          const timeout = window.setTimeout(() => {
            socket.close();
            reject(new Error("real operational WebSocket did not publish a frame"));
          }, 10_000);
          socket.onopen = () => socket.send("ping");
          socket.onmessage = (message) => {
            try {
              const payload = JSON.parse(String(message.data)) as Partial<{
                frame_id: number;
                sim_time_s: number;
              }>;
              if (typeof payload.frame_id !== "number") return;
              window.clearTimeout(timeout);
              socket.close();
              resolve({ frame_id: payload.frame_id, sim_time_s: payload.sim_time_s ?? 0 });
            } catch {
              // Ignore the text pong and heartbeat until the first frame arrives.
            }
          };
          socket.onerror = () => reject(new Error("real operational WebSocket failed"));
        }),
    );
    expect(wsFrame.frame_id).toBeGreaterThanOrEqual(0);
    expect(wsFrame.sim_time_s).toBeGreaterThanOrEqual(0);

    await page.getByRole("button", { name: "回放", exact: true }).click();
    await expect(page.getByText("历史态势 · 专家干预已锁定", { exact: true })).toBeVisible();
    await expect(page.getByLabel("回放控制")).toBeVisible();
    expect(consoleErrors).toEqual([]);
  });

  test("keeps the real data view usable on a narrow screen", async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto("/");
    await page.waitForLoadState("networkidle");
    await expect(page.locator("canvas")).toBeVisible();
    await page.getByRole("button", { name: "切换编队状态" }).click();
    const sidebar = page.getByRole("complementary", { name: "编队态势" });
    await expect(sidebar).toBeVisible();
    await sidebar.getByRole("button", { name: "关闭编队状态" }).click();
    await expect(sidebar).not.toBeVisible();
  });
});
