import { expect, test } from "../../src/underwater_tracking/ui/node_modules/@playwright/test";

const frame = {
  schema_version: "1.0",
  frame_id: 1,
  sim_time_s: 30,
  plan_version: 4,
  map_bounds: { min_x: -3200, min_y: -3200, max_x: 200, max_y: 200 },
  uuvs: [{
    uuv_id: "UUV-1", status: "returning", position: { x: -2740, y: -2780 }, heading_rad: 0,
    deployment_state: "returning",
    speed_mps: 2, energy_fraction: 0.82, group_id: null, current_waypoint: null,
    breadcrumb: [{ x: -2700, y: -2760 }, { x: -2740, y: -2780 }], sensor_mode: "passive", reserved: false,
  }, {
    uuv_id: "UUV-2", status: "tracking", position: { x: -200, y: 0 }, heading_rad: 0,
    deployment_state: "deployed",
    speed_mps: 2, energy_fraction: 0.76, group_id: "G-T1", current_waypoint: { x: 20, y: 0 },
    breadcrumb: [{ x: -250, y: 0 }, { x: -200, y: 0 }], sensor_mode: "active", reserved: false,
  }],
  target_estimates: [{
    target_id: "T1", mean: { x: 20, y: 0 },
    covariance_ellipse: { semimajor_m: 8, semiminor_m: 4, rotation_rad: 0 },
    intent: { label: "transit", confidence: 0.81, alternatives: {} },
    prediction: { horizon_s: 120, sample_step_s: 30, centerline_xy: [{ x: 20, y: 0 }, { x: 40, y: 5 }], radius_m: [5, 8] },
    quality: { quality_score: 0.88, estimated_rmse_m: 4.5, fim_min_eigenvalue: 0.01, fim_condition: 12 },
    classification: "submarine", last_ping_s: 30,
  }],
  bearing_rays: [{ observation_id: "obs-1", uuv_id: "UUV-2", target_id: "T1", origin: { x: -200, y: 0 }, azimuth_rad: 0, variance_rad2: 0.01, confidence: 0.9 }],
  groups: [{ group_id: "G-T1", target_id: "T1", member_ids: ["UUV-2"], quality: { instant: 0.9, window_mean: 0.88, ewma: 0.87, components: { fim: 0.9 }, hard_guard_reasons: [] } }],
  events: [{ event_id: "evt-1", sim_time_s: 30, event_type: "plan_committed", level: "strategic", entity_id: "T1", message: "方案已提交" }],
  plans: [{ plan_id: "plan-4", version: 4, status: "active", concept: "balanced", reason: "保证 T1 质量", affected_targets: ["T1"], group_changes: [], valid_from_s: 30, valid_until_s: 600, segment_plan: ["G-T1:30-600"] }],
  ledger: [{ decision_id: "decision-4", sim_time_s: 30, outcome: "committed", trigger_event_ids: ["evt-1"], evidence_ids: ["obs-1"], final_plan_id: "plan-4", final_plan_version: 4 }],
  metrics: [{ metric_id: "quality:T1", label: "T1 编组质量", value: 0.88, unit: "score", threshold: 0.7, window_s: 300, series: [0.8, 0.85, 0.88] }],
  scheme: {
    scheme_id: "scheme-1", version: 4, valid_from_s: 0, valid_until_s: 900,
    target_priorities: { T1: 1 }, minimum_quality: { T1: 0.8 }, constraints: ["keep-passive"],
  },
  intelligence: [{
    report_id: "intel-1", source: "technical_reconnaissance", target_id: "T1",
    confidence: 0.85, issued_at_s: 20, valid_until_s: 300,
    content_summary: "Propulsion signature changed.",
  }],
  carrier: {
    carrier_id: "carrier-01",
    position: { x: -3000, y: -3000 },
    heading_rad: 0,
    speed_mps: 1.5,
    status: "recovering",
    onboard_uuv_ids: [],
    deployed_uuv_ids: ["UUV-2"],
    returning_uuv_ids: ["UUV-1"],
  },
};

const sceneAssets = [
  { name: "background", path: "/assets/scene/background.png" },
  { name: "carrier", path: "/assets/scene/carrier.png" },
  { name: "uuv", path: "/assets/scene/uuv.png" },
  { name: "submarine", path: "/assets/scene/submarine.png" },
] as const;

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    class FakeWebSocket {
      static OPEN = 1;
      readyState = 0;
      onopen: (() => void) | null = null;
      onmessage: ((event: { data: string }) => void) | null = null;
      onerror: (() => void) | null = null;
      onclose: (() => void) | null = null;
      constructor(_url: string) {
        window.setTimeout(() => { this.readyState = FakeWebSocket.OPEN; this.onopen?.(); }, 0);
      }
      send(_message: string) {}
      close() { this.readyState = 3; this.onclose?.(); }
    }
    Object.defineProperty(window, "WebSocket", { configurable: true, value: FakeWebSocket });
  });
  await page.route("**/api/operational/snapshot", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(frame) }));
  await page.route("**/api/replay**", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ frames: [frame], count: 1 }) }));
  let assignmentApplied = false;
  await page.route("**/api/assignments", (route) => route.fulfill({
    status: 202,
    contentType: "application/json",
    body: JSON.stringify({ request_id: "assignment-job-1", status: "queued" }),
  }));
  await page.route("**/api/directives/assignment-job-1/apply", (route) => {
    assignmentApplied = true;
    return route.fulfill({
      status: 202,
      contentType: "application/json",
      body: JSON.stringify({ request_id: "assignment-job-1", status: "applying" }),
    });
  });
  await page.route("**/api/directives/assignment-job-1", (route) => route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify({
      request_id: "assignment-job-1",
      status: assignmentApplied ? "applied" : "preview",
      directive: {
        directive_id: "S1:assign:T1:UUV-2",
        raw_text: "assignment: UUV-2 -> T1",
        target_scope: ["T1"], locked_members: {}, target_priorities: {}, minimum_quality: {},
        disabled_uuv_ids: [], return_uuv_ids: [], directive_type: "assignment", assignment_target_id: "T1",
        assignment_uuv_ids: ["UUV-2"], confidence: 1, conflicts: [], status: assignmentApplied ? "applied" : "preview",
      },
    }),
  }));
});

test("operator can inspect live state, select a UUV, open details, and enter replay", async ({ page }) => {
  const consoleErrors: string[] = [];
  const sceneAssetPaths = sceneAssets.map(({ path }) => path);
  const sceneAssetStatuses = new Map<string, number>();
  page.on("console", (message) => { if (message.type() === "error") consoleErrors.push(message.text()); });
  page.on("response", (response) => {
    const path = new URL(response.url()).pathname;
    if (sceneAssetPaths.includes(path)) sceneAssetStatuses.set(path, response.status());
  });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/");
  await expect.poll(() => sceneAssetPaths.map((path) => sceneAssetStatuses.get(path))).toEqual([200, 200, 200, 200]);
  await expect(page.getByText("编队态势")).toBeVisible();
  await expect(page.getByText("方案约束")).toBeVisible();
  await expect(page.getByText("技侦 1 / 情报 1")).toBeVisible();
  await expect(page.getByText("carrier-01", { exact: true })).toBeVisible();
  await expect(page.getByText("回收 1", { exact: true })).toBeVisible();
  await expect(page.getByText("UUV-1", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("UUV-2", { exact: true }).first()).toBeVisible();
  await expect(page.getByRole("combobox", { name: "选择跟踪目标" })).toHaveValue("T1");
  await expect(page.locator('canvas[aria-label="水下跟踪态势地图，支持拖动、滚轮缩放、UUV 与区域选择"]')).toHaveScreenshot("command-center-carrier-returning.png", { animations: "disabled" });
  await page.getByRole("checkbox", { name: /UUV-2/ }).click();
  await page.getByRole("button", { name: "指派跟踪" }).click();
  await expect(page.getByRole("region", { name: "指派预览" })).toBeVisible();
  await page.getByRole("button", { name: "确认应用指派" }).click();
  await expect(page.getByText(/等待下一轮 LangGraph 重规划/)).toBeVisible();
  await page.getByRole("button", { name: /UUV-1/ }).first().click();
  await expect(page.getByText("UUV-1 详情")).toBeVisible();
  await page.getByRole("button", { name: "切换任务详情" }).click();
  await expect(page.getByRole("tab", { name: "方案" })).toBeVisible();
  await page.getByRole("button", { name: "回放" }).click();
  await expect(page.getByText("REPLAY")).toBeVisible();
  await page.screenshot({ path: "test-results/command-center-1440.png", fullPage: true });
  expect(consoleErrors).toEqual([]);
});

for (const missingAsset of sceneAssets) {
test(`missing ${missingAsset.name} scene image retains mixed asset/vector fallback without application console errors`, async ({ page }) => {
  const applicationConsoleErrors: string[] = [];
  const sceneAssetPaths = sceneAssets.map(({ path }) => path);
  const sceneAssetStatuses = new Map<string, number>();
  page.on("console", (message) => {
    if (message.type() === "error" && !message.text().includes("Failed to load resource")) {
      applicationConsoleErrors.push(message.text());
    }
  });
  page.on("response", (response) => {
    const path = new URL(response.url()).pathname;
    if (sceneAssetPaths.includes(path)) sceneAssetStatuses.set(path, response.status());
  });
  await page.route(`**${missingAsset.path}`, (route) => route.fulfill({ status: 404, body: "missing" }));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/");
  await expect.poll(() => sceneAssetPaths.map((path) => sceneAssetStatuses.get(path))).toEqual(
    sceneAssets.map(({ path }) => path === missingAsset.path ? 404 : 200),
  );
  const canvas = page.locator('canvas[aria-label="水下跟踪态势地图，支持拖动、滚轮缩放、UUV 与区域选择"]');
  await expect(canvas).toHaveScreenshot(`command-center-${missingAsset.name}-fallback.png`, { animations: "disabled" });
  await expect(page.getByText("carrier-01", { exact: true })).toBeVisible();
  expect(applicationConsoleErrors).toEqual([]);
});
}
