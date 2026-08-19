import { expect, test } from "@playwright/test";

const mainURL = process.env.PLAYWRIGHT_BASE_URL;

test.describe("main.py visual command center flow", () => {
  test.skip(!mainURL, "set PLAYWRIGHT_BASE_URL to a running main.py UI");

  test("desktop exposes live adversarial state, three cards, human controls, and replay", async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on("console", (message) => {
      if (message.type() === "error") consoleErrors.push(message.text());
    });
    page.on("pageerror", (error) => consoleErrors.push(error.message));
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto("/");

    await expect(page.locator('canvas[aria-label="水下跟踪态势地图，支持拖动、滚轮缩放、UUV 与区域选择"]')).toBeVisible();
    await expect(page.getByText("当前态势", { exact: true })).toBeVisible();
    await expect(page.getByText("预测与接力", { exact: true })).toBeVisible();
    await expect(page.getByText("LLM Client", { exact: true })).toBeVisible();
    await expect(page.getByText("目标潜艇脑", { exact: true })).toBeVisible();
    await expect(page.getByText(/目标侧估计|LLM 决策摘要/).first()).toBeVisible();
    await page.getByText("LLM Client", { exact: true }).click();
    await expect(page.getByLabel("统一 LLM Client")).toBeVisible();
    await expect(page.getByLabel("统一对话")).toBeEditable();

    const sidebar = page.getByRole("complementary", { name: "编队态势" });
    await expect(sidebar.locator("details.sidebar-collapsible")).toHaveCount(3);
    const uuvButton = page.getByRole("button", { name: /uuv_00/ }).first();
    await expect(uuvButton).toBeVisible();
    await uuvButton.click();
    await expect(page.getByText("uuv_00 详情", { exact: true })).toBeVisible();

    await page.getByRole("button", { name: "切换任务详情" }).click();
    for (const tab of ["时间线", "方案", "事件", "决策台账", "指标", "分段跟踪"]) {
      await expect(page.getByRole("tab", { name: tab, exact: true })).toBeVisible();
    }

    await page.getByRole("button", { name: "回放", exact: true }).click();
    await expect(page.getByText("REPLAY", { exact: true })).toBeVisible();
    await expect(page.getByLabel("回放控制")).toBeVisible();
    expect(consoleErrors).toEqual([]);
  });

  test("narrow screen keeps live state readable and supports sidebar close", async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on("console", (message) => {
      if (message.type() === "error") consoleErrors.push(message.text());
    });
    page.on("pageerror", (error) => consoleErrors.push(error.message));
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto("/");

    await expect(page.locator('canvas[aria-label="水下跟踪态势地图，支持拖动、滚轮缩放、UUV 与区域选择"]')).toBeVisible();
    await page.getByRole("button", { name: "切换编队状态" }).click();
    const sidebar = page.getByRole("complementary", { name: "编队态势" });
    await expect(sidebar).toBeVisible();
    await expect(sidebar.getByText("目标潜艇脑", { exact: true })).toBeVisible();
    const scrolling = await sidebar.evaluate((element) => (
      element.scrollHeight > element.clientHeight && getComputedStyle(element).overflowY === "auto"
    ));
    expect(scrolling).toBe(true);
    await sidebar.getByRole("button", { name: "关闭编队状态" }).click();
    await expect(sidebar).not.toBeVisible();
    expect(consoleErrors).toEqual([]);
  });
});
