import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { OperationalFrame, PlanAdjustmentSuggestionView } from "../../types/frames";
import DirectiveComposer from "./DirectiveComposer";

const frame: OperationalFrame = {
  schema_version: "1.0", frame_id: 1, sim_time_s: 30, plan_version: 4,
  map_bounds: { min_x: 0, min_y: 0, max_x: 100, max_y: 100 }, uuvs: [], target_estimates: [],
  bearing_rays: [], groups: [], events: [], plans: [], ledger: [], metrics: [],
  carrier: null,
};

const suggestion: PlanAdjustmentSuggestionView = {
  suggestion_id: "suggestion-1",
  category: "tracking_quality",
  title: "提高 T1 航迹稳定性",
  rationale: "当前估计不确定度正在上升。",
  proposed_feedback: "请优先保证 T1 的稳定跟踪，必要时选择性启用主动声纳。",
  target_ids: ["T1"],
  evidence_ids: ["evt-1"],
  confidence: 0.82,
};

describe("DirectiveComposer", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("queues a directive with the visible plan version and does not claim commitment", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({ request_id: "job-1", status: "queued" }) })
      .mockResolvedValue({ ok: true, json: async () => ({ request_id: "job-1", status: "processing" }) });
    vi.stubGlobal("fetch", fetchMock);
    render(<DirectiveComposer frame={frame} selectedTargetIds={["T1"]} />);
    fireEvent.change(screen.getByRole("textbox", { name: "专家指令" }), { target: { value: "优先保证 T1" } });
    fireEvent.click(screen.getByRole("button", { name: "提交预览" }));
    await waitFor(() => expect(screen.getByText(/已排队|解析中/)).toBeInTheDocument());
    expect(fetchMock.mock.calls[0][0]).toBe("/api/directives");
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toMatchObject({
      expected_plan_version: 4,
      target_ids: ["T1"],
    });
    expect(screen.queryByText("已提交方案")).not.toBeInTheDocument();
  });

  it("sends the clicked LLM suggestion text as operator feedback", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({ request_id: "suggestion-job", status: "queued" }) })
      .mockResolvedValue({ ok: true, json: async () => ({ request_id: "suggestion-job", status: "processing" }) });
    vi.stubGlobal("fetch", fetchMock);
    render(<DirectiveComposer frame={frame} selectedTargetIds={[]} suggestions={[suggestion]} />);

    fireEvent.click(screen.getByRole("button", { name: "发送建议：提高 T1 航迹稳定性" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());

    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toMatchObject({
      text: suggestion.proposed_feedback,
      expected_plan_version: 4,
      target_ids: ["T1"],
    });
  });
});
