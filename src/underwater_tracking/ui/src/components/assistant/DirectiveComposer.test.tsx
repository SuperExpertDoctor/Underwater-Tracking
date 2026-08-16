import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { OperationalFrame } from "../../types/frames";
import DirectiveComposer from "./DirectiveComposer";

const frame: OperationalFrame = {
  schema_version: "1.0", frame_id: 1, sim_time_s: 30, plan_version: 4,
  map_bounds: { min_x: 0, min_y: 0, max_x: 100, max_y: 100 }, uuvs: [], target_estimates: [],
  bearing_rays: [], groups: [], events: [], plans: [], ledger: [], metrics: [],
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
});
