import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { OperationalFrame } from "../../types/frames";
import ConversationPanel from "./ConversationPanel";

const frame: OperationalFrame = {
  schema_version: "1.0",
  frame_id: 1,
  sim_time_s: 30,
  plan_version: 4,
  map_bounds: { min_x: 0, min_y: 0, max_x: 100, max_y: 100 },
  uuvs: [],
  target_estimates: [],
  bearing_rays: [],
  groups: [],
  events: [],
  plans: [],
  ledger: [],
  metrics: [],
  carrier: null,
};

describe("ConversationPanel", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("renders only an input and send button, then sends the current plan context", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ conversation_id: "conversation-1", messages: [] }),
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<ConversationPanel frame={frame} selectedTargetIds={["T1"]} />);

    const input = screen.getByRole("textbox", { name: "LLM 输入" });
    fireEvent.change(input, { target: { value: "优先保证 T1" } });
    fireEvent.click(screen.getByRole("button", { name: "发送" }));

    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/conversation/messages",
        expect.objectContaining({ method: "POST" }),
      ),
    );
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toMatchObject({
      text: "优先保证 T1",
      expected_plan_version: 4,
      target_ids: ["T1"],
    });
    expect(input).toHaveValue("");
    expect(screen.queryByText("统一对话")).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "确认应用方案" }),
    ).not.toBeInTheDocument();
  });
});
