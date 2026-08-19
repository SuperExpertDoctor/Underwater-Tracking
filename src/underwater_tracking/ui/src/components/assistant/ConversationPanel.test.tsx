import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { OperationalFrame } from "../../types/frames";
import ConversationPanel from "./ConversationPanel";

const frame: OperationalFrame = {
  schema_version: "1.0", frame_id: 1, sim_time_s: 30, plan_version: 4,
  map_bounds: { min_x: 0, min_y: 0, max_x: 100, max_y: 100 }, uuvs: [], target_estimates: [],
  bearing_rays: [], groups: [], events: [], plans: [], ledger: [], metrics: [], carrier: null,
};

describe("ConversationPanel", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("renders a plan revision preview and requires an explicit apply", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          conversation_id: "conversation-1",
          classification: "plan_revision",
          expected_plan_version: 4,
          messages: [
            { message_id: "m1", role: "user", text: "优先保证 T1" },
            { message_id: "m2", role: "assistant", text: "已生成方案预览。", classification: "plan_revision" },
          ],
          proposal: {
            proposal_id: "proposal-1",
            directive: { directive_id: "D1", target_scope: ["T1"], target_priorities: { T1: 1 }, confidence: 0.9, status: "preview" },
            summary: "提高 T1 优先级",
            status: "preview",
          },
        }),
      })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ conversation_id: "conversation-1", classification: "plan_revision", proposal: { status: "applied" }, messages: [] }) });
    vi.stubGlobal("fetch", fetchMock);
    render(<ConversationPanel frame={frame} selectedTargetIds={["T1"]} />);

    fireEvent.change(screen.getByRole("textbox", { name: "统一对话" }), { target: { value: "优先保证 T1" } });
    fireEvent.click(screen.getByRole("button", { name: "发送" }));

    await waitFor(() => expect(screen.getByText("方案修正")).toBeInTheDocument());
    expect(screen.getByText("提高 T1 优先级")).toBeInTheDocument();
    expect(screen.getByText("证据 0 条")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "确认应用方案" })).toBeInTheDocument();
    expect(fetchMock.mock.calls[0][0]).toBe("/api/conversation/messages");
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toMatchObject({ expected_plan_version: 4, target_ids: ["T1"] });

    fireEvent.click(screen.getByRole("button", { name: "确认应用方案" }));
    await waitFor(() => expect(fetchMock.mock.calls[1][0]).toBe("/api/conversation/conversation-1/apply"));
  });

  it("shows evidence chips for an evidence query without an apply action", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          conversation_id: "conversation-2",
          classification: "evidence_query",
          messages: [
            { message_id: "m1", role: "user", text: "为什么保持当前编组？" },
            { message_id: "m2", role: "assistant", text: "因为观测质量稳定。", classification: "evidence_query", evidence_ids: ["E1"] },
          ],
          answer: { answer: "因为观测质量稳定。", evidence_ids: ["E1"] },
          evidence_ids: ["E1"],
        }),
      }),
    );
    render(<ConversationPanel frame={frame} selectedTargetIds={[]} />);

    fireEvent.change(screen.getByRole("textbox", { name: "统一对话" }), { target: { value: "为什么保持当前编组？" } });
    fireEvent.click(screen.getByRole("button", { name: "发送" }));

    await waitFor(() => expect(screen.getByText("因为观测质量稳定。")).toBeInTheDocument());
    expect(screen.getByRole("button", { name: "证据 E1" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "确认应用方案" })).not.toBeInTheDocument();
  });
});
