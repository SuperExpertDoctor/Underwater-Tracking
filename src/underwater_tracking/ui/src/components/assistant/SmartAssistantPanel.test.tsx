import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { OperationalFrame } from "../../types/frames";
import SmartAssistantPanel from "./SmartAssistantPanel";

const frame = { plan_version: 7 } as OperationalFrame;

function response(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), { status });
}

describe("SmartAssistantPanel", () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => vi.stubGlobal("fetch", fetchMock));
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("exposes the two assistant modes and applies only the backend proposal", async () => {
    fetchMock
      .mockResolvedValueOnce(
        response({
          conversation_id: "conversation-1",
          turn_id: "turn-1",
          user_id: "operator",
          assistant_mode: "plan_revision",
          classification: { classification: "plan_revision" },
          messages: [],
          expected_plan_version: 7,
          proposal: {
            proposal_id: "proposal-1",
            summary: "将下一交接窗口提前 30 秒",
            status: "preview",
            directive: { directive_id: "directive-1" },
            diff: { window_s: 30 },
          },
          memory_context: { user_id: "operator", memory_status: "pending", evidence_trace: [] },
        }),
      )
      .mockResolvedValueOnce(response({ status: "applied", turn_id: "turn-1" }));

    render(
      <SmartAssistantPanel
        frame={frame}
        selectedTargetIds={["target-1"]}
        conversationId="conversation-1"
        userId="operator"
      />,
    );

    expect(screen.getByRole("heading", { name: "智能助理" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "方案调整" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "证据回溯" })).toHaveAttribute("aria-pressed", "false");

    fireEvent.change(screen.getByRole("textbox", { name: "智能助理输入" }), {
      target: { value: "调整下一交接窗口" },
    });
    fireEvent.click(screen.getByRole("button", { name: "发送" }));

    await waitFor(() => expect(screen.getByText("将下一交接窗口提前 30 秒")).toBeInTheDocument());
    expect(screen.getByRole("button", { name: "应用方案预览" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "应用方案预览" }));
    await waitFor(() => expect(screen.getByText("方案已应用")).toBeInTheDocument());
    expect(fetchMock).toHaveBeenLastCalledWith(
      "/api/conversation/conversation-1/apply",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("renders an evidence answer as read-only with verified memory sources", async () => {
    fetchMock.mockResolvedValue(
      response({
        conversation_id: "conversation-1",
        turn_id: "turn-2",
        user_id: "operator",
        assistant_mode: "evidence_query",
        classification: { classification: "evidence_query" },
        messages: [],
        expected_plan_version: 7,
        answer: {
          answer: "当前方案保持重叠观测。",
          evidence_ids: ["event-1"],
          memory_ids: ["memory-1"],
          memory_status: "completed",
          evidence_trace: [
            {
              trace_id: "trace-1",
              user_id: "operator",
              status: "completed",
              memory_ids: ["memory-1"],
              source_event_ids: ["event-1"],
            },
          ],
        },
        memory_context: { user_id: "operator", memory_status: "completed", evidence_trace: [] },
      }),
    );

    render(
      <SmartAssistantPanel
        frame={frame}
        selectedTargetIds={[]}
        conversationId="conversation-1"
        userId="operator"
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "证据回溯" }));
    fireEvent.change(screen.getByRole("textbox", { name: "智能助理输入" }), {
      target: { value: "为什么保持方案？" },
    });
    fireEvent.click(screen.getByRole("button", { name: "发送" }));

    await waitFor(() => expect(screen.getByText("当前方案保持重叠观测。")).toBeInTheDocument());
    expect(screen.getByText("只读证据回答")).toBeInTheDocument();
    expect(screen.getByText("memory-1 · version 未提供")).toBeInTheDocument();
    expect(screen.getByText("已验证来源：event-1")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "应用方案预览" })).not.toBeInTheDocument();
  });

  it("shows the real request error instead of swallowing it", async () => {
    fetchMock.mockResolvedValue(response({ detail: "LLM unavailable" }, 503));
    render(
      <SmartAssistantPanel
        frame={frame}
        selectedTargetIds={[]}
        conversationId="conversation-1"
        userId="operator"
      />,
    );
    fireEvent.change(screen.getByRole("textbox", { name: "智能助理输入" }), {
      target: { value: "复核" },
    });
    fireEvent.click(screen.getByRole("button", { name: "发送" }));
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("HTTP 503"));
  });
});
