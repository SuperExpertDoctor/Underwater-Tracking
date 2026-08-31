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

  it("asks for a fresh preview when the plan version has advanced", async () => {
    fetchMock.mockResolvedValue(
      response(
        {
          detail: {
            message: "方案已更新",
            current_plan_version: 8,
            expected_plan_version: 7,
          },
        },
        409,
      ),
    );
    render(
      <SmartAssistantPanel
        frame={frame}
        selectedTargetIds={[]}
        conversationId="conversation-1"
        userId="operator"
      />,
    );
    fireEvent.change(screen.getByRole("textbox", { name: "智能助理输入" }), {
      target: { value: "保持当前任务区域" },
    });
    fireEvent.click(screen.getByRole("button", { name: "发送" }));
    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("重新生成预览"));
  });

  it("renders the backend classification even after switching the local tab", async () => {
    fetchMock.mockResolvedValue(
      response({
        conversation_id: "conversation-1",
        turn_id: "turn-3",
        user_id: "operator",
        assistant_mode: "auto",
        classification: { classification: "evidence_query", confidence: 0.98 },
        messages: [],
        expected_plan_version: 7,
        answer: {
          answer: "后端返回的证据答案。",
          evidence_ids: [],
          memory_ids: [],
          memory_status: "completed",
          evidence_trace: [],
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
    fireEvent.change(screen.getByRole("textbox", { name: "智能助理输入" }), {
      target: { value: "查询证据" },
    });
    fireEvent.click(screen.getByRole("button", { name: "发送" }));

    await waitFor(() => expect(screen.getByText("后端返回的证据答案。")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "方案调整" }));
    expect(screen.getByText("后端返回的证据答案。")).toBeInTheDocument();
  });

  it("renders both answer and proposal for a backend mixed classification", async () => {
    fetchMock.mockResolvedValue(
      response({
        conversation_id: "conversation-1",
        turn_id: "turn-4",
        user_id: "operator",
        assistant_mode: "auto",
        classification: { classification: "mixed" },
        messages: [],
        expected_plan_version: 7,
        proposal: { summary: "后端方案预览", status: "preview", diff: {} },
        answer: { answer: "后端混合回答。", evidence_trace: [] },
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
    fireEvent.change(screen.getByRole("textbox", { name: "智能助理输入" }), {
      target: { value: "同时回答并调整" },
    });
    fireEvent.click(screen.getByRole("button", { name: "发送" }));

    await waitFor(() => expect(screen.getByText("后端混合回答。")).toBeInTheDocument());
    expect(screen.getByText("后端方案预览")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "应用方案预览" })).toBeInTheDocument();
  });

  it("binds the request to the current frame and shows decision contributions", async () => {
    const onSelectEvidence = vi.fn();
    fetchMock.mockResolvedValue(
      response({
        conversation_id: "conversation-1",
        turn_id: "turn-context",
        user_id: "operator",
        assistant_mode: "evidence_query",
        execution_revision: 9,
        frame_id: 42,
        classification: { classification: "evidence_query" },
        messages: [],
        expected_plan_version: 7,
        answer: {
          answer: "依据当前执行快照生成的回答。",
          evidence_ids: ["event-1"],
          unresolved_evidence: ["missing-1"],
          decision_record: {
            algorithm_contributions: [
              { contributor: "algorithm", component: "global_track", summary: "全局轨迹" },
            ],
            llm_contributions: [
              { contributor: "llm", component: "strategy_revision", summary: "受限建议" },
            ],
            human_contributions: [
              { contributor: "human", component: "operator_feedback", summary: "人工确认" },
            ],
          },
        },
        memory_context: { user_id: "operator", memory_status: "completed", evidence_trace: [] },
      }),
    );

    render(
      <SmartAssistantPanel
        frame={{ plan_version: 7, frame_id: 42, execution: { execution_revision: 9 } } as OperationalFrame}
        selectedTargetIds={[]}
        conversationId="conversation-1"
        userId="operator"
        onSelectEvidence={onSelectEvidence}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "证据回溯" }));
    fireEvent.change(screen.getByRole("textbox", { name: "智能助理输入" }), {
      target: { value: "为何这样制定方案？" },
    });
    fireEvent.click(screen.getByRole("button", { name: "发送" }));

    await waitFor(() => expect(screen.getByText("依据当前执行快照生成的回答。")).toBeInTheDocument());
    expect(screen.getByText("执行版本 9 · 帧 42")).toBeInTheDocument();
    expect(screen.getByText("算法贡献")).toBeInTheDocument();
    expect(screen.getByText("全局轨迹")).toBeInTheDocument();
    expect(screen.getByText("LLM 贡献")).toBeInTheDocument();
    expect(screen.getByText("受限建议")).toBeInTheDocument();
    expect(screen.getByText("人工反馈")).toBeInTheDocument();
    expect(screen.getByText("人工确认")).toBeInTheDocument();
    expect(screen.getByText("未解析证据：missing-1")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "证据 event-1" }));
    expect(onSelectEvidence).toHaveBeenCalledWith("event-1");

    const request = JSON.parse(String(fetchMock.mock.calls.at(-1)?.[1]?.body));
    expect(request.execution_revision).toBe(9);
    expect(request.frame_id).toBe(42);
  });
});
