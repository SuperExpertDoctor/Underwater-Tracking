import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { MemorySnapshotView } from "../../services/memoryApi";
import MemoryWindow from "./MemoryWindow";

function response(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), { status });
}

const populatedSnapshot: MemorySnapshotView = {
  user_id: "operator",
  scenario_id: "scenario-1",
  conversation_id: "conversation-1",
  short_term: {
    user_id: "operator",
    scenario_id: "scenario-1",
    conversation_id: "conversation-1",
    summary_text: "短期上下文摘要",
    summary_version: 2,
    recent_messages: [],
    message_count: 2,
    estimated_tokens: 20,
    compression_count: 1,
    compression_status: "completed",
  },
  episodic: [{ memory_id: "episodic-1", memory_family_id: "family-e", version: 1, user_id: "operator", scenario_id: "scenario-1", memory_type: "episodic", summary: "一次目标机动", importance_score: 0.7, status: "active", source_event_ids: ["event-1"], access_count: 2 }],
  semantic: [{ memory_id: "semantic-1", memory_family_id: "family-s", version: 2, user_id: "operator", scenario_id: "scenario-1", memory_type: "semantic", summary: "语义结论", importance_score: 0.9, status: "active", source_knowledge_ids: ["knowledge-1"], access_count: 5 }],
  procedural: [{ memory_id: "procedural-1", memory_family_id: "family-p", version: 1, user_id: "operator", scenario_id: "scenario-1", memory_type: "procedural", summary: "程序步骤", importance_score: 0.6, status: "active", source_plan_ids: ["plan-1"], access_count: 1 }],
  retrieved_hits: [{ memory: { memory_id: "semantic-1", memory_family_id: "family-s", version: 2, user_id: "operator", scenario_id: "scenario-1", memory_type: "semantic", summary: "语义结论", importance_score: 0.9, status: "active", source_knowledge_ids: ["knowledge-1"], access_count: 5 }, similarity_score: 0.88, rerank_score: 0.91, retrieval_reason: "与当前问题相关" }],
  versions: [],
  memory_status: "completed",
  degraded_reason: null,
};

describe("MemoryWindow", () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => vi.stubGlobal("fetch", fetchMock));
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("renders all memory families, metadata, versions, and refreshes after deletion", async () => {
    fetchMock
      .mockResolvedValueOnce(response(populatedSnapshot))
      .mockResolvedValueOnce(response({ user_id: "operator", memory_family_id: "family-e", versions: [{ ...populatedSnapshot.episodic[0], version: 1 }] }))
      .mockResolvedValueOnce(response({ status: "deleted", memory_id: "episodic-1", user_id: "operator" }))
      .mockResolvedValueOnce(response({ ...populatedSnapshot, episodic: [] }));

    render(<MemoryWindow userId="operator" conversationId="conversation-1" scenarioId="scenario-1" />);

    await waitFor(() => expect(screen.getByText("短期上下文摘要")).toBeInTheDocument());
    expect(screen.getByText("短期记忆")).toBeInTheDocument();
    expect(screen.getByText("情景记忆")).toBeInTheDocument();
    expect(screen.getByText("语义记忆")).toBeInTheDocument();
    expect(screen.getByText("程序记忆")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "情景记忆" }));
    expect(screen.getByText("v1")).toBeInTheDocument();
    expect(screen.getByText("重要性 0.70")).toBeInTheDocument();
    expect(screen.getByText("访问 2 次")).toBeInTheDocument();
    expect(screen.getByText("来源 1")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "展开版本 episodic-1" }));
    await waitFor(() => expect(screen.getByText("历史版本")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "删除记忆 episodic-1" }));
    await waitFor(() => expect(screen.getByText("该记忆已删除")).toBeInTheDocument());
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/assistant/memory/episodic-1?user_id=operator&scenario_id=scenario-1&conversation_id=conversation-1",
      expect.objectContaining({ method: "DELETE" }),
    );
  });

  it("shows explicit empty and degraded states from the real snapshot", async () => {
    fetchMock.mockResolvedValue(
      response({
        user_id: "operator",
        scenario_id: "scenario-1",
        conversation_id: "conversation-1",
        short_term: null,
        episodic: [],
        semantic: [],
        procedural: [],
        retrieved_hits: [],
        versions: [],
        memory_status: "degraded",
        degraded_reason: "Embedding credentials are unavailable",
      }),
    );

    render(<MemoryWindow userId="operator" conversationId="conversation-1" scenarioId="scenario-1" />);
    await waitFor(() => expect(screen.getByText("记忆服务降级")).toBeInTheDocument());
    expect(screen.getByText("Embedding credentials are unavailable")).toBeInTheDocument();
    expect(screen.getByText("当前没有短期记忆")).toBeInTheDocument();
  });

  it("does not render a previous scope snapshot response after switching scenarios", async () => {
    let resolveScenarioA!: (value: Response) => void;
    const scenarioB = {
      ...populatedSnapshot,
      scenario_id: "scenario-2",
      conversation_id: "conversation-2",
      short_term: {
        ...populatedSnapshot.short_term,
        scenario_id: "scenario-2",
        conversation_id: "conversation-2",
        summary_text: "场景 B 摘要",
      },
      episodic: [{ ...populatedSnapshot.episodic[0], scenario_id: "scenario-2", summary: "场景 B 事件" }],
      semantic: [],
      procedural: [],
      retrieved_hits: [],
      versions: [],
    };
    fetchMock
      .mockReturnValueOnce(new Promise<Response>((resolve) => { resolveScenarioA = resolve; }))
      .mockResolvedValueOnce(response(scenarioB));

    const { rerender } = render(
      <MemoryWindow userId="operator" conversationId="conversation-1" scenarioId="scenario-1" />,
    );
    rerender(
      <MemoryWindow userId="operator" conversationId="conversation-2" scenarioId="scenario-2" />,
    );
    await waitFor(() => expect(screen.getByText("场景 B 摘要")).toBeInTheDocument());
    resolveScenarioA(response(populatedSnapshot));

    await waitFor(() => expect(screen.getByText("场景 B 摘要")).toBeInTheDocument());
    expect(screen.queryByText("短期上下文摘要")).not.toBeInTheDocument();
  });

  it("does not let a refresh overwrite a newer external snapshot in the same scope", async () => {
    let resolveRefresh!: (value: Response) => void;
    const externalSnapshot: MemorySnapshotView = {
      ...populatedSnapshot,
      short_term: {
        ...populatedSnapshot.short_term!,
        summary_text: "外部最新摘要",
      },
    };
    fetchMock.mockReturnValueOnce(new Promise<Response>((resolve) => { resolveRefresh = resolve; }));

    const { rerender } = render(
      <MemoryWindow
        userId="operator"
        conversationId="conversation-1"
        scenarioId="scenario-1"
        snapshot={populatedSnapshot}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "刷新记忆快照" }));
    rerender(
      <MemoryWindow
        userId="operator"
        conversationId="conversation-1"
        scenarioId="scenario-1"
        snapshot={externalSnapshot}
      />,
    );
    await waitFor(() => expect(screen.getByText("外部最新摘要")).toBeInTheDocument());
    resolveRefresh(response(populatedSnapshot));

    await waitFor(() => expect(screen.getByText("外部最新摘要")).toBeInTheDocument());
    expect(screen.queryByText("短期上下文摘要")).not.toBeInTheDocument();
  });

  it("does not render a previous scope version response after switching scenarios", async () => {
    let resolveVersions!: (value: Response) => void;
    const scenarioB = {
      ...populatedSnapshot,
      scenario_id: "scenario-2",
      conversation_id: "conversation-2",
      short_term: {
        ...populatedSnapshot.short_term,
        scenario_id: "scenario-2",
        conversation_id: "conversation-2",
      },
      episodic: [{ ...populatedSnapshot.episodic[0], scenario_id: "scenario-2", summary: "场景 B 事件" }],
      semantic: [],
      procedural: [],
      retrieved_hits: [],
      versions: [],
    };
    fetchMock
      .mockResolvedValueOnce(response(populatedSnapshot))
      .mockReturnValueOnce(new Promise<Response>((resolve) => { resolveVersions = resolve; }))
      .mockResolvedValueOnce(response(scenarioB));

    const { rerender } = render(
      <MemoryWindow userId="operator" conversationId="conversation-1" scenarioId="scenario-1" />,
    );
    await waitFor(() => expect(screen.getByText("短期上下文摘要")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "情景记忆" }));
    fireEvent.click(screen.getByRole("button", { name: "展开版本 episodic-1" }));
    rerender(
      <MemoryWindow userId="operator" conversationId="conversation-2" scenarioId="scenario-2" />,
    );
    resolveVersions(response({
      user_id: "operator",
      memory_family_id: "family-e",
      versions: [{ ...populatedSnapshot.episodic[0], summary: "场景 A 历史版本" }],
    }));

    await waitFor(() => expect(screen.getByText("场景 B 事件")).toBeInTheDocument());
    expect(screen.queryByText("场景 A 历史版本")).not.toBeInTheDocument();
  });

  it("waits for an authoritative scenario instead of querying a default", () => {
    render(<MemoryWindow userId="operator" conversationId="conversation-1" />);

    expect(screen.getByText("等待当前场景确定，记忆暂不可用。")).toBeInTheDocument();
    expect(screen.queryByText("正在读取真实记忆快照…")).not.toBeInTheDocument();
  });
});
