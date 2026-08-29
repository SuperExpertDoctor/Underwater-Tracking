import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { MemoryStatus, MemoryStreamEventView } from "../services/memoryApi";
import MemorySteam from "./MemorySteam";

function event(
  cursor: number,
  type: string,
  changes: Partial<MemoryStreamEventView> = {},
): MemoryStreamEventView {
  return {
    cursor,
    event_id: `event-${cursor}-${type}`,
    user_id: "operator",
    scenario_id: "scenario-1",
    conversation_id: "conversation-1",
    status: "completed",
    type,
    ...changes,
  };
}

describe("MemorySteam", () => {
  it("groups real memory event types, sorts by cursor, and removes duplicate ids and cursors", () => {
    const duplicateEventId = event(3, "memory_version_created", {
      event_id: "event-3-memory-version",
    });
    const duplicateCursor = event(3, "memory_accessed", {
      event_id: "event-3-access",
    });

    render(
      <MemorySteam
        events={[
          event(3, "memory_version_created", {
            event_id: "event-3-memory-version",
          }),
          event(1, "context_loaded"),
          event(2, "retrieval_started"),
          duplicateEventId,
          duplicateCursor,
          event(4, "short_term_compressed"),
          event(5, "evidence_trace_completed"),
          event(6, "memory_archived"),
          event(7, "memory_accessed"),
          event(8, "memory_filtered"),
          event(9, "memory_extracted"),
          event(10, "retrieval_completed"),
        ]}
        status="completed"
        loading={false}
        error=""
        cursor={10}
      />,
    );

    expect(screen.getByText("上下文已加载")).toBeInTheDocument();
    expect(screen.getByText("检索开始")).toBeInTheDocument();
    expect(screen.getByText("检索完成")).toBeInTheDocument();
    expect(screen.getByText("记忆已过滤")).toBeInTheDocument();
    expect(screen.getByText("记忆已提炼")).toBeInTheDocument();
    expect(screen.getByText("短期记忆压缩完成")).toBeInTheDocument();
    expect(screen.getByText("记忆版本创建")).toBeInTheDocument();
    expect(screen.getByText("记忆访问")).toBeInTheDocument();
    expect(screen.getByText("记忆归档")).toBeInTheDocument();
    expect(screen.getByText("证据追溯完成")).toBeInTheDocument();

    const rendered = screen.getAllByTestId("memory-steam-event");
    expect(rendered).toHaveLength(10);
    expect(rendered.map((item) => item.getAttribute("data-cursor"))).toEqual(
      ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    );
    expect(screen.getAllByText("#3")).toHaveLength(1);
  });

  it("keeps only the last 300 events without mutating the input", () => {
    const events = Array.from({ length: 302 }, (_, index) =>
      event(index + 1, "memory_accessed"),
    );
    const original = [...events];

    render(
      <MemorySteam
        events={events}
        status="completed"
        loading={false}
        error=""
        cursor={302}
      />,
    );

    const rendered = screen.getAllByTestId("memory-steam-event");
    expect(rendered).toHaveLength(300);
    expect(rendered[0]).toHaveAttribute("data-cursor", "3");
    expect(rendered.at(-1)).toHaveAttribute("data-cursor", "302");
    expect(events).toEqual(original);
  });

  it("drops expansion state for events that leave the bounded window", () => {
    const first = event(1, "evidence_trace_completed", {
      payload: { source_event_ids: ["event-1"] },
    });
    const second = event(2, "evidence_trace_completed", {
      payload: { source_event_ids: ["event-2"] },
    });
    const { rerender } = render(
      <MemorySteam
        events={[first]}
        status="completed"
        loading={false}
        error=""
        cursor={1}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /展开证据链 event-1/ }));
    expect(screen.getByText("event-1")).toBeInTheDocument();

    rerender(
      <MemorySteam
        events={[second]}
        status="completed"
        loading={false}
        error=""
        cursor={2}
      />,
    );
    rerender(
      <MemorySteam
        events={[first, second]}
        status="completed"
        loading={false}
        error=""
        cursor={2}
      />,
    );

    expect(screen.getByRole("button", { name: /展开证据链 event-1/ })).toBeInTheDocument();
    expect(screen.queryByText("event-1")).not.toBeInTheDocument();
  });

  it("shows explicit empty, loading, error, and degraded states without inventing events", () => {
    const { rerender } = render(
      <MemorySteam
        events={[]}
        status="idle"
        loading={false}
        error=""
        cursor={0}
      />,
    );
    expect(screen.getByText("暂无 Memory Steam 事件")).toBeInTheDocument();
    expect(screen.queryByTestId("memory-steam-event")).not.toBeInTheDocument();

    rerender(
      <MemorySteam
        events={[]}
        status="pending"
        loading
        error=""
        cursor={0}
      />,
    );
    expect(screen.getByText("正在读取 Memory Steam…")).toBeInTheDocument();
    expect(screen.queryByTestId("memory-steam-event")).not.toBeInTheDocument();

    rerender(
      <MemorySteam
        events={[]}
        status="failed"
        loading={false}
        error="Memory Stream 请求失败"
        degradedReason="Embedding credentials are unavailable"
        cursor={12}
      />,
    );
    expect(screen.getByRole("status")).toHaveTextContent("Memory Stream 请求失败");
    expect(screen.getByRole("status")).toHaveTextContent(
      "Embedding credentials are unavailable",
    );
    expect(screen.getByRole("status")).toHaveTextContent("已读取至游标 12");
    expect(screen.queryByTestId("memory-steam-event")).not.toBeInTheDocument();
  });

  it("expands a bounded causal chain and selects evidence references", () => {
    const onSelectEvidence = vi.fn();
    const streamEvent = event(12, "evidence_trace_completed", {
      memory_id: "memory-7",
      memory_family_id: "family-2",
      version: 3,
      payload: {
        work_id: "question-work-4",
        memory_family_id: "family-2",
        version: 3,
        memory_ids: ["memory-7"],
        source_ids: ["decision-9", "plan-12", "event-42", "source-very-long-id"],
        source_event_ids: ["event-42"],
        source_plan_ids: ["plan-12"],
        plan_version: 4,
      },
    });

    render(
      <MemorySteam
        events={[streamEvent]}
        status="completed"
        loading={false}
        error=""
        cursor={12}
        onSelectEvidence={onSelectEvidence}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /展开证据链/ }));
    expect(screen.getByText("问题 / 工作 ID")).toBeInTheDocument();
    expect(screen.getByText("question-work-4")).toBeInTheDocument();
    expect(screen.getByText("记忆 / 版本 / 家族")).toBeInTheDocument();
    expect(screen.getByText("memory-7")).toBeInTheDocument();
    expect(screen.getByText("方案 v4")).toBeInTheDocument();
    expect(screen.getByText("来源 IDs")).toBeInTheDocument();
    expect(screen.getByText("decision-9")).toBeInTheDocument();
    expect(screen.getByText("plan-12")).toBeInTheDocument();
    expect(screen.getByText("事件")).toBeInTheDocument();
    expect(screen.getByText("方案")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "证据 event-42" }));
    expect(onSelectEvidence).toHaveBeenCalledWith("event-42");

    fireEvent.click(screen.getByRole("button", { name: /展开全部来源/ }));
    expect(screen.getByText("source-very-long-id")).toBeInTheDocument();
    expect(screen.queryByText("retired-source-3")).not.toBeInTheDocument();
  });

  it.each<[MemoryStatus, string]>([
    ["processing", "记忆处理中"],
    ["degraded", "记忆服务降级"],
  ])("uses explicit labels for %s stream status", (status, label) => {
    render(
      <MemorySteam
        events={[]}
        status={status}
        loading={false}
        error=""
        cursor={0}
        degradedReason={status === "degraded" ? "LLM unavailable" : null}
      />,
    );

    expect(screen.getByRole("status")).toHaveTextContent(label);
  });
});
