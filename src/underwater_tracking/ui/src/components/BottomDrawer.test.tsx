import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { OperationalFrame } from "../types/frames";
import BottomDrawer from "./BottomDrawer";

const frame: OperationalFrame = {
  schema_version: "1.0",
  frame_id: 1,
  sim_time_s: 120,
  plan_version: 1,
  map_bounds: { min_x: 0, min_y: 0, max_x: 100, max_y: 100 },
  uuvs: [],
  target_estimates: [],
  bearing_rays: [],
  groups: [],
  events: [
    {
      event_id: "review-120",
      sim_time_s: 120,
      event_type: "strategic_review",
      level: "strategic",
      entity_id: "S1",
      message: "周期复盘",
    },
    {
      event_id: "rotation-120",
      sim_time_s: 120,
      event_type: "battery_rotation",
      level: "tactical",
      entity_id: "U1",
      message: "低能量轮换",
    },
  ],
  plans: [],
  ledger: [],
  metrics: [],
  carrier: null,
  llm_thinking: "基于方位观测建立首轮接力方案，并保持交接窗口内的观测重叠。",
  llm_thinking_trigger: "被动方位观测达到建轨门限。",
};

describe("BottomDrawer adaptive events", () => {
  it("labels strategy review and battery rotation events", () => {
    render(<BottomDrawer frame={frame} visible onToggle={() => undefined} />);

    expect(screen.getByText("战略复盘")).toBeInTheDocument();
    expect(screen.getByText("电量轮换")).toBeInTheDocument();
  });

  it("opens the regional tracking tab for frames without regional data", () => {
    render(<BottomDrawer frame={frame} visible onToggle={() => undefined} />);

    fireEvent.click(screen.getByRole("tab", { name: "分段跟踪" }));

    expect(screen.getByText("当前暂无区域任务")).toBeInTheDocument();
  });

  it("keeps the event tab on the accumulated event feed", () => {
    render(
      <BottomDrawer
        frame={frame}
        events={[
          {
            event_id: "found-60",
            sim_time_s: 60,
            event_type: "target_found",
            level: "informational",
            entity_id: "T1",
            message: "较早事件",
          },
          ...frame.events,
        ]}
        visible
        onToggle={() => undefined}
      />,
    );

    fireEvent.click(screen.getByRole("tab", { name: "事件" }));

    expect(screen.getByText("较早事件")).toBeInTheDocument();
    expect(screen.getByText("周期复盘")).toBeInTheDocument();
  });

  it("shows the LLM thinking paragraph", () => {
    render(
      <BottomDrawer
        frame={frame}
        thinkingHistory={[
          {
            sim_time_s: 60,
            plan_version: 1,
            content: "第一段思考。",
            trigger: "初始接触成立。",
          },
          {
            sim_time_s: 120,
            plan_version: 2,
            content: "第二段思考。",
            trigger: "目标发生机动。",
          },
        ]}
        visible
        onToggle={() => undefined}
      />,
    );

    fireEvent.click(screen.getByRole("tab", { name: "LLM 思考过程" }));

    expect(screen.getByText("第一段思考。")).toBeInTheDocument();
    expect(screen.getByText("第二段思考。")).toBeInTheDocument();
    expect(screen.getByText("目标发生机动。")).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: "思考演进至下一阶段" }),
    ).toBeInTheDocument();
  });

  it("shows memory stream failure status and reason without events", () => {
    render(
      <BottomDrawer
        frame={frame}
        memoryStatus="failed"
        memoryError="Memory Stream 请求失败"
        memoryDegradedReason="worker credentials unavailable"
        visible
        onToggle={() => undefined}
      />,
    );

    fireEvent.click(screen.getByRole("tab", { name: "Memory Steam" }));

    expect(screen.getByRole("status")).toHaveTextContent("failed");
    expect(screen.getByRole("status")).toHaveTextContent("worker credentials unavailable");
    expect(screen.getByRole("status")).toHaveTextContent("Memory Stream 请求失败");
    expect(screen.queryByText("暂无 Memory Stream 事件")).not.toBeInTheDocument();
  });
});
