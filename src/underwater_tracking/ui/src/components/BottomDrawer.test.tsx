import { render, screen } from "@testing-library/react";
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
};

describe("BottomDrawer adaptive events", () => {
  it("labels strategy review and battery rotation events", () => {
    render(<BottomDrawer frame={frame} visible onToggle={() => undefined} />);

    expect(screen.getByText("战略复盘")).toBeInTheDocument();
    expect(screen.getByText("电量轮换")).toBeInTheDocument();
  });
});
