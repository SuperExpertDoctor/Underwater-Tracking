import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { PlanView } from "../../types/frames";
import SegmentOverlay from "./SegmentOverlay";

const plan: PlanView = {
  plan_id: "plan-7", version: 4, status: "active", concept: "balanced", reason: "relay",
  affected_targets: ["T1"], group_changes: [], valid_from_s: 0, valid_until_s: 600,
  segment_plan: ["relay:G-T1:0-300", "relay:G-T2:300-600"],
};

describe("SegmentOverlay", () => {
  it("renders relay segments in order", () => {
    render(<SegmentOverlay plans={[plan]} />);
    expect(screen.getByText("2 段")).toBeInTheDocument();
    expect(screen.getAllByRole("listitem").map((item) => item.textContent)).toEqual(plan.segment_plan);
  });

  it("shows an empty state before segmentation is available", () => {
    render(<SegmentOverlay plans={[{ ...plan, segment_plan: [] }]} />);
    expect(screen.getByText("尚未生成分段接力方案")).toBeInTheDocument();
  });
});
