import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { OperationalFrame } from "../types/frames";
import RightSidebar from "./RightSidebar";

const frame: OperationalFrame = {
  schema_version: "1.0",
  frame_id: 4,
  sim_time_s: 120,
  plan_version: 3,
  map_bounds: { min_x: -4000, min_y: -4000, max_x: 4000, max_y: 4000 },
  uuvs: [],
  target_estimates: [],
  bearing_rays: [],
  groups: [],
  events: [],
  plans: [],
  ledger: [],
  metrics: [],
  carrier: null,
  scheme: {
    scheme_id: "scheme-1",
    version: 3,
    valid_from_s: 0,
    valid_until_s: 900,
    target_priorities: { T1: 1 },
    minimum_quality: { T1: 0.8 },
    constraints: ["keep-passive"],
  },
  intelligence: [
    {
      report_id: "intel-1",
      source: "technical_reconnaissance",
      target_id: "T1",
      confidence: 0.85,
      issued_at_s: 90,
      valid_until_s: 300,
      content_summary: "Propulsion signature changed.",
    },
  ],
};

describe("RightSidebar adaptive context", () => {
  it("renders the active scheme and intelligence summary", () => {
    render(
      <RightSidebar
        frame={frame}
        selectedUuvId={null}
        onSelectUuv={() => undefined}
        open
        onClose={() => undefined}
      />,
    );

    expect(screen.getByText("方案约束")).toBeInTheDocument();
    expect(screen.getByText("v3 · T1 质量 ≥ 80%")).toBeInTheDocument();
    expect(screen.getByText("技侦 1 / 情报 1")).toBeInTheDocument();
  });
});
