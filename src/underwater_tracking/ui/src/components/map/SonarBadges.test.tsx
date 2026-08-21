import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { UUVView } from "../../types/frames";
import SonarBadges from "./SonarBadges";

const uuv = (id: string, sensor_mode: "active" | "passive", reserved: boolean): UUVView => ({
  uuv_id: id, status: "tracking", deployment_state: "deployed", physically_exposed: true, position: { x: 0, y: 0 }, heading_rad: 0,
  speed_mps: 2, energy_fraction: 0.8, group_id: null, current_waypoint: null,
  breadcrumb: [], sensor_mode, reserved,
});

describe("SonarBadges", () => {
  it("badges active and reserved UUVs", () => {
    render(<SonarBadges uuvs={[uuv("UUV-1", "passive", false), uuv("UUV-2", "active", true)]} />);
    expect(screen.getByText("UUV-2 主动")).toBeInTheDocument();
    expect(screen.getByText("UUV-2 指派")).toBeInTheDocument();
    expect(screen.queryByText("全部被动")).not.toBeInTheDocument();
  });

  it("shows a passive marker when no special state exists", () => {
    render(<SonarBadges uuvs={[uuv("UUV-1", "passive", false)]} />);
    expect(screen.getByText("全部被动")).toBeInTheDocument();
  });
});
