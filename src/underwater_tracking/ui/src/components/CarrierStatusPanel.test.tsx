import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { OperationalFrame } from "../types/frames";
import CarrierStatusPanel from "./CarrierStatusPanel";

const frameWithoutCarrier: OperationalFrame = {
  schema_version: "1.0",
  frame_id: 1,
  sim_time_s: 30,
  plan_version: 4,
  map_bounds: { min_x: -4000, min_y: -4000, max_x: 4000, max_y: 4000 },
  uuvs: [{
    uuv_id: "uuv_01",
    status: "available",
    deployment_state: "deployed",
    position: { x: -1200, y: -900 },
    heading_rad: 0,
    speed_mps: 1,
    energy_fraction: 0.8,
    group_id: null,
    current_waypoint: null,
    breadcrumb: [],
    sensor_mode: "passive",
    reserved: false,
  }],
  target_estimates: [],
  bearing_rays: [],
  groups: [],
  events: [],
  plans: [],
  ledger: [],
  metrics: [],
  carrier: null,
};

const frameWithCarrier: OperationalFrame = {
  ...frameWithoutCarrier,
  carrier: {
    carrier_id: "carrier-01",
    position: { x: -3000, y: -3000 },
    heading_rad: 0,
    speed_mps: 1.5,
    status: "recovering",
    onboard_uuv_ids: ["uuv_04"],
    deployed_uuv_ids: ["uuv_01", "uuv_02"],
    returning_uuv_ids: ["uuv_03"],
  },
  uuvs: [{ ...frameWithoutCarrier.uuvs[0], uuv_id: "uuv_03", deployment_state: "returning" }],
};

describe("CarrierStatusPanel", () => {
  it("shows carrier deployment and recovery counts", () => {
    render(<CarrierStatusPanel frame={frameWithCarrier} />);

    expect(screen.getByText("载体舰 / 发送回收")).toBeInTheDocument();
    expect(screen.getByText("回收 1")).toBeInTheDocument();
    expect(screen.getByText("uuv_03")).toBeInTheDocument();
  });

  it("keeps legacy frames usable without a carrier", () => {
    render(<CarrierStatusPanel frame={null} />);

    expect(screen.getByText("等待载体态势")).toBeInTheDocument();
  });
});
