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

const frameWithMission: OperationalFrame = {
  ...frameWithoutCarrier,
  uuv_only: true,
  carrier_missions: [{
    carrier_id: "carrier-01",
    home_battle_group_id: "BG-01",
    mission_type: "DEPLOY_AND_RECOVER",
    route_status: "EN_ROUTE_NEXT_DEPLOY",
    route: [{ x: 0, y: 0 }, { x: 100, y: 100 }, { x: 0, y: 0 }],
    stop_ids: ["R01", "R02", "R03"],
    onboard_uuv_ids: ["uuv_01"],
    ready_uuv_ids: ["uuv_02"],
    reserved_uuv_ids: ["uuv_03"],
    recoverable_uuv_ids: ["uuv_04"],
  }],
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

  it("shows UUV reserve inventory and a home-returning carrier route", () => {
    render(<CarrierStatusPanel frame={frameWithMission} />);

    expect(screen.getByText("carrier-01")).toBeInTheDocument();
    expect(screen.getByText("预留 1")).toBeInTheDocument();
    expect(screen.getByText("航线闭合返回母港")).toBeInTheDocument();
    expect(screen.getByText("R03")).toBeInTheDocument();
  });
});
