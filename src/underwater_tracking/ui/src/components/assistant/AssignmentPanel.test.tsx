import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { TargetEstimateView, UUVView } from "../../types/frames";
import { isDeployableUuv } from "../../domain/availability";
import AssignmentPanel from "./AssignmentPanel";

const target: TargetEstimateView = {
  target_id: "T1", mean: { x: 10, y: 20 },
  covariance_ellipse: { semimajor_m: 10, semiminor_m: 5, rotation_rad: 0 },
  intent: { label: "transit", confidence: 0.8, alternatives: {} }, prediction: null,
  quality: { quality_score: 0.9, estimated_rmse_m: 8, fim_min_eigenvalue: 1, fim_condition: 2 },
  classification: "unknown", last_ping_s: null,
};

const uuv = (id: string, reserved: boolean): UUVView => ({
  uuv_id: id, status: "tracking", deployment_state: "deployed", position: { x: 0, y: 0 }, heading_rad: 0,
  speed_mps: 2, energy_fraction: 0.8, group_id: null, current_waypoint: null,
  breadcrumb: [], sensor_mode: "passive", reserved,
});

describe("AssignmentPanel", () => {
  it("shares the deployability contract with assignment filtering", () => {
    expect(isDeployableUuv(uuv("UUV-deployed", false))).toBe(true);
    expect(isDeployableUuv({ ...uuv("UUV-onboard", false), deployment_state: "onboard" })).toBe(false);
    expect(isDeployableUuv({ ...uuv("UUV-failed", false), status: "failed" })).toBe(false);
  });

  it("lists only non-reserved UUVs and reports the sorted assignment", () => {
    const onAssign = vi.fn();
    render(<AssignmentPanel targets={[target]} uuvs={[uuv("UUV-2", true), uuv("UUV-1", false)]} onAssign={onAssign} />);
    expect(screen.getByRole("checkbox", { name: /UUV-1/ })).toBeInTheDocument();
    expect(screen.queryByRole("checkbox", { name: /UUV-2/ })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("checkbox", { name: /UUV-1/ }));
    fireEvent.click(screen.getByRole("button", { name: "指派跟踪" }));
    expect(onAssign).toHaveBeenCalledWith(["UUV-1"], "T1");
  });

  it("disables the action until a UUV is selected and counts reservations", () => {
    render(<AssignmentPanel targets={[target]} uuvs={[uuv("UUV-1", false), uuv("UUV-2", true)]} onAssign={vi.fn()} />);
    expect(screen.getByRole("button", { name: "指派跟踪" })).toBeDisabled();
    expect(screen.getByText("已指派 1 艇")).toBeInTheDocument();
  });

  it("does not offer onboard or returning UUVs for manual assignment", () => {
    render(<AssignmentPanel targets={[target]} uuvs={[
      { ...uuv("UUV-onboard", false), deployment_state: "onboard" },
      { ...uuv("UUV-returning", false), deployment_state: "returning" },
      uuv("UUV-deployed", false),
    ]} onAssign={vi.fn()} />);
    expect(screen.getByRole("checkbox", { name: /UUV-deployed/ })).toBeInTheDocument();
    expect(screen.queryByRole("checkbox", { name: /UUV-onboard/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("checkbox", { name: /UUV-returning/ })).not.toBeInTheDocument();
  });
});
