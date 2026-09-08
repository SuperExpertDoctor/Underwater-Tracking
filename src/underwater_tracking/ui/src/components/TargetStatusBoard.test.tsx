import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { createContractFrame } from "../testSupport/contractFrame";
import TargetStatusBoard from "./TargetStatusBoard";

describe("TargetStatusBoard", () => {
  it("keeps owner, exiting group and handoff evidence visible in one frame", () => {
    const { container } = render(<TargetStatusBoard frame={createContractFrame()} />);
    expect(screen.getByLabelText("每目标运行状态")).toBeInTheDocument();
    expect(container.querySelector('[data-estimate-status="live"]')).toBeInTheDocument();
    expect(container.querySelector('[data-entry-confirmation="2/2"]')).toBeInTheDocument();
    expect(container.querySelector('[data-handoff-progress="3/3"]')).toBeInTheDocument();
    expect(container.querySelector('[data-group-lifecycle="exiting"]')).toBeInTheDocument();
    expect(screen.getByText(/OWNER \/ 当前负责/)).toBeInTheDocument();
  });

  it("shows expired estimate while the run is still executing", () => {
    const { container } = render(<TargetStatusBoard frame={createContractFrame({ simTimeS: 100, validUntilS: 90 })} />);
    expect(container.querySelector('[data-estimate-status="expired"]')).toBeInTheDocument();
    expect(screen.getByText("目标估计已过期 · 等待刷新")).toBeInTheDocument();
    expect(screen.getByText(/物理仿真仍在运行/)).toBeInTheDocument();
  });

  it("keeps dedicated release evidence explicit", () => {
    const { container } = render(<TargetStatusBoard frame={createContractFrame({ dedicated: true })} />);
    expect(screen.getByText("DEDICATED_TRACK · RELEASE_PENDING")).toBeInTheDocument();
    expect(screen.getByText("剩余里程最低")).toBeInTheDocument();
    expect(container.querySelector(".target-status-dedicated")?.textContent).toContain("6850 m");
    expect(container.querySelector('[data-release-threshold-m="7000"]')).toBeInTheDocument();
  });
});
