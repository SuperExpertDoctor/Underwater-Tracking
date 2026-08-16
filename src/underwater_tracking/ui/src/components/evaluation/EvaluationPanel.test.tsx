import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import EvaluationPanel from "./EvaluationPanel";

describe("EvaluationPanel", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("does not request evaluation data when the build gate is off", () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    render(<EvaluationPanel enabled={false} simTimeS={30} />);
    expect(fetchMock).not.toHaveBeenCalled();
    expect(screen.queryByText(/TRUTH ENABLED/)).not.toBeInTheDocument();
  });

  it("shows a distinct truth-side banner only when explicitly enabled", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({
      ok: true,
      json: async () => ({ frames: [{ frame_id: 1, sim_time_s: 30, targets: [{ target_id: "T1", position_xy: [1, 2], intent_label: "transit" }] }] }),
    })));
    render(<EvaluationPanel enabled simTimeS={30} />);
    await waitFor(() => expect(screen.getByText("EVALUATION / TRUTH ENABLED")).toBeInTheDocument());
    expect(screen.getByText(/T1/)).toBeInTheDocument();
  });
});
