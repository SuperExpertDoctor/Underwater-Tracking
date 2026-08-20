import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import PlaybackBar from "./PlaybackBar";

describe("PlaybackBar", () => {
  it("exposes a simulation-time range instead of a frame counter", () => {
    render(
      <PlaybackBar
        visible
        isPlaying={false}
        onPlayPause={vi.fn()}
        frameIndex={1}
        totalFrames={3}
        onSeek={vi.fn()}
        startTimeS={30}
        endTimeS={90}
        onSeekTime={vi.fn()}
        playSpeed={1}
        onSpeedChange={vi.fn()}
        frame={{ sim_time_s: 60 } as never}
        markers={[]}
      />,
    );

    const timeline = screen.getByRole("slider", { name: "Replay timeline" });
    expect(timeline).toHaveAttribute("min", "30");
    expect(timeline).toHaveAttribute("max", "90");
    expect(timeline).toHaveValue("60");
    expect(screen.queryByText("2 / 3")).not.toBeInTheDocument();
  });

  it("renders adversarial tracking milestones and outcome metrics", () => {
    render(
      <PlaybackBar
        visible
        isPlaying={false}
        onPlayPause={vi.fn()}
        frameIndex={1}
        totalFrames={3}
        onSeek={vi.fn()}
        startTimeS={30}
        endTimeS={90}
        onSeekTime={vi.fn()}
        playSpeed={1}
        onSpeedChange={vi.fn()}
        frame={{
          sim_time_s: 60,
          plan_version: 4,
          events: [
            { event_id: "e1", sim_time_s: 45, event_type: "target_maneuver", level: "tactical", entity_id: "T1", message: "目标转向" },
            { event_id: "e2", sim_time_s: 50, event_type: "prediction_revision", level: "tactical", entity_id: "T1", message: "预测修订" },
            { event_id: "e3", sim_time_s: 60, event_type: "region_activation", level: "tactical", entity_id: "T1:cell:1", message: "区域激活" },
            { event_id: "e4", sim_time_s: 65, event_type: "handoff", level: "tactical", entity_id: "T1:cell:2", message: "接力" },
            { event_id: "e5", sim_time_s: 70, event_type: "plan_revision", level: "strategic", entity_id: "T1", message: "方案修订" },
            { event_id: "e6", sim_time_s: 72, event_type: "degradation", level: "tactical", entity_id: "T1:cell:3", message: "质量降级" },
            { event_id: "e7", sim_time_s: 75, event_type: "expert_confirmation", level: "strategic", entity_id: null, message: "专家确认" },
          ],
          metrics: [
            { metric_id: "coverage", label: "coverage_ratio", value: 0.82, unit: "%", threshold: null, window_s: 30, series: [], reason: "group_quality_proxy" },
            { metric_id: "quality", label: "tracking_quality", value: 0.91, unit: "%", threshold: null, window_s: 30, series: [], reason: "region_telemetry" },
            { metric_id: "deviation", label: "target_deviation_m", value: 12, unit: "m", threshold: null, window_s: 30, series: [] },
            { metric_id: "latency", label: "handoff_latency_s", value: 8, unit: "s", threshold: null, window_s: 30, series: [] },
          ],
        } as never}
        markers={[
          { frameIndex: 1, timeS: 60, type: "region_activation", label: "区域激活" },
        ]}
      />,
    );

    expect(screen.getByText("覆盖 82% · 质量 91% · 偏差 12m")).toBeInTheDocument();
    expect(screen.getByText("接力时延 8s · 响应修订 v4")).toBeInTheDocument();
    expect(screen.getByText("代理指标")).toBeInTheDocument();
    expect(screen.getByText("60s")).toBeInTheDocument();
    ["目标机动", "预测修订", "区域激活", "接力", "方案修订", "降级", "专家确认"].forEach((label) => {
      expect(screen.getByTitle(label)).toBeInTheDocument();
    });
  });

  it("offers only the supported replay speeds", () => {
    const onSpeedChange = vi.fn();
    render(
      <PlaybackBar
        visible
        isPlaying={false}
        onPlayPause={vi.fn()}
        frameIndex={0}
        totalFrames={2}
        onSeek={vi.fn()}
        startTimeS={0}
        endTimeS={5}
        onSeekTime={vi.fn()}
        playSpeed={1}
        onSpeedChange={onSpeedChange}
        frame={{ sim_time_s: 0 } as never}
        markers={[]}
      />,
    );

    const speed = screen.getByRole("combobox", { name: "回放速度" });
    expect(screen.getByRole("option", { name: "1x" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "4x" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "10x" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "20x" })).not.toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "100x" })).not.toBeInTheDocument();

    fireEvent.change(speed, { target: { value: "10" } });

    expect(onSpeedChange).toHaveBeenCalledWith(10);
  });
});
