import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import type { OperationalFrame } from "../types/frames";
import { formatSimTime } from "./RightSidebar";

const SPEEDS = [1, 4, 10];
const EVENT_COLORS: Record<string, string> = {
  target_found: "#ff6f7f", target_added: "#ff6f7f", target_lost: "#f6b94a",
  target_maneuver: "#ff6f7f", adversary_maneuver: "#ff6f7f", intent_change_confirmed: "#ff6f7f", state_changed: "#ff6f7f",
  plan_commit: "#62e6a7", plan_committed: "#62e6a7", plan_revision: "#62e6a7", plan_decision: "#62e6a7",
  prediction_revision: "#b29cff", region_activation: "#37b8bd", handoff: "#f6b94a",
  degradation: "#ff9e72", expert_confirmation: "#e7c25b", directive_applied: "#e7c25b",
  target_intent_change_suspected: "#d88b16", imm_motion_mode_changed: "#687f92", target_intent_changed: "#cc3f4d",
};

const EVENT_LABELS: Record<string, string> = {
  target_found: "目标发现", target_added: "目标发现", target_lost: "目标丢失",
  target_maneuver: "目标机动", adversary_maneuver: "目标机动", target_maneuver_detected: "目标机动", intent_change_confirmed: "目标机动", state_changed: "目标机动",
  prediction_revision: "预测修订", prediction_revised: "预测修订",
  region_activation: "区域激活", region_activated: "区域激活",
  handoff: "接力", handoff_ready: "接力", region_handoff: "接力",
  plan_commit: "方案修订", plan_committed: "方案修订", plan_decision: "方案修订", plan_revision: "方案修订", plan_revised: "方案修订",
  degradation: "降级", quality_degraded: "降级", quality_warning: "降级", quality_critical: "降级",
  expert_confirmation: "专家确认", directive_applied: "专家确认", expert_confirmed: "专家确认",
  target_intent_change_suspected: "预测分歧", imm_motion_mode_changed: "IMM 模式变化", target_intent_changed: "意图确认",
};

export interface ReplayMarker { frameIndex: number; timeS: number; type: string; label?: string }

interface PlaybackBarProps {
  visible: boolean;
  isPlaying: boolean;
  onPlayPause: () => void;
  frameIndex: number;
  totalFrames: number;
  onSeek: (index: number) => void;
  startTimeS: number;
  endTimeS: number;
  onSeekTime: (timeS: number) => void;
  playSpeed: number;
  onSpeedChange: (speed: number) => void;
  frame: OperationalFrame | null;
  markers: ReplayMarker[];
}

export default function PlaybackBar({ visible, isPlaying, onPlayPause, frameIndex, totalFrames, onSeek, startTimeS, endTimeS, onSeekTime, playSpeed, onSpeedChange, frame, markers }: PlaybackBarProps) {
  if (!visible) return null;
  const disabled = totalFrames === 0;
  const timelineStart = Number.isFinite(startTimeS) ? startTimeS : 0;
  const timelineEnd = Math.max(timelineStart, Number.isFinite(endTimeS) ? endTimeS : timelineStart);
  const currentTime = frame?.sim_time_s ?? timelineStart;
  const timelineMarkers = [...markers];
  const markerKeys = new Set(timelineMarkers.map((marker) => `${marker.timeS}:${marker.type}`));
  frame?.events?.forEach((event) => {
    if (!EVENT_LABELS[event.event_type] || markerKeys.has(`${event.sim_time_s}:${event.event_type}`)) return;
    markerKeys.add(`${event.sim_time_s}:${event.event_type}`);
    timelineMarkers.push({ frameIndex, timeS: event.sim_time_s, type: event.event_type, label: EVENT_LABELS[event.event_type] });
  });
  return <section className="playback-bar" aria-label="回放控制">
    <span className="replay-chip">REPLAY</span>
    <button className="transport-btn primary" onClick={onPlayPause} disabled={disabled} aria-label={isPlaying ? "暂停" : "播放"}>{isPlaying ? <Pause size={16} /> : <Play size={16} />}</button>
    <button className="transport-btn" onClick={() => onSeek(frameIndex - 1)} disabled={disabled || frameIndex === 0} aria-label="上一帧"><SkipBack size={15} /></button>
    <button className="transport-btn" onClick={() => onSeek(frameIndex + 1)} disabled={disabled || frameIndex >= totalFrames - 1} aria-label="下一帧"><SkipForward size={15} /></button>
    <div className="timeline-control"><div className="event-marks" aria-hidden="true">{timelineMarkers.map((marker, index) => <i key={`${marker.frameIndex}-${marker.type}-${index}`} title={marker.label ?? EVENT_LABELS[marker.type] ?? marker.type} style={{ left: `${timelineEnd > timelineStart ? Math.max(0, Math.min(100, (marker.timeS - timelineStart) / (timelineEnd - timelineStart) * 100)) : 0}%`, background: EVENT_COLORS[marker.type] ?? "#7e9bb8" }} />)}</div><input type="range" min={timelineStart} max={timelineEnd} step="1" value={Math.min(timelineEnd, Math.max(timelineStart, currentTime))} onChange={(event) => onSeekTime(Number(event.target.value))} disabled={disabled} aria-label="Replay timeline" aria-valuetext={formatSimTime(currentTime)} /><div className="timeline-scale" aria-hidden="true"><span>{formatSimTime(timelineStart)}</span><span>{formatSimTime(timelineEnd)}</span></div></div>
    <span className="playback-readout time">{frame ? `${frame.sim_time_s}s` : "—"}</span>
    <select className="speed-select" value={playSpeed} onChange={(event) => onSpeedChange(Number(event.target.value))} aria-label="回放速度">{SPEEDS.map((speed) => <option key={speed} value={speed}>{speed}x</option>)}</select>
    {frame && <TrackingEffectSummary frame={frame} />}
  </section>;
}

function TrackingEffectSummary({ frame }: { frame: OperationalFrame }) {
  const coverage = findMetric(frame, ["coverage", "coverage_ratio"]);
  const quality = findMetric(frame, ["quality", "tracking_quality"])
    ?? frame.target_estimates?.[0]?.quality.quality_score ?? null;
  const deviation = findMetric(frame, ["target_deviation", "deviation", "tracking_error"]);
  const latency = findMetric(frame, ["handoff_latency", "relay_latency"]);
  const proxy = (frame.metrics ?? []).some((metric) => /proxy/i.test(metric.reason ?? ""))
    || Object.values(frame.regional_plans ?? {}).some((plan) => plan.regions.some((region) => region.effect.quality_source === "group_quality_proxy"));
  return <div className="playback-effects" aria-label="跟踪效果">
    <span>{`覆盖 ${formatPercent(coverage)} · 质量 ${formatPercent(quality)} · 偏差 ${formatValue(deviation, "m")}`}</span>
    <span>{`接力时延 ${formatValue(latency, "s")} · 响应修订 v${frame.plan_version}`}</span>
    {proxy && <small>代理指标</small>}
  </div>;
}

function findMetric(frame: OperationalFrame, needles: string[]) {
  const metric = (frame.metrics ?? []).find((candidate) => needles.some((needle) => `${candidate.metric_id} ${candidate.label}`.toLowerCase().includes(needle)));
  return metric?.value ?? null;
}

function formatPercent(value: number | null) {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${Math.round((value <= 1 ? value * 100 : value))}%`;
}

function formatValue(value: number | null, unit: string) {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${Number.isInteger(value) ? value : value.toFixed(1)}${unit}`;
}
