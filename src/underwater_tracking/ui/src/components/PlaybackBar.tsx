import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import type { OperationalFrame } from "../types/frames";
import { formatSimTime } from "./RightSidebar";

const SPEEDS = [0.5, 1, 2, 4];
const EVENT_COLORS: Record<string, string> = { target_found: "#ff6f7f", target_added: "#ff6f7f", plan_commit: "#62e6a7", plan_committed: "#62e6a7", target_lost: "#f6b94a", plan_decision: "#b29cff" };

export interface ReplayMarker { frameIndex: number; type: string }

interface PlaybackBarProps {
  visible: boolean;
  isPlaying: boolean;
  onPlayPause: () => void;
  frameIndex: number;
  totalFrames: number;
  onSeek: (index: number) => void;
  playSpeed: number;
  onSpeedChange: (speed: number) => void;
  frame: OperationalFrame | null;
  markers: ReplayMarker[];
}

export default function PlaybackBar({ visible, isPlaying, onPlayPause, frameIndex, totalFrames, onSeek, playSpeed, onSpeedChange, frame, markers }: PlaybackBarProps) {
  if (!visible) return null;
  const disabled = totalFrames === 0;
  return <section className="playback-bar" aria-label="回放控制">
    <span className="replay-chip">REPLAY</span>
    <button className="transport-btn primary" onClick={onPlayPause} disabled={disabled} aria-label={isPlaying ? "暂停" : "播放"}>{isPlaying ? <Pause size={16} /> : <Play size={16} />}</button>
    <button className="transport-btn" onClick={() => onSeek(frameIndex - 1)} disabled={disabled || frameIndex === 0} aria-label="上一帧"><SkipBack size={15} /></button>
    <button className="transport-btn" onClick={() => onSeek(frameIndex + 1)} disabled={disabled || frameIndex >= totalFrames - 1} aria-label="下一帧"><SkipForward size={15} /></button>
    <div className="timeline-control"><div className="event-marks" aria-hidden="true">{markers.map((marker, index) => <i key={`${marker.frameIndex}-${marker.type}-${index}`} style={{ left: `${totalFrames > 1 ? marker.frameIndex / (totalFrames - 1) * 100 : 0}%`, background: EVENT_COLORS[marker.type] ?? "#7e9bb8" }} />)}</div><input type="range" min="0" max={Math.max(0, totalFrames - 1)} value={frameIndex} onChange={(event) => onSeek(Number(event.target.value))} disabled={disabled} aria-label="回放时间轴" /></div>
    <span className="playback-readout">{disabled ? "0 / 0" : `${frameIndex + 1} / ${totalFrames}`}</span>
    <span className="playback-readout time">{frame ? formatSimTime(frame.sim_time_s) : "--:--:--"}</span>
    <select className="speed-select" value={playSpeed} onChange={(event) => onSpeedChange(Number(event.target.value))} aria-label="回放速度">{SPEEDS.map((speed) => <option key={speed} value={speed}>{speed}x</option>)}</select>
  </section>;
}
