import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import type { OperationalFrame } from "../frameTypes";

/** Plan-specified playback speeds (Task 7 of the UI plan). */
const SPEEDS = [0.5, 1, 2, 4];

const EVENT_COLORS: Record<string, string> = {
  target_found: "#DC2626",
  plan_decision: "#7C3AED",
  target_lost: "#EA580C",
};

export interface ReplayMarker {
  frameIndex: number;
  type: string;
}

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

/**
 * Replay transport bar (migrated from the reference project's
 * PlaybackBar; component boundary preserved).  MP4 export is not part of
 * this plan, so the export controls were dropped.  Full playback controls
 * land with the frame store in Task 7.
 */
export default function PlaybackBar({
  visible,
  isPlaying,
  onPlayPause,
  frameIndex,
  totalFrames,
  onSeek,
  playSpeed,
  onSpeedChange,
  frame,
  markers = [],
}: PlaybackBarProps) {
  if (!visible) return null;
  const disabled = totalFrames === 0;

  return (
    <section className="playback-bar" aria-label="回放控制">
      <button
        className="transport-btn primary"
        onClick={onPlayPause}
        disabled={disabled}
        title={isPlaying ? "暂停" : "播放"}
        aria-label={isPlaying ? "暂停" : "播放"}
      >
        {isPlaying ? <Pause size={17} /> : <Play size={17} />}
      </button>
      <button
        className="transport-btn"
        onClick={() => onSeek(frameIndex - 1)}
        disabled={disabled || frameIndex === 0}
        title="上一帧"
        aria-label="上一帧"
      >
        <SkipBack size={16} />
      </button>
      <button
        className="transport-btn"
        onClick={() => onSeek(frameIndex + 1)}
        disabled={disabled || frameIndex >= totalFrames - 1}
        title="下一帧"
        aria-label="下一帧"
      >
        <SkipForward size={16} />
      </button>
      <div className="timeline-control">
        <div className="event-marks" aria-hidden="true">
          {markers.map((marker, index) => (
            <i
              key={`${marker.frameIndex}-${marker.type}-${index}`}
              style={{
                left: `${totalFrames > 1 ? (marker.frameIndex / (totalFrames - 1)) * 100 : 0}%`,
                background: EVENT_COLORS[marker.type] || "#94A3B8",
              }}
            />
          ))}
        </div>
        <input
          type="range"
          min="0"
          max={Math.max(0, totalFrames - 1)}
          value={frameIndex}
          onChange={(event) => onSeek(Number(event.target.value))}
          disabled={disabled}
          aria-label="回放时间轴"
        />
      </div>
      <span className="playback-readout">帧 {disabled ? 0 : frameIndex + 1} / {totalFrames}</span>
      <span className="playback-readout time">{frame?.timestamp || "--:--:--"}</span>
      <select
        className="speed-select"
        value={playSpeed}
        onChange={(event) => onSpeedChange(Number(event.target.value))}
        aria-label="回放速度"
      >
        {SPEEDS.map((speed) => <option key={speed} value={speed}>{speed}x</option>)}
      </select>
    </section>
  );
}
