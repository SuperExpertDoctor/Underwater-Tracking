import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { MOCK_FRAMES } from "./mockFrames";
import type { ReplayMarker } from "../hooks/useReplay";

const MARKER_EVENT_TYPES = new Set([
  "target_found", "region_entry_confirmed", "passive_track_started", "handoff_ready",
  "handoff", "region_replacement", "dedicated_release_pending", "estimate_expired",
]);

const MARKER_LABELS: Record<string, string> = {
  target_found: "目标发现",
  region_entry_confirmed: "入区确认",
  passive_track_started: "被动跟踪",
  handoff_ready: "接力待确认",
  handoff: "接力完成",
  region_replacement: "替补部署",
  dedicated_release_pending: "专用释放待定",
  estimate_expired: "估计过期",
};

/** Replay-compatible deterministic mock; it shares the same frame objects as mock live. */
export default function useMockReplay(enabled: boolean) {
  const framesRef = useRef(MOCK_FRAMES);
  const [index, setIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  useEffect(() => {
    setIndex(0);
    setIsPlaying(false);
  }, [enabled]);

  const seek = useCallback((nextIndex: number) => {
    const upper = Math.max(0, framesRef.current.length - 1);
    setIndex(Math.max(0, Math.min(upper, Math.round(nextIndex) || 0)));
  }, []);

  const seekTime = useCallback((nextTimeS: number) => {
    if (!Number.isFinite(nextTimeS)) return;
    const nearest = framesRef.current.reduce((best, frame, frameIndex) =>
      Math.abs(frame.sim_time_s - nextTimeS) < Math.abs(framesRef.current[best].sim_time_s - nextTimeS)
        ? frameIndex
        : best, 0);
    setIndex(nearest);
  }, []);

  useEffect(() => {
    if (!enabled || !isPlaying || index >= framesRef.current.length - 1) {
      if (index >= framesRef.current.length - 1) setIsPlaying(false);
      return undefined;
    }
    const current = framesRef.current[index];
    const next = framesRef.current[index + 1];
    const delay = Math.max(120, ((next.sim_time_s - current.sim_time_s) * 40) / Math.max(0.25, speed));
    const timer = window.setTimeout(() => setIndex((value) => Math.min(value + 1, framesRef.current.length - 1)), delay);
    return () => window.clearTimeout(timer);
  }, [enabled, index, isPlaying, speed]);

  const markers = useMemo<ReplayMarker[]>(() => {
    const seen = new Set<string>();
    return framesRef.current.flatMap((frame, frameIndex) => frame.events.flatMap((event) => {
      if (!MARKER_EVENT_TYPES.has(event.event_type)) return [];
      const key = `${event.event_id}:${event.event_type}`;
      if (seen.has(key)) return [];
      seen.add(key);
      return [{ frameIndex, timeS: frame.sim_time_s, type: event.event_type, label: MARKER_LABELS[event.event_type] ?? event.event_type }];
    }));
  }, []);

  const loadRange = useCallback(async (_startS = 0, _endS?: number) => {
    setIndex(0);
    setIsPlaying(false);
  }, []);

  return {
    files: ["mock://three-uuv-seed-42"],
    selectedFile: "mock://three-uuv-seed-42",
    load: loadRange,
    loadRange,
    frames: enabled ? framesRef.current : [],
    total: enabled ? framesRef.current.length : 0,
    frame: enabled ? framesRef.current[index] ?? null : null,
    index: enabled ? index : 0,
    seek,
    seekTime,
    startTimeS: enabled ? framesRef.current[0]?.sim_time_s ?? 0 : 0,
    endTimeS: enabled ? framesRef.current.at(-1)?.sim_time_s ?? 0 : 0,
    isPlaying: enabled && isPlaying,
    setIsPlaying,
    speed,
    setSpeed,
    loading: false,
    error: "",
    markers: enabled ? markers : [],
  };
}

