import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { OperationalFrame } from "../types/frames";
import { MAX_REPLAY_FRAMES, mergeReplayFrames } from "../state/frameStore";

const MARKER_EVENT_TYPES = new Set([
  "target_found", "target_added", "plan_decision", "plan_commit", "plan_committed", "target_lost",
]);

export interface ReplayMarker {
  frameIndex: number;
  type: string;
}

interface ReplayResponse {
  frames: OperationalFrame[];
  count: number;
}

export default function useReplay(enabled: boolean) {
  const [frames, setFrames] = useState<OperationalFrame[]>([]);
  const [index, setIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const framesRef = useRef<OperationalFrame[]>([]);

  const loadRange = useCallback(async (startS = 0, endS?: number) => {
    setLoading(true);
    setError("");
    setIsPlaying(false);
    const params = new URLSearchParams({ start_s: String(Math.max(0, startS)) });
    if (endS !== undefined) params.set("end_s", String(Math.max(startS, endS)));
    try {
      const response = await fetch(`/api/replay?${params.toString()}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = (await response.json()) as ReplayResponse;
      const next = mergeReplayFrames([], payload.frames ?? [], MAX_REPLAY_FRAMES);
      framesRef.current = next;
      setFrames(next);
      setIndex(0);
    } catch {
      framesRef.current = [];
      setFrames([]);
      setIndex(0);
      setError("无法读取该时间范围的回放");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (enabled) void loadRange();
  }, [enabled, loadRange]);

  const seek = useCallback((nextIndex: number) => {
    const upper = Math.max(0, framesRef.current.length - 1);
    setIndex(Math.max(0, Math.min(upper, Math.round(nextIndex) || 0)));
  }, []);

  useEffect(() => {
    if (!enabled || !isPlaying || frames.length === 0) return undefined;
    const timer = window.setInterval(() => {
      setIndex((current) => {
        if (current >= framesRef.current.length - 1) {
          setIsPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 1000 / Math.max(0.25, speed));
    return () => window.clearInterval(timer);
  }, [enabled, frames.length, isPlaying, speed]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!enabled) return;
      const tag = (event.target as HTMLElement | null)?.tagName ?? "";
      if (/INPUT|SELECT|TEXTAREA/.test(tag)) return;
      if (event.code === "Space") {
        event.preventDefault();
        setIsPlaying((current) => !current);
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        seek(index - 1);
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        seek(index + 1);
      } else if (/^[0-9]$/.test(event.key)) {
        seek((Number(event.key) / 10) * Math.max(0, framesRef.current.length - 1));
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [enabled, index, seek]);

  const markers = useMemo(() => {
    const seen = new Set<string>();
    const result: ReplayMarker[] = [];
    frames.forEach((frame, frameIndex) => {
      frame.events.forEach((event) => {
        if (!MARKER_EVENT_TYPES.has(event.event_type)) return;
        const key = `${event.event_id}:${event.event_type}`;
        if (seen.has(key)) return;
        seen.add(key);
        result.push({ frameIndex, type: event.event_type });
      });
    });
    return result;
  }, [frames]);

  return {
    files: [] as string[],
    selectedFile: "",
    load: loadRange,
    loadRange,
    frames,
    total: frames.length,
    frame: frames[index] ?? null,
    index,
    seek,
    isPlaying,
    setIsPlaying,
    speed,
    setSpeed,
    loading,
    error,
    markers,
  };
}
