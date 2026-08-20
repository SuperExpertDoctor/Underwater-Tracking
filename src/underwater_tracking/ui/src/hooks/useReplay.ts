import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { OperationalFrame } from "../types/frames";
import { getReplayDelayMs, loadReplayRange } from "./replayApi";

const MARKER_EVENT_TYPES = new Set([
  "target_found", "target_added", "plan_decision", "plan_commit", "plan_committed", "target_lost",
  "target_maneuver", "target_maneuver_detected", "adversary_maneuver", "intent_change_confirmed", "state_changed", "prediction_revision", "prediction_revised",
  "region_activation", "region_activated", "handoff", "handoff_ready", "region_handoff", "plan_revision", "plan_revised",
  "degradation", "quality_degraded", "quality_warning", "quality_critical", "expert_confirmation", "expert_confirmed", "directive_applied",
]);

const MARKER_LABELS: Record<string, string> = {
  target_found: "目标发现", target_added: "目标发现", target_lost: "目标丢失", target_maneuver: "目标机动", target_maneuver_detected: "目标机动", adversary_maneuver: "目标机动", intent_change_confirmed: "目标机动", state_changed: "目标机动",
  prediction_revision: "预测修订", prediction_revised: "预测修订", region_activation: "区域激活", region_activated: "区域激活",
  handoff: "区域接力", handoff_ready: "区域接力", region_handoff: "区域接力", plan_decision: "方案修订", plan_commit: "方案修订", plan_committed: "方案修订", plan_revision: "方案修订", plan_revised: "方案修订",
  degradation: "跟踪降级", quality_degraded: "跟踪降级", quality_warning: "跟踪降级", quality_critical: "跟踪降级", expert_confirmation: "专家确认", expert_confirmed: "专家确认", directive_applied: "专家确认",
};

export interface ReplayMarker {
  frameIndex: number;
  timeS: number;
  type: string;
  label?: string;
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
    try {
      const { frames: next } = await loadReplayRange(fetch, startS, endS);
      framesRef.current = next;
      setFrames(next);
      setIndex(0);
    } catch (cause) {
      framesRef.current = [];
      setFrames([]);
      setIndex(0);
      setError(cause instanceof Error ? cause.message : "无法读取该时间范围的回放");
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

  const seekTime = useCallback((nextTimeS: number) => {
    if (!Number.isFinite(nextTimeS) || framesRef.current.length === 0) return;
    let nearestIndex = 0;
    let nearestDistance = Number.POSITIVE_INFINITY;
    framesRef.current.forEach((candidate, candidateIndex) => {
      const distance = Math.abs(candidate.sim_time_s - nextTimeS);
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearestIndex = candidateIndex;
      }
    });
    setIndex(nearestIndex);
  }, []);

  useEffect(() => {
    if (!enabled || !isPlaying || frames.length === 0) return undefined;
    const current = framesRef.current[index];
    const next = framesRef.current[index + 1];
    if (!current || !next) {
      setIsPlaying(false);
      return undefined;
    }
    const timer = window.setTimeout(() => {
      setIndex((current) => {
        if (current >= framesRef.current.length - 1) {
          setIsPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, getReplayDelayMs(current, next, speed));
    return () => window.clearTimeout(timer);
  }, [enabled, frames.length, index, isPlaying, speed]);

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
        result.push({ frameIndex, timeS: frame.sim_time_s, type: event.event_type, label: MARKER_LABELS[event.event_type] ?? event.event_type });
      });
      frame.plan_timeline?.forEach((item) => {
        const key = `plan-timeline:${item.adjustment_id}`;
        if (seen.has(key)) return;
        seen.add(key);
        result.push({ frameIndex, timeS: item.sim_time_s, type: "plan_revision", label: item.plan ? `方案修订 v${item.plan.version}` : "方案修订" });
      });
    });
    return result;
  }, [frames]);

  const startTimeS = frames[0]?.sim_time_s ?? 0;
  const endTimeS = frames.at(-1)?.sim_time_s ?? startTimeS;

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
    seekTime,
    startTimeS,
    endTimeS,
    isPlaying,
    setIsPlaying,
    speed,
    setSpeed,
    loading,
    error,
    markers,
  };
}
